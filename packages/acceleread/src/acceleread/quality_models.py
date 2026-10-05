# SPDX-License-Identifier: Apache-2.0
"""Model weights for the `quality` Profile: pinned revisions, hashes and the Workspace layout.

`acceleread models fetch` downloads the weights Docling needs into `<workspace>/models/docling/`,
the directory Docling's `artifacts_path` reads, so extraction never downloads anything at runtime.
Every model is pinned to a commit and to the SHA-256 of each file.

A fetch is all or nothing. Each model is downloaded into a temporary directory beside its final
place, every hash is checked, and only then are the directories renamed into place and
`manifest.json` written (atomically, last). Any mismatch leaves the previous install and manifest
untouched. With `ACCELEREAD_OFFLINE=1` fetching is an error (spec §2).

Extraction asks `check_model` before it uses a weight: a file that is missing or doesn't match its
pin is never loaded. The answer is cached per process, keyed on the files' size and mtime, so a
worker does not re-hash 300 MB per Document.

Nothing here imports Docling at module level: the base install has neither it nor torch.
"""

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from acceleread.languages import offline

MANIFEST = "manifest.json"
HF_CACHE_DIR = ".cache"  # huggingface_hub's bookkeeping, not model files
TEMP_PREFIX = ".fetch-"

type ModelState = Literal["ok", "absent", "corrupt"]


class ModelFetchError(Exception):
    """The models could not be fetched or did not match their pins."""


@dataclass(frozen=True)
class ModelPin:
    """One model: where it comes from, at which revision, and the SHA-256 of every file we use."""

    name: str
    source: Literal["hf", "rapidocr"]
    folder: str
    revision: str
    files: Mapping[str, str]
    label: str = ""
    repo_id: str | None = None
    optional: bool = False

    def __post_init__(self) -> None:
        if self.source == "hf" and not self.repo_id:
            raise ValueError(f"model {self.name!r} comes from Hugging Face and needs a repo_id")


PINS: tuple[ModelPin, ...] = (
    ModelPin(
        name="layout",
        label="Docling layout",
        source="hf",
        folder="docling-project--docling-layout-heron",
        repo_id="docling-project/docling-layout-heron",
        revision="8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
        files={
            "model.safetensors": "00333a43451945aaf89db8ca9c0a17e75d1537c17db60fdb91aa95f4c7929e0c",
            "config.json": "fdea30805ce2f5666b147fca941dcdd27ad468e27d6ed21902207d3da056a97d",
            "preprocessor_config.json": (
                "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1"
            ),
        },
    ),
    ModelPin(
        name="ocr",
        label="RapidOCR (PP-OCR latin)",
        source="rapidocr",
        folder="RapidOcr",
        revision="rapidocr==3.9.2 PP-OCRv5 latin, onnxruntime",
        files={
            "ch_PP-OCRv5_det_mobile.onnx": (
                "4d97c44a20d30a81aad087d6a396b08f786c4635742afc391f6621f5c6ae78ae"
            ),
            "ch_ppocr_mobile_v2.0_cls_mobile.onnx": (
                "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"
            ),
            "latin_PP-OCRv5_rec_mobile.onnx": (
                "b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a"
            ),
        },
    ),
    ModelPin(
        name="tables",
        label="TableFormer (optional)",
        source="hf",
        folder="docling-project--docling-models",
        repo_id="docling-project/docling-models",
        revision="fc0f2d45e2218ea24bce5045f58a389aed16dc23",  # tag v2.3.0
        files={
            "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors": (
                "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9"
            ),
            "model_artifacts/tableformer/accurate/tm_config.json": (
                "984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd"
            ),
            "model_artifacts/tableformer/fast/tableformer_fast.safetensors": (
                "3119563aab5a7c96fda4d621119b63fd8806272b86c30936d15507616422f718"
            ),
            "model_artifacts/tableformer/fast/tm_config.json": (
                "dca6762508dddfae6d57d6cb4ef822c6000119dff0f3b6489db7413118c2622a"
            ),
        },
        optional=True,
    ),
)


class Downloader(Protocol):
    def hf(self, repo_id: str, revision: str, local_dir: Path, files: list[str]) -> None:
        """Download `files` at `revision` into `local_dir`."""

    def rapidocr(self, local_dir: Path) -> None:
        """Download the PP-OCR latin ONNX models into `local_dir`."""


class DoclingDownloader:
    """The real downloader: Hugging Face for weights, RapidOCR's own registry for PP-OCR."""

    def hf(self, repo_id: str, revision: str, local_dir: Path, files: list[str]) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id, revision=revision, local_dir=local_dir, allow_patterns=files
        )

    def rapidocr(self, local_dir: Path) -> None:
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

        RapidOcrModel.download_models(backend="onnxruntime", local_dir=local_dir, lang="latin")


def default_downloader() -> Downloader:
    return DoclingDownloader()


def models_dir(workspace: Path) -> Path:
    """Docling's `artifacts_path`: where `models fetch` puts the weights."""
    return workspace / "models" / "docling"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


_cache: dict[tuple[Any, ...], ModelState] = {}


def _signature(folder: Path, pin: ModelPin) -> tuple[Any, ...] | None:
    """Pin identity plus each file's size and mtime; None when a pinned file is missing."""
    stats = []
    for rel in sorted(pin.files):
        try:
            stat = (folder / rel).stat()
        except OSError:
            return None
        stats.append((rel, stat.st_size, stat.st_mtime_ns))
    return (str(folder), pin.name, pin.revision, tuple(sorted(pin.files.items())), tuple(stats))


def _state_of(folder: Path, pin: ModelPin) -> ModelState:
    key = _signature(folder, pin)
    if key is None:
        # Missing files: "absent" if nothing of the model is there, else damaged.
        return "corrupt" if any((folder / rel).exists() for rel in pin.files) else "absent"
    if key not in _cache:
        good = all(_sha256(folder / rel) == digest for rel, digest in pin.files.items())
        _cache[key] = "ok" if good else "corrupt"
    return _cache[key]


def check_model(workspace: Path, pin: ModelPin) -> ModelState:
    """Whether the Workspace has `pin`'s files, with the pinned hashes.

    `corrupt` covers a truncated, tampered or partly missing install. Cached per process.
    """
    return _state_of(models_dir(workspace) / pin.folder, pin)


def models_status(workspace: Path, pins: Sequence[ModelPin] | None = None) -> dict[str, bool]:
    """Whether each model is installed and verified, by name."""
    return {p.name: check_model(workspace, p) == "ok" for p in (PINS if pins is None else pins)}


def _entry(pin: ModelPin) -> dict[str, Any]:
    """The manifest entry for a verified model: its revision and every file's hash."""
    return {
        "revision": pin.revision,
        "files": {f"{pin.folder}/{rel}": digest for rel, digest in sorted(pin.files.items())},
    }


def _download(pin: ModelPin, target: Path, downloader: Downloader) -> None:
    if pin.source == "hf":
        assert pin.repo_id  # guaranteed by ModelPin
        downloader.hf(pin.repo_id, pin.revision, target, sorted(pin.files))
    else:
        downloader.rapidocr(target)
    shutil.rmtree(target / HF_CACHE_DIR, ignore_errors=True)
    for rel, digest in pin.files.items():
        path = target / rel
        if not path.is_file():
            raise ModelFetchError(f"{pin.name}: {rel} was not downloaded")
        if _sha256(path) != digest:
            raise ModelFetchError(f"{pin.name}: {rel} does not match its pinned hash")


def fetch_models(
    workspace: Path,
    *,
    tables: bool = False,
    pins: Sequence[ModelPin] | None = None,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    """Install the pinned models into the Workspace and write `manifest.json`.

    Models already installed with the pinned hashes aren't downloaded again, and an installed
    optional model is kept when `tables` is off. Everything that needs downloading is downloaded
    and verified before anything is replaced; a mismatch raises `ModelFetchError` and leaves the
    previous install and manifest as they were.
    """
    if offline():
        raise ModelFetchError("ACCELEREAD_OFFLINE=1 forbids downloading models; run without it")
    root = models_dir(workspace)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    staged: dict[str, tuple[ModelPin, Path]] = {}
    scratch = root / f"{TEMP_PREFIX}{uuid.uuid4().hex}"
    try:
        for pin in PINS if pins is None else pins:
            if _state_of(root / pin.folder, pin) == "ok":
                manifest[pin.name] = _entry(pin)
            elif not pin.optional or tables:
                target = scratch / pin.folder  # beside the final place: same filesystem
                _download(pin, target, downloader or default_downloader())
                staged[pin.name] = (pin, target)
                manifest[pin.name] = _entry(pin)
        for pin, target in staged.values():
            final = root / pin.folder
            old = scratch / f"old-{pin.folder}"
            if final.exists():
                final.rename(old)
            target.rename(final)
        text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / MANIFEST).write_text(text)
        os.replace(scratch / MANIFEST, root / MANIFEST)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return manifest
