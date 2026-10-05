# SPDX-License-Identifier: Apache-2.0
"""Model weights for the `quality` Profile: pinned revisions, hashes and the Workspace layout.

`acceleread models fetch` downloads the weights Docling needs into `<workspace>/models/docling/`,
the directory Docling's `artifacts_path` reads, so extraction never downloads anything at runtime.
Each model is pinned to a revision and, where we know them, to file hashes; every file that lands
is hashed into `manifest.json`. With `ACCELEREAD_OFFLINE=1` fetching is an error (spec §2).

Nothing here imports Docling at module level: the base install has neither it nor torch. The real
downloader imports it when it runs.
"""

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from acceleread.languages import offline

MANIFEST = "manifest.json"
HF_CACHE_DIR = ".cache"  # huggingface_hub's bookkeeping, not model files


class ModelFetchError(Exception):
    """The models could not be fetched or did not match their pins."""


@dataclass(frozen=True)
class ModelPin:
    """One model: where it comes from, at which revision, and the hash of each file we pin.

    `files` maps a path inside `folder` to its SHA-256. An empty mapping means the whole
    snapshot is fetched and recorded, but not pinned file by file.
    """

    name: str
    source: Literal["hf", "rapidocr"]
    folder: str
    revision: str
    label: str = ""
    repo_id: str | None = None
    files: Mapping[str, str] = field(default_factory=dict)
    optional: bool = False


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
        revision="v2.3.0",
        optional=True,
    ),
)


class Downloader(Protocol):
    def hf(self, repo_id: str, revision: str, local_dir: Path, files: list[str]) -> None:
        """Download `files` (all of the repo when empty) at `revision` into `local_dir`."""

    def rapidocr(self, local_dir: Path) -> None:
        """Download the PP-OCR latin ONNX models into `local_dir`."""


class DoclingDownloader:
    """The real downloader: Hugging Face for weights, RapidOCR's own registry for PP-OCR."""

    def hf(self, repo_id: str, revision: str, local_dir: Path, files: list[str]) -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            allow_patterns=files or None,
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


def _files(root: Path, pin: ModelPin) -> list[Path]:
    folder = root / pin.folder
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and HF_CACHE_DIR not in p.relative_to(folder).parts
    )


def _verified(root: Path, pin: ModelPin) -> bool:
    """Every pinned file is present with the pinned hash (and something is there, if none are)."""
    folder = root / pin.folder
    if pin.files:
        return all(
            (folder / rel).is_file() and _sha256(folder / rel) == digest
            for rel, digest in pin.files.items()
        )
    return bool(_files(root, pin))


def _fetch_one(root: Path, pin: ModelPin, downloader: Downloader) -> dict[str, Any]:
    folder = root / pin.folder
    if pin.source == "hf":
        assert pin.repo_id is not None
        downloader.hf(pin.repo_id, pin.revision, folder, list(pin.files))
    else:
        downloader.rapidocr(folder)
    shutil.rmtree(folder / HF_CACHE_DIR, ignore_errors=True)
    for rel, digest in pin.files.items():
        path = folder / rel
        if not path.is_file() or _sha256(path) != digest:
            path.unlink(missing_ok=True)
            raise ModelFetchError(f"{pin.name}: {rel} is missing or does not match its pinned hash")
    return {
        "revision": pin.revision,
        "files": {str(p.relative_to(root)): _sha256(p) for p in _files(root, pin)},
    }


def fetch_models(
    workspace: Path,
    *,
    tables: bool = False,
    pins: Sequence[ModelPin] | None = None,
    downloader: Downloader | None = None,
) -> dict[str, Any]:
    """Install the pinned models into the Workspace and write `manifest.json`.

    Models already present with the pinned hashes aren't downloaded again. A downloaded file that
    doesn't match its pin is deleted and fails the fetch, with no manifest written.
    """
    if offline():
        raise ModelFetchError("ACCELEREAD_OFFLINE=1 forbids downloading models; run without it")
    root = models_dir(workspace)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    loader = downloader
    for pin in PINS if pins is None else pins:
        installed = _verified(root, pin)
        if pin.optional and not tables and not installed:
            continue
        if installed:
            manifest[pin.name] = {
                "revision": pin.revision,
                "files": {str(p.relative_to(root)): _sha256(p) for p in _files(root, pin)},
            }
            continue
        loader = loader or default_downloader()
        manifest[pin.name] = _fetch_one(root, pin, loader)
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def models_status(workspace: Path, pins: Sequence[ModelPin] | None = None) -> dict[str, bool]:
    """Whether each model is installed, by name. A cheap check: presence, not hashes."""
    root = models_dir(workspace)
    status = {}
    for pin in PINS if pins is None else pins:
        folder = root / pin.folder
        if pin.files:
            status[pin.name] = all((folder / rel).is_file() for rel in pin.files)
        else:
            status[pin.name] = bool(_files(root, pin))
    return status
