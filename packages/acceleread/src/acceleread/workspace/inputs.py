# SPDX-License-Identifier: Apache-2.0
"""A Job's inputs: uploads and fetched URLs copied by content hash, local paths referenced."""

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import BaseModel


class InputChangedError(RuntimeError):
    """A referenced local file is missing or no longer matches the size and hash recorded."""


class InputRef(BaseModel):
    kind: Literal["copy", "path"]
    path: str
    sha256: str
    bytes: int


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class JobInputs:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def add_bytes(self, data: bytes) -> InputRef:
        """Copy an upload or fetched URL body into `inputs/`, named by its SHA-256.

        The stored path is relative to the Job directory, so a moved Workspace keeps working.
        """
        sha = hashlib.sha256(data).hexdigest()
        target = self.directory / sha
        if not target.exists():
            self.directory.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, target)  # atomic: a crash never leaves a half-written input
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
        return InputRef(
            kind="copy", path=f"{self.directory.name}/{sha}", sha256=sha, bytes=len(data)
        )

    def reference_path(self, path: Path) -> InputRef:
        """Record a local file's size and hash without copying it."""
        return InputRef(
            kind="path",
            path=str(path),
            sha256=_hash_file(path),
            bytes=path.stat().st_size,
        )

    def open(self, ref: InputRef) -> BinaryIO:
        """Open the input, failing if a referenced local file is missing or has changed."""
        if ref.kind == "copy":
            return (self.directory.parent / ref.path).open("rb")
        path = Path(ref.path)
        if ref.kind == "path":
            if not path.is_file():
                raise InputChangedError(f"input file is missing: {path}")
            if path.stat().st_size != ref.bytes or _hash_file(path) != ref.sha256:
                raise InputChangedError(f"input file has changed since submission: {path}")
        return path.open("rb")
