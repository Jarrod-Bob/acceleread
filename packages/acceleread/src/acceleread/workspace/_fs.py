# SPDX-License-Identifier: Apache-2.0
"""Detect network filesystems, which SQLite WAL cannot run on (ADR 0010)."""

import subprocess
import sys
from pathlib import Path

NETWORK_FS_TYPES = frozenset(
    {"nfs", "nfs3", "nfs4", "smbfs", "smb", "smb2", "smb3", "cifs", "afpfs", "webdav", "9p"}
    | {"sshfs", "fuse.sshfs", "ceph", "glusterfs", "fuse.glusterfs", "lustre"}
)


def is_network_fs(fs_type: str) -> bool:
    return fs_type.lower() in NETWORK_FS_TYPES


def parse_mounts(text: str) -> list[tuple[str, str]]:
    """Parse Linux `/proc/mounts` or macOS `mount` output into (mountpoint, fstype) pairs."""
    mounts: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if " on " in line and line.rstrip().endswith(")"):  # macOS: dev on /mnt (smbfs, local)
            mountpoint = line.split(" on ", 1)[1].rsplit(" (", 1)[0]
            fs_type = line.rsplit("(", 1)[1].split(",")[0].strip(") ")
            mounts.append((mountpoint, fs_type))
        elif len(parts) >= 3:  # Linux: dev /mnt type opts ...
            mounts.append((parts[1], parts[2]))
    return mounts


def detect_filesystem_type(path: Path) -> str:
    """The filesystem type of the longest mountpoint containing `path`, or "unknown"."""
    try:
        if sys.platform == "linux":
            text = Path("/proc/mounts").read_text(encoding="utf-8")
        else:
            text = subprocess.run(["mount"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    target = path.resolve()
    best = ("", "unknown")
    for mountpoint, fs_type in parse_mounts(text):
        mount = Path(mountpoint)
        if (target == mount or mount in target.parents) and len(mountpoint) >= len(best[0]):
            best = (mountpoint, fs_type)
    return best[1]
