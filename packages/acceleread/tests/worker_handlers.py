# SPDX-License-Identifier: Apache-2.0
"""Handlers that misbehave on demand, for the worker manager's tests.

Workers import them by dotted path (`worker_handlers:crash`), so they must be module-level.
"""

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Report = Callable[[int, int], None]


def echo(payload: Any, report: Report) -> Any:
    return payload


def pid(payload: Any, report: Report) -> int:
    time.sleep(payload or 0)
    return os.getpid()


def omp_thread_limit(payload: Any, report: Report) -> str | None:
    return os.environ.get("OMP_THREAD_LIMIT")


def crash(payload: Any, report: Report) -> None:
    os._exit(7)


def crash_once(payload: Any, report: Report) -> int:
    """Crash the first time it runs (marker file absent), succeed on the retry."""
    marker = Path(payload)
    if not marker.exists():
        marker.write_text("crashed")
        os._exit(7)
    return os.getpid()


def hang(payload: Any, report: Report) -> None:
    time.sleep(3600)


def sleep_after_report(payload: Any, report: Report) -> str:
    """payload = (pages, ocr_pages, seconds): announce the Page counts, then work a while."""
    pages, ocr_pages, seconds = payload
    report(pages, ocr_pages)
    time.sleep(seconds)
    return "done"


def hog(payload: Any, report: Report) -> None:
    """Hold `payload` megabytes, then wait to be killed."""
    held = bytearray(int(payload) * 1024 * 1024)
    for i in range(0, len(held), 4096):
        held[i] = 1
    time.sleep(3600)


def fail(payload: Any, report: Report) -> None:
    raise ValueError("this Document is broken")


def _explode() -> None:
    raise RuntimeError("cannot be unpickled")


class Unloadable:
    """Pickles fine in the worker but raises when the parent unpickles it."""

    def __reduce__(self) -> tuple[Callable[[], None], tuple[()]]:
        return (_explode, ())


def unloadable(payload: Any, report: Report) -> Unloadable:
    return Unloadable()


def spawn_child_and_hang(payload: Any, report: Report) -> None:
    """Start a grandchild that outlives its parent unless the whole tree is killed."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)"])
    Path(payload).write_text(str(child.pid))
    time.sleep(3600)
