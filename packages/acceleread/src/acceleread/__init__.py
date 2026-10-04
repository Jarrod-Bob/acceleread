# SPDX-License-Identifier: Apache-2.0
"""acceleread: turn PDFs and HTML into structured, classified Document Records."""

from importlib.metadata import version

__version__ = version("acceleread")

from acceleread.models import DocumentRecord, JobSpec, Taxonomy
from acceleread.pipeline import run

__all__ = ["DocumentRecord", "JobSpec", "Taxonomy", "__version__", "run"]
