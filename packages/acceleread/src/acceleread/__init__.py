# SPDX-License-Identifier: Apache-2.0
"""acceleread: turn PDFs and HTML into structured, classified Document Records."""

from importlib.metadata import version

__version__ = version("acceleread")

from acceleread.models import DocumentRecord, JobSpec, Question, QuestionSet, Taxonomy
from acceleread.pipeline import run
from acceleread.validate import resolve, validate

__all__ = [
    "DocumentRecord",
    "JobSpec",
    "Question",
    "QuestionSet",
    "Taxonomy",
    "__version__",
    "resolve",
    "run",
    "validate",
]
