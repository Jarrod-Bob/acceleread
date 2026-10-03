# acceleread

Turns documents into structured, classified records quickly and cheaply, for people triaging reading and for ML/LLM pipelines that need ingestion at scale.

## Ingestion

**Document**:
One source file submitted for ingestion (a PDF or HTML file in v0).
_Avoid_: File, article, item

**Page**:
A single page of a Document, the unit at which extraction method is decided.

**Section**:
A named, contiguous part of a Document (e.g. a filing's "Risk Factors"), used to choose what text a Classifier sees.
_Avoid_: Chunk, part, segment

**Extraction**:
Turning a Document's Pages into text, either from the embedded text layer or by OCR.
_Avoid_: Parsing, reading

**Extraction Profile**:
The chosen way of turning a Job's PDF Pages into text, either `fast` or `quality`; HTML is unaffected.
_Avoid_: Mode, engine, pipeline

**Section Verification**:
A cheap Classifier judgment that confirms or rejects a detected Section before Questions rely on it.
_Avoid_: Validation, QA, check

**Document Record**:
The structured output of ingesting one Document: its extracted text, page metadata, and Classification.
_Avoid_: Result, output, doc

**Job**:
A batch of Documents ingested together against one Taxonomy.
_Avoid_: Run, batch, task

## Classification

**Taxonomy**:
The user-defined, flat set of Categories a Job classifies against.
_Avoid_: Labels, schema, category list

**Category**:
One named, described option within a Taxonomy.
_Avoid_: Label, class, tag

**Classifier**:
Anything that assigns a Classification to a Document Record given a Taxonomy; Jev is the default.
_Avoid_: Model, engine

**Classification**:
A Classifier's verdict on one Document: the chosen Category plus the full probability distribution and confidence.
_Avoid_: Prediction, label

**Question**:
A user-defined judgment asked of every Document in a Job beyond its Taxonomy, such as a yes/no condition or a graded score.
_Avoid_: Check, metric, probe, signal

**Answer**:
A Classifier's response to one Question for one Document, with its probabilities.
_Avoid_: Result, score

**Escalation**:
Handing a low-confidence Classification to a more expensive Classifier (e.g. an LLM) or a human.
_Avoid_: Fallback, retry
