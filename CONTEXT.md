# acceleread

Turns documents into structured, classified records quickly and cheaply, for people triaging reading and for ML/LLM pipelines that need ingestion at scale.

## Ingestion

**Document**:
One source file submitted for ingestion (a PDF or HTML file in v0). An exhibit filed alongside a filing is its own Document.
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
The structured output of ingesting one Document: its extracted text, Pages, Sections, metadata, Classification and Answers. Every Document yields exactly one, even when ingestion fails.
_Avoid_: Result, output, doc

**Coverage**:
The part of a Document a single Classification or Answer actually read (which Sections or Pages, and whether it was truncated).
_Avoid_: Context, window, scope

**Job**:
A batch of Documents ingested together, judged against at most one Taxonomy and any number of Questions (at least one of either).
_Avoid_: Run, batch, task

## Classification

**Taxonomy**:
The user-defined, flat set of Categories a Job classifies against.
_Avoid_: Labels, schema, category list

**Category**:
One named, described option within a Taxonomy.
_Avoid_: Label, class, tag

**Classifier**:
Anything that answers Judgments about a piece of text; Jev is the default.
_Avoid_: Model, engine

**Judgment**:
A single yes/no, graded or multiple-choice decision put to a Classifier, with its probabilities. Classifications, Answers and acceleread's own checks (such as Section Verification) are all Judgments.
_Avoid_: Question (reserved for user-defined Judgments), query, prompt

**Classification**:
The Judgment over a Job's Taxonomy for one Document: the chosen Category plus the full probability distribution and confidence.
_Avoid_: Prediction, label

**Question**:
A user-defined judgment asked of every Document in a Job beyond its Taxonomy: a yes/no condition, a graded score, or a choice among named options. It may name the Sections it reads.
_Avoid_: Check, metric, probe, signal

**Question Set**:
A named, versioned collection of Questions (and optionally a Taxonomy) that Jobs reuse, such as a "filing risk" set.
_Avoid_: Pack, template, preset

**Answer**:
The Judgment for one Question on one Document, with its probabilities.
_Avoid_: Result, score

**Escalation**:
Handing a low-confidence Classification or Answer to a more expensive Classifier (e.g. an LLM) or flagging it for a human.
_Avoid_: Fallback, retry
