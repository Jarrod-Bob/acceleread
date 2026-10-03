# Thin Classifier seam; planning and Escalation live above it

A Classifier implements one call: answer a set of Judgments (Noul, Score, Choice) about one state, and declare its `capabilities` (kinds supported, max Choice options, token budget, chars-per-token estimate). Everything else lives in one shared planner above the seam: grouping Judgments by the Sections they read, fitting state to the budget (head+tail, proportional truncation), turning the Taxonomy into a Choice, Coverage, caching, and Escalation. We chose this over a thick `classify(record, taxonomy, questions)` interface because those rules are acceleread's decisions, not a Classifier's. A thick seam would make Jev, the LLM escalation target and any future local model each re-implement them, and drift apart. It also lets Extraction ask a single Noul (Section Verification, the OCR rule's "real words?" check) through the same call.

## Consequences

- Adding a Classifier means implementing one async call plus `capabilities`, and the planner adapts. For example, an LLM's larger budget means escalated Judgments read untruncated Sections.
- A Classifier can't apply its own smarter document handling (e.g. native PDF input) without a planner change.
- Escalation re-plans the same Judgments against the target Classifier's `capabilities`. The first-pass result is kept beside the escalated one, and escalated Judgments carry no probabilities rather than invented ones.
- Details: [What does the Classifier seam look like, and when does it escalate?](https://github.com/Jarrod-Bob/acceleread/issues/8)
