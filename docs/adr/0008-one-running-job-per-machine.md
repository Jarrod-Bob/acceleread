# One running Job per machine; single-Document ingests jump the queue

A machine runs one Job at a time. Further Jobs wait in a FIFO, whether they were submitted by the CLI or the API. We chose this over running several Jobs at once because the scarce resources are machine-wide: the Classifier's rate limit (Jev's tokens/sec and requests/sec) and the OCR process pool. Parallel Jobs would only split that throughput, make every ETA wrong, and make a Job's cost and duration depend on what else happened to be running. The one exception is the synchronous single-Document ingest. It interleaves between the running Job's Documents rather than waiting behind it, because otherwise "classify this one URL now" could wait hours behind a 100k-Document Job.

## Considered Options

- **N concurrent Jobs sharing the limiter and pool:** fairer for many small Jobs, but no faster in total, and it needs a scheduler to divide throughput between them.
- **Priorities on the FIFO:** deferred. The queue can gain a priority field later without changing this decision.

## Consequences

- A small Job submitted behind a large one waits for it to finish. Cancel-then-resume is the escape hatch, and the Judgment cache makes resume free.
- The sync ingest is kept deliberately small (one Document, a cap on Pages needing OCR, a timeout) so that jumping the queue barely delays the running Job.
- Details: [What API and CLI surface does v0 expose?](https://github.com/Jarrod-Bob/acceleread/issues/9)
