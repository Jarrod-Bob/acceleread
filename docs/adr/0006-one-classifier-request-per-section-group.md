# One Classifier request per group of Questions reading the same Sections

Each Question (and the Taxonomy) declares the Sections it reads. For each Document, acceleread sends one Jev request per distinct set of Sections, and every Question reading that set shares the request. A 10-K asked about sector (`business`), outlook (`mdna`) and going concern plus material weakness (`controls` + `financial_statements`) therefore makes three requests, not one. Jev ingests a shared state once and charges extra questions only their own tokens, so one composed request holding every Section would be cheapest. We rejected it because Jev "suffers from context rot": accuracy falls as the state fills with text unrelated to the decision. In the corpus probe, an outlook Score that read Risk Factors instead of MD&A landed on "cautious" for Apple and JPMorgan.

## Considered Options

- **One composed request per Document:** the cheapest option and one call, but every Question sees every Section.
- **One request per Question:** the cleanest state, but it pays for the same Section text again for each Question (12× costlier than batching in TypeSafe's parallel-questions test).

## Consequences

- Requests per Document equal the number of distinct `reads` sets, typically 2–4 for a filing Question Set. Throughput planning (the rate-limit budgeting item on the map) must count requests, not Documents.
- A group over Jev's budget truncates each of its Sections head+tail in proportion, and Coverage records what each judgment actually read.
- Details: [How are Questions defined and attached to a Job?](https://github.com/Jarrod-Bob/acceleread/issues/15)
