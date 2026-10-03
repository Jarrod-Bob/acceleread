# Server-rendered htmx UI on the shared service layer

The v0 UI is rendered on the server by FastAPI with Jinja2 templates and htmx (plus its SSE extension). It ships inside the wheel with its JS and CSS vendored, and `acceleread serve` serves it beside `/v1`. Its routes call the same in-process service layer that `/v1` calls, rather than making HTTP calls to `/v1` themselves. We chose this because the UI is a thin console (tables, filters, a drawer, a form, a review screen) in a Python-first library: a single-page app would add a Node toolchain to CI and packaging, and a second language for contributors, for interactions htmx's partial swaps already cover. "The UI is a client of the API" is kept as a rule rather than a transport: every UI action has a `/v1` equivalent, and the UI never gains an ability the API lacks.

## Considered Options

- **SPA (React or Svelte + Vite) consuming `/v1` JSON:** a literal API client with richer interactions, but it needs a Node build to produce wheel assets.
- **Python UI frameworks (NiceGUI, Streamlit, Reflex):** quick to write, but they own the server, state and styling. Streamlit's rerun model also fits live progress over 100k Documents badly.

## Consequences

- A test maps each UI route to its `/v1` route, so API completeness doesn't drift.
- The UI authenticates with a signed HttpOnly cookie obtained by entering the API token. `/v1` accepts either the cookie or the bearer token.
- Replacing the UI with an SPA later means rewriting the templates, but `/v1` doesn't change.
- Details: [How is the UI built and served?](https://github.com/Jarrod-Bob/acceleread/issues/16)
