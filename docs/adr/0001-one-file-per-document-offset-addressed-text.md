# One file per Document, text addressed by Document-level character offsets

A Document is exactly one source file. An EDGAR exhibit such as Exhibit 13 is its own Document, joined to its filing by a shared `source.group` (the accession number), not bundled into the filing's Document. Each Document Record holds one Document-level `text`. A PDF's Pages are character ranges into it, carrying their extraction provenance, and HTML Records have no Pages. Section spans and Coverage are character offsets into the same `text`. We chose this because one offset space works across PDF and HTML and keeps "one file in, one Record out" for pipelines. It also keeps Page to its glossary meaning: the unit where the extraction method is decided.

## Considered Options

- **Bundle a filing's primary file and exhibits into one Document.** This would let Sections span files and fix filers whose substance sits in Exhibit 13. It was rejected for v0 because Pages, Sections and offsets would all need a file dimension. Section Verification already flags pointer-only Sections, so the gap is visible.
- **Text stored per Page, with HTML as one pseudo-Page.** This was rejected because spans would need (page, offset) pairs, and a pseudo-Page breaks the meaning of Page.

## Consequences

- `text` is optional on export (`include_text=false`). Offsets stay valid, so a consumer can re-attach text from storage.
- Adding bundles later means a new `files[]` level above `text`. That is a major `schema_version` bump.
- Details: [What is the Document Record schema?](https://github.com/Jarrod-Bob/acceleread/issues/7)
