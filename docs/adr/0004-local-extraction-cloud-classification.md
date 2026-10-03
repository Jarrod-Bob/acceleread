# Extraction stays local; classification may go to the cloud

Every Extraction step (text layer, OCR, Section detection) runs on the user's machine, and Extraction never requires a network connection. Classification uses Jev by default, and Jev is cloud-only, with no self-hosting option. So Document text leaves the machine at that step, and we accept that for v0 because Jev is several times cheaper than LLM alternatives. The boundary is deliberate: cloud OCR (Mistral, Textract) is banned in v0, even though it would be easy to add.

## Consequences

- Extraction-time judgments that use the Classifier (Section Verification, and the per-Page "real words?" check in the OCR rule) run only when a judgment-capable Classifier is configured. Without one they are skipped, the Record says so, and Extraction falls back to rules alone.
- Jev's zero data retention is enterprise-only, and the TypeSafe SDK's debug logging dumps request bodies unredacted. acceleread keeps that logging off by default and documents where Document text goes.
- A self-hosted Classifier can be plugged in through the Classifier seam for users who can't send text out. Self-hosted *models* remain out of scope.
- Details: [Should Jobs default to quality or fast extraction?](https://github.com/Jarrod-Bob/acceleread/issues/11), [When should the fast extraction path send a Page to OCR?](https://github.com/Jarrod-Bob/acceleread/issues/12)
