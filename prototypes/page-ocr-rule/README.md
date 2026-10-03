# PROTOTYPE (throwaway): when should a Page go to OCR?

Answers [When should the fast extraction path send a Page to OCR?](https://github.com/Jarrod-Bob/acceleread/issues/12).
Open `page-ocr-rule.html` (self-contained) to tune the rule against 244 measured Pages.

Rebuild (needs the corpus PDFs in `corpus/data/pdf/` and a TypeSafe key in `.env`):

```sh
uv run --no-project --with pypdfium2 --with pillow --with pikepdf --with reportlab --with matplotlib prototypes/page-ocr-rule/make_tricky.py
uv run --no-project --with pypdfium2 --with pillow --with typesafe-sdk prototypes/page-ocr-rule/extract_signals.py
python3 prototypes/page-ocr-rule/build_demo.py
```

The rule lives in the `OcrRule` module inside `demo.template.html` and is the part worth lifting into acceleread.
