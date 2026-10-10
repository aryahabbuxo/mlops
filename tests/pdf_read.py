from pathlib import Path
import pdfplumber

for f in sorted(Path("data/raw/cea/manual").glob("*.pdf")):
    print(f"\n=== {f.name} ===")
    with pdfplumber.open(f) as pdf:
        print("pages:", len(pdf.pages))
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for line in text.splitlines():
                if "karnataka" in line.lower():
                    print(f"  p{i+1}: {line}")
            if i == 0:
                print(text[:800])   # first-page header: look for units