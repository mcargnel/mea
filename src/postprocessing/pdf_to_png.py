# /// script
# requires-python = ">=3.11"
# dependencies = ["pymupdf"]
# ///
"""Convert all PDFs in a directory to PNGs.

Usage:
    uv run src/postprocessing/pdf_to_png.py                  # output/ at 300 dpi
    uv run src/postprocessing/pdf_to_png.py -d some/dir
    uv run src/postprocessing/pdf_to_png.py -d output --dpi 200
"""

import argparse
from pathlib import Path

import fitz


def convert(pdf_path: Path, dpi: int) -> list[Path]:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    outputs = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        suffix = "" if doc.page_count == 1 else f"_p{i + 1}"
        out = pdf_path.with_name(f"{pdf_path.stem}{suffix}.png")
        pix.save(out)
        outputs.append(out)
    doc.close()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDFs in a directory to PNGs.")
    parser.add_argument("-d", "--dir", default="output/empirical", type=Path)
    parser.add_argument("--dpi", default=300, type=int)
    args = parser.parse_args()

    pdfs = sorted(args.dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {args.dir}")
        return

    for pdf in pdfs:
        outs = convert(pdf, args.dpi)
        for out in outs:
            print(f"{pdf.name} -> {out.name}")


if __name__ == "__main__":
    main()
