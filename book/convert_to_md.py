#!/usr/bin/env python3
"""
Convert LaTeX thesis chapters to Markdown and DOCX.

Usage:
    python convert_to_md.py          # Generate Markdown + DOCX
    python convert_to_md.py --md     # Only Markdown
    python convert_to_md.py --docx   # Only DOCX from existing Markdown

Output is written to book/markdown/.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BOOK_DIR = Path(__file__).resolve().parent
BIB_FILE = BOOK_DIR / "references.bib"
IMAGES_DIR = BOOK_DIR / "images"
OUTPUT_DIR = BOOK_DIR / "markdown"
OUTPUT_IMAGES = OUTPUT_DIR / "images"

# Ordered list of (output_filename, chapter_title, tex_source)
CHAPTERS = [
    ("00_abstract.md", "Abstract", BOOK_DIR / "abstract.tex"),
    ("01_introduction.md", "Chapter 1: Introduction", BOOK_DIR / "chapters" / "introduction.tex"),
    ("02_difference_in_differences.md", "Chapter 2: Difference in Differences", BOOK_DIR / "chapters" / "chapter02.tex"),
    ("03_double_machine_learning.md", "Chapter 3: Double Machine Learning", BOOK_DIR / "chapters" / "chapter03.tex"),
    ("04_applications.md", "Chapter 4: Applications", BOOK_DIR / "chapters" / "chapter04.tex"),
    ("05_conclusion.md", "Chapter 5: Conclusion", BOOK_DIR / "chapters" / "conclusion.tex"),
    ("06_appendix.md", "Appendix", BOOK_DIR / "chapters" / "appendix.tex"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dirs():
    """Create output directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_IMAGES.mkdir(exist_ok=True)


def copy_and_convert_images():
    """Copy images to the output dir, converting PDFs to PNG."""
    if not IMAGES_DIR.exists():
        return

    for img in IMAGES_DIR.iterdir():
        if img.suffix.lower() == ".pdf":
            # Convert PDF to PNG using sips (macOS built-in)
            png_name = img.stem + ".png"
            dest = OUTPUT_IMAGES / png_name
            try:
                # First try sips
                subprocess.run(
                    ["sips", "-s", "format", "png", str(img), "--out", str(dest)],
                    check=True,
                    capture_output=True,
                )
                print(f"  Converted {img.name} → {png_name}")
            except subprocess.CalledProcessError:
                # Fallback: just copy the PDF
                shutil.copy2(img, OUTPUT_IMAGES / img.name)
                print(f"  Copied {img.name} (PDF conversion failed)")
        else:
            shutil.copy2(img, OUTPUT_IMAGES / img.name)
            print(f"  Copied {img.name}")


def preprocess_tex(tex_content: str) -> str:
    """
    Preprocess LaTeX content before passing to Pandoc:
    - Remove comments (lines starting with %)
    - Replace TikZ environments with a text placeholder
    - Fix image paths to point to the output images dir
    - Strip LaTeX-only formatting commands
    """
    # Remove full-line comments (but keep % inside text)
    lines = tex_content.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        cleaned.append(line)
    tex_content = "\n".join(cleaned)

    # Replace TikZ picture blocks with a placeholder
    # Match \begin{tikzpicture}...\end{tikzpicture}
    tikz_pattern = re.compile(
        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
        re.DOTALL,
    )
    tex_content = tikz_pattern.sub(
        r"\\textit{[Diagram: Directed Acyclic Graph showing the confounding structure. "
        r"Covariates X affect both treatment D (through m(X)) and outcome Y (through g(X)), "
        r"while the causal effect of interest is $\\theta$ (from D to Y).]}",
        tex_content,
    )

    # Replace \begin{figure}...\end{figure} that contained tikz
    # (already handled above, the figure wrapper remains and Pandoc handles it)

    # Fix image paths: images/foo.pdf → images/foo.png
    tex_content = re.sub(
        r"\\includegraphics(\[.*?\])?\{images/([^}]+)\.pdf\}",
        r"\\includegraphics\1{images/\2.png}",
        tex_content,
    )

    # Remove \thispagestyle, \vspace, centering commands that confuse Pandoc
    tex_content = re.sub(r"\\thispagestyle\{[^}]*\}", "", tex_content)
    tex_content = re.sub(r"\\vspace\{[^}]*\}", "", tex_content)
    tex_content = re.sub(r"\\vfill", "", tex_content)
    tex_content = re.sub(r"\\graphicspath\{[^}]*\}", "", tex_content)

    return tex_content


def wrap_as_document(tex_content: str, title: str) -> str:
    """
    Wrap chapter content in a minimal LaTeX document so Pandoc can parse it.
    """
    preamble = r"""\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
"""
    return f"""{preamble}
\\title{{{title}}}
\\begin{{document}}

{tex_content}

\\end{{document}}
"""


def convert_tex_to_md(tex_path: Path, title: str, output_path: Path):
    """Convert a single .tex file to .md using Pandoc."""
    tex_content = tex_path.read_text(encoding="utf-8")
    tex_content = preprocess_tex(tex_content)
    wrapped = wrap_as_document(tex_content, title)

    # Write to a temp file for Pandoc
    tmp_tex = OUTPUT_DIR / "_temp.tex"
    tmp_tex.write_text(wrapped, encoding="utf-8")

    try:
        cmd = [
            "pandoc",
            str(tmp_tex),
            "-f", "latex",
            "-t", "markdown",
            "--citeproc",
            f"--bibliography={BIB_FILE}",
            "--csl=",  # will be removed, use default
            "--wrap=none",
            "--columns=9999",
            "-o", str(output_path),
        ]
        # Remove empty --csl= arg
        cmd = [c for c in cmd if c != "--csl="]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BOOK_DIR),
        )

        if result.returncode != 0:
            print(f"  WARNING: Pandoc stderr for {tex_path.name}:")
            print(f"    {result.stderr[:500]}")

        print(f"  ✓ {tex_path.name} → {output_path.name}")

    finally:
        tmp_tex.unlink(missing_ok=True)


def postprocess_md(md_path: Path, title: str):
    """
    Post-process the generated Markdown to clean up Pandoc artifacts
    and produce clean, standard Markdown.
    """
    content = md_path.read_text(encoding="utf-8")

    # Remove Pandoc's auto-generated title block if present
    content = re.sub(r"^---\n.*?---\n", "", content, flags=re.DOTALL)

    # --- Demote headings: # → ##, ## → ###, etc. (chapter title will be #) ---
    lines = content.split("\n")
    demoted = []
    for line in lines:
        if re.match(r"^#{1,5}\s", line):
            demoted.append("#" + line)
        else:
            demoted.append(line)
    content = "\n".join(demoted)

    # --- Convert HTML <figure> + <img> blocks to standard Markdown ---
    # Pattern: <figure ...> <img src="..." ...> <figcaption>...</figcaption> </figure>
    def replace_html_figure(m):
        img_match = re.search(r'src="([^"]+)"', m.group(0))
        caption_match = re.search(
            r"<figcaption>(.*?)</figcaption>", m.group(0), re.DOTALL
        )
        if img_match:
            src = img_match.group(1)
            # Strip HTML tags from caption
            caption = ""
            if caption_match:
                caption = re.sub(r"<[^>]+>", "", caption_match.group(1)).strip()
            return f"![{caption}]({src})\n"
        # Fallback for figures without images (e.g., TikZ placeholders)
        text = re.sub(r"<[^>]+>", "", m.group(0)).strip()
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return f"\n*{text}*\n"

    content = re.sub(
        r"<figure[^>]*>.*?</figure>",
        replace_html_figure,
        content,
        flags=re.DOTALL,
    )

    # --- Clean up cross-reference attributes ---
    # Remove {reference-type="ref" reference="..."} and keep just the text
    content = re.sub(
        r'\[(\d+)\]\(#[^)]+\)\{reference-type="ref"\s+reference="[^"]+"\}',
        r"(\1)",
        content,
    )
    # Also handle Figure/Table references like [1](#fig:xxx){...}
    content = re.sub(
        r'\[([^\]]+)\]\(#[^)]+\)\{[^}]*reference-type="[^"]*"[^}]*\}',
        r"\1",
        content,
    )

    # --- Remove Pandoc fenced div blocks (::: {.class} ... :::) ---
    # Remove opening div markers like ::: center, :::: center, :::::: {#refs ...}
    content = re.sub(r"^:{2,}\s*\{[^}]*\}\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^:{2,}\s+\w+\s*$", "", content, flags=re.MULTILINE)
    # Remove closing div markers (lines that are only colons)
    content = re.sub(r"^:{2,}\s*$", "", content, flags=re.MULTILINE)

    # --- Clean up the references section ---
    # Format references as a clean "References" section
    # Remove Pandoc's div-based reference formatting
    content = re.sub(r"^:{2,}.*$", "", content, flags=re.MULTILINE)

    # --- Remove escaped backslash-quotes ---
    content = content.replace('\\"', '"')

    # --- Fix image paths to be relative ---
    content = content.replace("images/", "./images/")
    content = content.replace("././images/", "./images/")

    # --- Clean up excessive blank lines ---
    content = re.sub(r"\n{3,}", "\n\n", content)

    # --- Add title heading at the top ---
    final = f"# {title}\n\n{content.strip()}\n"

    md_path.write_text(final, encoding="utf-8")


def build_combined_md():
    """Combine all chapter Markdown files into a single document."""
    combined = OUTPUT_DIR / "full_thesis.md"
    parts = []

    # Add a title page
    parts.append("# Double Machine Learning for Difference in Differences: Fundamentals and Applications\n")
    parts.append("**Author**: Martín Gabriel Cargnel\n")
    parts.append("**Director**: Dra. María Noelia Romero\n")
    parts.append("**Universidad de Buenos Aires, Facultad de Ciencias Económicas**\n")
    parts.append("**February 2025**\n")
    parts.append("---\n")

    for filename, _, _ in CHAPTERS:
        md_path = OUTPUT_DIR / filename
        if md_path.exists():
            parts.append(md_path.read_text(encoding="utf-8"))
            parts.append("\n---\n\n")

    combined.write_text("\n".join(parts), encoding="utf-8")
    print(f"  ✓ Combined → {combined.name}")
    return combined


def build_docx(combined_md: Path):
    """Convert the combined Markdown to DOCX."""
    docx_path = OUTPUT_DIR / "full_thesis.docx"

    cmd = [
        "pandoc",
        str(combined_md),
        "-f", "markdown",
        "-t", "docx",
        "--standalone",
        "-o", str(docx_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(OUTPUT_DIR))

    if result.returncode != 0:
        print(f"  WARNING: DOCX generation stderr: {result.stderr[:500]}")
    else:
        print(f"  ✓ DOCX → {docx_path.name}")

    return docx_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    only_md = "--md" in sys.argv
    only_docx = "--docx" in sys.argv

    print("=" * 60)
    print("LaTeX → Markdown/DOCX Conversion")
    print("=" * 60)

    ensure_dirs()

    if not only_docx:
        print("\n📁 Copying and converting images...")
        copy_and_convert_images()

        print("\n📄 Converting chapters...")
        for filename, title, tex_path in CHAPTERS:
            output_path = OUTPUT_DIR / filename
            convert_tex_to_md(tex_path, title, output_path)
            postprocess_md(output_path, title)

    print("\n📚 Building combined Markdown...")
    combined = build_combined_md()

    if not only_md:
        print("\n📝 Generating DOCX...")
        build_docx(combined)

    print("\n✅ Done! Output in:", OUTPUT_DIR)
    print()

    # List generated files
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file() and not f.name.startswith("_"):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:45s} {size_kb:6.1f} KB")


if __name__ == "__main__":
    main()
