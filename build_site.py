#!/usr/bin/env python3
"""Build the project page (index.html and static/) from the arXiv version of the paper.

The page shows the teaser and the main-text figures that carry the findings, with the
paper's numbers and captions. Figures are the paper's own PDFs, so they match the PDF.

Steps:
  1. Compile the paper so main.aux holds the current figure numbers.
  2. Walk main.tex from \\begin{document} to \\appendix, following \\input, and find the
     subsection that each shown figure is \\input under.
  3. Render the shown figures (PDF -> PNG) into static/images/.
  4. Convert the captions and the abstract to HTML (math is left to KaTeX).
  5. Write index.html.

Usage:  python3 build_site.py [path/to/valid-thoughts-arxiv]
Needs pdflatex, bibtex and pdftocairo on the PATH, and Pillow.
"""

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PAPER = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "valid-thoughts-arxiv"
IMAGES = HERE / "static" / "images"
PDFS = HERE / "static" / "pdfs"
MAX_WIDTH_PX = 2400
PAGE_URL = "https://fard-lab.github.io/valid-thoughts/"
CODE_URL = "https://github.com/FARD-Lab/valid-thoughts"
ARXIV_ID = "2606.27378"

TITLE = "What Makes a Latent Thought Valid? Defining and Auditing Thought Representations in LLMs"
AUTHORS = ["Fahd Seddik", "Fatemeh Fard"]
AFFILIATION = "FARD Lab, University of British Columbia"

TEASER = "figures/teaser"
FINDINGS = ["figures/properties_figure", "figures/design_choices", "figures/measures_accuracy"]
LABEL_PREFIX = {"fig": "Figure", "tab": "Table", "sec": "Section", "app": "Appendix", "eq": "Eq.", "thm": "Theorem"}


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compile_paper():
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], PAPER)
    run(["bibtex", "main"], PAPER)
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main"], PAPER)


def read_labels():
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}\{([^}]*)\}", (PAPER / "main.aux").read_text()):
        labels[m.group(1)] = m.group(2)
    return labels


def braced(text, start):
    """Return (content, end) of the {...} group whose opening brace is at text[start]."""
    assert text[start] == "{"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def command_arg(text, command):
    m = re.search(r"\\" + command + r"\*?(\[[^\]]*\])?\{", text)
    return braced(text, m.end() - 1)[0] if m else None


def body_floats():
    """(section, subsection, float) for every figure and table of the main body, in reading
    order, where section and subsection are the paper headings the float is \\input under."""
    main = (PAPER / "main.tex").read_text()
    body = main.split("\\begin{document}", 1)[1].split("\\appendix", 1)[0]
    found, state = [], {"section": None, "subsection": None}
    token = re.compile(r"\\(section|subsection)\*?\{|\\input\{([^}]+)\}")

    def walk(text):
        for m in token.finditer(text):
            if m.group(1):
                title = braced(text, m.end() - 1)[0]
                if m.group(1) == "section":
                    state["section"], state["subsection"] = title, None
                else:
                    state["subsection"] = title
                continue
            name = m.group(2)
            if re.match(r"(figures|tables)/", name):
                found.append((state["section"], state["subsection"], name))
            else:
                path = PAPER / (name if name.endswith(".tex") else name + ".tex")
                if path.exists():
                    walk(path.read_text())

    walk(body)
    return found


def latex_to_html(text, labels):
    """Convert caption-level LaTeX to HTML. Inline math stays as $...$ for KaTeX."""
    out, i = [], 0
    while i < len(text):
        if text[i] == "$":
            j = text.index("$", i + 1)
            out.append(html.escape(text[i:j + 1], quote=False))
            i = j + 1
            continue
        m = re.match(r"\\(textbf|emph|textit|texttt|cref|Cref|ref)\{", text[i:])
        if m:
            arg, end = braced(text, i + m.end() - 1)
            cmd = m.group(1)
            if cmd in {"cref", "Cref", "ref"}:
                parts = []
                for key in arg.split(","):
                    key = key.strip()
                    number = labels.get(key, "??")
                    prefix = LABEL_PREFIX.get(key.split(":")[0], "")
                    parts.append(number if cmd == "ref" else f"{prefix} {number}".strip())
                out.append(", ".join(parts))
            else:
                tag = {"textbf": "strong", "emph": "em", "textit": "em", "texttt": "code"}[cmd]
                out.append(f"<{tag}>{latex_to_html(arg, labels)}</{tag}>")
            i = end
            continue
        for src, dst in [("\\ ", " "), ("~", "&nbsp;"), ("\\%", "%"), ("\\&", "&amp;"), ("---", "\u2013"), ("--", "\u2013"), ("\\,", "\u2009")]:
            if text.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            if text[i] == "\\":
                m = re.match(r"\\[a-zA-Z]+\*?", text[i:])
                raise ValueError(f"unsupported LaTeX command in caption: {m.group(0) if m else text[i:i + 10]!r}")
            out.append(html.escape(text[i], quote=False))
            i += 1
    return re.sub(r"\s+", " ", "".join(out)).strip()


def png_from_pdf(pdf, png, dpi=300):
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "page"
        run(["pdftocairo", "-png", "-r", str(dpi), "-singlefile", str(pdf), str(stem)], tmp)
        im = Image.open(str(stem) + ".png").convert("RGB")
    if im.width > MAX_WIDTH_PX:
        im = im.resize((MAX_WIDTH_PX, round(im.height * MAX_WIDTH_PX / im.width)), Image.LANCZOS)
    im.save(png, optimize=True)


def plain(text):
    return re.sub(r"<[^>]+>", "", html.unescape(text))


def render_figure(name, labels):
    """Render one figure to static/images/ and return its HTML block."""
    tex = (PAPER / f"{name}.tex").read_text()
    caption = latex_to_html(command_arg(tex, "caption"), labels)
    number = labels[command_arg(tex, "label")]
    graphic = command_arg(tex, "includegraphics")
    pdf = PAPER / (graphic if graphic.endswith(".pdf") else graphic + ".pdf")
    png = f"{Path(name).name}.png"
    png_from_pdf(pdf, IMAGES / png)
    return f'''      <figure class="paper-float paper-figure">
        <img src="static/images/{png}" alt="{html.escape(plain(caption))}" loading="lazy" style="width: 100%;">
        <figcaption><strong>Figure {number}.</strong> {caption}</figcaption>
      </figure>'''


def main():
    compile_paper()
    labels = read_labels()
    IMAGES.mkdir(parents=True, exist_ok=True)
    PDFS.mkdir(parents=True, exist_ok=True)
    for old in IMAGES.glob("*.png"):
        old.unlink()

    where = {name: (section, subsection) for section, subsection, name in body_floats()}
    missing = [name for name in [TEASER, *FINDINGS] if name not in where]
    if missing:
        sys.exit(f"ERROR: not in the main body of the paper: {missing}")

    teaser = render_figure(TEASER, labels)
    sections = []
    for name in FINDINGS:
        section, subsection = where[name]
        heading = latex_to_html(subsection or section, labels)
        block = render_figure(name, labels)
        if sections and sections[-1][0] == heading:
            sections[-1][1].append(block)
        else:
            sections.append((heading, [block]))

    main_tex = (PAPER / "main.tex").read_text()
    abstract_tex = main_tex.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    abstract = latex_to_html(abstract_tex, labels)
    shutil.copy(PAPER / "main.pdf", PDFS / "valid-thoughts.pdf")

    section_html = []
    for i, (heading, blocks) in enumerate(sections):
        shade = " is-light" if i % 2 == 1 else ""
        section_html.append(f'''  <section class="section hero paper-section{shade}">
    <div class="container is-max-desktop">
      <h2 class="title is-3 has-text-centered">{heading}</h2>
{chr(10).join(blocks)}
    </div>
  </section>''')

    template = (HERE / "index.template.html").read_text()
    arxiv_url = f"https://arxiv.org/abs/{ARXIV_ID}"
    bibtex = (
        "@misc{seddik2026makeslatentthoughtvalid,\n"
        f"  title         = {{{TITLE}}},\n"
        f"  author        = {{{' and '.join(AUTHORS)}}},\n"
        "  year          = {2026},\n"
        f"  eprint        = {{{ARXIV_ID}}},\n"
        "  archivePrefix = {arXiv},\n"
        "  primaryClass  = {cs.CL},\n"
        f"  url           = {{{arxiv_url}}}\n"
        "}"
    )
    arxiv_button = (
        f'<a href="{arxiv_url}" target="_blank" rel="noopener" class="external-link button is-normal is-rounded is-dark">'
        '<span class="icon"><i class="ai ai-arxiv"></i></span><span>arXiv</span></a>'
    )
    page = (template
            .replace("{{TITLE}}", TITLE)
            .replace("{{AUTHORS_META}}", ", ".join(AUTHORS))
            .replace("{{AUTHORS}}", "\n              ".join(
                f'<span class="author-block"><span class="author-name">{a}</span><sup>1</sup>'
                f'{"," if i < len(AUTHORS) - 1 else ""}</span>' for i, a in enumerate(AUTHORS)))
            .replace("{{AFFILIATION}}", AFFILIATION)
            .replace("{{DESCRIPTION}}", html.escape(plain(abstract).split(". ")[0] + "."))
            .replace("{{ABSTRACT}}", abstract)
            .replace("{{ABSTRACT_PLAIN}}", html.escape(plain(abstract)))
            .replace("{{TEASER}}", teaser)
            .replace("{{SECTIONS}}", "\n".join(section_html))
            .replace("{{CODE_URL}}", CODE_URL)
            .replace("{{PAGE_URL}}", PAGE_URL)
            .replace("{{ARXIV_BUTTON}}", arxiv_button)
            .replace("{{BIBTEX}}", html.escape(bibtex)))
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", page)
    if leftover:
        sys.exit(f"ERROR: unfilled placeholders {leftover}")
    if "\u2014" in page:
        sys.exit("ERROR: the page contains an em-dash")
    (HERE / "index.html").write_text(page)
    print(f"index.html: teaser + {len(FINDINGS)} figures in {len(sections)} sections")
    for heading, blocks in sections:
        print(f"   {heading}: {len(blocks)} figure(s)")


if __name__ == "__main__":
    main()
