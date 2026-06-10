"""Build the Master's Thesis DOCX from the stitched markdown and the figure paths.

Frankfurt School format:
- Times New Roman 12 pt body, 10 pt footnotes
- 1.5 line spacing, ≥9 pt after paragraph
- Margins: 4 cm left, 2 cm right, 2.5 cm top/bottom
- Cover page: FS logo CENTERED, no header, no page number
- Subsequent pages: FS logo top-right, header per section
- Front matter: lowercase Roman numerals; main text: Arabic from 1
- Headings: H1 16 pt bold, H2 14 pt bold, H3 12 pt bold
- Justified body
- Harvard citations
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, Inches, Emu, RGBColor

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "thesis_build"
LOGO = ROOT / "fs_logo_blue.png"
OUT = ROOT / "Causal_Explainability_Thesis.docx"

FRONT = BUILD / "front_matter.md"
TABLES = BUILD / "tables.md"
APPENDIX = BUILD / "appendix.md"
BIBLIO = BUILD / "bibliography.md"


def parse_tables_md(path: Path) -> dict[int, str]:
    """Return {table_no: markdown_block} from tables.md. Each block is the
    `## Table N. ...` heading and its body up to the next `## Table` heading."""
    out: dict[int, str] = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^##\s*Table\s+(\d+)\.", text)
    # parts = [preamble, num1, body1, num2, body2, ...]
    for i in range(1, len(parts), 2):
        try:
            n = int(parts[i])
        except ValueError:
            continue
        body = parts[i + 1] if i + 1 < len(parts) else ""
        out[n] = f"## Table {n}.{body}"
    return out


TABLE_BLOCKS: dict[int, str] = {}
TABLE_CAPTIONS: dict[int, str] = {
    1: "Table 1. Equity universe — 20 DAX constituents.",
    2: "Table 2. Engineered features and definitions.",
    3: "Table 3. Stationarity test results (Augmented Dickey-Fuller).",
    4: "Table 4. CatBoost hyperparameters.",
    5: "Table 5. Causal discovery methods and core assumptions.",
    6: "Table 6. Holdout evaluation metrics (time vs random split).",
    7: "Table 7. LEWIS scores (max-Necessity-Sufficiency, normalised).",
    8: "Table 8. LEWIS vs SHAP rankings on the nine-feature panel.",
    9: "Table 9. Survey demographics and descriptive statistics (n = 25).",
    10: "Table 10. OLS coefficient table — drivers of expert trust (n = 25).",
}
CHAPTERS = [
    BUILD / "chapter_1_introduction.md",
    BUILD / "chapter_2_literature.md",
    BUILD / "chapter_3_methodology.md",
    BUILD / "chapter_4_data.md",
    BUILD / "chapter_5_results.md",
    BUILD / "chapter_6_discussion.md",
    BUILD / "chapter_7_conclusion.md",
]

# Map figure number -> (caption, png path) — from SPINE §9
FIGURES = {
    2: ("Discovered causal DAG over the nine features and the investment decision (DirectLiNGAM, prior b).",
        ROOT / "results/investment/causal_graph_DirectLiNGAM(b).png"),
    3: ("Necessity, Sufficiency, and max-Necessity-Sufficiency scores per feature.",
        ROOT / "results/investment/nesuf_comparison_DirectLiNGAM(b).png"),
    4: ("SHAP summary plot.",
        ROOT / "results/investment/shap_summary_DirectLiNGAM(b).png"),
    5: ("LEWIS (causal) vs SHAP (correlational) normalised importance.",
        ROOT / "results/plots/lewis_vs_shap_bw.png"),
    6: ("Reversal probabilities — counterfactual feasibility per feature.",
        ROOT / "results/investment/reversal_DirectLiNGAM(b).png"),
    7: ("Confusion matrix on time-split holdout.",
        ROOT / "results/plots/confusion_matrix.png"),
    8: ("Calibration curve.",
        ROOT / "results/plots/calibration_curve.png"),
    9: ("Rolling holdout accuracy.",
        ROOT / "results/plots/rolling_accuracy.png"),
    10: ("Accuracy by sector.",
         ROOT / "results/plots/accuracy_by_sector.png"),
    11: ("Accuracy by macro-regime.",
         ROOT / "results/plots/accuracy_by_regime.png"),
    12: ("DAG stability heatmap across discovery methods × priors.",
         ROOT / "results/plots/dag_stability_heatmap.png"),
    13: ("Expert preference distribution among the 25 evaluations.",
         ROOT / "results/plots/expert_preference.png"),
    14: ("Trust score distribution.",
         ROOT / "results/plots/trust_distribution.png"),
    15: ("Trust vs confidence scatter with OLS line.",
         ROOT / "results/plots/trust_vs_confidence.png"),
    16: ("Mechanics-accuracy feedback distribution.",
         ROOT / "results/plots/mechanics_distribution.png"),
    17: ("OLS diagnostic panel: actual vs predicted, residuals, Q-Q, coefficients.",
         ROOT / "results/plots/trust_drivers_plot.png"),
}


# ---------------------------------------------------------------------------
# Document setup helpers
# ---------------------------------------------------------------------------

def set_default_font(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for tag in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(tag), "Times New Roman")
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_after = Pt(9)


def set_section_margins(section, *, header_distance: Cm = Cm(1.0)) -> None:
    section.left_margin = Cm(4)
    section.right_margin = Cm(2)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.header_distance = header_distance
    section.footer_distance = Cm(1.0)
    section.different_first_page_header_footer = False


def add_logo_header(section, *, alignment: int) -> None:
    """Insert the FS logo as a header image at the requested alignment.
    Unlinks from the previous section's header so it doesn't bleed back to the cover."""
    section.header.is_linked_to_previous = False
    header = section.header
    # Wipe any existing paragraphs
    for p in list(header.paragraphs):
        p.clear()
    p = header.paragraphs[0]
    p.alignment = alignment
    run = p.add_run()
    run.add_picture(str(LOGO), width=Cm(3.0))


def add_page_number(section, *, numfmt: str = "decimal", start: int | None = 1) -> None:
    """Add a page number field to the footer and configure numbering format/start.
    numfmt: 'decimal' for Arabic, 'lowerRoman' for i/ii/iii."""
    section.footer.is_linked_to_previous = False
    footer = section.footer
    for p in list(footer.paragraphs):
        p.clear()
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()

    # The PAGE field instruction needs the right format switch:
    # \* roman = lowercase Roman; \* Arabic = Arabic (default)
    if numfmt == "lowerRoman":
        field_instr = " PAGE \\* roman \\* MERGEFORMAT "
    else:
        field_instr = " PAGE \\* Arabic \\* MERGEFORMAT "

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field_instr
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(fld_end)

    # Configure section page numbering format and start value via sectPr
    sectPr = section._sectPr
    pgNumType = sectPr.find(qn("w:pgNumType"))
    if pgNumType is None:
        pgNumType = OxmlElement("w:pgNumType")
        sectPr.append(pgNumType)
    pgNumType.set(qn("w:fmt"), numfmt)
    if start is not None:
        pgNumType.set(qn("w:start"), str(start))
    else:
        # Remove any existing start so numbering continues from previous section
        if pgNumType.get(qn("w:start")) is not None:
            pgNumType.attrib.pop(qn("w:start"), None)


def disable_header(section) -> None:
    """Make the section's header empty (used for the cover page)."""
    header = section.header
    header.is_linked_to_previous = False
    for p in list(header.paragraphs):
        p.clear()


def disable_footer(section) -> None:
    footer = section.footer
    footer.is_linked_to_previous = False
    for p in list(footer.paragraphs):
        p.clear()


def add_page_break(doc: Document) -> None:
    p = doc.add_paragraph()
    run = p.add_run()
    run.add_break()
    # Use proper page break
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run._r.append(br)


def add_field(p, field_code: str) -> None:
    """Insert a Word field into paragraph p (e.g. TOC, PAGEREF).
    The field will auto-update when the user opens the doc in Word and presses Ctrl+A, F9."""
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    run._r.append(fld_begin)

    run2 = p.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field_code
    run2._r.append(instr)

    run3 = p.add_run()
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    run3._r.append(fld_sep)

    # Placeholder text visible before the user updates the field
    run4 = p.add_run("Right-click → Update Field to populate")
    run4.font.name = "Times New Roman"
    run4.font.size = Pt(11)
    run4.italic = True

    run5 = p.add_run()
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run5._r.append(fld_end)


def add_toc(doc: Document) -> None:
    """Insert a Word Table of Contents field (updates on open in Word)."""
    add_heading(doc, 1, "Table of Contents")
    p = doc.add_paragraph()
    # TOC \o "1-3" shows headings levels 1-3, \h creates hyperlinks, \z hides tab leaders in web
    add_field(p, ' TOC \\o "1-3" \\h \\z \\u ')


def _add_pageref_field(p, bookmark_name: str) -> None:
    """Insert a PAGEREF field pointing to a bookmark. Resolves to the page number
    of the bookmarked location when the user updates fields in Word."""
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    run._r.append(fld_begin)

    run2 = p.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" PAGEREF {bookmark_name} \\h "
    run2._r.append(instr)

    run3 = p.add_run()
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    run3._r.append(fld_sep)

    run4 = p.add_run("??")
    run4.font.name = "Times New Roman"
    run4.font.size = Pt(12)

    run5 = p.add_run()
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run5._r.append(fld_end)


def _add_bookmark(p, name: str) -> None:
    """Add a bookmark start+end to a paragraph so PAGEREF can target it."""
    import random
    bm_id = str(random.randint(10000, 99999))
    bm_start = OxmlElement("w:bookmarkStart")
    bm_start.set(qn("w:id"), bm_id)
    bm_start.set(qn("w:name"), name)
    bm_end = OxmlElement("w:bookmarkEnd")
    bm_end.set(qn("w:id"), bm_id)
    p._p.append(bm_start)
    p._p.append(bm_end)


def add_list_of_figures(doc: Document) -> None:
    """Build a manual List of Figures with PAGEREF fields."""
    add_heading(doc, 1, "List of Figures")
    for fig_no, (caption, _path) in sorted(FIGURES.items()):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        # "Figure N. Caption text ......... pagenum"
        run = p.add_run(f"Figure {fig_no}. {caption}")
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        # Tab + page ref
        run2 = p.add_run("\t")
        _add_pageref_field(p, f"_Fig{fig_no}")


def add_list_of_tables(doc: Document) -> None:
    """Build a manual List of Tables with PAGEREF fields."""
    add_heading(doc, 1, "List of Tables")
    for t_no, cap_text in sorted(TABLE_CAPTIONS.items()):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        run = p.add_run(cap_text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        run2 = p.add_run("\t")
        _add_pageref_field(p, f"_Tbl{t_no}")


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------

INLINE_FIG_RE = re.compile(r"See Figure (\d+)")
INLINE_TABLE_RE = re.compile(r"See Table (\d+)")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"\*([^*]+)\*|_([^_]+)_")
PIPE_RE = re.compile(r"^\s*\|.*\|\s*$")
HBAR_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")


def _ensure_heading_styles(doc: Document) -> None:
    """Configure built-in Heading 1/2/3 styles to match FS format (TNR bold, sized)."""
    sizes = {1: 16, 2: 14, 3: 12}
    for level, size in sizes.items():
        style_name = f"Heading {level}"
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        for tag in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(tag), "Times New Roman")
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.keep_with_next = True


def add_heading(doc: Document, level: int, text: str) -> None:
    """Add a heading using Word's built-in Heading styles (required for TOC)."""
    level = min(level, 3)  # cap at Heading 3
    p = doc.add_paragraph(text, style=f"Heading {level}")


def add_paragraph_with_inline_formatting(doc: Document, text: str) -> None:
    """Add a paragraph and parse **bold** / *italic* / _italic_ inline.
    Justified, TNR 12 pt, 1.5 spacing. Fig and table refs are left as plain text
    (rendered as 'Figure N' / 'Table N').
    """
    text = INLINE_FIG_RE.sub(lambda m: f"Figure {m.group(1)}", text)
    text = INLINE_TABLE_RE.sub(lambda m: f"Table {m.group(1)}", text)
    # Strip the [ … ] brackets around the now-rewritten "See Figure / See Table" refs
    text = re.sub(r"\[\s*(Figure \d+(?:\s*,\s*Table \d+)*)\s*\]", r"\1", text)
    text = re.sub(r"\[\s*(Table \d+(?:\s*,\s*Figure \d+)*)\s*\]", r"\1", text)
    text = re.sub(r"\[\s*(Figure \d+)\s*\]", r"\1", text)
    text = re.sub(r"\[\s*(Table \d+)\s*\]", r"\1", text)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    p.paragraph_format.space_after = Pt(9)

    _emit_inline_formatted(p, text)


def _emit_inline_formatted(p, text: str) -> None:
    """Tokenise **bold**, *italic*, `code` in text and emit runs on paragraph p.
    Strategy: split on backticks first (they never nest), then handle bold/italic
    in the non-code segments."""
    # Pre-strip: if the entire text is wrapped in single * (italic paragraph), unwrap it
    stripped_check = text.strip()
    force_italic = False
    if stripped_check.startswith("*") and stripped_check.endswith("*") and not stripped_check.startswith("**"):
        inner = stripped_check[1:-1]
        if "*" not in inner:
            text = inner
            force_italic = True
    # Replace backtick code spans with a sentinel, process separately
    code_spans: list[str] = []
    def _stash_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans)-1}\x00"
    text = re.sub(r"`([^`]+)`", _stash_code, text)
    pattern = re.compile(r"\x00CODE(\d+)\x00|\*\*(.+?)\*\*|\*(.+?)\*|_([^_]+)_")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            run = p.add_run(text[pos:m.start()])
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)
            if force_italic:
                run.italic = True
        code_idx = m.group(1)
        bold_chunk = m.group(2)
        ital_chunk = m.group(3) or m.group(4)
        if code_idx is not None:
            run = p.add_run(code_spans[int(code_idx)])
            run.font.name = "Courier New"
            run.font.size = Pt(11)
        elif bold_chunk:
            run = p.add_run(bold_chunk)
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)
            run.bold = True
        else:
            run = p.add_run(ital_chunk)
            run.font.name = "Times New Roman"
            run.font.size = Pt(12)
            run.italic = True
        pos = m.end()
    if pos < len(text):
        run = p.add_run(text[pos:])
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        if force_italic:
            run.italic = True


def add_bullet(doc: Document, text: str) -> None:
    text = INLINE_FIG_RE.sub(lambda m: f"Figure {m.group(1)}", text)
    text = INLINE_TABLE_RE.sub(lambda m: f"Table {m.group(1)}", text)
    text = re.sub(r"\[\s*(Figure \d+)\s*\]", r"\1", text)
    text = re.sub(r"\[\s*(Table \d+)\s*\]", r"\1", text)
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    # Clear default run and use inline formatter
    for r in p.runs:
        r.text = ""
    _emit_inline_formatted(p, text)


def add_numbered(doc: Document, text: str) -> None:
    text = INLINE_FIG_RE.sub(lambda m: f"Figure {m.group(1)}", text)
    text = INLINE_TABLE_RE.sub(lambda m: f"Table {m.group(1)}", text)
    text = re.sub(r"\[\s*(Figure \d+)\s*\]", r"\1", text)
    text = re.sub(r"\[\s*(Table \d+)\s*\]", r"\1", text)
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    for r in p.runs:
        r.text = ""
    _emit_inline_formatted(p, text)


def add_table_from_markdown(doc: Document, lines: list[str]) -> None:
    """Build a docx table from a markdown table block (the lines list is the
    block, including header and separator rows). Also handle note lines below
    the table (starting with *) as italic paragraphs."""
    table_rows = []
    note_lines = []
    for raw in lines:
        if HBAR_RE.match(raw):
            continue
        # Check if this is a note/caption line (starts with * for italic)
        stripped_r = raw.strip().strip("|").strip()
        if stripped_r.startswith("*") and stripped_r.endswith("*") and "|" not in raw[1:]:
            note_lines.append(stripped_r.strip("*").strip())
            continue
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        table_rows.append(cells)
    if not table_rows:
        return
    n_cols = max(len(r) for r in table_rows)
    table = doc.add_table(rows=len(table_rows), cols=n_cols)
    table.style = "Light Grid Accent 1"
    for i, row in enumerate(table_rows):
        while len(row) < n_cols:
            row.append("")
        for j, cell_text in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            # Strip any markdown inline from cell text
            clean = re.sub(r"`([^`]+)`", r"\1", cell_text)
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
            clean = re.sub(r"\*([^*]+)\*", r"\1", clean)
            run = p.add_run(clean)
            run.font.name = "Times New Roman"
            run.font.size = Pt(11)
            if i == 0:
                run.bold = True
    # Render note lines as italic paragraphs below the table
    for note in note_lines:
        note = re.sub(r"`([^`]+)`", r"\1", note)
        p = doc.add_paragraph()
        run = p.add_run(note)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.italic = True


def _ensure_caption_style(doc: Document) -> None:
    """Configure the built-in Caption style for FS formatting."""
    style = doc.styles["Caption"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    style.font.italic = True
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style.paragraph_format.space_before = Pt(6)
    style.paragraph_format.space_after = Pt(12)


def _add_seq_field(p, seq_id: str, number: int) -> None:
    """Insert a SEQ field into paragraph p. Word's TOC \\c uses SEQ fields to build
    the List of Figures / List of Tables. The field auto-numbers, but we also put
    the known number as placeholder text so it renders correctly without updating."""
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    run._r.append(fld_begin)

    run2 = p.add_run()
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" SEQ {seq_id} \\r {number} "
    run2._r.append(instr)

    run3 = p.add_run()
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    run3._r.append(fld_sep)

    # Placeholder: the actual number (visible before field update)
    run4 = p.add_run(str(number))
    run4.font.name = "Times New Roman"
    run4.font.size = Pt(11)
    run4.italic = True

    run5 = p.add_run()
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run5._r.append(fld_end)


def add_figure(doc: Document, fig_no: int) -> None:
    if fig_no not in FIGURES:
        return
    caption, path = FIGURES[fig_no]
    if not path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(f"[Figure {fig_no} — image not found at {path}]")
        run.italic = True
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(path), width=Cm(13.0))
    # Caption with bookmark for PAGEREF targeting from the List of Figures
    cap = doc.add_paragraph(style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _add_bookmark(cap, f"_Fig{fig_no}")
    run = cap.add_run(f"Figure {fig_no}. {caption}")
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"
    run.italic = True


def render_markdown(doc: Document, md_text: str, *, embed_figures: bool = True, embed_tables: bool = False) -> None:
    """Walk the markdown line-by-line and emit docx content. Tables and bullets
    are recognised; bold/italic inline formatting is applied; `[See Figure N]`
    triggers an inline figure embed (when embed_figures is True)."""
    lines = md_text.split("\n")
    i = 0
    figures_seen = set()
    tables_seen = set()
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # Custom anchor lines like <<<COVER>>> and <<<PAGEBREAK>>> consumed
        # outside this function — pass them through transparently.
        if stripped.startswith("<<<") and stripped.endswith(">>>"):
            i += 1
            continue
        # Code blocks (triple backtick) — render as monospace paragraphs
        if stripped.startswith("```"):
            i += 1  # skip opening fence
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # skip closing fence
            for cl in code_lines:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                run = p.add_run(cl)
                run.font.name = "Courier New"
                run.font.size = Pt(10)
            continue
        # Horizontal rule (---) — render as empty space
        if re.match(r"^-{3,}$", stripped):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            i += 1
            continue
        # Headings
        h = HEADING_RE.match(stripped)
        if h:
            level = len(h.group(1))
            add_heading(doc, level, h.group(2).strip())
            i += 1
            continue
        # Markdown table block
        if PIPE_RE.match(line):
            block = []
            while i < len(lines) and (PIPE_RE.match(lines[i]) or HBAR_RE.match(lines[i])):
                block.append(lines[i])
                i += 1
            add_table_from_markdown(doc, block)
            continue
        # Bulleted list
        if stripped.startswith("- ") or stripped.startswith("* "):
            add_bullet(doc, stripped[2:])
            i += 1
            continue
        # Numbered list (1. 2. 3.)
        if re.match(r"^\d+\.\s", stripped):
            add_numbered(doc, re.sub(r"^\d+\.\s", "", stripped))
            i += 1
            continue
        # Body paragraph — accumulate until blank line
        para_lines = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not HEADING_RE.match(lines[i].strip()) \
                and not PIPE_RE.match(lines[i]) and not lines[i].strip().startswith("- ") \
                and not lines[i].strip().startswith("* ") \
                and not re.match(r"^\d+\.\s", lines[i].strip()):
            para_lines.append(lines[i].strip())
            i += 1
        para = " ".join(para_lines)
        # Embed figures referenced in this paragraph (each fig at most once)
        if embed_figures:
            for m in INLINE_FIG_RE.finditer(para):
                fig_no = int(m.group(1))
                if fig_no not in figures_seen:
                    figures_seen.add(fig_no)
        # Detect newly-referenced tables on this paragraph
        new_tables = []
        if embed_tables:
            for m in INLINE_TABLE_RE.finditer(para):
                t_no = int(m.group(1))
                if t_no not in tables_seen and t_no in TABLE_BLOCKS:
                    tables_seen.add(t_no)
                    new_tables.append(t_no)
        add_paragraph_with_inline_formatting(doc, para)
        # Embed any new figures right after the paragraph that referenced them
        if embed_figures:
            for fig_no in list(figures_seen):
                if fig_no not in getattr(doc, "_emitted_figs", set()):
                    add_figure(doc, fig_no)
                    if not hasattr(doc, "_emitted_figs"):
                        doc._emitted_figs = set()
                    doc._emitted_figs.add(fig_no)
        # Embed any new tables right after the paragraph that referenced them
        if embed_tables:
            for t_no in new_tables:
                if t_no not in getattr(doc, "_emitted_tables", set()):
                    if not hasattr(doc, "_emitted_tables"):
                        doc._emitted_tables = set()
                    doc._emitted_tables.add(t_no)
                    # Render the table block (recursive call, no further table-embedding)
                    render_markdown(doc, TABLE_BLOCKS[t_no], embed_figures=False, embed_tables=False)
                    # Add a Caption-styled label with bookmark for PAGEREF from LoT
                    cap_text = TABLE_CAPTIONS.get(t_no, f"Table {t_no}.")
                    cap_p = doc.add_paragraph(style="Caption")
                    cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    _add_bookmark(cap_p, f"_Tbl{t_no}")
                    cap_run = cap_p.add_run(cap_text)
                    cap_run.font.size = Pt(11)
                    cap_run.font.name = "Times New Roman"
                    cap_run.italic = True


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

def write_cover(doc: Document) -> None:
    """Cover page: FS logo CENTERED, no header, no page number."""
    section = doc.sections[0]
    set_section_margins(section)
    disable_header(section)
    disable_footer(section)

    # Top spacer
    for _ in range(3):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)

    # Big centered logo
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(LOGO), width=Cm(8.0))

    # Title block
    for _ in range(3):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Master's Thesis")
    run.bold = True
    run.font.size = Pt(20)
    run.font.name = "Times New Roman"

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Causal Explainability for ML-Driven Investment Decisions\nby German Market Investors")
    run.bold = True
    run.font.size = Pt(18)
    run.font.name = "Times New Roman"

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("An Empirical Study on the DAX 30")
    run.italic = True
    run.font.size = Pt(14)
    run.font.name = "Times New Roman"

    for _ in range(4):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("[Author Name 1]")
    run.font.size = Pt(13)
    run.font.name = "Times New Roman"

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("[Author Name 2]")
    run.font.size = Pt(13)
    run.font.name = "Times New Roman"


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------

def build() -> Path:
    global TABLE_BLOCKS
    TABLE_BLOCKS = parse_tables_md(TABLES)
    doc = Document()
    set_default_font(doc)
    _ensure_heading_styles(doc)
    _ensure_caption_style(doc)

    # === SECTION 1: Cover ===
    write_cover(doc)

    # === SECTION 2: Front matter (Roman numerals, logo top-right) ===
    front_section = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section_margins(front_section)
    add_logo_header(front_section, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    add_page_number(front_section, numfmt="lowerRoman", start=1)

    front_md = FRONT.read_text(encoding="utf-8")
    # Split on the <<<...>>> markers to control which sections get special treatment.
    # Sections we handle: TITLE, CERTIFICATION, ACKNOWLEDGEMENTS, ABSTRACT, TOC,
    # ABBREVIATIONS, LIST_FIGURES, LIST_TABLES
    sections_raw = re.split(r"<<<(\w+)>>>", front_md)
    # sections_raw = [pre, "COVER", text, "PAGEBREAK", text, "TITLE", text, ...]
    # Walk pairs of (marker, content)
    front_sections: dict[str, str] = {}
    i_fm = 1  # skip preamble before the first marker
    while i_fm < len(sections_raw) - 1:
        marker = sections_raw[i_fm].strip()
        content = sections_raw[i_fm + 1] if i_fm + 1 < len(sections_raw) else ""
        front_sections[marker] = content
        i_fm += 2

    # Render: title page
    if "TITLE" in front_sections:
        render_markdown(doc, front_sections["TITLE"])
        add_page_break(doc)
    # Statement of certification
    if "CERTIFICATION" in front_sections:
        render_markdown(doc, front_sections["CERTIFICATION"])
        add_page_break(doc)
    # Acknowledgements
    if "ACKNOWLEDGEMENTS" in front_sections:
        render_markdown(doc, front_sections["ACKNOWLEDGEMENTS"])
        add_page_break(doc)
    # Abstract
    if "ABSTRACT" in front_sections:
        render_markdown(doc, front_sections["ABSTRACT"])
        add_page_break(doc)
    # Table of Contents — INSERT REAL WORD FIELD
    add_toc(doc)
    add_page_break(doc)
    # List of Abbreviations (this one is a real markdown table — render it)
    if "ABBREVIATIONS" in front_sections:
        render_markdown(doc, front_sections["ABBREVIATIONS"])
        add_page_break(doc)
    # List of Figures — WORD FIELD (auto-populates from figure captions)
    add_list_of_figures(doc)
    add_page_break(doc)
    # List of Tables — WORD FIELD
    add_list_of_tables(doc)
    add_page_break(doc)

    # === SECTION 3: Main text (Arabic numerals, logo top-right) ===
    main_section = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section_margins(main_section)
    add_logo_header(main_section, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    add_page_number(main_section, numfmt="decimal", start=1)

    # Reset the figures-emitted tracker for the main body
    doc._emitted_figs = set()

    for ch_path in CHAPTERS:
        if not ch_path.exists():
            continue
        ch_text = ch_path.read_text(encoding="utf-8")
        # Strip the per-chapter "Bibliography fragment" — we merge them at the end.
        ch_text = re.split(r"^##\s*Bibliography fragment", ch_text, flags=re.MULTILINE)[0]
        render_markdown(doc, ch_text, embed_figures=True, embed_tables=True)
        add_page_break(doc)

    # === SECTION 4: Back matter (Arabic numerals continue, logo top-right) ===
    back_section = doc.add_section(WD_SECTION.NEW_PAGE)
    set_section_margins(back_section)
    add_logo_header(back_section, alignment=WD_ALIGN_PARAGRAPH.RIGHT)
    add_page_number(back_section, numfmt="decimal", start=None)

    if APPENDIX.exists():
        render_markdown(doc, APPENDIX.read_text(encoding="utf-8"), embed_figures=False)
        add_page_break(doc)
    if BIBLIO.exists():
        render_markdown(doc, BIBLIO.read_text(encoding="utf-8"), embed_figures=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


if __name__ == "__main__":
    out_path = build()
    print(f"DOCX written: {out_path}")
