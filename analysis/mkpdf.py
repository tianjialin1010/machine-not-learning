#!/usr/bin/env python3
"""通用中文 PDF 排版器（ReportLab）。

背景：本机 Chrome headless 在受限环境下会被 SIGKILL（exit 137），HTML→PDF 走不通；
WeasyPrint 又缺 gobject-introspection。因此统一用 ReportLab 直接排版。
中文字体使用 /Library/Fonts/Arial Unicode.ttf（系统自带，覆盖简繁与数字）。

用法：
    from mkpdf import build_pdf
    build_pdf("out.pdf", title, subtitle, blocks, footer)

blocks 支持的节点：
    {"h2": "章节标题"}
    {"h3": "小节标题"}
    {"p": "正文"}
    {"table": {"header": [...], "rows": [[...]], "num_cols": [1,2]}}
    {"note": "提示", "level": "info"|"warn"|"bad"}
    {"kpi": [("数值", "标签"), ...]}
"""
from __future__ import annotations

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

FONT_PATH = "/Library/Fonts/Arial Unicode.ttf"
FONT_NAME = "ArialUni"

# 配色
INK = colors.HexColor("#1F2328")
INK2 = colors.HexColor("#4B5563")
LINE = colors.HexColor("#E5E7EB")
SOFT = colors.HexColor("#F7F8FA")
NV = colors.HexColor("#76B900")
NV_DARK = colors.HexColor("#4F7A00")
GREEN = colors.HexColor("#1B7F3B")
GREEN_BG = colors.HexColor("#F0FDF4")
AMBER = colors.HexColor("#8A5A00")
AMBER_BG = colors.HexColor("#FFFBEB")
RED = colors.HexColor("#B02020")
RED_BG = colors.HexColor("#FEF2F2")
BLUE_BG = colors.HexColor("#EFF6FF")
BLUE = colors.HexColor("#1D4ED8")


def _register_font() -> None:
    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, FONT_PATH))


def _styles():
    _register_font()
    base = dict(fontName=FONT_NAME, textColor=INK, leading=1.55 * 10)
    return {
        "title": ParagraphStyle("t", fontName=FONT_NAME, fontSize=19, leading=25,
                                textColor=INK, spaceAfter=4),
        "kicker": ParagraphStyle("k", fontName=FONT_NAME, fontSize=8.5, leading=12,
                                 textColor=NV_DARK, spaceAfter=2),
        "sub": ParagraphStyle("s", fontName=FONT_NAME, fontSize=9.5, leading=14,
                              textColor=INK2),
        "h2": ParagraphStyle("h2", fontName=FONT_NAME, fontSize=12.5, leading=17,
                             textColor=INK, spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", fontName=FONT_NAME, fontSize=10.5, leading=15,
                             textColor=INK2, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", fontName=FONT_NAME, fontSize=9, leading=14,
                               textColor=INK),
        "cell": ParagraphStyle("c", fontName=FONT_NAME, fontSize=8, leading=11.5,
                               textColor=INK),
        "celln": ParagraphStyle("cn", fontName=FONT_NAME, fontSize=8, leading=11.5,
                                textColor=INK, alignment=2),  # 右对齐（数字）
        "head": ParagraphStyle("hd", fontName=FONT_NAME, fontSize=8, leading=11.5,
                               textColor=INK2),
        "headn": ParagraphStyle("hdn", fontName=FONT_NAME, fontSize=8, leading=11.5,
                                textColor=INK2, alignment=2),
        "note": ParagraphStyle("n", fontName=FONT_NAME, fontSize=8.5, leading=13.5,
                               textColor=INK),
        "footer": ParagraphStyle("f", fontName=FONT_NAME, fontSize=7.5, leading=11,
                                 textColor=INK2, alignment=TA_CENTER),
    }


def _note_colors(level: str):
    return {"bad": (RED_BG, RED), "warn": (AMBER_BG, AMBER),
            "info": (BLUE_BG, BLUE), "good": (GREEN_BG, GREEN)}.get(
        level, (BLUE_BG, BLUE))


def build_pdf(path: str, title: str, subtitle: str, blocks: list[dict],
              footer: str = "") -> str:
    st = _styles()
    doc = BaseDocTemplate(path, pagesize=A4,
                          leftMargin=16 * mm, rightMargin=16 * mm,
                          topMargin=15 * mm, bottomMargin=15 * mm,
                          title=title, author="哨兵 EdgeSentinel")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")

    def deco(canvas, doc_):
        canvas.saveState()
        canvas.setFillColor(NV)
        canvas.rect(0, A4[1] - 3 * mm, A4[0], 3 * mm, stroke=0, fill=1)
        canvas.setFont(FONT_NAME, 7.5)
        canvas.setFillColor(INK2)
        canvas.drawString(doc_.leftMargin, 9 * mm, title[:70])
        canvas.drawRightString(A4[0] - doc_.rightMargin, 9 * mm,
                               f"第 {doc_.page} 页")
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=deco)])

    story = []
    story.append(Paragraph("MODEL EVALUATION REPORT · 技术交接", st["kicker"]))
    story.append(Paragraph(title, st["title"]))
    if subtitle:
        story.append(Paragraph(subtitle, st["sub"]))
    story.append(Spacer(1, 4 * mm))

    for blk in blocks:
        if "h2" in blk:
            story.append(Paragraph(blk["h2"], st["h2"]))
        elif "h3" in blk:
            story.append(Paragraph(blk["h3"], st["h3"]))
        elif "p" in blk:
            story.append(Paragraph(blk["p"], st["body"]))
            story.append(Spacer(1, 2 * mm))
        elif "kpi" in blk:
            cells, rows = [], []
            for val, lbl in blk["kpi"]:
                cells.append(Paragraph(
                    f'<font size="15" color="#4F7A00"><b>{val}</b></font><br/>'
                    f'<font size="7.5" color="#4B5563">{lbl}</font>', st["cell"]))
            per = min(len(cells), 4)
            for i in range(0, len(cells), per):
                rows.append(cells[i:i + per] + [Spacer(1, 1)] * max(
                    0, per - len(cells[i:i + per])))
            t = Table(rows, colWidths=[doc.width / per] * per)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ]))
            story.append(t)
            story.append(Spacer(1, 3 * mm))
        elif "table" in blk:
            spec = blk["table"]
            header = spec.get("header") or []
            rows = spec.get("rows") or []
            num_cols = set(spec.get("num_cols") or [])
            data = []
            if header:
                data.append([Paragraph(f"<b>{h}</b>",
                                       st["headn"] if i in num_cols else st["head"])
                             for i, h in enumerate(header)])
            for r in rows:
                data.append([Paragraph(str(c),
                                       st["celln"] if i in num_cols else st["cell"])
                             for i, c in enumerate(r)])
            ncol = len(header) if header else (len(rows[0]) if rows else 1)
            avail = doc.width
            widths = spec.get("widths")
            if not widths:
                widths = [avail / ncol] * ncol
            else:
                s = sum(widths)
                widths = [w / s * avail for w in widths]
            t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
            style = [
                ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINE),
                ("LINEBELOW", (0, 1), (-1, -2), 0.4, LINE),
                ("LINEBEFORE", (0, 0), (0, -1), 0.6, LINE),
                ("LINEAFTER", (-1, 0), (-1, -1), 0.6, LINE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
            for r in spec.get("bold_rows", []):
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F2F6EC")))
            t.setStyle(TableStyle(style))
            story.append(t)
            story.append(Spacer(1, 3 * mm))
        elif "note" in blk:
            bg, fg = _note_colors(blk.get("level", "info"))
            inner = Table([[Paragraph(blk["note"], st["note"])]],
                          colWidths=[doc.width])
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("LINEBEFORE", (0, 0), (0, -1), 2.5, fg),
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(KeepTogether([inner, Spacer(1, 3 * mm)]))
        elif "spacer" in blk:
            story.append(Spacer(1, blk["spacer"] * mm))

    if footer:
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(footer, st["footer"]))

    doc.build(story)
    return os.path.abspath(path)
