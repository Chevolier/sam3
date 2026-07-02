#!/usr/bin/env python3
"""Build SAM3 fine-tuning progress deck from scratch — Chinese, custom design.

Design system:
- Palette: Deep navy dominant, teal accent, warm orange for gains, sage for savings
- Motif: Left-side vertical accent bar on every content slide
- Fonts: Noto Sans CJK SC for body, larger sizes for headings
- Layout variety: title, two-column, KPI grid, comparison table, cost callouts
- 16:9, 13.333in × 7.5in
"""

from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn
from lxml import etree

OUT = Path("/home/ec2-user/SageMaker/efs/Projects/sam3/deck/sam3_finetune_progress.pptx")

# --------------------------------------------------------------------------
# Palette — "Midnight Executive" adapted for ML report
# --------------------------------------------------------------------------
NAVY_DEEP  = RGBColor(0x0F, 0x1E, 0x3C)  # slide background for dark slides
NAVY       = RGBColor(0x1E, 0x2A, 0x4A)  # primary
NAVY_MID   = RGBColor(0x2C, 0x3E, 0x5F)
TEAL       = RGBColor(0x22, 0xB8, 0xC5)  # accent 1 — cool, highlight
TEAL_DIM   = RGBColor(0x15, 0x8A, 0x93)
ORANGE     = RGBColor(0xFF, 0x8A, 0x3C)  # accent 2 — warm, gains / metrics
SAGE       = RGBColor(0x7A, 0xC5, 0x8A)  # positive, cost savings
CORAL      = RGBColor(0xFF, 0x6B, 0x6B)  # attention, cost line
CREAM      = RGBColor(0xF8, 0xF6, 0xF0)  # light content background
GRAY_TXT   = RGBColor(0x3A, 0x3A, 0x48)
GRAY_MUTED = RGBColor(0x8A, 0x92, 0xA6)
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)

FONT_CN = "Noto Sans CJK SC"
FONT_EN = "Segoe UI"
FONT_MONO = "Consolas"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def set_slide_bg(slide, color: RGBColor):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, x, y, w, h, fill=None, line=None, line_w=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shape.line.fill.background()
    if fill is not None:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line is not None:
        shape.line.color.rgb = line
        if line_w is not None:
            shape.line.width = line_w
    shape.shadow.inherit = False
    return shape


def add_text(slide, x, y, w, h, text, *, size=14, bold=False, color=None,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT_CN, italic=False):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    if color is not None:
        r.font.color.rgb = color
    # Ensure eastAsia typeface for CJK
    rPr = r._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", FONT_CN)
    return tb


def add_multi_text(slide, x, y, w, h, lines, *, default_size=14, default_color=None,
                   anchor=MSO_ANCHOR.TOP, line_spacing=1.15):
    """
    lines: list of dicts { text, size, bold, color, align, indent, space_before, italic, font }
    """
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = anchor
    first = True
    for line in lines:
        text = line.get("text", "")
        size = line.get("size", default_size)
        bold = line.get("bold", False)
        color = line.get("color", default_color)
        align = line.get("align", PP_ALIGN.LEFT)
        indent = line.get("indent", 0)
        space_before = line.get("space_before", 0)
        italic = line.get("italic", False)
        font = line.get("font", FONT_CN)
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = align
        p.level = indent
        if space_before:
            p.space_before = Pt(space_before)
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = text
        r.font.name = font
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        if color is not None:
            r.font.color.rgb = color
        rPr = r._r.get_or_add_rPr()
        ea = rPr.find(qn("a:ea"))
        if ea is None:
            ea = etree.SubElement(rPr, qn("a:ea"))
        ea.set("typeface", FONT_CN)
    return tb


def add_accent_bar(slide, color=TEAL):
    """The visual motif — vertical accent bar on the left of every content slide."""
    add_rect(slide, Inches(0), Inches(0), Inches(0.15), SLIDE_H, fill=color)


def add_page_number(slide, page_num, total, color=GRAY_MUTED):
    add_text(
        slide, Inches(12.5), Inches(7.15), Inches(0.7), Inches(0.25),
        f"{page_num:02d} / {total:02d}",
        size=9, color=color, align=PP_ALIGN.RIGHT, font=FONT_EN,
    )


def add_footer(slide, text, color=GRAY_MUTED):
    add_text(
        slide, Inches(0.6), Inches(7.15), Inches(11.5), Inches(0.25),
        text, size=9, color=color, font=FONT_CN,
    )


# --------------------------------------------------------------------------
# slide builders
# --------------------------------------------------------------------------

TOTAL = 8  # for page numbers


def slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    set_slide_bg(slide, NAVY_DEEP)

    # Decorative geometric shapes (right side) — teal + orange accents
    for (x_in, y_in, sz_in, col) in [
        (10.5, 0.8,  1.6, TEAL_DIM),
        (11.9, 2.4,  0.9, ORANGE),
        (10.9, 4.2,  1.3, TEAL),
        (12.4, 5.5,  0.6, ORANGE),
        (9.8,  5.9,  0.4, SAGE),
    ]:
        add_rect(slide, Inches(x_in), Inches(y_in), Inches(sz_in), Inches(sz_in),
                 fill=None, line=col, line_w=Pt(1.2))

    # Small tag pill
    tag = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Inches(0.8), Inches(1.2), Inches(2.1), Inches(0.4))
    tag.adjustments[0] = 0.5
    tag.fill.solid()
    tag.fill.fore_color.rgb = TEAL
    tag.line.fill.background()
    tag.shadow.inherit = False
    tf = tag.text_frame
    tf.margin_left = tf.margin_right = Emu(0)
    tf.margin_top = tf.margin_bottom = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "PROJECT UPDATE"
    r.font.name = FONT_EN
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = NAVY_DEEP

    # Main title
    add_text(slide, Inches(0.8), Inches(2.1), Inches(9.5), Inches(1.4),
             "SAM3 微调项目阶段性进展",
             size=52, bold=True, color=WHITE)

    # Subtitle
    add_text(slide, Inches(0.8), Inches(3.7), Inches(10.5), Inches(0.5),
             "AWS_SAM 数据集微调  ·  训练与部署流水线  ·  性能评测",
             size=22, color=TEAL)

    # Highlight strip
    add_rect(slide, Inches(0.8), Inches(4.55), Inches(0.5), Inches(0.06), fill=ORANGE)

    # Key stats preview — 3 mini stat callouts
    stats = [
        ("+105.6%",  "mIoU @ 0-click 提升"),
        ("+22.5%",   "mIoU @ 1-click 提升"),
        ("↓ 0.87",   "NoC@95 减少 (clicks)"),
    ]
    for i, (val, label) in enumerate(stats):
        x = Inches(0.8 + i * 3.2)
        add_text(slide, x, Inches(5.0), Inches(3.0), Inches(0.7),
                 val, size=36, bold=True, color=ORANGE, font=FONT_EN)
        add_text(slide, x, Inches(5.75), Inches(3.0), Inches(0.35),
                 label, size=12, color=GRAY_MUTED)

    # Byline
    add_text(slide, Inches(0.8), Inches(6.7), Inches(6.0), Inches(0.35),
             "Chevolier Zhang  ·  SAM3 Fine-tune Project",
             size=12, color=GRAY_MUTED, font=FONT_EN)
    add_text(slide, Inches(0.8), Inches(7.05), Inches(6.0), Inches(0.35),
             "2026 年 7 月", size=11, color=GRAY_MUTED)


def slide_content_frame(prs, page_num, title_zh, title_en=None, kicker=None):
    """Shared frame for content slides. Returns the slide handle."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, CREAM)
    add_accent_bar(slide, TEAL)

    # Kicker (section label)
    if kicker:
        add_text(slide, Inches(0.6), Inches(0.45), Inches(6.0), Inches(0.28),
                 kicker, size=11, bold=True, color=ORANGE, font=FONT_EN)

    # Main slide title
    add_text(slide, Inches(0.6), Inches(0.75), Inches(11.5), Inches(0.7),
             title_zh, size=30, bold=True, color=NAVY)

    # Optional English subtitle
    if title_en:
        add_text(slide, Inches(0.6), Inches(1.35), Inches(11.5), Inches(0.35),
                 title_en, size=13, color=GRAY_MUTED, italic=True, font=FONT_EN)

    # Divider bar under title
    add_rect(slide, Inches(0.6), Inches(1.75), Inches(0.6), Inches(0.05), fill=ORANGE)

    # Footer
    add_footer(slide, "SAM3 Fine-tune Project  ·  Chevolier Zhang")
    add_page_number(slide, page_num, TOTAL)
    return slide


# --- slide 2: overview ---
def slide_overview(prs):
    s = slide_content_frame(prs, 2, "项目概览与工作总结",
                             title_en="Project overview and work summary",
                             kicker="01  ·  OVERVIEW")

    # 2×2 card grid
    cards = [
        (0.6, 2.1, TEAL,   "数据与任务",       [
            "AWS_SAM 数据集：LabelMe → COCO 格式转换",
            "9:1 划分训练集 / 测试集（测试 5,181 实例）",
            "文本 / 点击双模式 promptable 分割任务",
        ]),
        (6.9, 2.1, ORANGE, "两条微调配方",     [
            "配方 A：text-only prompt（detection + segmentation）",
            "配方 B：+ SAM-1 风格 click 分支联合训练",
            "两种配方互补 → 后续尝试双分支联合训练",
        ]),
        (0.6, 4.6, SAGE,   "训练 & 部署 pipeline", [
            "SageMaker 一键 TrainingJob (单机 / 多机 DDP)",
            "SageMaker 一键实时推理端点 (text + click)",
            "Per-epoch checkpoint 实时同步到 S3",
        ]),
        (6.9, 4.6, CORAL, "评测与对比工具",   [
            "交互式评测: mIoU / Boundary IoU / NoC@95",
            "三面板 Web 对比工具: GT / 预训练 / 微调",
            "端点性能测试脚本 (延迟 + 吞吐)",
        ]),
    ]
    for (x_in, y_in, accent, header, items) in cards:
        # card background
        card = add_rect(s, Inches(x_in), Inches(y_in), Inches(5.85), Inches(2.3),
                        fill=WHITE)
        card.line.color.rgb = RGBColor(0xE5, 0xE0, 0xD5)
        card.line.width = Pt(0.75)
        # accent stripe on left of card
        add_rect(s, Inches(x_in), Inches(y_in), Inches(0.08), Inches(2.3), fill=accent)
        # header
        add_text(s, Inches(x_in + 0.3), Inches(y_in + 0.2), Inches(5.4), Inches(0.4),
                 header, size=17, bold=True, color=NAVY)
        # bullet list
        bullet_lines = [{"text": "▸  " + it, "size": 13, "color": GRAY_TXT,
                         "space_before": 6 if i > 0 else 0} for i, it in enumerate(items)]
        add_multi_text(s, Inches(x_in + 0.3), Inches(y_in + 0.75),
                        Inches(5.4), Inches(1.6), bullet_lines, line_spacing=1.2)


# --- slide 3: main contributions ---
def slide_contributions(prs):
    s = slide_content_frame(prs, 3, "主要贡献",
                             title_en="Key contributions",
                             kicker="02  ·  CONTRIBUTIONS")

    # Six numbered cards in a 2×3 grid
    items = [
        ("01", TEAL,   "两条 SAM3 微调配方",
         "aws_sam_finetune.yaml (text-only) 与 aws_sam_finetune_click.yaml (+ click training) — 完整配置化"),
        ("02", ORANGE, "移植 SAM-1 click 训练损失",
         "多步 click 采样 + best-of-3 mask selection + focal / dice + MSE IoU"),
        ("03", SAGE,   "SageMaker 多机训练",
         "单机 8×A100 及多节点 DDP (multi-instance-train 分支已跑通)"),
        ("04", TEAL,   "一键部署 notebook",
         "私有 fork 源码 vendored 打包，规避 checkpoint / 上游 state_dict 不匹配"),
        ("05", ORANGE, "交互式评测框架",
         "mIoU / Boundary IoU / NoC@95 @ 0/1/3 clicks — 覆盖 text + click 双模式"),
        ("06", SAGE,   "训练 checkpoint 持续同步 S3",
         "支持任意 epoch 恢复 & Spot 实例训练 (成本约降 60%)"),
    ]
    cols, rows = 3, 2
    card_w, card_h = 4.0, 2.4
    gap_x, gap_y = 0.15, 0.15
    x0 = 0.6
    y0 = 2.15
    for i, (num, color, header, body) in enumerate(items):
        r, c = i // cols, i % cols
        x = x0 + c * (card_w + gap_x)
        y = y0 + r * (card_h + gap_y)
        # card
        card = add_rect(s, Inches(x), Inches(y), Inches(card_w), Inches(card_h),
                        fill=WHITE)
        card.line.color.rgb = RGBColor(0xE5, 0xE0, 0xD5)
        card.line.width = Pt(0.75)
        # big number, top left
        add_text(s, Inches(x + 0.25), Inches(y + 0.2), Inches(1.2), Inches(0.7),
                 num, size=32, bold=True, color=color, font=FONT_EN)
        # small colored square accent, top right
        add_rect(s, Inches(x + card_w - 0.4), Inches(y + 0.35), Inches(0.18), Inches(0.18),
                 fill=color)
        # header
        add_text(s, Inches(x + 0.25), Inches(y + 0.95), Inches(card_w - 0.5), Inches(0.5),
                 header, size=15, bold=True, color=NAVY)
        # body
        add_text(s, Inches(x + 0.25), Inches(y + 1.4), Inches(card_w - 0.5), Inches(1.0),
                 body, size=11, color=GRAY_TXT)


# --- slide 4: experiment results (the big table) ---
def slide_experiments(prs):
    s = slide_content_frame(prs, 4, "实验结果 — 相对于官方预训练模型",
                             title_en="Performance improvements vs pretrained SAM3 · AWS_SAM test set (5,181 instances)",
                             kicker="03  ·  EXPERIMENTS")

    rows_data = [
        ("指标", "Pretrained", "Finetune A · epoch 2 (text-only)", "Finetune B · epoch 8 (+ click)"),
        ("mIoU @ 0-click (纯文本)",      "0.2733", "0.5620   +105.6%", "0.3583   +31.1%"),
        ("mIoU @ 1-click",               "0.5667", "0.6309   +11.3%",  "0.6942   +22.5%"),
        ("mIoU @ 3-click",               "0.7613", "0.8186   +7.5%",   "0.8359   +9.8%"),
        ("Boundary IoU @ 0-click",       "0.2199", "0.4976   +126.3%", "0.3047   +38.6%"),
        ("Boundary IoU @ 1-click",       "0.4976", "0.5589   +12.3%",  "0.6095   +22.5%"),
        ("Boundary IoU @ 3-click",       "0.6749", "0.7327   +8.6%",   "0.7569   +12.2%"),
        ("NoC@95  (↓ 越好)",             "15.80",  "15.44   ↓ 0.36",   "14.94   ↓ 0.87"),
        ("Reach rate @ 95",              "0.2822", "0.3011   +6.7%",   "0.3216   +14.0%"),
    ]
    best = {(1, 2), (4, 2),
            (2, 3), (3, 3), (5, 3), (6, 3), (7, 3), (8, 3)}

    n_rows = len(rows_data)
    n_cols = 4
    left = Inches(0.6)
    top = Inches(2.0)
    total_w = Inches(12.15)
    height = Inches(4.0)

    table_shape = s.shapes.add_table(n_rows, n_cols, left, top, total_w, height)
    tbl = table_shape.table
    col_pcts = [0.33, 0.15, 0.26, 0.26]
    for i, p in enumerate(col_pcts):
        tbl.columns[i].width = Inches(12.15 * p)

    # rows — first row header, rest body
    header = rows_data[0]
    for j, txt in enumerate(header):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        tf = cell.text_frame
        tf.clear()
        tf.margin_left = Inches(0.08)
        tf.margin_right = Inches(0.08)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
        r = p.add_run()
        r.text = txt
        r.font.name = FONT_CN
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = WHITE
        rPr = r._r.get_or_add_rPr()
        ea = rPr.find(qn("a:ea"))
        if ea is None:
            ea = etree.SubElement(rPr, qn("a:ea"))
        ea.set("typeface", FONT_CN)

    for i in range(1, n_rows):
        for j in range(n_cols):
            cell = tbl.cell(i, j)
            tf = cell.text_frame
            tf.clear()
            tf.margin_left = Inches(0.08)
            tf.margin_right = Inches(0.08)
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            is_best = (i, j) in best
            # alternate row shading
            if is_best:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xE8, 0xF4, 0xF0)  # pale sage tint
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if i % 2 else RGBColor(0xFA, 0xF7, 0xF1)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = rows_data[i][j]
            r.font.name = FONT_CN
            r.font.size = Pt(11)
            r.font.bold = (j == 0) or is_best
            if j == 0:
                r.font.color.rgb = NAVY
            elif is_best:
                r.font.color.rgb = TEAL_DIM
            else:
                r.font.color.rgb = GRAY_TXT
            rPr = r._r.get_or_add_rPr()
            ea = rPr.find(qn("a:ea"))
            if ea is None:
                ea = etree.SubElement(rPr, qn("a:ea"))
            ea.set("typeface", FONT_CN)

    # Key observations panel (below table)
    obs_top = 6.15
    add_rect(s, Inches(0.6), Inches(obs_top), Inches(12.15), Inches(0.85),
             fill=NAVY)
    add_text(s, Inches(0.85), Inches(obs_top + 0.08), Inches(3.0), Inches(0.3),
             "▸  关键结论", size=12, bold=True, color=ORANGE)
    add_multi_text(
        s, Inches(0.85), Inches(obs_top + 0.35), Inches(11.8), Inches(0.55),
        [
            {"text": "· 0-click 纯文本场景：配方 A 优势显著 (+105.6% / +126.3%)     "
                     "· 1/3-click 交互式：配方 B 全面领先 (NoC@95 ↓ 0.87)     "
                     "· 两配方互补 → 后续尝试双分支联合训练",
             "size": 11, "color": WHITE},
        ],
        line_spacing=1.15,
    )


# --- slide 5: training cost ---
def slide_training_cost(prs):
    s = slide_content_frame(prs, 5, "训练成本",
                             title_en="Training cost on SageMaker (ml.p4de.24xlarge)",
                             kicker="04  ·  TRAINING COST")

    # Big cost callout — total per experiment
    add_rect(s, Inches(0.6), Inches(2.1), Inches(5.85), Inches(2.4), fill=NAVY)
    add_text(s, Inches(0.9), Inches(2.3), Inches(5.3), Inches(0.35),
             "▸  单次完整实验 · 20 epochs", size=12, bold=True, color=ORANGE)
    add_text(s, Inches(0.9), Inches(2.75), Inches(5.3), Inches(1.05),
             "$50 – $80", size=64, bold=True, color=WHITE, font=FONT_EN)
    add_text(s, Inches(0.9), Inches(3.75), Inches(5.3), Inches(0.35),
             "on-demand · 1.5 – 2.5 小时", size=13, color=GRAY_MUTED)
    # spot savings pill
    pill = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(0.9), Inches(4.05), Inches(4.6), Inches(0.32))
    pill.adjustments[0] = 0.5
    pill.fill.solid()
    pill.fill.fore_color.rgb = SAGE
    pill.line.fill.background()
    pill.shadow.inherit = False
    tf = pill.text_frame
    tf.margin_left = tf.margin_right = Emu(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Spot 实例：$20 – $32 · 省 ~60%"
    r.font.name = FONT_CN
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = NAVY_DEEP
    rPr = r._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", FONT_CN)

    # Right side: 4 stat rows
    stats = [
        ("实例",           "ml.p4de.24xlarge",         "8 × A100 80GB"),
        ("按需价格",       "$32 / 小时",               "on-demand"),
        ("单 epoch 时长",  "5 – 7 分钟",               "配方 A / 配方 B"),
        ("Batch 配置",     "8 / GPU · gas=2",          "gradient accumulation"),
    ]
    for i, (label, val, note) in enumerate(stats):
        y = 2.2 + i * 0.72
        # left label
        add_text(s, Inches(6.9), Inches(y), Inches(1.9), Inches(0.35),
                 label, size=11, color=GRAY_MUTED)
        # main value
        add_text(s, Inches(8.85), Inches(y - 0.05), Inches(3.9), Inches(0.45),
                 val, size=18, bold=True, color=NAVY)
        # sub note
        add_text(s, Inches(8.85), Inches(y + 0.35), Inches(3.9), Inches(0.28),
                 note, size=10, color=GRAY_MUTED, font=FONT_EN, italic=True)
        # divider line between rows
        if i < len(stats) - 1:
            add_rect(s, Inches(6.9), Inches(y + 0.7), Inches(5.85), Inches(0.01),
                     fill=RGBColor(0xE0, 0xDC, 0xCF))

    # Bottom note
    add_text(s, Inches(0.6), Inches(6.5), Inches(12.15), Inches(0.35),
             "▸  Per-epoch checkpoint 自动同步到 S3 → 支持任意 epoch 恢复 & Spot 实例训练",
             size=12, bold=True, color=SAGE)


# --- slide 6: inference cost ---
def slide_inference_cost(prs):
    s = slide_content_frame(prs, 6, "推理成本 — SageMaker 实时端点",
                             title_en="Inference cost on SageMaker real-time endpoint (ml.g5.xlarge)",
                             kicker="05  ·  INFERENCE COST")

    # KPI grid — 3×2
    kpis = [
        ("$1.40",    "美元 / 小时",         "on-demand",             ORANGE),
        ("~ $34",    "美元 / 天",            "= ~ $1,020 / 月",       CORAL),
        ("200–300",  "ms · text p50",       "1008 × 1008 输入",       TEAL),
        ("150–250",  "ms · click p50",      "SAM-1 兼容 click",      TEAL),
        ("30–60",    "s · 冷启动",           "CUDA + cuDNN plan",     GRAY_MUTED),
        ("3–5",      "req/s (并发=1)",       "单 client 吞吐",         SAGE),
    ]
    cols = 3
    card_w, card_h = 4.0, 1.9
    gap = 0.15
    x0, y0 = 0.6, 2.1
    for i, (val, label, note, accent) in enumerate(kpis):
        r, c = i // cols, i % cols
        x = x0 + c * (card_w + gap)
        y = y0 + r * (card_h + gap)
        card = add_rect(s, Inches(x), Inches(y), Inches(card_w), Inches(card_h),
                        fill=WHITE)
        card.line.color.rgb = RGBColor(0xE5, 0xE0, 0xD5)
        card.line.width = Pt(0.75)
        # top accent bar
        add_rect(s, Inches(x), Inches(y), Inches(card_w), Inches(0.08), fill=accent)
        # big value
        add_text(s, Inches(x + 0.3), Inches(y + 0.25), Inches(card_w - 0.6), Inches(0.9),
                 val, size=40, bold=True, color=NAVY, font=FONT_EN)
        # label
        add_text(s, Inches(x + 0.3), Inches(y + 1.05), Inches(card_w - 0.6), Inches(0.35),
                 label, size=12, bold=True, color=accent)
        # note
        add_text(s, Inches(x + 0.3), Inches(y + 1.42), Inches(card_w - 0.6), Inches(0.35),
                 note, size=10, color=GRAY_MUTED, italic=True)

    # Bottom row: benchmark tool callout
    add_rect(s, Inches(0.6), Inches(6.5), Inches(12.15), Inches(0.4), fill=NAVY)
    add_text(s, Inches(0.85), Inches(6.53), Inches(11.7), Inches(0.35),
             "▸  benchmark_endpoint.py :  p50/p90/p95/p99 延迟 + 并发吞吐 (线程池模拟真实负载)",
             size=11, bold=True, color=WHITE)


# --- slide 7: next steps ---
def slide_next_steps(prs):
    s = slide_content_frame(prs, 7, "下一步计划",
                             title_en="Next steps",
                             kicker="06  ·  NEXT STEPS")

    # Timeline-style — 5 numbered steps
    steps = [
        ("Q3 · 观察",     "训练与收敛观察",
         "跟踪配方 B 在 epoch 8 之后的收敛趋势 · 目前 ckpt8 为最佳", TEAL),
        ("Q3 · 优化",     "配方 B 的 0-click 性能",
         "更长 warmup + 更大 effective batch · 缩小相对配方 A 的差距", TEAL),
        ("Q3 · 融合",     "双分支联合训练",
         "text + click 损失共同优化 · 统一两种 prompt 模式的性能优势", ORANGE),
        ("Q4 · 验证",     "生产环境 A/B 测试",
         "两个微调模型同时部署为 SageMaker 端点 · 用真实业务流量对比", ORANGE),
        ("Q4 · 降本",     "训练成本优化",
         "启用 Spot 训练 (借助已有 checkpoint_s3_uri) · 降本约 60%", SAGE),
    ]
    y0 = 2.15
    row_h = 0.85
    gap = 0.1
    for i, (tag, header, body, color) in enumerate(steps):
        y = y0 + i * (row_h + gap)
        # card
        card = add_rect(s, Inches(0.6), Inches(y), Inches(12.15), Inches(row_h),
                        fill=WHITE)
        card.line.color.rgb = RGBColor(0xE5, 0xE0, 0xD5)
        card.line.width = Pt(0.5)
        # left circle with number
        circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.85), Inches(y + 0.15),
                                   Inches(0.55), Inches(0.55))
        circ.fill.solid()
        circ.fill.fore_color.rgb = color
        circ.line.fill.background()
        circ.shadow.inherit = False
        tf = circ.text_frame
        tf.margin_left = tf.margin_right = Emu(0)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = f"{i + 1:02d}"
        r.font.name = FONT_EN
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = WHITE
        # tag
        add_text(s, Inches(1.6), Inches(y + 0.13), Inches(1.5), Inches(0.28),
                 tag, size=10, bold=True, color=color, font=FONT_EN)
        # header
        add_text(s, Inches(1.6), Inches(y + 0.36), Inches(4.5), Inches(0.4),
                 header, size=15, bold=True, color=NAVY)
        # body
        add_text(s, Inches(6.3), Inches(y + 0.28), Inches(6.5), Inches(0.5),
                 body, size=11, color=GRAY_TXT)


# --- slide 8: thank you ---
def slide_thanks(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(slide, NAVY_DEEP)

    # Same decorative geometric motif as slide 1
    for (x_in, y_in, sz_in, col) in [
        (0.5, 0.5,  1.2, TEAL_DIM),
        (2.0, 1.4,  0.7, ORANGE),
        (0.9, 5.4,  0.9, TEAL),
        (2.5, 6.2,  0.4, SAGE),
        (11.8, 5.6, 1.0, ORANGE),
    ]:
        add_rect(slide, Inches(x_in), Inches(y_in), Inches(sz_in), Inches(sz_in),
                 fill=None, line=col, line_w=Pt(1.2))

    # Center content
    add_text(slide, Inches(0), Inches(2.6), SLIDE_W, Inches(1.4),
             "谢谢", size=100, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    # Divider dot pattern
    for i in range(3):
        add_rect(slide, Inches(6.35 + i * 0.35), Inches(4.2),
                 Inches(0.15), Inches(0.15), fill=ORANGE)
    add_text(slide, Inches(0), Inches(4.6), SLIDE_W, Inches(0.5),
             "欢迎讨论与反馈", size=22, color=TEAL, align=PP_ALIGN.CENTER)

    # Bottom info
    add_text(slide, Inches(0), Inches(6.5), SLIDE_W, Inches(0.35),
             "SAM3 Fine-tune Project",
             size=12, color=GRAY_MUTED, align=PP_ALIGN.CENTER, font=FONT_EN)
    add_text(slide, Inches(0), Inches(6.85), SLIDE_W, Inches(0.35),
             "Chevolier Zhang  ·  2026 年 7 月",
             size=11, color=GRAY_MUTED, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    # Build
    slide_title(prs)
    slide_overview(prs)
    slide_contributions(prs)
    slide_experiments(prs)
    slide_training_cost(prs)
    slide_inference_cost(prs)
    slide_next_steps(prs)
    slide_thanks(prs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"✓ saved: {OUT}")
    print(f"  slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
