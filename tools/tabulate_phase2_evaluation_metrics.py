from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS, "m": M_NS}
DOCUMENT_XML = "word/document.xml"
ACCENT_BLUE = "1F4E79"
HEADER_FILL = "1F4E79"
ALT_FILL = "F3F6FA"
BORDER_COLOR = "9EADBA"


def qn(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


def visible_text(element) -> str:
    return "".join(
        element.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS)
    )


def formula_text(element) -> str:
    return "".join(element.xpath(".//m:t/text()", namespaces=NS))


def add_on_off(parent, local: str) -> etree._Element:
    return etree.SubElement(parent, qn(W_NS, local))


def word_run(
    text: str,
    *,
    bold: bool = False,
    color: str | None = None,
    size: int = 20,
) -> etree._Element:
    run = etree.Element(qn(W_NS, "r"))
    properties = etree.SubElement(run, qn(W_NS, "rPr"))
    fonts = etree.SubElement(properties, qn(W_NS, "rFonts"))
    fonts.set(qn(W_NS, "ascii"), "Times New Roman")
    fonts.set(qn(W_NS, "hAnsi"), "Times New Roman")
    fonts.set(qn(W_NS, "eastAsia"), "宋体")
    if bold:
        add_on_off(properties, "b")
        add_on_off(properties, "bCs")
    if color is not None:
        color_element = etree.SubElement(properties, qn(W_NS, "color"))
        color_element.set(qn(W_NS, "val"), color)
    size_element = etree.SubElement(properties, qn(W_NS, "sz"))
    size_element.set(qn(W_NS, "val"), str(size))
    size_cs = etree.SubElement(properties, qn(W_NS, "szCs"))
    size_cs.set(qn(W_NS, "val"), str(size))
    text_element = etree.SubElement(run, qn(W_NS, "t"))
    if text.startswith(" ") or text.endswith(" "):
        text_element.set(qn(XML_NS, "space"), "preserve")
    text_element.text = text
    return run


def paragraph_properties(
    *,
    alignment: str = "left",
    keep_next: bool = False,
    before: int = 0,
    after: int = 0,
) -> etree._Element:
    properties = etree.Element(qn(W_NS, "pPr"))
    if keep_next:
        add_on_off(properties, "keepNext")
    add_on_off(properties, "keepLines")
    spacing = etree.SubElement(properties, qn(W_NS, "spacing"))
    spacing.set(qn(W_NS, "before"), str(before))
    spacing.set(qn(W_NS, "after"), str(after))
    spacing.set(qn(W_NS, "line"), "300")
    spacing.set(qn(W_NS, "lineRule"), "auto")
    justification = etree.SubElement(properties, qn(W_NS, "jc"))
    justification.set(qn(W_NS, "val"), alignment)
    return properties


def paragraph(
    parts: list[etree._Element],
    *,
    alignment: str = "left",
    keep_next: bool = False,
    before: int = 0,
    after: int = 0,
) -> etree._Element:
    element = etree.Element(qn(W_NS, "p"))
    element.append(
        paragraph_properties(
            alignment=alignment,
            keep_next=keep_next,
            before=before,
            after=after,
        )
    )
    element.extend(parts)
    return element


def label_paragraph(text: str) -> etree._Element:
    return paragraph(
        [word_run(text, bold=True, color=ACCENT_BLUE, size=21)],
        keep_next=True,
        before=120,
        after=60,
    )


def math_paragraph(*formulas: etree._Element) -> etree._Element:
    parts: list[etree._Element] = []
    for index, formula in enumerate(formulas):
        if index:
            parts.append(word_run("、", size=19))
        parts.append(deepcopy(formula))
    return paragraph(parts, alignment="center")


def text_paragraph(text: str, *, alignment: str = "left", bold: bool = False) -> etree._Element:
    return paragraph(
        [word_run(text, bold=bold, size=19 if not bold else 20)],
        alignment=alignment,
    )


def set_cell_properties(cell: etree._Element, width: int, *, fill: str | None) -> None:
    properties = etree.SubElement(cell, qn(W_NS, "tcPr"))
    width_element = etree.SubElement(properties, qn(W_NS, "tcW"))
    width_element.set(qn(W_NS, "w"), str(width))
    width_element.set(qn(W_NS, "type"), "dxa")
    vertical = etree.SubElement(properties, qn(W_NS, "vAlign"))
    vertical.set(qn(W_NS, "val"), "center")
    if fill is not None:
        shading = etree.SubElement(properties, qn(W_NS, "shd"))
        shading.set(qn(W_NS, "val"), "clear")
        shading.set(qn(W_NS, "fill"), fill)


def make_cell(
    width: int,
    paragraphs: list[etree._Element],
    *,
    fill: str | None = None,
) -> etree._Element:
    cell = etree.Element(qn(W_NS, "tc"))
    set_cell_properties(cell, width, fill=fill)
    cell.extend(paragraphs)
    return cell


def make_table(
    widths: list[int],
    headers: list[str],
    rows: list[list[list[etree._Element]]],
) -> etree._Element:
    if sum(widths) != 9360:
        raise ValueError(f"table width must be 9360 DXA, got {sum(widths)}")
    if len(widths) != len(headers):
        raise ValueError("header/width mismatch")
    table = etree.Element(qn(W_NS, "tbl"))
    properties = etree.SubElement(table, qn(W_NS, "tblPr"))
    style = etree.SubElement(properties, qn(W_NS, "tblStyle"))
    style.set(qn(W_NS, "val"), "33")
    table_width = etree.SubElement(properties, qn(W_NS, "tblW"))
    table_width.set(qn(W_NS, "w"), "9360")
    table_width.set(qn(W_NS, "type"), "dxa")
    indent = etree.SubElement(properties, qn(W_NS, "tblInd"))
    indent.set(qn(W_NS, "w"), "120")
    indent.set(qn(W_NS, "type"), "dxa")
    layout = etree.SubElement(properties, qn(W_NS, "tblLayout"))
    layout.set(qn(W_NS, "type"), "fixed")
    margins = etree.SubElement(properties, qn(W_NS, "tblCellMar"))
    for side, value in (("top", 90), ("left", 120), ("bottom", 90), ("right", 120)):
        margin = etree.SubElement(margins, qn(W_NS, side))
        margin.set(qn(W_NS, "w"), str(value))
        margin.set(qn(W_NS, "type"), "dxa")
    borders = etree.SubElement(properties, qn(W_NS, "tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = etree.SubElement(borders, qn(W_NS, edge))
        border.set(qn(W_NS, "val"), "single")
        border.set(qn(W_NS, "sz"), "4")
        border.set(qn(W_NS, "space"), "0")
        border.set(qn(W_NS, "color"), BORDER_COLOR)

    grid = etree.SubElement(table, qn(W_NS, "tblGrid"))
    for width in widths:
        column = etree.SubElement(grid, qn(W_NS, "gridCol"))
        column.set(qn(W_NS, "w"), str(width))

    header_row = etree.SubElement(table, qn(W_NS, "tr"))
    header_properties = etree.SubElement(header_row, qn(W_NS, "trPr"))
    add_on_off(header_properties, "tblHeader")
    add_on_off(header_properties, "cantSplit")
    for width, header in zip(widths, headers):
        header_row.append(
            make_cell(
                width,
                [
                    paragraph(
                        [word_run(header, bold=True, color="FFFFFF", size=20)],
                        alignment="center",
                    )
                ],
                fill=HEADER_FILL,
            )
        )

    for row_index, row_cells in enumerate(rows):
        if len(row_cells) != len(widths):
            raise ValueError("row/width mismatch")
        row = etree.SubElement(table, qn(W_NS, "tr"))
        row_properties = etree.SubElement(row, qn(W_NS, "trPr"))
        add_on_off(row_properties, "cantSplit")
        fill = ALT_FILL if row_index % 2 == 0 else None
        for width, cell_paragraphs in zip(widths, row_cells):
            row.append(make_cell(width, cell_paragraphs, fill=fill))
    return table


def find_formula(formulas: list[etree._Element], expected: str) -> etree._Element:
    normalize = lambda value: value.replace("−", "-")
    matches = [
        formula
        for formula in formulas
        if normalize(formula_text(formula)) == normalize(expected)
    ]
    if not matches:
        raise RuntimeError(f"formula not found: {expected}")
    return matches[0]


def build_section(formulas: list[etree._Element]) -> list[etree._Element]:
    symbol = lambda name: find_formula(formulas, name)

    intro = paragraph(
        [
            word_run(
                "为避免符号、状态和结果字段混在一个段落中，以下按三类表格说明。状态名统一写为"
                "DAx_REGy：第一位DA表示是否完成域适应，第二位REG表示是否完成新类注册，0/1分别"
                "表示未完成/已完成。准确率与调和均值以%报告，差值以百分点（percentage points，pp）"
                "报告，运行时间以秒（second，s）报告；Δ表示相对正式LEO结果的变化量。",
                size=21,
            )
        ],
        after=80,
    )

    symbol_rows = [
        [[math_paragraph(symbol("Qold"), symbol("Qnew"))], [text_paragraph("旧类query集合与新类query集合。")]],
        [[math_paragraph(symbol("Qc"))], [text_paragraph("旧类c的query子集。")]],
        [[math_paragraph(symbol("i"))], [text_paragraph("query样本索引。")]],
        [[math_paragraph(symbol("yi"))], [text_paragraph("第i个query样本的真实类别。")]],
        [
            [math_paragraph(symbol("yiDA0_REG0"), symbol("yiDA1_REG0"))],
            [text_paragraph("Stage2-B域适应前与域适应后的预测类别。")],
        ],
        [[math_paragraph(symbol("I[⋅]"))], [text_paragraph("指示函数：条件成立取1，否则取0。")]],
        [[math_paragraph(symbol("Yold"))], [text_paragraph("Phase1已经注册的旧发射机类别集合。")]],
    ]
    symbol_table = make_table([1800, 7560], ["符号", "定义与物理意义"], symbol_rows)

    state_rows = [
        [[text_paragraph("DA0_REG0", alignment="center")], [text_paragraph("否", alignment="center")], [text_paragraph("否", alignment="center")], [text_paragraph("域适应前、注册前；Stage2-B的直接基座状态。")]],
        [[text_paragraph("DA1_REG0", alignment="center")], [text_paragraph("是", alignment="center")], [text_paragraph("否", alignment="center")], [text_paragraph("域适应后、注册前；Stage2-B的主要评价状态。")]],
        [[text_paragraph("DA0_REG1", alignment="center")], [text_paragraph("否", alignment="center")], [text_paragraph("是", alignment="center")], [text_paragraph("域适应前、注册后；用于分离新类注册带来的变化。")]],
        [[text_paragraph("DA1_REG1", alignment="center")], [text_paragraph("是", alignment="center")], [text_paragraph("是", alignment="center")], [text_paragraph("域适应后、注册后；Stage2-C的旧新联合评价状态。")]],
    ]
    state_table = make_table(
        [1500, 1400, 1500, 4960],
        ["状态", "完成域适应", "完成新类注册", "对应阶段与含义"],
        state_rows,
    )

    metric_specs = [
        (
            ["AoldDA0_REG0=1Qoldi∈Qold\u200bIyiDA0_REG0=yi"],
            "old_acc_before",
            "Stage2-B域适应前（DA0_REG0）的旧类准确率；数值越大越好。",
        ),
        (
            ["AoldDA1_REG0=1Qoldi∈Qold\u200bIyiDA1_REG0=yi"],
            "old_acc_after",
            "Stage2-B域适应后（DA1_REG0）的旧类准确率；数值越大越好。",
        ),
        (
            ["Gold=AoldDA1_REG0-AoldDA0_REG0"],
            "派生量",
            "旧类适配收益。正值表示正迁移，负值表示适配损伤；数值越大越好。",
        ),
        (
            ["Anew=1Qnewi∈Qnew\u200bIyiDA1_REG1=yi"],
            "seen_new_acc",
            "已注册新类准确率；取0表示没有新类query被正确识别，数值越大越好。",
        ),
        (
            ["Hold,new=2AoldDA1_REG1AnewAoldDA1_REG1+Anew"],
            "H_old_new",
            "旧类与新类准确率的调和均值；任一侧接近0时该值也接近0，数值越大越好。",
        ),
        (
            ["Fold=AoldDA1_REG0-AoldDA1_REG1"],
            "forgetting",
            "注册新类前后旧类准确率之差；正值表示旧知识受损，数值越小越好。",
        ),
        (
            ["Ac=1Qci∈Qc\u200bIyiDA1_REG1=yi", "Amin,old=minc∈YoldAc"],
            "min_old",
            "全部旧发射机中的最低单类准确率，用于暴露局部类别崩塌；数值越大越好。",
        ),
    ]
    metric_rows: list[list[list[etree._Element]]] = []
    for formula_names, field, meaning in metric_specs:
        metric_formula_paragraphs = [
            math_paragraph(find_formula(formulas, formula_name))
            for formula_name in formula_names
        ]
        metric_rows.append(
            [
                metric_formula_paragraphs,
                [text_paragraph(field, alignment="center")],
                [text_paragraph(meaning)],
            ]
        )
    metric_table = make_table(
        [4300, 1500, 3560],
        ["指标与公式", "结果字段", "物理意义与判读"],
        metric_rows,
    )

    return [
        intro,
        label_paragraph("（1）数据对象与基础符号"),
        symbol_table,
        label_paragraph("（2）四状态编码"),
        state_table,
        label_paragraph("（3）核心评价指标"),
        metric_table,
    ]


def patch_document_xml(xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml, parser)
    body = root.find(qn(W_NS, "body"))
    if body is None:
        raise RuntimeError("document body not found")
    children = list(body)
    starts = [i for i, child in enumerate(children) if visible_text(child).strip() == "2.3评价指标"]
    ends = [i for i, child in enumerate(children) if visible_text(child).strip().startswith("2.4")]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise RuntimeError(f"unable to isolate section 2.3: starts={starts}, ends={ends}")
    start, end = starts[0], ends[0]
    old_section = children[start + 1 : end]
    if any(element.tag != qn(W_NS, "p") for element in old_section):
        raise RuntimeError("section 2.3 already contains non-paragraph content")
    old_text = "".join(visible_text(element) for element in old_section)
    for required in ("2.3.1旧类适应", "2.3.2新类注册", "2.3.3旧新联合评价"):
        if required not in old_text:
            raise RuntimeError(f"expected section label not found: {required}")

    formulas = [
        formula
        for element in old_section
        for formula in element.xpath(".//m:oMath", namespaces=NS)
    ]
    replacements = build_section(formulas)
    for element in old_section:
        body.remove(element)
    insertion_index = start + 1
    for offset, element in enumerate(replacements):
        body.insert(insertion_index + offset, element)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def tabulate_evaluation_metrics(source: Path | str, output: Path | str) -> None:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("source and output must differ")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(source_path, "r") as source_archive, ZipFile(
        output_path, "w", compression=ZIP_DEFLATED
    ) as output_archive:
        for info in source_archive.infolist():
            data = source_archive.read(info.filename)
            if info.filename == DOCUMENT_XML:
                data = patch_document_xml(data)
            output_archive.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabulate Phase2 evaluation metrics")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tabulate_evaluation_metrics(args.source, args.output)


if __name__ == "__main__":
    main()
