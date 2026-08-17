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
MARKER = "PHASE2_METRIC_DEFINITIONS_V1"
ACCENT_BLUE = "1F4E79"


def qn(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


def set_on_off(properties, local: str, enabled: bool) -> None:
    element = properties.find(qn(W_NS, local))
    if enabled and element is None:
        properties.append(etree.Element(qn(W_NS, local)))
    elif not enabled and element is not None:
        properties.remove(element)


def paragraph_properties(template, *, keep_next: bool) -> etree._Element:
    existing = template.find(qn(W_NS, "pPr"))
    properties = deepcopy(existing) if existing is not None else etree.Element(qn(W_NS, "pPr"))
    set_on_off(properties, "keepNext", keep_next)
    set_on_off(properties, "keepLines", True)

    spacing = properties.find(qn(W_NS, "spacing"))
    if spacing is None:
        spacing = etree.SubElement(properties, qn(W_NS, "spacing"))
    spacing.set(qn(W_NS, "before"), "0")
    spacing.set(qn(W_NS, "after"), "60")
    return properties


def word_run(text: str, *, bold: bool = False, color: str | None = None):
    run = etree.Element(qn(W_NS, "r"))
    properties = etree.SubElement(run, qn(W_NS, "rPr"))
    fonts = etree.SubElement(properties, qn(W_NS, "rFonts"))
    fonts.set(qn(W_NS, "ascii"), "Times New Roman")
    fonts.set(qn(W_NS, "hAnsi"), "Times New Roman")
    fonts.set(qn(W_NS, "eastAsia"), "宋体")
    if bold:
        etree.SubElement(properties, qn(W_NS, "b"))
        etree.SubElement(properties, qn(W_NS, "bCs"))
    if color is not None:
        color_element = etree.SubElement(properties, qn(W_NS, "color"))
        color_element.set(qn(W_NS, "val"), color)
    size = etree.SubElement(properties, qn(W_NS, "sz"))
    size.set(qn(W_NS, "val"), "21")
    size_cs = etree.SubElement(properties, qn(W_NS, "szCs"))
    size_cs.set(qn(W_NS, "val"), "21")
    text_element = etree.SubElement(run, qn(W_NS, "t"))
    if text.startswith(" ") or text.endswith(" "):
        text_element.set(qn(XML_NS, "space"), "preserve")
    text_element.text = text
    return run


def math_run(text: str):
    run = etree.Element(qn(M_NS, "r"))
    text_element = etree.SubElement(run, qn(M_NS, "t"))
    text_element.text = text
    return run


def math_container(local: str, *children):
    element = etree.Element(qn(M_NS, local))
    for child in children:
        element.append(child)
    return element


def math_sub(base: str, sub: str):
    return math_container(
        "sSub",
        etree.Element(qn(M_NS, "sSubPr")),
        math_container("e", math_run(base)),
        math_container("sub", math_run(sub)),
    )


def math_sub_sup(base: str, sub: str, sup: str):
    return math_container(
        "sSubSup",
        etree.Element(qn(M_NS, "sSubSupPr")),
        math_container("e", math_run(base)),
        math_container("sub", math_run(sub)),
        math_container("sup", math_run(sup)),
    )


def math_fraction(numerator: list, denominator: list):
    return math_container(
        "f",
        etree.Element(qn(M_NS, "fPr")),
        math_container("num", *numerator),
        math_container("den", *denominator),
    )


def old_accuracy(state: str):
    return math_sub_sup("A", "old", state)


def new_accuracy():
    return math_sub("A", "new")


def metric_formula(metric: str):
    equation = etree.Element(qn(M_NS, "oMath"))
    if metric == "K":
        equation.append(math_run("K"))
    elif metric == "old_accuracy":
        equation.extend(
            [
                old_accuracy("DA0_REG1"),
                math_run("="),
                math_fraction(
                    [math_sub_sup("N", "old", "correct")],
                    [math_sub_sup("N", "old", "query")],
                ),
            ]
        )
    elif metric == "new_accuracy":
        equation.extend(
            [
                new_accuracy(),
                math_run("="),
                math_fraction(
                    [math_sub_sup("N", "new", "correct")],
                    [math_sub_sup("N", "new", "query")],
                ),
            ]
        )
    elif metric == "harmonic":
        equation.extend(
            [
                math_sub("H", "old,new"),
                math_run("="),
                math_fraction(
                    [math_run("2"), old_accuracy("DA0_REG1"), new_accuracy()],
                    [old_accuracy("DA0_REG1"), math_run("+"), new_accuracy()],
                ),
            ]
        )
    elif metric == "forgetting":
        equation.extend(
            [
                math_sub("F", "old"),
                math_run("="),
                old_accuracy("DA0_REG0"),
                math_run("-"),
                old_accuracy("DA0_REG1"),
            ]
        )
    else:
        raise ValueError(metric)
    return equation


def metric_paragraph(template, index: int, parts: list, *, keep_next: bool = False):
    paragraph = etree.Element(qn(W_NS, "p"))
    paragraph.append(paragraph_properties(template, keep_next=keep_next))
    bookmark_id = str(9100 + index)
    bookmark_start = etree.SubElement(paragraph, qn(W_NS, "bookmarkStart"))
    bookmark_start.set(qn(W_NS, "id"), bookmark_id)
    bookmark_start.set(qn(W_NS, "name"), f"{MARKER}_{index}")
    for part in parts:
        paragraph.append(part)
    bookmark_end = etree.SubElement(paragraph, qn(W_NS, "bookmarkEnd"))
    bookmark_end.set(qn(W_NS, "id"), bookmark_id)
    return paragraph


def build_metric_block(template) -> list:
    return [
        metric_paragraph(
            template,
            1,
            [word_run("表中指标定义与判读", bold=True, color=ACCENT_BLUE)],
            keep_next=True,
        ),
        metric_paragraph(
            template,
            2,
            [
                word_run("配置标题中的"),
                metric_formula("K"),
                word_run(
                    "表示每个新类可用的独立support样本数，“新类数”表示本次同时注册的新发射机"
                    "类别数量。以下准确率和调和均值均在冻结模型的query上计算，query不参与训练"
                    "或参数更新；准确率与调和均值按百分比（%）报告，遗忘量按百分点"
                    "（percentage point, pp）判读。"
                ),
            ],
        ),
        metric_paragraph(
            template,
            3,
            [
                metric_formula("old_accuracy"),
                word_run(
                    "（注册后旧类准确率）：分子为正确识别的旧类query样本数，分母为旧类query"
                    "总数。上标表示未执行目标域适应、但已完成新类注册。该指标衡量注册新类后"
                    "模型保留原有发射机身份的能力，数值越大越好。"
                ),
            ],
        ),
        metric_paragraph(
            template,
            4,
            [
                metric_formula("new_accuracy"),
                word_run(
                    "（已注册新类准确率）：分子为正确识别的新类query样本数，分母为新类query"
                    "总数。本节表格省略状态上标，但统计状态同样为未执行目标域适应且已完成新类"
                    "注册。该指标衡量少量support是否建立了可用的新发射机身份边界，数值越大越好；"
                    "取0表示没有任何新类query被正确识别。"
                ),
            ],
        ),
        metric_paragraph(
            template,
            5,
            [
                metric_formula("harmonic"),
                word_run(
                    "（旧新类调和均值）：同时衡量旧类保留与新类注册，分母为两项准确率之和；"
                    "当分母为0时，该指标按0计。任一侧准确率为0都会使调和均值为0，因此它比"
                    "算术平均更能暴露“只保旧类”或“只学新类”的单侧失效，数值越大越好。"
                ),
            ],
        ),
        metric_paragraph(
            template,
            6,
            [
                metric_formula("forgetting"),
                word_run(
                    "（旧类遗忘量）：用新类注册前的旧类准确率减去注册后的旧类准确率。前一状态"
                    "表示未执行域适应且尚未注册新类，后一状态表示未执行域适应但已经完成新类注册。"
                    "表中由两个百分比数值直接相减，单位按百分点（pp）理解；正值表示旧类性能下降，"
                    "0表示没有下降，负值表示注册后旧类性能反而提高，数值越小越好。"
                ),
            ],
        ),
    ]


def patch_document_xml(xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml, parser)
    existing = root.xpath(
        ".//w:body/w:p[w:bookmarkStart[starts-with(@w:name, $marker)]]",
        namespaces=NS,
        marker=MARKER,
    )
    if existing:
        raise RuntimeError("metric definition block already exists")

    captions = [
        paragraph
        for paragraph in root.xpath(
            ".//w:body/w:p[starts-with(normalize-space(string(.)), '配置：')]",
            namespaces=NS,
        )
        if "".join(
            paragraph.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS)
        )
        == "配置：K=5，新类数=1"
    ]
    if len(captions) != 1:
        raise RuntimeError(f"expected one first Stage2-C result caption, found {len(captions)}")
    caption = captions[0]
    template = caption.getprevious()
    while template is not None and template.tag != qn(W_NS, "p"):
        template = template.getprevious()
    if template is None:
        raise RuntimeError("unable to locate Stage2-C result introduction paragraph")
    parent = caption.getparent()
    insertion_index = parent.index(caption)
    for offset, paragraph in enumerate(build_metric_block(template)):
        parent.insert(insertion_index + offset, paragraph)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def add_metric_definitions(source: Path | str, output: Path | str) -> None:
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
    parser = argparse.ArgumentParser(description="Add detailed Stage2-C metric definitions")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    add_metric_definitions(args.source, args.output)


if __name__ == "__main__":
    main()
