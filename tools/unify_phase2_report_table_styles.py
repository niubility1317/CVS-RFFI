from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
DOCUMENT_XML = "word/document.xml"
HEADER_FILL = "D9E2F3"
HEADER_TEXT = "1F4E79"
ALT_FILL = "F7F9FC"
WHITE_FILL = "FFFFFF"
BORDER_COLOR = "A6B4C2"


def qn(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def ensure_properties(parent, local: str) -> etree._Element:
    properties = parent.find(qn(local))
    if properties is None:
        properties = etree.Element(qn(local))
        parent.insert(0, properties)
    return properties


def replace_single(parent, local: str) -> etree._Element:
    for element in list(parent.findall(qn(local))):
        parent.remove(element)
    return etree.SubElement(parent, qn(local))


def ensure_on_off(parent, local: str) -> etree._Element:
    elements = parent.findall(qn(local))
    if elements:
        for duplicate in elements[1:]:
            parent.remove(duplicate)
        return elements[0]
    return etree.SubElement(parent, qn(local))


def style_header_run(run: etree._Element) -> None:
    properties = ensure_properties(run, "rPr")
    ensure_on_off(properties, "b")
    ensure_on_off(properties, "bCs")
    color = replace_single(properties, "color")
    color.set(qn("val"), HEADER_TEXT)


def style_cell(cell: etree._Element, *, fill: str, header: bool) -> None:
    properties = ensure_properties(cell, "tcPr")
    shading = replace_single(properties, "shd")
    shading.set(qn("val"), "clear")
    shading.set(qn("color"), "auto")
    shading.set(qn("fill"), fill)
    vertical = replace_single(properties, "vAlign")
    vertical.set(qn("val"), "center")
    if header:
        for paragraph in cell.xpath(".//w:p", namespaces=NS):
            paragraph_properties = ensure_properties(paragraph, "pPr")
            justification = replace_single(paragraph_properties, "jc")
            justification.set(qn("val"), "center")
        for run in cell.xpath(".//w:r", namespaces=NS):
            style_header_run(run)


def style_table(table: etree._Element) -> None:
    properties = ensure_properties(table, "tblPr")
    style = replace_single(properties, "tblStyle")
    style.set(qn("val"), "33")

    for old in list(properties.findall(qn("tblBorders"))):
        properties.remove(old)
    borders = etree.SubElement(properties, qn("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = etree.SubElement(borders, qn(edge))
        border.set(qn("val"), "single")
        border.set(qn("sz"), "4")
        border.set(qn("space"), "0")
        border.set(qn("color"), BORDER_COLOR)

    for old in list(properties.findall(qn("tblCellMar"))):
        properties.remove(old)
    margins = etree.SubElement(properties, qn("tblCellMar"))
    for side, value in (("top", 90), ("left", 120), ("bottom", 90), ("right", 120)):
        margin = etree.SubElement(margins, qn(side))
        margin.set(qn("w"), str(value))
        margin.set(qn("type"), "dxa")

    rows = table.xpath("./w:tr", namespaces=NS)
    if not rows:
        return
    for row_index, row in enumerate(rows):
        row_properties = ensure_properties(row, "trPr")
        ensure_on_off(row_properties, "cantSplit")
        if row_index == 0:
            ensure_on_off(row_properties, "tblHeader")
            fill = HEADER_FILL
        else:
            fill = ALT_FILL if row_index % 2 == 1 else WHITE_FILL
        for cell in row.xpath("./w:tc", namespaces=NS):
            style_cell(cell, fill=fill, header=row_index == 0)


def patch_document_xml(xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml, parser)
    tables = root.xpath(".//w:tbl", namespaces=NS)
    if not tables:
        raise RuntimeError("no tables found")
    for table in tables:
        style_table(table)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def unify_table_styles(source: Path | str, output: Path | str) -> None:
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
    parser = argparse.ArgumentParser(description="Apply one visual style to every report table")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    unify_table_styles(args.source, args.output)


if __name__ == "__main__":
    main()
