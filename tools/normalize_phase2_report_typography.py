from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}
DOCUMENT_XML = "word/document.xml"
STYLES_XML = "word/styles.xml"

EAST_ASIA_FONT = "宋体"
LATIN_FONT = "Times New Roman"

# Word stores font size in half-points.
BODY_SIZE = "24"  # 12 pt
TABLE_SIZE = "20"  # 10 pt
STYLE_SIZES = {
    "1": BODY_SIZE,  # Normal
    "19": BODY_SIZE,  # Body Text
    "3": "36",  # Heading 1, 18 pt
    "4": "28",  # Heading 2, 14 pt
    "5": "24",  # Heading 3, 12 pt
}


def qn(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def ensure_child(parent: etree._Element, local: str, *, first: bool = False) -> etree._Element:
    children = parent.findall(qn(local))
    if children:
        for duplicate in children[1:]:
            parent.remove(duplicate)
        return children[0]
    child = etree.Element(qn(local))
    if first:
        parent.insert(0, child)
    else:
        parent.append(child)
    return child


def ensure_run_properties(parent: etree._Element) -> etree._Element:
    properties = parent.find(qn("rPr"))
    if properties is None:
        properties = etree.Element(qn("rPr"))
        parent.insert(0, properties)
    return properties


def set_run_fonts(properties: etree._Element) -> None:
    fonts = ensure_child(properties, "rFonts", first=True)
    for theme_attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(qn(theme_attribute), None)
    fonts.set(qn("ascii"), LATIN_FONT)
    fonts.set(qn("hAnsi"), LATIN_FONT)
    fonts.set(qn("cs"), LATIN_FONT)
    fonts.set(qn("eastAsia"), EAST_ASIA_FONT)


def set_run_size(properties: etree._Element, size: str) -> None:
    font_size = ensure_child(properties, "sz")
    font_size.set(qn("val"), size)
    complex_size = ensure_child(properties, "szCs")
    complex_size.set(qn("val"), size)


def paragraph_size(paragraph: etree._Element) -> str:
    if paragraph.xpath("ancestor::w:tc", namespaces=NS):
        return TABLE_SIZE
    style_ids = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    if style_ids:
        return STYLE_SIZES.get(style_ids[0], BODY_SIZE)
    return BODY_SIZE


def patch_document_xml(xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml, parser)
    paragraphs = root.xpath(".//w:p", namespaces=NS)
    if not paragraphs:
        raise RuntimeError("no paragraphs found")
    for paragraph in paragraphs:
        size = paragraph_size(paragraph)
        for run in paragraph.xpath(".//w:r", namespaces=NS):
            properties = ensure_run_properties(run)
            set_run_fonts(properties)
            set_run_size(properties, size)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def ensure_style_run_properties(style: etree._Element) -> etree._Element:
    properties = style.find(qn("rPr"))
    if properties is None:
        properties = etree.SubElement(style, qn("rPr"))
    return properties


def patch_styles_xml(xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml, parser)

    defaults = root.find(qn("docDefaults"))
    if defaults is None:
        defaults = etree.Element(qn("docDefaults"))
        root.insert(0, defaults)
    run_defaults = ensure_child(defaults, "rPrDefault", first=True)
    default_properties = ensure_child(run_defaults, "rPr", first=True)
    set_run_fonts(default_properties)
    set_run_size(default_properties, BODY_SIZE)

    for style in root.xpath("./w:style", namespaces=NS):
        properties = ensure_style_run_properties(style)
        style_id = style.get(qn("styleId"), "")
        set_run_fonts(properties)
        if style_id in STYLE_SIZES:
            set_run_size(properties, STYLE_SIZES[style_id])

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def normalize_typography(source: Path | str, output: Path | str) -> None:
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
            elif info.filename == STYLES_XML:
                data = patch_styles_xml(data)
            output_archive.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Phase2 report fonts and sizes by text hierarchy"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    normalize_typography(args.source, args.output)


if __name__ == "__main__":
    main()
