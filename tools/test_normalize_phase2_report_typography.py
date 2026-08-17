from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from normalize_phase2_report_typography import (
    BODY_SIZE,
    DOCUMENT_XML,
    EAST_ASIA_FONT,
    LATIN_FONT,
    STYLES_XML,
    STYLE_SIZES,
    TABLE_SIZE,
    normalize_typography,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "docs/weekly_reports/"
    "CVS-RFFI_Phase2详细复现报告1_ERTB-IDR版_评价指标与表格样式统一_截至20260817.docx"
)
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}


def qn(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def package_entries(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def expected_paragraph_size(paragraph: etree._Element) -> str:
    if paragraph.xpath("ancestor::w:tc", namespaces=NS):
        return TABLE_SIZE
    style_ids = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return STYLE_SIZES.get(style_ids[0], BODY_SIZE) if style_ids else BODY_SIZE


def assert_typography(test: unittest.TestCase, properties: etree._Element, size: str) -> None:
    fonts = properties.xpath("./w:rFonts", namespaces=NS)
    test.assertEqual(len(fonts), 1)
    test.assertEqual(fonts[0].get(qn("eastAsia")), EAST_ASIA_FONT)
    test.assertEqual(fonts[0].get(qn("ascii")), LATIN_FONT)
    test.assertEqual(fonts[0].get(qn("hAnsi")), LATIN_FONT)
    test.assertEqual(fonts[0].get(qn("cs")), LATIN_FONT)
    for theme_attribute in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        test.assertIsNone(fonts[0].get(qn(theme_attribute)))
    test.assertEqual(properties.xpath("./w:sz/@w:val", namespaces=NS), [size])
    test.assertEqual(properties.xpath("./w:szCs/@w:val", namespaces=NS), [size])


def strip_typography(root: etree._Element) -> bytes:
    root = copy.deepcopy(root)
    for properties in root.xpath(".//w:rPr", namespaces=NS):
        for local in ("rFonts", "sz", "szCs"):
            for child in list(properties.findall(qn(local))):
                properties.remove(child)
        if len(properties) == 0 and not properties.attrib and properties.getparent() is not None:
            properties.getparent().remove(properties)
    return etree.tostring(root, method="c14n")


class NormalizePhase2ReportTypographyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.output = Path(self.tempdir.name) / "normalized.docx"
        normalize_typography(SOURCE, self.output)
        self.before = package_entries(SOURCE)
        self.after = package_entries(self.output)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_normalizes_document_runs_by_hierarchy(self):
        root = etree.fromstring(self.after[DOCUMENT_XML])
        paragraphs = root.xpath(".//w:p", namespaces=NS)
        self.assertTrue(paragraphs)
        for paragraph in paragraphs:
            size = expected_paragraph_size(paragraph)
            for run in paragraph.xpath(".//w:r", namespaces=NS):
                properties = run.find(qn("rPr"))
                self.assertIsNotNone(properties)
                assert_typography(self, properties, size)

    def test_updates_defaults_and_named_styles(self):
        root = etree.fromstring(self.after[STYLES_XML])
        defaults = root.xpath("./w:docDefaults/w:rPrDefault/w:rPr", namespaces=NS)
        self.assertEqual(len(defaults), 1)
        assert_typography(self, defaults[0], BODY_SIZE)

        for style_id, size in STYLE_SIZES.items():
            styles = root.xpath(f'./w:style[@w:styleId="{style_id}"]', namespaces=NS)
            self.assertEqual(len(styles), 1, style_id)
            properties = styles[0].find(qn("rPr"))
            self.assertIsNotNone(properties)
            assert_typography(self, properties, size)

    def test_preserves_content_formulas_layout_and_unrelated_parts(self):
        self.assertEqual(self.before.keys(), self.after.keys())
        for name in self.before:
            if name not in {DOCUMENT_XML, STYLES_XML}:
                self.assertEqual(self.before[name], self.after[name], name)

        before_document = etree.fromstring(self.before[DOCUMENT_XML])
        after_document = etree.fromstring(self.after[DOCUMENT_XML])
        self.assertEqual(
            before_document.xpath(".//w:t/text()", namespaces=NS),
            after_document.xpath(".//w:t/text()", namespaces=NS),
        )
        self.assertEqual(
            before_document.xpath(".//m:t/text()", namespaces=NS),
            after_document.xpath(".//m:t/text()", namespaces=NS),
        )
        self.assertEqual(
            [etree.tostring(node, method="c14n") for node in before_document.xpath(".//m:r", namespaces=NS)],
            [etree.tostring(node, method="c14n") for node in after_document.xpath(".//m:r", namespaces=NS)],
        )
        self.assertEqual(strip_typography(before_document), strip_typography(after_document))

        before_styles = etree.fromstring(self.before[STYLES_XML])
        after_styles = etree.fromstring(self.after[STYLES_XML])
        self.assertEqual(strip_typography(before_styles), strip_typography(after_styles))


if __name__ == "__main__":
    unittest.main()
