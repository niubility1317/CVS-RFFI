from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from integrate_phase2_trusted_support_context import (
    DOCUMENT_XML,
    HEADING_RENAMES,
    INSERTED_PARAGRAPHS,
    integrate_context,
)
from normalize_phase2_report_typography import (
    BODY_SIZE,
    EAST_ASIA_FONT,
    LATIN_FONT,
    STYLE_SIZES,
    TABLE_SIZE,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "docs/weekly_reports/"
    "CVS-RFFI_Phase2详细复现报告1_ERTB-IDR版_评价指标表格与字体统一_截至20260817.docx"
)
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}


def qn(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def package_entries(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def table_text(table: etree._Element) -> tuple[str, ...]:
    return tuple(table.xpath(".//w:t/text()", namespaces=NS))


def expected_size(paragraph: etree._Element) -> str:
    if paragraph.xpath("ancestor::w:tc", namespaces=NS):
        return TABLE_SIZE
    style_ids = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return STYLE_SIZES.get(style_ids[0], BODY_SIZE) if style_ids else BODY_SIZE


class IntegratePhase2TrustedSupportContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.output = Path(self.tempdir.name) / "integrated.docx"
        integrate_context(SOURCE, self.output)
        self.before = package_entries(SOURCE)
        self.after = package_entries(self.output)
        self.before_root = etree.fromstring(self.before[DOCUMENT_XML])
        self.after_root = etree.fromstring(self.after[DOCUMENT_XML])

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_adds_context_in_logical_order_and_renumbers_section_one(self):
        paragraphs = self.after_root.xpath(".//w:body/w:p", namespaces=NS)
        texts = [paragraph_text(paragraph) for paragraph in paragraphs]
        headings = [
            paragraph_text(paragraph)
            for paragraph in paragraphs
            if paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS) == ["4"]
            and paragraph_text(paragraph).startswith("1.")
        ]
        expected_headings = [
            "1.1研究动机：为什么在卫星上做RFFI",
            "1.2CVS具体场景：地面训练、星上部署与受控注册",
            "1.3Phase2可信标签与K-shot support如何获取",
            *HEADING_RENAMES.values(),
        ]
        self.assertEqual(headings[: len(expected_headings)], expected_headings)
        self.assertTrue(any("可信support形成链" in text for text in texts))
        self.assertTrue(any("unknown query不得事后改成support" in text for text in texts))
        self.assertTrue(any("RFFI定位为物理一致性辅助证据" in text for text in texts))
        self.assertEqual(
            sum(1 for text in texts if text.startswith("1.1研究动机")),
            1,
        )
        self.assertGreater(len(INSERTED_PARAGRAPHS), 20)

    def test_preserves_existing_tables_formulas_and_reference_list(self):
        before_tables = self.before_root.xpath(".//w:tbl", namespaces=NS)
        after_tables = self.after_root.xpath(".//w:tbl", namespaces=NS)
        self.assertEqual(len(before_tables), 40)
        self.assertEqual(
            [table_text(table) for table in before_tables],
            [table_text(table) for table in after_tables],
        )
        self.assertEqual(
            self.before_root.xpath(".//m:t/text()", namespaces=NS),
            self.after_root.xpath(".//m:t/text()", namespaces=NS),
        )

        before_texts = [
            paragraph_text(paragraph)
            for paragraph in self.before_root.xpath(".//w:body/w:p", namespaces=NS)
        ]
        after_texts = [
            paragraph_text(paragraph)
            for paragraph in self.after_root.xpath(".//w:body/w:p", namespaces=NS)
        ]
        before_refs = before_texts[before_texts.index("5.参考文献") :]
        after_refs = after_texts[after_texts.index("5.参考文献") :]
        self.assertEqual(before_refs, after_refs)

    def test_keeps_fonts_and_sizes_uniform(self):
        for paragraph in self.after_root.xpath(".//w:p", namespaces=NS):
            size = expected_size(paragraph)
            for run in paragraph.xpath(".//w:r", namespaces=NS):
                properties = run.find(qn("rPr"))
                self.assertIsNotNone(properties)
                fonts = properties.find(qn("rFonts"))
                self.assertIsNotNone(fonts)
                self.assertEqual(fonts.get(qn("eastAsia")), EAST_ASIA_FONT)
                self.assertEqual(fonts.get(qn("ascii")), LATIN_FONT)
                self.assertEqual(fonts.get(qn("hAnsi")), LATIN_FONT)
                self.assertEqual(properties.xpath("./w:sz/@w:val", namespaces=NS), [size])
                self.assertEqual(properties.xpath("./w:szCs/@w:val", namespaces=NS), [size])


if __name__ == "__main__":
    unittest.main()
