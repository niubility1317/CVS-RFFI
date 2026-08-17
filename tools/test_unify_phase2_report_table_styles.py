from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from unify_phase2_report_table_styles import unify_table_styles


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "docs/weekly_reports/"
    "CVS-RFFI_Phase2详细复现报告1_ERTB-IDR版_评价指标表格优化_截至20260817.docx"
)
DOCUMENT_XML = "word/document.xml"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}


def package_entries(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def table_geometry(table) -> tuple:
    return (
        tuple(table.xpath("./w:tblPr/w:tblW/@w:w", namespaces=NS)),
        tuple(table.xpath("./w:tblPr/w:tblInd/@w:w", namespaces=NS)),
        tuple(table.xpath("./w:tblGrid/w:gridCol/@w:w", namespaces=NS)),
        tuple(table.xpath(".//w:tc/w:tcPr/w:tcW/@w:w", namespaces=NS)),
    )


class UnifyPhase2ReportTableStylesTests(unittest.TestCase):
    def test_applies_one_visual_style_to_every_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "unified.docx"
            unify_table_styles(SOURCE, output)
            root = etree.fromstring(package_entries(output)[DOCUMENT_XML])
            tables = root.xpath(".//w:tbl", namespaces=NS)
            self.assertEqual(len(tables), 40)

            for table in tables:
                self.assertEqual(
                    table.xpath("./w:tblPr/w:tblStyle/@w:val", namespaces=NS),
                    ["33"],
                )
                for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                    borders = table.xpath(f"./w:tblPr/w:tblBorders/w:{edge}", namespaces=NS)
                    self.assertEqual(len(borders), 1)
                    self.assertEqual(borders[0].get(f"{{{W_NS}}}val"), "single")
                    self.assertEqual(borders[0].get(f"{{{W_NS}}}color"), "A6B4C2")
                    self.assertEqual(borders[0].get(f"{{{W_NS}}}sz"), "4")

                margins = {
                    node.tag.rsplit("}", 1)[-1]: node.get(f"{{{W_NS}}}w")
                    for node in table.xpath("./w:tblPr/w:tblCellMar/*", namespaces=NS)
                }
                self.assertEqual(
                    margins,
                    {"top": "90", "left": "120", "bottom": "90", "right": "120"},
                )

                rows = table.xpath("./w:tr", namespaces=NS)
                self.assertTrue(rows)
                self.assertTrue(rows[0].xpath("./w:trPr/w:tblHeader", namespaces=NS))
                for row in rows:
                    self.assertTrue(row.xpath("./w:trPr/w:cantSplit", namespaces=NS))
                for cell in rows[0].xpath("./w:tc", namespaces=NS):
                    self.assertEqual(
                        cell.xpath("./w:tcPr/w:shd/@w:fill", namespaces=NS),
                        ["D9E2F3"],
                    )
                    for run in cell.xpath(".//w:r", namespaces=NS):
                        self.assertTrue(run.xpath("./w:rPr/w:b", namespaces=NS))
                        self.assertEqual(
                            run.xpath("./w:rPr/w:color/@w:val", namespaces=NS),
                            ["1F4E79"],
                        )
                for row_index, row in enumerate(rows[1:], 1):
                    expected_fill = "F7F9FC" if row_index % 2 == 1 else "FFFFFF"
                    for cell in row.xpath("./w:tc", namespaces=NS):
                        self.assertEqual(
                            cell.xpath("./w:tcPr/w:shd/@w:fill", namespaces=NS),
                            [expected_fill],
                        )
                        self.assertEqual(
                            cell.xpath("./w:tcPr/w:vAlign/@w:val", namespaces=NS),
                            ["center"],
                        )

    def test_preserves_all_text_formulas_geometry_and_non_document_parts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "unified.docx"
            unify_table_styles(SOURCE, output)
            before = package_entries(SOURCE)
            after = package_entries(output)

            self.assertEqual(before.keys(), after.keys())
            for name in before:
                if name != DOCUMENT_XML:
                    self.assertEqual(before[name], after[name], name)

            before_root = etree.fromstring(before[DOCUMENT_XML])
            after_root = etree.fromstring(after[DOCUMENT_XML])
            self.assertEqual(
                before_root.xpath(".//w:t/text()", namespaces=NS),
                after_root.xpath(".//w:t/text()", namespaces=NS),
            )
            self.assertEqual(
                before_root.xpath(".//m:t/text()", namespaces=NS),
                after_root.xpath(".//m:t/text()", namespaces=NS),
            )
            before_tables = before_root.xpath(".//w:tbl", namespaces=NS)
            after_tables = after_root.xpath(".//w:tbl", namespaces=NS)
            self.assertEqual(
                [table_geometry(table) for table in before_tables],
                [table_geometry(table) for table in after_tables],
            )


if __name__ == "__main__":
    unittest.main()
