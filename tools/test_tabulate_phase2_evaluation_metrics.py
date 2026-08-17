from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from tabulate_phase2_evaluation_metrics import tabulate_evaluation_metrics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "docs/weekly_reports/"
    "CVS-RFFI_Phase2详细复现报告1_ERTB-IDR版_指标定义补充_截至20260817.docx"
)
DOCUMENT_XML = "word/document.xml"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}


def package_entries(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def text(element) -> str:
    return "".join(element.xpath(".//w:t/text() | .//m:t/text()", namespaces=NS))


def section_bounds(root) -> tuple[etree._Element, int, int]:
    body = root.find(f"{{{W_NS}}}body")
    children = list(body)
    starts = [i for i, child in enumerate(children) if text(child).strip() == "2.3评价指标"]
    ends = [i for i, child in enumerate(children) if text(child).strip().startswith("2.4")]
    if len(starts) != 1 or len(ends) != 1:
        raise AssertionError((starts, ends))
    return body, starts[0], ends[0]


def canonical_outside_section(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    body, start, end = section_bounds(root)
    for child in list(body)[start + 1 : end]:
        body.remove(child)
    return etree.tostring(root, method="c14n")


class TabulatePhase2EvaluationMetricsTests(unittest.TestCase):
    def test_replaces_dense_metric_prose_with_three_readable_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "metrics-tables.docx"
            tabulate_evaluation_metrics(SOURCE, output)
            entries = package_entries(output)
            root = etree.fromstring(entries[DOCUMENT_XML])
            body, start, end = section_bounds(root)
            section = list(body)[start + 1 : end]

            tables = [element for element in section if element.tag == f"{{{W_NS}}}tbl"]
            self.assertEqual(len(tables), 3)
            section_text = "".join(text(element) for element in section)
            for phrase in (
                "数据对象与基础符号",
                "四状态编码",
                "核心评价指标",
                "第一位DA表示是否完成域适应",
                "第二位REG表示是否完成新类注册",
                "结果字段",
                "数值越大越好",
                "数值越小越好",
                "percentage points",
            ):
                self.assertIn(phrase, section_text)
            self.assertNotIn("为同时保证数学表达和实验字段可追溯", section_text)
            self.assertNotIn("2.3.1旧类适应", section_text)
            self.assertNotIn("2.3.2新类注册", section_text)
            self.assertNotIn("2.3.3旧新联合评价", section_text)

            formula_texts = [
                "".join(node.xpath(".//m:t/text()", namespaces=NS)).replace("−", "-")
                for element in section
                for node in element.xpath(".//m:oMath", namespaces=NS)
            ]
            for formula in (
                "AoldDA0_REG0=1Qoldi∈Qold\u200bIyiDA0_REG0=yi",
                "AoldDA1_REG0=1Qoldi∈Qold\u200bIyiDA1_REG0=yi",
                "Gold=AoldDA1_REG0-AoldDA0_REG0",
                "Anew=1Qnewi∈Qnew\u200bIyiDA1_REG1=yi",
                "Hold,new=2AoldDA1_REG1AnewAoldDA1_REG1+Anew",
                "Fold=AoldDA1_REG0-AoldDA1_REG1",
                "Ac=1Qci∈Qc\u200bIyiDA1_REG1=yi",
                "Amin,old=minc∈YoldAc",
            ):
                self.assertIn(formula, formula_texts)

    def test_uses_fixed_table_geometry_and_repeating_headers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "metrics-tables.docx"
            tabulate_evaluation_metrics(SOURCE, output)
            root = etree.fromstring(package_entries(output)[DOCUMENT_XML])
            body, start, end = section_bounds(root)
            tables = [
                element
                for element in list(body)[start + 1 : end]
                if element.tag == f"{{{W_NS}}}tbl"
            ]
            expected_grids = ([1800, 7560], [1500, 1400, 1500, 4960], [4300, 1500, 3560])
            for table, expected in zip(tables, expected_grids):
                grid = [
                    int(column.get(f"{{{W_NS}}}w"))
                    for column in table.xpath("./w:tblGrid/w:gridCol", namespaces=NS)
                ]
                self.assertEqual(grid, list(expected))
                self.assertEqual(
                    int(table.xpath("./w:tblPr/w:tblW/@w:w", namespaces=NS)[0]),
                    9360,
                )
                self.assertEqual(
                    int(table.xpath("./w:tblPr/w:tblInd/@w:w", namespaces=NS)[0]),
                    120,
                )
                self.assertTrue(
                    table.xpath("./w:tr[1]/w:trPr/w:tblHeader", namespaces=NS)
                )

    def test_preserves_every_other_package_part_and_all_content_outside_section_2_3(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "metrics-tables.docx"
            tabulate_evaluation_metrics(SOURCE, output)
            before = package_entries(SOURCE)
            after = package_entries(output)

            self.assertEqual(before.keys(), after.keys())
            for name in before:
                if name != DOCUMENT_XML:
                    self.assertEqual(before[name], after[name], name)
            self.assertEqual(
                canonical_outside_section(before[DOCUMENT_XML]),
                canonical_outside_section(after[DOCUMENT_XML]),
            )


if __name__ == "__main__":
    unittest.main()
