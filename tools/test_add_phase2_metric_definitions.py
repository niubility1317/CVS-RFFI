from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from add_phase2_metric_definitions import add_metric_definitions


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / (
    "docs/weekly_reports/"
    "CVS-RFFI_Phase2详细复现报告1_正式中英文命名与ERTB-IDR版_截至20260817.docx"
)
DOCUMENT_XML = "word/document.xml"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
NS = {"w": W_NS, "m": M_NS}
MARKER = "PHASE2_METRIC_DEFINITIONS_V1"


def package_entries(path: Path) -> dict[str, bytes]:
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def canonical_without_inserted_metrics(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    for paragraph in root.xpath(
        ".//w:body/w:p[w:bookmarkStart[starts-with(@w:name, $marker)]]",
        namespaces=NS,
        marker=MARKER,
    ):
        paragraph.getparent().remove(paragraph)
    return etree.tostring(root, method="c14n")


class AddPhase2MetricDefinitionsTests(unittest.TestCase):
    def test_inserts_detailed_word_equation_definitions_before_first_result_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "metrics.docx"
            add_metric_definitions(SOURCE, output)
            entries = package_entries(output)
            root = etree.fromstring(entries[DOCUMENT_XML])

            inserted = root.xpath(
                ".//w:body/w:p[w:bookmarkStart[starts-with(@w:name, $marker)]]",
                namespaces=NS,
                marker=MARKER,
            )
            self.assertEqual(len(inserted), 6)
            flattened = "".join(
                text
                for inserted_paragraph in inserted
                for text in inserted_paragraph.xpath(
                    ".//w:t/text() | .//m:t/text()", namespaces=NS
                )
            )
            for phrase in (
                "表中指标定义与判读",
                "注册后旧类准确率",
                "已注册新类准确率",
                "旧新类调和均值",
                "旧类遗忘量",
                "数值越大越好",
                "数值越小越好",
            ):
                self.assertIn(phrase, flattened)

            formula_texts = [
                "".join(node.xpath(".//m:t/text()", namespaces=NS))
                for paragraph in inserted
                for node in paragraph.xpath("./m:oMath", namespaces=NS)
            ]
            self.assertEqual(len(formula_texts), 5)
            for token in (
                "K",
                "AoldDA0_REG1=NoldcorrectNoldquery",
                "Anew=NnewcorrectNnewquery",
                "Hold,new=2AoldDA0_REG1AnewAoldDA0_REG1+Anew",
                "Fold=AoldDA0_REG0-AoldDA0_REG1",
            ):
                self.assertIn(token, formula_texts)

            body = root.find(f"{{{W_NS}}}body")
            children = list(body)
            last_inserted_index = children.index(inserted[-1])
            self.assertEqual(
                "".join(
                    children[last_inserted_index + 1].xpath(
                        ".//w:t/text() | .//m:t/text()", namespaces=NS
                    )
                ),
                "配置：K=5，新类数=1",
            )

    def test_preserves_every_non_document_part_and_all_preexisting_document_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "metrics.docx"
            add_metric_definitions(SOURCE, output)
            before = package_entries(SOURCE)
            after = package_entries(output)

            self.assertEqual(before.keys(), after.keys())
            for name in before:
                if name != DOCUMENT_XML:
                    self.assertEqual(before[name], after[name], name)
            self.assertEqual(
                canonical_without_inserted_metrics(before[DOCUMENT_XML]),
                canonical_without_inserted_metrics(after[DOCUMENT_XML]),
            )


if __name__ == "__main__":
    unittest.main()
