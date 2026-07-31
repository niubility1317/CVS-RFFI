from __future__ import annotations

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Sequence

from docx import Document
from docx.document import Document as _Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt

from revise_phase2_report_numbering_and_results import (
    HEADING_BLUE,
    add_result_table_before,
    append_paragraph_before,
    apply_captured_cell_content,
    capture_cell_content,
    clean_text,
    find_formula_header_table,
    find_paragraph,
    normalize_all_visible_run_fonts,
    replace_paragraph_text,
    set_cell_shading,
    set_repeat_table_header,
    set_row_cant_split,
    style_run,
    table_matrix,
)


STAGE2B_D92_TABLE = [
    [
        "K",
        "直接ADV3B02",
        "MRIOR-SDA",
        "DADDA-SDA",
        "ProtoNet CDA",
        "D92注册前旧类准确率",
    ],
    ["1", "75.21%", "69.88%", "72.58%", "59.47%", "68.144%"],
    ["5", "75.21%", "79.17%", "76.74%", "70.28%", "81.267%"],
    ["10", "75.21%", "84.50%", "79.36%", "70.86%", "86.111%"],
]


STAGE2C_D92_COMPARISON_TABLE = [
    ["切片", "方法", "适应前", "", "", "", ""],
    ["K1/new20", "D92", "68.144%", "44.033%", "27.150%", "33.410%", "24.111pp"],
    [
        "K1/new20",
        "CSIL官方流程",
        "42.833%",
        "42.833%",
        "0.000%",
        "0.000%",
        "0.000pp",
    ],
    [
        "K1/new20",
        "MoPC-HR官方流程",
        "45.322%",
        "40.722%",
        "1.363%",
        "2.603%",
        "4.600pp",
    ],
    ["K5/new20", "D92", "81.267%", "63.711%", "58.883%", "60.955%", "17.556pp"],
    [
        "K5/new20",
        "CSIL官方流程",
        "42.833%",
        "0.200%",
        "5.557%",
        "0.316%",
        "42.633pp",
    ],
    [
        "K5/new20",
        "MoPC-HR官方流程",
        "45.322%",
        "13.511%",
        "17.433%",
        "14.309%",
        "31.811pp",
    ],
    ["K10/new5", "D92", "86.111%", "76.189%", "74.133%", "74.803%", "9.922pp"],
    [
        "K10/new5",
        "CSIL官方流程",
        "42.833%",
        "0.689%",
        "20.413%",
        "1.264%",
        "42.144pp",
    ],
    [
        "K10/new5",
        "MoPC-HR官方流程",
        "45.322%",
        "9.322%",
        "49.547%",
        "14.947%",
        "36.000pp",
    ],
    ["K10/new10", "D92", "86.111%", "72.533%", "66.353%", "69.106%", "13.578pp"],
    [
        "K10/new10",
        "CSIL官方流程",
        "42.833%",
        "0.000%",
        "10.460%",
        "0.000%",
        "42.833pp",
    ],
    [
        "K10/new10",
        "MoPC-HR官方流程",
        "45.322%",
        "9.500%",
        "32.900%",
        "13.770%",
        "35.822pp",
    ],
    ["K10/new20", "D92", "86.111%", "71.333%", "68.150%", "69.555%", "14.778pp"],
    [
        "K10/new20",
        "CSIL官方流程",
        "42.833%",
        "38.222%",
        "1.660%",
        "2.979%",
        "4.611pp",
    ],
    [
        "K10/new20",
        "MoPC-HR官方流程",
        "45.322%",
        "7.611%",
        "25.187%",
        "10.695%",
        "37.711pp",
    ],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_table_by_math_signature(
    doc: _Document,
    signature: Sequence[str],
) -> object:
    matches = []
    for table in doc.tables:
        if len(table.columns) != len(signature):
            continue
        math_values = [
            "".join(cell._tc.xpath(".//m:t/text()"))
            for cell in table.rows[0].cells
        ]
        if math_values == list(signature):
            matches.append(table)
    if len(matches) != 1:
        raise ValueError(
            f"expected one table with math signature {signature!r}, found {len(matches)}"
        )
    return matches[0]


def style_inserted_formula_headers(table, formula_contents: Sequence[tuple[int, Sequence]]) -> None:
    for column_index, captured in formula_contents:
        apply_captured_cell_content(
            table.rows[0].cells[column_index],
            captured,
            font_size=7.8,
        )
        set_cell_shading(table.rows[0].cells[column_index], "E7E6E6")
    set_repeat_table_header(table.rows[0])
    for row in table.rows:
        set_row_cant_split(row)


def set_table_header_keep_with_next(table) -> None:
    for cell in table.rows[0].cells:
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for paragraph in cell.paragraphs:
            paragraph.paragraph_format.keep_with_next = True
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                style_run(run, bold=True, color=HEADING_BLUE)


def revise(source: Path, output: Path) -> None:
    doc = Document(str(source))
    original_tables = [table_matrix(table) for table in doc.tables]
    if len(original_tables) != 14:
        raise RuntimeError(f"expected 14 source tables, found {len(original_tables)}")

    formal_table = find_formula_header_table(
        doc,
        column_count=7,
        math_signature=["", "", "", "Aoldpost", "Anew", "Hold,new", "Fold"],
        word_signature=["K-shot", "方法", "新类数", "", "", "", ""],
    )
    stage2b_overall = find_table_by_math_signature(
        doc,
        ["", "Aold", "Aold", "Gold", "", "", ""],
    )
    pre_header = capture_cell_content(stage2b_overall.rows[0].cells[1])
    post_header = capture_cell_content(formal_table.rows[0].cells[3])
    new_header = capture_cell_content(formal_table.rows[0].cells[4])
    h_header = capture_cell_content(formal_table.rows[0].cells[5])
    forgetting_header = capture_cell_content(formal_table.rows[0].cells[6])

    boundary_heading = find_paragraph(doc, "3.6.4结果边界")
    replace_paragraph_text(boundary_heading, "3.6.5结果边界")
    d92_stage2b_heading = append_paragraph_before(
        doc,
        boundary_heading,
        "3.6.4D92注册前共同LEO场景对照",
        style="Heading 3",
        keep_with_next=True,
    )
    append_paragraph_before(
        doc,
        boundary_heading,
        (
            "本表补充D92正式retry2在注册新类之前的旧类状态，并与D92技术报告中保留的"
            "共同LEO弱信道场景矩阵并列。它不是在3.6.2周报表上直接追加一列：域适应矩阵"
            "使用5个receiver、seed 713101–713105，D92使用相同三类LEO场景、5个receiver"
            "和seed 713102–713106，数据矩阵及seed集合并非严格一一配对，因此只作描述性比较。"
        ),
        style="Body Text",
    )
    stage2b_table = add_result_table_before(
        doc,
        boundary_heading,
        STAGE2B_D92_TABLE,
        fractions=[0.07, 0.17, 0.17, 0.17, 0.17, 0.25],
        font_size=8.2,
        left_columns=(1, 2, 3, 4, 5),
    )
    set_table_header_keep_with_next(stage2b_table)
    append_paragraph_before(
        doc,
        boundary_heading,
        (
            "在这一描述性口径下，D92注册前旧类准确率在K=5和K=10时分别为81.267%和"
            "86.111%，高于同表中的三个论文适配头；K=1时MRIOR-SDA为69.88%，高于D92的"
            "68.144%。注册前尚未加入新类，D92的旧/新任务均衡协方差尚未启用，因此这里"
            "反映的是D92旧类support稳健状态构造，而不是Stage2-C任务均衡模块的增益。"
        ),
        style="Body Text",
    )

    boundary_paragraphs = [
        paragraph
        for paragraph in doc.paragraphs
        if paragraph._p.getprevious() is not None
        and paragraph._p.getprevious() is boundary_heading._p
    ]
    if boundary_paragraphs:
        boundary_text = clean_text(boundary_paragraphs[0].text)
        if boundary_text:
            replace_paragraph_text(
                boundary_paragraphs[0],
                boundary_text
                + "D92补充表只报告注册前旧类状态，不应被解释为新类注册结果。",
            )

    stage2c_anchor = find_paragraph(doc, "4.6matched无LEO新类归因诊断")
    append_paragraph_before(
        doc,
        stage2c_anchor,
        "4.5.3D92与类增量方法的共同LEO切片对照",
        style="Heading 3",
        keep_with_next=True,
    )
    append_paragraph_before(
        doc,
        stage2c_anchor,
        (
            "D92正式retry2完成125/125个任务和375/375个LEO场景，失败数为0。下表在"
            "K1/new20、K5/new20、K10/new5、K10/new10和K10/new20五个共同切片上，"
            "对照D92与CSIL、MoPC-HR官方流程的三场景平均结果。CSIL和MoPC-HR使用seed "
            "713101–713105，D92使用713102–713106；三者的base训练、状态构造和训练权限"
            "也不同，因此该表是同LEO场景的描述性对照，不是严格paired显著性比较。"
        ),
        style="Body Text",
    )
    stage2c_table = add_result_table_before(
        doc,
        stage2c_anchor,
        STAGE2C_D92_COMPARISON_TABLE,
        fractions=[0.14, 0.23, 0.13, 0.13, 0.12, 0.12, 0.13],
        font_size=7.8,
        left_columns=(0, 1),
        group_column=0,
    )
    style_inserted_formula_headers(
        stage2c_table,
        [
            (2, pre_header),
            (3, post_header),
            (4, new_header),
            (5, h_header),
            (6, forgetting_header),
        ],
    )
    set_table_header_keep_with_next(stage2c_table)
    append_paragraph_before(
        doc,
        stage2c_anchor,
        (
            "D92在五个共同切片上的旧新调和均值均高于两种论文流程，但不能把差距全部"
            "归因于D92算法：D92的注册前旧类状态明显更强，方法生命周期和允许使用的历史"
            "状态也不同。D92在K10/new20得到注册后旧类71.333%、新类68.150%、调和均值"
            "69.555%，遗忘14.778个百分点；相对其matched D81控制，它改善旧类和遗忘，"
            "但新类下降且全部绝对性能门仍失败。正式结论保持"
            "COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE，不能表述为D92已完成晋级。"
        ),
        style="Body Text",
    )

    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith("Heading"):
            paragraph.paragraph_format.keep_with_next = True
            for run in paragraph.runs:
                style_run(run)
    normalize_all_visible_run_fonts(doc)

    doc.core_properties.title = "CVS-RFFI Phase2详细复现报告（D92同场景结果补充版）"
    doc.core_properties.subject = "Phase2对比方法、D92共同LEO场景结果与证据边界"
    doc.core_properties.comments = (
        "在Stage2-B与Stage2-C对应结果章节补充D92正式retry2共同LEO场景描述性对照。"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output))

    check = Document(str(output))
    headings = [
        clean_text(paragraph.text)
        for paragraph in check.paragraphs
        if paragraph.style.name.startswith("Heading")
    ]
    required = {
        "3.6.4D92注册前共同LEO场景对照",
        "3.6.5结果边界",
        "4.5.3D92与类增量方法的共同LEO切片对照",
    }
    missing = required.difference(headings)
    if missing:
        raise RuntimeError(f"missing inserted headings: {sorted(missing)}")
    if "3.6.4结果边界" in headings:
        raise RuntimeError("obsolete Stage2-B boundary heading remains")
    if len(check.tables) != 16:
        raise RuntimeError(f"expected 16 final tables, found {len(check.tables)}")

    final_tables = [table_matrix(table) for table in check.tables]
    for original in original_tables:
        if original not in final_tables:
            raise RuntimeError(f"source table was not preserved: {original[0]}")
    if STAGE2B_D92_TABLE not in final_tables:
        raise RuntimeError("Stage2-B D92 table was not preserved exactly")
    if not any(
        len(matrix) == len(STAGE2C_D92_COMPARISON_TABLE)
        and matrix[1:] == STAGE2C_D92_COMPARISON_TABLE[1:]
        for matrix in final_tables
    ):
        raise RuntimeError("Stage2-C D92 table body was not preserved exactly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert formal D92 same-LEO-scene descriptive results into the "
            "Stage2-B and Stage2-C comparison sections of an existing DOCX."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("source and output must be different files")
    if not source.is_file():
        raise FileNotFoundError(source)
    revise(source, output)
    print(f"source_sha256={sha256(source)}")
    print(f"output_sha256={sha256(output)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
