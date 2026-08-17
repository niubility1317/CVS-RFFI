# Phase2 Inline Parenthetical Definitions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将Phase2详细复现报告中的25处独立说明迁移到相关术语或符号首次正文出现处的中文全角括号内，并交付通过结构与逐页版式检查的新DOCX。

**Architecture:** 以2026-08-01符号缩写说明版为内容基准，以其前一版DOCX为无独立说明的生成源。新增一个确定性python-docx修订脚本，根据固定段落锚点在首次正文使用处追加含Word原生公式的括号定义；脚本内执行表格、说明标签、公式、字体、页眉页脚和关键术语校验。生成后使用文档技能的渲染链导出页面PNG并逐页检查。

**Tech Stack:** Python 3.11、python-docx 1.2.0、WordprocessingML/OMML、Git Bash、Microsoft Word或文档技能渲染器、Poppler。

## Global Constraints

- Windows终端外壳固定为`C:\Program Files\Git\bin\bash.exe`，`login: false`；禁止执行`pwsh`或`pwsh.exe`。
- 中文与英文、数字、缩写、变量名之间不增加空格；英文短语内部保留必要空格。
- 中文字体为宋体，英文与数字字体为Times New Roman。
- 所有数学变量、上下标、集合和运算符继续使用Word原生公式对象。
- 17张实验表、实验数值、方法损失函数、训练权限、结果边界和5篇参考文献均保持不变。
- 输出新文件，不覆盖用户当前可能打开的2026-08-01版本。
- Git提交只包含本任务文件，不改动或提交工作区内其他用户文件。

---

### Task 1: 实现首次出现括号定义转换器

**Files:**
- Create: `tools/inline_first_use_parenthetical_definitions.py`
- Test: `tools/test_inline_first_use_parenthetical_definitions.py`
- Read: `tools/add_first_use_symbol_abbreviation_explanations.py`
- Read: `docs/superpowers/specs/2026-08-17-phase2-inline-parenthetical-definitions-design.md`

**Interfaces:**
- Consumes: `Path source`，指向`CVS-RFFI_Phase2详细复现报告1_qKNN域适应与类增量结果补充版_截至20260731.docx`。
- Produces: `revise(source: Path, output: Path) -> None`，生成不含独立说明标签、含25组首次出现括号定义的新DOCX。
- Produces: `append_parenthetical(paragraph, latex_text: str) -> None`，在指定首次使用段落结尾追加中文全角括号，并把`\(...\)`片段转换成OMML。

- [ ] **Step 1: 写出失败的最小结构测试**

测试建立仅含标题、首次使用段落、一个公式和一个表格的临时DOCX，调用`append_parenthetical`后断言：段落出现全角括号、OMML数量增加、表格数量不变、正文中没有`符号说明：`等标签。

```python
def test_append_parenthetical_adds_omml_and_no_standalone_label(tmp_path):
    doc = Document()
    paragraph = doc.add_paragraph("一次N-way K-shot任务记为")
    doc.add_table(rows=1, cols=1)
    before_tables = len(doc.tables)
    append_parenthetical(paragraph, r"\(N\)表示类别数；\(K\)表示每类support数。")
    assert paragraph.text.startswith("一次N-way K-shot任务记为（")
    assert paragraph.text.endswith("）")
    assert len(paragraph._p.xpath(".//m:oMath")) == 2
    assert len(doc.tables) == before_tables
    assert "符号说明：" not in paragraph.text
```

- [ ] **Step 2: 串行运行测试并确认失败原因**

Run:

```bash
conda run -n ssr-gpu python tools/test_inline_first_use_parenthetical_definitions.py
```

Expected: `ImportError`或`ModuleNotFoundError`，因为转换器尚未创建。

- [ ] **Step 3: 实现最小转换器与25组锚点映射**

实现要求：

- 从2026-07-31无说明版生成，而不是在现有说明版上反复改写。
- 复用`add_text_with_inline_math`，但由`append_parenthetical`显式添加`（`和`）`。
- 25组定义按固定正文锚点附着；标题中首次出现的术语在标题后的第一句正文中解释。
- 每个锚点必须唯一命中；0个或多个命中均立即报错。
- 保存后重新打开文件，核对说明组数、表格数、OMML数量和关键全称。

```python
def append_parenthetical(paragraph, latex_text: str) -> None:
    paragraph.add_run("（")
    add_text_with_inline_math(paragraph, latex_text)
    paragraph.add_run("）")
```

- [ ] **Step 4: 运行最小结构测试并确认通过**

Run:

```bash
conda run -n ssr-gpu python tools/test_inline_first_use_parenthetical_definitions.py
```

Expected: `PASS`，括号、OMML和表格断言全部成立。

- [ ] **Step 5: 提交转换器与测试**

```bash
git add tools/inline_first_use_parenthetical_definitions.py tools/test_inline_first_use_parenthetical_definitions.py
git commit -m "tools: inline Phase2 definitions"
```

### Task 2: 生成报告并执行内容与结构校验

**Files:**
- Create: `docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx`
- Modify: `docs/weekly_reports/CVS_RFFI_Phase2报告符号缩写首次出现说明_20260801.md`

**Interfaces:**
- Consumes: Task 1的`revise(source, output)`。
- Produces: 通过结构校验的正式DOCX和更新后的修订记录。

- [ ] **Step 1: 运行转换器生成新DOCX**

Run:

```bash
conda run -n ssr-gpu python tools/inline_first_use_parenthetical_definitions.py \
  'docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN域适应与类增量结果补充版_截至20260731.docx' \
  'docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx'
```

Expected: 输出源/目标SHA256、`definitions=25`、`tables=17`和`standalone_labels=0`。

- [ ] **Step 2: 运行完整结构校验**

校验必须同时断言：

- 17张表和153个表格行保持不变；
- 25组括号定义全部出现，独立说明标签为0；
- OMML数量不低于2026-08-01说明版；
- 页眉页脚为空；
- 可见文本不含反斜杠、LaTeX分隔符、D92旧称或替换字符`U+FFFD`；
- 可见run的中文字体为宋体，英文与数字字体为Times New Roman；
- `RFFI`、`FSL`、`FSCIL`、`MMD`、`EWC`、`KD`、`qKNN`等关键缩写在首次正文使用段落中紧邻全角括号。

Run:

```bash
conda run -n ssr-gpu python tools/inline_first_use_parenthetical_definitions.py --check \
  'docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx'
```

Expected: `STRUCTURE_QA=PASS`。

- [ ] **Step 3: 更新修订记录**

在现有修订记录中追加2026-08-17条目，写明独立说明改为首次出现括号定义、未改实验数据和参考文献，并记录结构校验结果。

- [ ] **Step 4: 检查差异并提交报告产物**

```bash
git diff --check -- tools/inline_first_use_parenthetical_definitions.py \
  docs/weekly_reports/CVS_RFFI_Phase2报告符号缩写首次出现说明_20260801.md
git add -f docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx
git add docs/weekly_reports/CVS_RFFI_Phase2报告符号缩写首次出现说明_20260801.md
git commit -m "docs: inline Phase2 definitions at first use"
```

### Task 3: 渲染、逐页检查和交付

**Files:**
- Read: `docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx`
- Create: `local_artifacts/phase2_inline_parenthetical_qa_20260817_v1/`，仅作为内部QA，不提交。
- Create: `C:/Users/lh594/Downloads/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx`

**Interfaces:**
- Consumes: Task 2正式DOCX。
- Produces: 全页视觉QA证据和Downloads交付副本。

- [ ] **Step 1: 使用文档技能渲染全部页面**

优先调用`render_docx.py`；若仅因LibreOffice缺失而失败，则调用Microsoft Word的Windows原生导出路径，并用Poppler生成PNG。所有Windows原生程序均从Git Bash调用，不使用`pwsh`。

Expected: PDF非空，且每页对应一张`page-<N>.png`。

- [ ] **Step 2: 逐页以100%比例检查**

检查每页：括号换行自然、没有孤立右括号、公式不被拆坏、表格不裁切、标题不孤行、字体无替换、页眉页脚为空、参考文献完整。

Expected: 每页均无裁切、重叠、缺字或异常留白。

- [ ] **Step 3: 若发现版式问题则局部修复并重新执行Task 2校验与Task 3渲染**

只允许调整新增括号说明的段落间距、keep-with-next或局部换行；不得改实验数据、公式含义、表格内容或参考文献。

- [ ] **Step 4: 复制最终DOCX并进行哈希回读**

```bash
cp 'docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx' \
  '/c/Users/lh594/Downloads/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx'
sha256sum 'docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx' \
  '/c/Users/lh594/Downloads/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx'
```

Expected: 两个SHA256完全一致。

- [ ] **Step 5: 最终状态检查**

Run:

```bash
git status -sb
git log -2 --oneline
```

Expected: 本任务文件已提交；已有的其他未跟踪或用户文件保持不变；未执行push。
