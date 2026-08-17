# Phase2 Symbol Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份符号无冲突、状态语义明确且实验数据不变的Phase2报告。

**Architecture:** 在已通过视觉检查的首次出现括号定义版上执行确定性OOXML修改。转换器只改正文符号、公式文本和指定表头；独立检查器比较数值token、表格规模、参考文献和公式数量，Word渲染负责最终视觉验证。

**Tech Stack:** Python 3、python-docx、lxml、Word COM导出、PyMuPDF、unittest。

## Global Constraints

- 输入：`docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN首次出现括号定义版_截至20260817.docx`
- 输出：`docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN符号统一版_截至20260817.docx`
- 保留17张表、153行记录、全部数值和5条参考文献。
- 禁止改动实验机制、训练配置、损失权重和结论边界。
- 只使用Git Bash外壳和已验证的bundled Python；不使用PowerShell。

---

### Task 1: 建立符号转换器和回归检查

**Files:**
- Create: `tools/normalize_phase2_report_symbols.py`
- Create: `tools/test_normalize_phase2_report_symbols.py`

**Interfaces:**
- Consumes: 输入DOCX路径。
- Produces: `normalize(source: Path, output: Path) -> None`和`audit(output: Path, source: Path) -> dict[str, int]`。

- [ ] **Step 1: 写失败测试**

测试须覆盖类别集合、域下标、概率、prototype、温度、稳定项、噪声、优化步和状态记号，并比较数值token、表格规模及参考文献。

- [ ] **Step 2: 运行测试并确认因缺少实现而失败**

Run: `python tools/test_normalize_phase2_report_symbols.py`

- [ ] **Step 3: 实现最小确定性OOXML转换**

按段落锚点和OMML文本执行替换；任何锚点缺失或替换次数异常均抛出异常。

- [ ] **Step 4: 重跑测试、编译检查和diff检查**

Run: `python tools/test_normalize_phase2_report_symbols.py`

Run: `python -m py_compile tools/normalize_phase2_report_symbols.py tools/test_normalize_phase2_report_symbols.py`

Run: `git diff --check`

- [ ] **Step 5: 提交转换器与测试**

Commit message: `tools: normalize Phase2 report symbols`

### Task 2: 生成报告并执行结构验证

**Files:**
- Create: `docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN符号统一版_截至20260817.docx`
- Modify: `docs/weekly_reports/CVS_RFFI_Phase2报告符号缩写首次出现说明_20260801.md`

**Interfaces:**
- Consumes: Task 1转换器和输入DOCX。
- Produces: 结构检查通过的最终候选DOCX。

- [ ] **Step 1: 生成输出DOCX**

Run: `python tools/normalize_phase2_report_symbols.py SOURCE OUTPUT`

- [ ] **Step 2: 执行源文件一致性审计**

Run: `python tools/normalize_phase2_report_symbols.py --check OUTPUT --reference SOURCE`

- [ ] **Step 3: 更新报告交接记录并提交**

记录统一规则、检查计数与输出哈希。

### Task 3: 渲染、逐页检查与交付

**Files:**
- Create: `local_artifacts/phase2_symbol_normalization_qa_20260817_v1/`（仅内部QA）
- Copy: `C:/Users/lh594/Downloads/CVS-RFFI_Phase2详细复现报告1_qKNN符号统一版_截至20260817.docx`

**Interfaces:**
- Consumes: Task 2候选DOCX。
- Produces: 逐页视觉检查通过、哈希一致的下载副本。

- [ ] **Step 1: 用Word导出PDF并转为逐页PNG**

- [ ] **Step 2: 检查每页公式、括号、表格和分页**

- [ ] **Step 3: 重跑测试、结构审计、哈希比较和Git状态检查**

- [ ] **Step 4: 保留独立分支，不合并或推送未获授权的改动**

