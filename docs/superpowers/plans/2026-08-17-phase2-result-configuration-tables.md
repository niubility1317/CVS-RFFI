# Phase2 Result Configuration Tables Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将Phase2报告后半部分的大结果表无损拆分为按`K-shot`与新类规模组织的配置表。

**Architecture:** 使用独立的OOXML/python-docx转换脚本定位结果表，按原始行的配置键分组，复制原单元格内容以保留公式与样式，再替换原表。测试脚本验证行级数据守恒、配置分组、表头和章节隔离。

**Tech Stack:** Python 3、python-docx、lxml/OOXML、Microsoft Word渲染验证。

## Global Constraints

- 不修改任何实验数值、单位、方法名称、引用或结论边界。
- 正式LEO、共同切片和matched无LEO诊断必须保持分离。
- 数学符号继续使用Word公式对象；程序字段名继续使用代码式文本。
- 中文与英文、数字、缩写之间不增加额外空格。

---

### Task 1: 配置分组转换器与回归测试

**Files:**
- Create: `tools/regroup_phase2_result_tables.py`
- Create: `tools/test_regroup_phase2_result_tables.py`

**Interfaces:**
- Consumes: 符号统一版Phase2 DOCX中的正式LEO、共同切片、无LEO诊断三张结果表。
- Produces: `regroup_report(input_path, output_path)`和按配置拆分的DOCX。

- [ ] **Step 1: 编写失败测试**

验证原三表的数据行集合与拆分后数据行集合一致，且共同切片的每个配置表包含3种方法。

- [ ] **Step 2: 运行测试并确认转换器尚不存在而失败**

Run: `python tools/test_regroup_phase2_result_tables.py`

- [ ] **Step 3: 实现最小转换器**

定位三张目标表，按配置键分组，复制方法与指标单元格，插入配置标题并移除原表。

- [ ] **Step 4: 运行测试并确认通过**

Run: `python tools/test_regroup_phase2_result_tables.py`

- [ ] **Step 5: 提交转换器与测试**

Run: `git add tools/regroup_phase2_result_tables.py tools/test_regroup_phase2_result_tables.py && git commit -m "tools: regroup Phase2 result tables by configuration"`

### Task 2: 生成报告并完成结构与视觉验证

**Files:**
- Create: `docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_qKNN结果按配置拆分版_截至20260817.docx`
- Modify: `docs/weekly_reports/CVS_RFFI_Phase2报告符号缩写首次出现说明_20260801.md`

**Interfaces:**
- Consumes: Task 1的转换器和符号统一版DOCX。
- Produces: 可交付DOCX及审计记录。

- [ ] **Step 1: 生成按配置拆分的DOCX**

Run: `python tools/regroup_phase2_result_tables.py <input.docx> <output.docx>`

- [ ] **Step 2: 执行结构审计**

检查数据行守恒、公式对象、参考文献数量、章节标题和配置表数量。

- [ ] **Step 3: 渲染并逐页检查**

使用Word导出PDF并生成PNG，检查配置标题、分页、表格宽度、公式和参考文献页面。

- [ ] **Step 4: 修复视觉问题并重复验证**

仅调整分页、间距或列宽，不改变数据内容。

- [ ] **Step 5: 更新说明、提交并复制到下载目录**

Run: `git add docs/weekly_reports && git commit -m "docs: split Phase2 results by experiment configuration"`
