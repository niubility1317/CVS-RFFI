# Phase2 Formal Project Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为Phase2报告加入正式项目与阶段方法名称，删除内部版本代号，并用克制的红色加粗突出项目重点。

**Architecture:** 使用独立python-docx/OOXML转换器在第1章正文中插入重点块，按上下文替换正文和表格中的实验代号，并对关键句应用统一红色加粗样式。回归测试检查名称、代号清零、格式属性和数据守恒。

**Tech Stack:** Python 3、python-docx、OOXML、Microsoft Word渲染验证。

## Global Constraints

- 不修改实验数值、单位、方法机制、训练权限、状态定义或结论边界。
- 不把内部版本号改写成未经验证的新算法贡献。
- 中文使用宋体，英文与数字使用Times New Roman。
- 重点使用红色加粗，但只覆盖项目名称、阶段方法名称、核心任务和少量关键句。

---

### Task 1: 正式命名转换器与回归测试

**Files:**
- Create: `tools/formalize_phase2_project_naming.py`
- Create: `tools/test_formalize_phase2_project_naming.py`

**Interfaces:**
- Consumes: 按实验配置拆分版Phase2 DOCX。
- Produces: `formalize_report(source, output)`及正式命名版DOCX。

- [ ] **Step 1: 编写失败测试**

检查正式名称、红色加粗、`ADV3B02`清零、表格与实验数值守恒。

- [ ] **Step 2: 运行测试并确认转换器不存在而失败**

Run: `python tools/test_formalize_phase2_project_naming.py`

- [ ] **Step 3: 实现最小转换器**

插入重点块，按上下文替换代号，对Stage2-B/Stage2-C核心目标句应用红色加粗。

- [ ] **Step 4: 运行测试并确认通过**

Run: `python tools/test_formalize_phase2_project_naming.py`

- [ ] **Step 5: 提交工具与测试**

Run: `git add tools && git commit -m "tools: formalize Phase2 project naming"`

### Task 2: 生成、审计和交付正式命名版报告

**Files:**
- Create: `docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_正式命名与重点突出版_截至20260817.docx`
- Modify: `docs/weekly_reports/CVS_RFFI_Phase2报告符号缩写首次出现说明_20260801.md`

**Interfaces:**
- Consumes: Task 1转换器与按配置拆分版DOCX。
- Produces: 结构和视觉验证后的可交付DOCX。

- [ ] **Step 1: 生成DOCX并执行结构审计**

核对代号、名称、红色强调、表格、结果数字、引用和公式。

- [ ] **Step 2: 使用Microsoft Word渲染PDF和PNG**

检查第1页重点块、章节分页、结果表和参考文献。

- [ ] **Step 3: 修复视觉问题并重新渲染**

只调整强调范围、间距和分页，不改变技术内容。

- [ ] **Step 4: 更新修订说明并提交**

Run: `git add docs tools && git commit -m "docs: add formal project naming and emphasis"`

- [ ] **Step 5: 复制到下载目录并核对哈希**

交付最终DOCX，保留独立分支，不合并、不推送。
