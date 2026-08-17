# Phase2 Bilingual and ERTB-IDR Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有Phase2 DOCX中加入项目及两阶段的英文名称，并把实际使用的D92 E0对比方法正式命名为ERTB-IDR。

**Architecture:** 继续复用`formalize_phase2_project_naming.py`执行确定性DOCX修改。测试直接读取生成后的OOXML可见文本、红色加粗属性、表格数值和公式对象；最终用原生Word渲染全部页面进行视觉复核。

**Tech Stack:** Python 3、python-docx、lxml、Word OOXML、Git Bash、Microsoft Word PDF导出。

## Global Constraints

- 仅修改名称、首次定义和关联叙述，不改变实验数值、公式、表格结构或参考文献。
- 中文使用宋体，英文使用Times New Roman；重点命名保持红色加粗。
- `ERTB-IDR`严格对应D92 E0_FULL_ONLY，不代表原始D92，也不增加论文引用。
- 最终DOCX必须完成全部页面渲染与逐页视觉检查。

---

### Task 1: 锁定中英文名称和D92 E0边界

**Files:**
- Create: `docs/superpowers/specs/2026-08-17-phase2-bilingual-and-ertb-idr-naming-design.md`
- Create: `docs/superpowers/plans/2026-08-17-phase2-bilingual-and-ertb-idr-naming.md`

**Interfaces:**
- Consumes: `docs/D92_METHOD_COMPLETE_REPORT_20260727.md`与D92 E0 Target125报告中的机制证据。
- Produces: 供生成工具和测试复用的唯一命名文本。

- [ ] **Step 1: 写入正式中英文名称和机制边界**
- [ ] **Step 2: 检查无占位内容或含混的D92/D92 E0指代**
- [ ] **Step 3: 提交规格与计划**

### Task 2: 测试先行并修改生成工具

**Files:**
- Modify: `tools/test_formalize_phase2_project_naming.py`
- Modify: `tools/formalize_phase2_project_naming.py`

**Interfaces:**
- Consumes: 现有按配置拆分版DOCX。
- Produces: 含中英文正式名称和ERTB-IDR定义的新版DOCX。

- [ ] **Step 1: 增加失败测试，要求中英文名称、ERTB-IDR及D92 E0边界出现，并禁止`qKNN`残留**
- [ ] **Step 2: 运行测试，确认因功能缺失而失败**
- [ ] **Step 3: 最小修改生成工具，重写首次定义并替换方法标签**
- [ ] **Step 4: 运行全部相关测试并确认通过**

### Task 3: 生成、审计、渲染与交付

**Files:**
- Create: `docs/weekly_reports/CVS-RFFI_Phase2详细复现报告1_正式中英文命名与ERTB-IDR版_截至20260817.docx`
- Modify: `docs/weekly_reports/CVS_RFFI_Phase2报告符号缩写首次出现说明_20260801.md`

**Interfaces:**
- Consumes: Task 2的确定性生成工具。
- Produces: 下载目录中的最终DOCX。

- [ ] **Step 1: 生成DOCX并核对52张表、纯数值实验单元格、491个公式和5条参考文献不变**
- [ ] **Step 2: 渲染全部页面并逐页检查**
- [ ] **Step 3: 运行可复现性和符号审计**
- [ ] **Step 4: 提交最终产物并复制到下载目录，核对SHA-256一致**
