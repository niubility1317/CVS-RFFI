# 控制面职责

维护日期：2026-09-05。此文件是导航，不是第二套权限或gate。

| 内容 | 唯一维护位置 | 读取时机 |
|---|---|---|
| 工作约定、安全与版本管理 | ../AGENTS.md | 每个任务 |
| 科学场景和数据权限 | E:/type10-7/项目.md；克隆镜像docs/PROJECT_PROTOCOL.md | 科学/实验任务 |
| 候选、矩阵、预算、停止规则 | 当前用户目标及对应run报告 | 本次实验 |
| 最小实验步骤 | optimizer_workflow_contract.md | 实验执行 |
| N607连接和修复 | ../docs/workflows/n607.md | 远端操作前 |
| 技能入口 | ../.agents/skills/cvs-experiment-workflow/SKILL.md | 实验任务 |
| 当前证据 | 当前run的报告、日志与产物 | 监控/分析 |
| 历史optimizer实现 | optimizer_*工具与历史prompt/state | 明确复现或维护旧optimizer时 |

无需每回合加载全部控制文件、全量state、历史报告或所有技能。
现有`stage2_optimizer_state.json`可供定位run，优先`optimizer_state_current_view.py`提取相关当前字段，再通过该run日志/进程核实；历史字段不授予启动权限。
旧`automation_prompt_backups/.../stage2_prompt.md`不再是新任务的默认指令入口；保留用于历史复现。实际调度器保存的prompt是否采用新入口须单独核实，修改本文件不等于更新调度器。

旧64-row生成器、`optimizer_validate_matrix.py`与`optimizer_preflight_decision.py`保留兼容性，不作为新候选的必经流程。其输出是旧schema的诊断，不能授予数据权限，也不能因缺少旧字段阻断新候选。需要旧工具时核对当前协议；直接数据/进程/输出错误仍须处理，不能简单忽略。
旧工具代码、历史schema/state和调度任务不随本次文档改动迁移。调用方应使用当前候选已有launcher和最小流程，不为运行一个小候选先改造完整optimizer平台。

控制文档变更验证编码、路径、必要权限和调用关系；只有修改相应工具行为时才运行其聚焦测试。缺少可选manifest/state不是实验停止理由；缺少本次实际输入或无法确定权限才影响相关步骤。
