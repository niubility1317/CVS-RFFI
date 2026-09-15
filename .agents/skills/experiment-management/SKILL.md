---
name: experiment-management
description: Register, name, locate, and maintain experiment records for CVS, external comparison methods, federated runs, diagnostics, and evaluation. Use before every new experiment launch and when retrieving or organizing historical experiments in E:/type10-7.
---

# 实验登记与查找

本skill把用户要求的实验管理落实到既有预登记与交接，不新增实验审批、数据重验或哈希链。适用于CVS、外部对比、联邦、诊断、消融、seed扫描，以及独立评估/评分；手动命令、脚本、队列和授权自动化启动都适用。仅查找时不执行启动流程。

## 查找现有实验

1. 先运行`tools/experiment_registry.py search "方法或问题关键词"`，或看[总索引](../../../experiment_registry/README.md)。用`--seed`、`--tag`、`--record-type`缩小范围，再`show <完整id>`。精确命令、字段和历史边界见[管理规范](../../../docs/EXPERIMENT_MANAGEMENT.md)。
2. 读取命中的report和config；历史条目可用`show <id> --section facts`或`--section artifacts`定位配置、日志、权重和结果。facts保留文件和JSON位置/行号，多个row的值不能拼成一行。
3. 先核实索引更新时间和证据路径。历史状态、目录修改时间、旧RUNNING声明不代表实时状态；远端只按已有权限读回。查不到时检查覆盖文件，针对缺少的目录增补，不每次全盘扫描。
4. 对话原因/授权未记录时才搜索`conversation_index`；原始实验路径优先于对话回忆。恢复先核实原run，不能重复启动。

## 每次启动前

- 已有登记的同一run继续补全原记录；真正的新启动、失败替代或独立评估使用新run ID。一个run只有一个launch owner，禁止覆盖旧输出。
- 用`template --output <新spec文件>`生成模板；按本次真实配置填写。命名采用`YYYYMMDD-stage-method-data-sSEED-rNN`；多seed矩阵用`mN`，`group_id`稳定描述方法与研究问题，`display_name`用中文说明目的。可沿用已有不可变run ID，增加aliases，不追溯改目录名。
- `experiment.json`是登记字段与逐行矩阵的唯一维护位置，`report.md`保留说明和证据解释，`events.jsonl`保留状态变化。常规run由`new --spec <文件>`建立在`automation_reports/CV-SincNet/<run_id>/`。报告、登记与实际生效config互相指向，不复制维护多份矩阵。
- 必填含义：研究问题/对照差异、方法实际启用组件、CVS或外部方法权限、数据版本/表示/契约和物理ID索引、RX/day/TX与标签映射、角色比例、support/query/K/LEO、checkpoint完整来源与选择规则、逐row的六类seed、优化器/LR/预算、精确命令/commit/环境/CWD、host/GPU/owner、输出/log/配置/预期artifact位置。无适用项写null并解释；尚缺证据写未知及证据来源，不能猜默认值。
- seed明确区分model/split/data/augmentation/support/evaluation；固定划分只扫描模型seed时，保留其他seed值。外部方法若仅暴露一个seed，明确它实际控制哪些角色，未控制项写null加说明。每个row一份配置和独占输出，不把seed列表自动展开为未授权矩阵。
- 在既有最小流程第1/5/6项内完成登记、输入核对和输出防冲突。可用`validate <spec> --launch-ready`发现漏字段；它只检查登记结构，不能证明checkpoint合法、数据有效、获得启动许可或实验成功。填写规范见[管理规范](../../../docs/EXPERIMENT_MANAGEMENT.md)。

## 启动后、评分后与交接

- 实际启动与N607规则仍由[cvs-experiment-workflow](../cvs-experiment-workflow/SKILL.md)负责；本工具从不提交任务、读取query truth或改变进程。
- 启动后记录独立PID/CWD/argv/GPU/log证据、实际resolved_config位置和读取时刻；使用`record`追加run或row状态，自动增量更新索引。混合矩阵不能因一个row完成就标整个run完成。
- 训练、预测、评分、分析分别标状态。Phase1源域训练等待冻结/授权时，保留`SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW`，不能为凑齐登记自动读取目标。Phase1最终clean及三种LEO结果、Phase2固定prediction及独立truth-last评分仍按科学协议执行。
- 最终保留每row及适用的RX/day/TX/scene/K/seed指标、Macro-F1、跨seed统计、失败/缺项、资源成本、协议有效性与声明边界；索引只存路径和摘要，训练/评分不得消费索引来调参或选择候选。
- 技术失败、用户停止、替代、source-only、仅配置、partial和历史记录都保留。替代用`replaces_run_id`，模型继承用`parent_run_ids`加checkpoint来源，不能只写“续上次”。
- 登记、工具、模板、报告和索引元数据按AGENTS精确镜像、提交、push并读回；大体积IQ、checkpoint、预测/原始日志保持原存储，用路径引用。索引重建不用重跑实验或解压历史归档。

## 历史整理

运行`build`生成本地证据组、artifact路径表、带出处的配置字段和覆盖清单；完整读取小型元数据，不把日志抽样叫完整分析。历史索引中的seed是有出处的字段提及，状态一律保持未实时核实；路径标签只是查找提示。
既有早期CSV日志目录按原行导入；远端目录使用单独只读快照及其读取时刻，采集命令见管理规范。元数据细目按证据组压缩；show可按字段、row或artifact类别筛选，达到展示上限会明确标记has_more。
用户只问一项时不重建全库。批量整理按覆盖清单核对报告、runs、日志备份、对比方法目录和Git检出；必要时扩展工具的扫描范围并说明遗漏。禁止为整理而移动/删除/重命名数据、权重、日志和既有产物。
