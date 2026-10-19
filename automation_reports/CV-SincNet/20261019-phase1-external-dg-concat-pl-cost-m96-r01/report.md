# Phase1拼接增强/普通PL：96冻结模型统一推理成本

- run_id：`20261019-phase1-external-dg-concat-pl-cost-m96-r01`
- group_id：`phase1-nonpamf-concat-PL-cost`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

219性能训练全部闭合后，同一空闲GPU0/1测96个冻结checkpoint；合成IQ、零新增训练、无query。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 本次发布预登记

完整数据、接收机数量与预算依据见 [原执行设计](../../../docs/PHASE1_EXTERNAL_DG_EXECUTION_PLAN_20261019.md) 及 [当前增强与PL修订](../../../docs/PHASE1_DG_AUG_PL_REVISION_20261019.md)。当前矩阵由 experiments/phase1_external_dg_aug/matrix.json 固定；正式训练从零开始，不继承无增强历史权重。

所有正式训练均使用 clean＋satellite 拼接和 0.68 卫星 CE；PL 行在同一次训练额外使用阈值 0.95、权重 1、U/L=1 的普通 PL。监督与 PL 共用相同增强、模型种子、最大预算和源域选模规则。

RIEI/DRIFT E100（用户指定实现预算，阶段按比例缩放）；IB-RFF 两阶段各最多 E200、源域降学习率及早停；SDG E200；通用 DG 固定 5000 次更新；源数量固定预算组 10000 次更新；强 ResNet CE E200。IB 早停以实际执行轮数报告。训练首批启用卫星 CE，默认场景概率按相对最大预算推进。

仅物理 GPU 0、1，每卡一个本任务训练/预测 worker，其他 owner 占用时等待。每行全部七视图预测固定后独立 CPU truth-last 评分。两次相同技术故障暂停未启动行，健康行继续；低分、负收益或零 PL 接受率均不触发重跑、调参或停跑。

针对性检查：8 项通过；独立 P0/P1 审查 PASS，唯一 stages 字段缺陷已修复并完成针对性复核。实际 Git commit、发布包、运行路径和提交回执在发布后登记，真实 checkpoint 无 query 联合目标 smoke 为启动第一步，PASS 后立即继续。

本成本 run 等待新性能 219 行全部闭合且调度进程实际退出，才使用同一空闲卡测量固定 96 个 checkpoint，不新增训练或访问目标数据。
