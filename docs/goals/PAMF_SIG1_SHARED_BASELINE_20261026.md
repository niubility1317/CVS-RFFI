# 单视图完整 PAMF 共享基座

2026-10-26 用户在两个协同对话中均指定：原完整 PAMF，保留 SIG_TIME，改为单视图；在普通 PL 上优化，使 clean、星地及同 DURA 配对均提高。此文记录共享控制，避免重复发布，不是额外审批门槛。

**用户最新要求：对照全部复用已完成产物，不重复训练。** 本次刚创建的控制 run `20261026-phase1-PAMF-SIG1-REPL-base-source-m2-r01` 已按用户指令停止，controller 和两行 worker 均核实退出 GPU，其他 owner 改动 0，部分产物保留；不重启、不补齐、不把 partial 当作 E30/E50 父或对照。

另一对话“优化 PAMF-REPL 超越普通 PL”研究低置信 U／可靠负类／历史稳定性，使用独立模块和 run。本对话研究 REPL 外推与身份间距。两个方向使用相同单视图结构和已完成的普通 PL E20 前缀；后续只训练改进候选，不另发公共对照。

## 固定结构与来源

`experiments/pamf_channel_pl/sig1_model.py` 使用既有 `CompositePool(96,'SIG_TIME',seed)`：一组 `6×96×1` 局部投影、四窗口含时间坐标的 112 维描述、`160×112` 输出投影、原 LayerNorm／有界残差增益；支路 18976 参数。原时频、PA、参考响应与 `6×160` 默认头联合训练保持。没有删 SIG_TIME、没有把 X3 权重裁切或平均成单视图。

三个 seed 的已核实 E20 前缀：`runs/20261024-phase1-PAMF-REPL-completion-m201-r01/prefixes/F0-A1B1-SIG_TIME-F21-sSEED/segment20`，SEED 为 2026092701／02／03。实际张量均符合单视图、有限、原默认头；从零 E20/4440、无祖先，目标／U 真值／另训头均关闭。custom ordinary PL 开启，native U/EMA 关闭。来源核实见本控制报告中的 `prefix_tensor_evidence.json`；这不代替正式 launch 的真实 checkpoint smoke。

数据沿用原 F0 VALIDATED_ONCE：L6300、隐藏 TX 的 U56700、V27000，源 RX[1,3,4,6,8]、day[1,2,3]，原六 TX；以每 seed 原契约和物理 ID 对照，具体私有句柄沿用 `restored_source_context.json`，不把新架构变化当作数据重验触发。G 继续同一原 F0 G-LU／DURA v1.0，F 不反向更新 G。

## 复用已完成对照

主性能参照继续复用 corrected binding run `20261022-phase1-T1-DURA-F-ablation-binding-m63-r01` 的三 seed `F20`（完整监督）和 `F21`（普通 PL），以及原 HardPL-ReceiverExtrap 的完整 REPL 和同 G-LU 冻结结果，不重训或重复预测。

已完成单视图参照包括 m201 的 `REPL_SIG_TIME`、`CE_SIG_TIME` 三 seed，以及 binding-m63 的 `F06` 三 seed。必须保留配方差异：F20/F21 是 X3 结构；F06 保留原 native U/EMA/门控，不能改称普通 PL；CE_SIG_TIME 是严格 CE 结构控制，不能改称完整 L 骨架的监督对照。它们可比较系统性能和相应结构条件，不能冒充已经存在“单视图＋完整 L”的严格监督／普通 PL 因果消融。缺少这种匹配项时披露缺项，遵从用户要求不新增训练。

新候选共同从上述三个已完成的单视图普通 PL E20 前缀出发，自己原生 +10＋20 到 E50，累计 13L＋37LU、11100 更新。新机制启用阶段在候选发布前固定；不再为了创建普通 PL E30 公共父单独补训十轮，不加载中止控制的 partial。

## 已停止的两行源控制计划

- `SIG1-PL`：先单 seed01，复用自己的单视图普通 PL E20，原生 +10 至 E30、再 +20 至 E50；累计 13L＋37LU、11100 更新。E30 留作两研究方向共同父模型。
- `SIG1-Sup`：完整匹配 L 骨架，从零 20＋10＋20 至 E50；无 U loader／伪标签／EMA，不能继承 PL 前缀。

上述为已停止 run 的原计划，当前不继续执行。真实单视图 E20/G 与两行完整 L/U backward 检查已经 PASS，实际默认头和单视图梯度非零；PL649、Sup696 条完整可读更新记录保留，两行均无完整 final，未进入源或目标最终评估。[停止证据](../../automation_reports/CV-SincNet/20261026-phase1-PAMF-SIG1-REPL-base-source-m2-r01/user_stop_evidence.json)。

已停止 run root：`PROJECT/runs/20261026-phase1-PAMF-SIG1-REPL-base-source-m2-r01`。该 root 仅用于保留停止证据，不能作为后续 E30/E50 来源。

GPU 不做静态占卡。当前本对话没有活动 GPU worker；原 NoSIG 六行全部完成，新增重复控制已停止。新候选仍可用物理卡 0 至 7，每次派发核实全局 CUDA＋pending<2、free≥10000 MiB，计入其他 owner。其他方向发布前独立核实实时进程，不能从本文件推断 GPU 仍然空闲。禁止中止任何其他 owner 的健康任务。

新改进学生另行最小登记、实际 smoke 后启动。最终同一候选三 seed、原目标 168000 包/seed、raw 与同 G clean＋六 practical、全部预测冻结后独立 truth-last；原联合门槛保持，不以源分数或共享控制的完成替代目标达成。
