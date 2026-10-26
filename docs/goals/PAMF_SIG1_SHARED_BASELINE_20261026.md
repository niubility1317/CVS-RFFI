# 单视图完整 PAMF 共享基座

2026-10-26 用户在两个协同对话中均指定：原完整 PAMF，保留 SIG_TIME，改为单视图；在普通 PL 上优化，使 clean、星地及同 DURA 配对均提高。此文记录共享控制，避免重复发布，不是额外审批门槛。

当前对话 owner `codex-pamf-sig1-repl-base-20261026` 独占发布控制 run `20261026-phase1-PAMF-SIG1-REPL-base-source-m2-r01`。另一对话“优化 PAMF-REPL 超越普通 PL”研究低置信 U／可靠负类／历史稳定性，使用独立模块和 run，不重复训练这两行公共控制。本对话研究 REPL 外推与身份间距，两个方向使用相同基座、数据和默认头。

## 固定结构与来源

`experiments/pamf_channel_pl/sig1_model.py` 使用既有 `CompositePool(96,'SIG_TIME',seed)`：一组 `6×96×1` 局部投影、四窗口含时间坐标的 112 维描述、`160×112` 输出投影、原 LayerNorm／有界残差增益；支路 18976 参数。原时频、PA、参考响应与 `6×160` 默认头联合训练保持。没有删 SIG_TIME、没有把 X3 权重裁切或平均成单视图。

三个 seed 的已核实 E20 前缀：`runs/20261024-phase1-PAMF-REPL-completion-m201-r01/prefixes/F0-A1B1-SIG_TIME-F21-sSEED/segment20`，SEED 为 2026092701／02／03。实际张量均符合单视图、有限、原默认头；从零 E20/4440、无祖先，目标／U 真值／另训头均关闭。custom ordinary PL 开启，native U/EMA 关闭。来源核实见本控制报告中的 `prefix_tensor_evidence.json`；这不代替正式 launch 的真实 checkpoint smoke。

数据沿用原 F0 VALIDATED_ONCE：L6300、隐藏 TX 的 U56700、V27000，源 RX[1,3,4,6,8]、day[1,2,3]，原六 TX；以每 seed 原契约和物理 ID 对照，具体私有句柄沿用 `restored_source_context.json`，不把新架构变化当作数据重验触发。G 继续同一原 F0 G-LU／DURA v1.0，F 不反向更新 G。

## 首批两行源控制

- `SIG1-PL`：先单 seed01，复用自己的单视图普通 PL E20，原生 +10 至 E30、再 +20 至 E50；累计 13L＋37LU、11100 更新。E30 留作两研究方向共同父模型。
- `SIG1-Sup`：完整匹配 L 骨架，从零 20＋10＋20 至 E50；无 U loader／伪标签／EMA，不能继承 PL 前缀。

两行均保持原 L 六前向、As KL0、ordinary/raw RDrop、LS.01、浅中层跨 RX 监督、CENTER_25K/CENTER_UNITS 和原分段预算／优化设置。PL 门槛 .95、U 权重 1。新 E50 的训练总成本分别 30 新轮／50 新轮；共享 E20 训练成本另计。实现 `sig1_repl_base.py`，发布 `publish_sig1_repl_base.py`。每行 final 自动 source V 八视图：raw、G(raw)、三 H、三 G(H)，保存完整 logits、TX/RX/day 与所有评估分组；当前不是目标测试或三学生 seed 完整结果。

运行 root：`PROJECT/runs/20261026-phase1-PAMF-SIG1-REPL-base-source-m2-r01`。PL E30：`F/F0-SIG1-PL-s2026092701/source/segment30`。PL/Sup E50：`F/F0-SIG1-PL-s2026092701/source`／`F/F0-SIG1-Sup-s2026092701/source`。另一方向可引用完整父链与预算，不改/停止公共控制。

GPU 不做静态占卡。首批最多两个本 run GPU worker，可用物理卡 0 至 7；每次派发核实全局 CUDA＋pending<2、free≥10000 MiB，计入其他 owner。原 NoSIG 六行已经全部完成，不保留它们的占卡预约。其他方向发布前独立核实实时进程，不能从本文件推断 GPU 仍然空闲。禁止中止任何其他 owner 的健康任务。

新改进学生另行最小登记、实际 smoke 后启动。最终同一候选三 seed、原目标 168000 包/seed、raw 与同 G clean＋六 practical、全部预测冻结后独立 truth-last；原联合门槛保持，不以源分数或共享控制的完成替代目标达成。
