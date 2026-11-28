# WiSig R17 与实增益双先验的源域无标签筛查

- run_id：`20261128-phase1-wisig-r17-realgain-source-s2026112865-r01`
- group_id：`phase1-wisig-r17-real-gain-two-prior`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：源域训练、两景预测和独立评分完成；候选未晋级。

## 目的与对照

首轮 WiSig RX1 筛查未超过 R17 后，固定 R17 和独立 WiSig 实增益先验，学习物理伴随及实增益差异两条逐频实增益，使用独立 RX3 源验证筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
发布归档传输校验与远端编译通过。控制器 PID 3542492，首个训练 worker PID 3542516，物理 GPU 7；实际进程、CWD、显存与日志增长已读回。真实 R17／实增益 checkpoint 首步确认两条新增方向均有有限非零梯度。

## 结果与覆盖

源 RX 3 的 360 条记录在 clean／low urban 两景全部预测固定后，由独立 CPU 连接源 V 标签评分。该 RX 与第一轮 RX 1 物理记录互斥；历史源 V 与目标基准已有曝光。

| 路径 | Clean | Low urban |
|---|---:|---:|
| 直通 | 347/360 | 204/360 |
| 原始 R17 | 347/360 | 292/360 |
| WiSig 实增益 | 348/360 | 181/360 |
| 前轮融合 | 347/360 | 283/360 |
| 本轮双方向融合 | 348/360 | 274/360 |

相对 R17，low urban 同包改对 9 条、改错 27 条，净减 18 条；95% 配对区间为 −30 至 −6 条，两个预定半样本分别净减 12 条、6 条。Clean 相对 R17 净增 1 条。候选未满足晋级规则，未启动完整目标测试。

新增实增益头 11,394 参数，源 U 训练 2,000 步、48,000 次暴露，耗时 414.58 秒，峰值已分配显存 57,982,976 字节；不包含 R17 和实增益父模型历史训练成本。训练未读取识别器、TX 真值、源 V 或目标。详细固定预测标记、训练记录及评分见 `local_artifacts/20261128-phase1-wisig-r17-realgain-source-s2026112865-r01/readback/`。

## 交接

本候选已完成预登记筛查并未晋级。后续候选另行固定结构和数据范围；目标七景及 DURA v1 正式配对比较待完成。
