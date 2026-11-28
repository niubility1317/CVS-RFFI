# WiSig R17 与实增益双先验的源域无标签筛查

- run_id：`20261128-phase1-wisig-r17-realgain-source-s2026112865-r01`
- group_id：`phase1-wisig-r17-real-gain-two-prior`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源训练正在 N607 GPU 7 执行。

## 目的与对照

首轮 WiSig RX1 筛查未超过 R17 后，固定 R17 和独立 WiSig 实增益先验，学习物理伴随及实增益差异两条逐频实增益，使用独立 RX3 源验证筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
发布归档传输校验与远端编译通过。控制器 PID 3542492，首个训练 worker PID 3542516，物理 GPU 7；实际进程、CWD、显存与日志增长已读回。真实 R17／实增益 checkpoint 首步确认两条新增方向均有有限非零梯度。

## 结果与覆盖

训练完成后，队列自动固定源 RX 3 的 360 条记录的 clean／low urban 预测，再由独立 CPU 评分；该 RX 与第一轮的 RX 1 物理记录互斥。本候选尚无筛查结果；目标集七景和 DURA v1 正式配对比较待完成。历史源 V 与目标基准已有曝光。

## 交接

恢复时先只读核对本 run `state.json`、所属 PID 与输出，保留唯一 owner；不要重复启动。训练、预测与评分按原预登记顺序自动接续。
