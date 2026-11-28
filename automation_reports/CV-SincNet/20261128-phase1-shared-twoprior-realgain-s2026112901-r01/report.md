# LoRa 与 WiSig 共用双恢复分支实增益融合筛查

- run_id：`20261128-phase1-shared-twoprior-realgain-s2026112901-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两数据集源训练已在 N607 启动。

## 目的与对照

两个数据集都输入各自合法冻结的实增益与 R17 两个恢复结果，由同一物理残差与逐频点实增益核心融合，独立源域训练与筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `cf2d93c91b9d53cd43da3dc0b9137ac850e3674e`；独立 P0/P1 审查和新增对照的针对性复核完成，本地 8 项测试通过。N607 发布归档传输校验和远端编译通过；控制器 PID 3647832，LoRa worker PID 3648255 / GPU 0，WiSig worker PID 3648256 / GPU 5。两行已核实 CWD、命令、真实专家权重、首步零初始化和有限非零梯度；WiSig 日志增长到 50 步。启动前两卡空闲约 20,013 MiB 和 18,220 MiB，按用户最新显存充足即启动要求，各卡只新增一个本 run worker，其他 owner 任务保留。

## 结果与覆盖

尚无本候选筛查或正式结果。两数据集共用同一个双恢复分支融合核心、训练目标与推理公式；输入适配、冻结实增益/R17 专家实现和独立训练权重不同。LoRa 实增益专家历史上使用授权的 E104 指导及 U 伪标签，WiSig 专家无 F 指导；新融合头仅用本数据集源 U 的 IQ，不读 F、TX、V 或目标。LoRa 和 WiSig 继续使用既有已曝光源 V 包，各自固定低城区和 clean；样本、门槛、同包对照和停止规则见 `experiment.json`。LoRa 的匹配对照是合法源训练的 R69 结构适配，区别于原 WiSig 固定 DURA v1 checkpoint。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
