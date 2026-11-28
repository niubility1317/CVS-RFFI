# LoRa 与 WiSig 共用 R17 物理回投影实增益筛查

- run_id：`20261128-phase1-shared-r17-realgain-s2026112893-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；源域训练已在 N607 启动。

## 目的与对照

使用同一共享核心、相同源域目标和推理公式，分别在 LoRa 与 WiSig 合法源 U 训练，并对单 RX 的低城区与 clean 做预登记同包筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `16bb009f485ed41790d64da32bc07269beb8de83`，独立 P0/P1 审查发现的 clean 归一化问题已修复并经针对性复核，5 项本地测试通过。N607 发布归档传输校验和远端编译通过。控制器 PID 3626302；LoRa worker PID 3626317、GPU 0，WiSig worker PID 3626318、GPU 5；已核实工作目录、命令、GPU 和首步真实权重加载、零初始输出及有限非零梯度。WiSig 日志增长到第 50 步。启动前两卡空闲分别约 20,013 MiB 与 18,234 MiB，保留其他 owner 的进程；按用户最新“显存充足即启动”要求，每卡仅新增本 run 一个 worker。

## 结果与覆盖

当前仅有源训练启动证据，尚无本版本冻结权重、筛查结果或正式七景结果。两数据集共享同一个增益核心、训练目标及推理规则，分别训练权重；输入为 LoRa 8 × 1,024 和 WiSig 8 × 32，分别连接同契约 R02/R07/R17。两套预登记源 V 筛查包及历史基准已曝光，目标数据保留；具体门槛与数据角色见 `experiment.json`。LoRa 同口径固定 DURA v1 对照仍待补齐，未齐全前不得晋级。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
