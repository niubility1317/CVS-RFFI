# LoRa Full R69 结构适配同包 W24 源对照

- run_id：`20261128-diagnostic-lora-r69-full-w24-s2026112895-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；仅为已曝光源 V 对照。

## 目的与对照

复用 LoRa 合法源训练的 256 点 R69 架构适配及其源门控，投影回同一 8192 点 Full 接收包并接冻结 E104，对既有 60 个源 V 包做 low urban 与 clean 对照。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `cdee6cf829de5aaa6ef644d2dc065697c6cbdd56`；独立 P0/P1 审查修复并复核了 smoke 门控旁路问题，2 项本地测试通过。N607 归档传输校验与远端编译通过；控制器 PID 3635201，发布后已完成。独立产物证实 2 个合法 U 包强制经过实际 R69、回写和 E104 的无查询 smoke；具体 GPU 与短时 worker PID 未在其结束前独立读回，标记 UNKNOWN。已有其他 owner 进程保持。

## 结果与覆盖

源 V 的 b200_1 共 60 包，低城区和 clean 的预测均固定后独立 CPU 连接真值；同包 ID 和重放的直通预测与原实增益筛查一致。LoRa R69 结构适配低城区正确 33/60，直通 30/60、实增益 34/60；相对实增益改对 5、改错 6、净减 1，95% 配对区间 [-7, 5]。Clean 三者均为 60/60。源门控 clean 激活 0 包，低城区激活 26 包。

候选与对照各自预测固定后，独立按同物理 ID 配对：统一方法低城区 15/60，相对 LoRa R69 结构适配改对 4、改错 22、净减 18，95% 区间 [-27, -9]；clean 59/60 对 60/60，净减 1。源 V 已在历史研发中曝光，结果只供开发筛查，未读取目标。LoRa R69 采用 LoRa 训练权重与适配，保留与原固定 WiSig DURA v1 checkpoint 的身份差异；完整目标验收仍未完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
