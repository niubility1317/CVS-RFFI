# LoRa 与 WiSig 共用双扰动一致性选择器筛查

- run_id：`20261128-phase1-shared-consistency-selector-s2026112913-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两数据集源训练已在 N607 启动。

## 目的与对照

同一整包门控结合实增益、R17 和物理作用残差，以源 U 同包两次模拟扰动的一致性与错包对照生成无 TX 训练标签；分别源训练并筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `d31d77bd7fd7110eb53bdf2b4fef6ce1246ab223`；独立 P0/P1 审查未发现直接缺陷，12 项本地测试通过。N607 归档传输校验及远端编译通过；控制器 PID 3701752，LoRa worker PID 3701766 / GPU 7，WiSig worker PID 3701767 / GPU 2，已核实 CWD、命令、真实专家首步与有限非零梯度；WiSig 日志增长到第 50 步。启动前两卡空闲约 16.9 GiB 和 15.9 GiB；每卡仅一个本 run worker，保留其他 owner 任务。

## 结果与覆盖

本候选尚无筛查或正式结果。两数据集共用同一个整包选择器及双扰动训练目标，分别使用各自合法源训练的实增益与 R17 专家和独立头权重。源 U 中对每个物理包模拟两次受扰，比较专家同包相关性与循环错包相关性的差值，构造无 TX 标签训练目标；错包可能来自同一 TX，属于该代理目标的限制。clean 输入和第二观测只在训练使用，部署逐包只使用当前接收 IQ 与冻结专家。LoRa 专家历史上继承授权 E104 指导，WiSig 专家 F-free；新头不读 F、TX、V 或目标。源 V 面板已有多轮曝光，门槛和各自同包对照已固定；LoRa 自训 R69 结构适配与原 WiSig 固定 DURA v1 checkpoint 身份分列。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
