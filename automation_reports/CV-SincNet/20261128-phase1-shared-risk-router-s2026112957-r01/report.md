# LoRa 与 WiSig 共用源伪标签保守专家路由筛查

- run_id：`20261128-phase1-shared-risk-router-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两侧源域训练进行中。

## 目的与对照

共享物理特征网络依据当前包选择实增益或 R17，源域 clean 教师伪标签构造保守切换目标。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 09dc5582d58535793e9312aa1a7f7190c41cb154；N607 控制器 3896268，LoRa 3896335/GPU 7，WiSig 3896336/GPU 1。真实专家首步、有限非零路由梯度、PID/CWD/GPU 与日志增长均已核实；预检、单包 SHA、远端编译及一次独立 P0/P1 审查通过。

## 结果与覆盖

共同物理特征路由网络和损失，LoRa 实增益及 WiSig R17 为分别独立训练时的固定默认锚点。源 U clean 教师 argmax 为伪标签，只有其他专家正确且锚点错误时目标为切换；锚点正确的负例权重 2，两者均错忽略，并以 clean BCE 防止切换。使用 LoRa W24 E104 和 WiSig Tri-SIG-TIME SSL F0 冻结原默认头，推理只使用当前包和恢复专家。预定 LoRa/WiSig 教师样本前向 12000/72000 次，专家和教师祖先成本另计。源 V 多轮曝光并用于锚点及方案选择；不使用 U 的真实 TX、V 或目标训练，筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
