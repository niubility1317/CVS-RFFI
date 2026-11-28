# LoRa 与 WiSig 共用源伪标签纠错加权专家路由筛查

- run_id：`20261128-phase1-shared-risk8-router-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；两侧源域训练进行中。

## 目的与对照

共享逐包专家路由：将源伪标签切换纠错正例权重固定为 8、锚点正确负例权重为 1。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 8af783b1e1b79a5bbf4408e6fc15013e8f6bbc65；N607 控制器 3904145，LoRa 3904219/GPU 2，WiSig 3904220/GPU 3。真实专家首步、有限非零路由梯度、PID/CWD/GPU 和日志增长已核实；预检、单包 SHA、远端编译及一次独立 P0/P1 审查通过。

## 结果与覆盖

共同物理特征路由网络和损失；源 U clean 冻结教师伪标签仅用于训练，切换纠错正例权重 8、锚点正确负例 1，两者均错忽略，并加 clean 不切换损失。LoRa 实增益、WiSig R17 为源 V 已曝光结果选定的默认锚点。WiSig 首版路由对 360 个低城区样本与 R17 正误完全一致时提出本版，LoRa 首版当时尚在训练。每侧独立权重，推理仅用当前包与恢复专家；预定 LoRa/WiSig 教师样本前向 12000/72000，专家和识别器祖先成本另计。无 U TX、V 或目标训练；筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
