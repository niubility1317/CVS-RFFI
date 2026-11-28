# 共享置信度路由 LoRa 技术补跑

- run_id：`20261128-phase1-shared-confidence-lora-retry-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；LoRa 技术补跑已通过真实权重首步。

## 目的与对照

LoRa 首步因历史模块 NamespacePath 初始化失败；先加载合法专家再加载冻结教师，仅补 LoRa，WiSig 原行继续。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 26d14407d12f5cc9ba896fe012383d405aaaa09a；N607 控制器 3922715，LoRa worker 3922730 / GPU 7。先用已验证的 experts 初始化再加载教师，真实权重首步和有限非零路由梯度通过，PID/CWD/GPU 已核实。原 WiSig 行保留运行，没有重复启动。

## 结果与覆盖

技术补跑保持原候选结构、损失、seed、源 U、1000 步和源筛查矩阵，仅修正初始化顺序。原 LoRa 首次失败发生在训练之前，日志保留。LoRa W24 E104 训练和推理各参与，推理每包额外两次 F 前向；源 V 已多轮曝光。筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
