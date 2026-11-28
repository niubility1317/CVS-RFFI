# LoRa 与 WiSig 共用教师一致性门控锚点损失筛查

- run_id：`20261128-phase1-shared-agreement-loss-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；WiSig 源域训练与 LoRa 源模拟池生成进行中。

## 目的与对照

仅在源模拟包的 clean 和锚点教师预测类别一致时保持锚点 KL，避免不一致时抑制 clean 教师纠错。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 36fe3ee02a6fdc279709ee9c118aebedbcd294ac；N607 控制器 4036548，LoRa 4036625/GPU 6，WiSig 4036626/GPU 4。两侧前 16 包真实专家和教师梯度 smoke 通过，PID/CWD/GPU 与日志增长核实；预检、单包 SHA、远端编译和独立 P0/P1 审查通过。

## 结果与覆盖

两数据集共用 5% 实增益与 DURA 物理残差网络，分别训练权重。仅在源 U 模拟包的 clean 与锚点冻结教师预测类别一致时施加锚点 KL，clean 教师 KL、波形损失和 clean 锚点保持继续；门控仅训练时使用，聚合一致次数将记录，推理无需识别器。LoRa 实增益/WiSig R17 锚点由已曝光源 V 选择；每侧源 U 4096 包、clean 512、低城区 1792、其他五景各 358/359，源训练专用模拟流。未读取 U 真 TX、V 或目标参与训练；教师前向预定 12008/72032（含 smoke），源模拟及专家/教师祖先成本另计，筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
