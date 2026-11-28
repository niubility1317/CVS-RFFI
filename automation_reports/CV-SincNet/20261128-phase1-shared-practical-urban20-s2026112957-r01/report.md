# LoRa 与 WiSig 共同恢复的 20% 低城区加权源域模拟筛查

- run_id：`20261128-phase1-shared-practical-urban20-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；WiSig 源域训练与 LoRa 源模拟池生成进行中。

## 目的与对照

两数据集共同使用 20% 逐包 RMS 校正上限；沿用低城区加权源模拟、网络和训练预算。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 b1cfe9582ebad04f458f861700ff842158b37446；N607 控制器 3989731，LoRa 3990080/GPU 6，WiSig 3990081/GPU 4。两侧前 16 包真实专家和教师梯度 smoke 通过，PID/CWD/GPU 与日志增长已核实；预检、归档 SHA、远端编译和独立 P0/P1 审查通过。

## 结果与覆盖

共同网络与损失、20% 训练和推理 RMS 修正上限，分别训练 LoRa/WiSig 权重；每侧源 U 4096 包中 clean 512、低城区 1792、其他五景各 358 或 359。源训练专用 namespace 和随机流，未读 U 真实 TX、V、目标 IQ/真值/统计；LoRa 实增益与 WiSig R17 锚点和扩大修正空间的想法来自已曝光源 V。与 5% 低城区加权版保持相同源样本和模拟、1000 步及筛查；不同 GPU 数值可能有差异。预定教师样本前向 12008/72032（含 smoke），源模拟、专家和教师祖先成本另计；筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
