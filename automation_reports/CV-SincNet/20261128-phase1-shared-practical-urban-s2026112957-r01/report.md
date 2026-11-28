# LoRa 与 WiSig 共同恢复的低城区加权源域模拟训练筛查

- run_id：`20261128-phase1-shared-practical-urban-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；WiSig 源域训练与 LoRa 源模拟池生成进行中。

## 目的与对照

按用户优先低城区要求调整源 U 模拟场景比例，保留 clean 和其他五景；共同模型、固定筛查不变。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 8b7e1aeb2a96ac7a2db85f5013527df381317a80；N607 控制器 3981450，LoRa 3981463/GPU 2，WiSig 3981464/GPU 3。两侧前 16 包真实权重及教师梯度 smoke 通过，PID/CWD/GPU 和日志增长已核实；预检、归档 SHA、远端编译和独立 P0/P1 审查通过。

## 结果与覆盖

两数据集各自从合法源 U 固定抽 4096 包，其中 clean 512、低城区 1792，其他五景各 358 或 359；使用独立源训练模拟命名空间和随机流。该比例响应用户优先关注低城区的要求，其他场景与 clean 保留。共同 5% 网络、损失及 1000 步，分别训练权重；LoRa 实增益和 WiSig R17 锚点来自已曝光源 V。未读取 U 真正 TX、V 或目标进行训练；WiSig 均衡 4096 包版为 333/360，半区 +1/0 未晋级，LoRa 当时尚未读取。预定教师样本前向 12008/72032，含 smoke；源模拟、专家和教师祖先成本另计，筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
