# DURA 源记录对比约束 GPU 训练

- run_id：`20261107-phase1-dura-instance-contrast-source-s2026092701-r01`
- group_id：`phase1-dura-source-instance-contrast`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：QUEUED；N607 队列 PID 3978828 等待 GPU 0 容量，正式训练尚未开始。

## 目的与对照

源域辅助编码器以同一物理记录的模拟观测与 clean 为正对，使用对称多正样本对比损失；每步先更新编码器，再冻结它约束 DURA 修正。部署只需当前 received 和冻结 DURA，不使用编码器、clean 或其他观测。候选包括 117062 个修正参数与 53424 个仅训练期辅助参数，固定 32768 条源记录 × 三视图、五轮。

重复出现的同一源记录均为正对，不同记录可能属于同一设备；对比目标也可能学习包内容或接收机信息，不能称为身份监督。思路受到同发射记录对比学习研究启发，但该恢复方案不是论文方法复现，其效果需单独验证。[原论文](https://arxiv.org/abs/2403.04036)

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

本机使用真实 DURA 权重和合法源缓存完成 6 次 GPU 技术更新；18 项针对性检查通过，一次独立 P0/P1 审查未发现阻塞问题。远端源 smoke、正式训练、七景预测及独立评分尚未完成，暂无识别结果。

## 交接

队列满足 GPU 0 资源条件后自动执行源 smoke、固定五轮训练和七景预测，再独立评分；恢复时核实唯一 owner，不重复启动。
