# 独立大容量时频恢复七景测试

- run_id：`20261108-phase1-direct-large-timefreq-ground-test-s2026092701-r01`
- group_id：`phase1-direct-large-timefreq`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：QUEUED；源训练固定最终轮后自动预测。

## 目的与对照

从单条 received IQ 直接恢复；时域、频域联合全窗口注意力。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

固定最终第 20 轮后自动执行七景预测，每景 168000 条；全部预测固定后独立 truth-last，并与原 DURA 逐记录配对。尚无正式识别结果。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 七景最终评分

七景各 168000 条预测固定后，已完成独立 truth-last 与 DURA 配对评分。六个实际场景均低于 DURA，且均未达到相对下降 3% 门槛。详细场景、配对纠错及 TX/RX/day 分组见 [结果分析](analysis.md)。本结果来自已曝光的 Phase1 地面模拟测试，未用于其他已固定实验的训练或选模。
