# 源训练样本扰动相对残差分布

- run_id：`20261109-phase1-source-train-response-visibility-s2026092701-r01`
- group_id：`phase1-source-train-response-visibility`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（源训练分布诊断完成）

## 目的与对照

对 32768 源训练记录的三个视图量化现有四方向训练扰动相对观测残差的分布，不使用 F 或 TX 标签。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完成 83,450 条受损源训练观测；城市 low 的 1% 扰动低于残差能量 1% 的比例为 70.21%，固定 action 下 10% 推算为 15.84%。详见 [analysis.md](analysis.md)。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 固定口径和验证

使用与 1% 和 10% 响应训练完全相同的 32,768 个合法源 fit ID 和 source epochs 21/22/23，每个视图选择与训练相同的四个人工方向之一。统计六种受损场景；clean 可能存在零残差，单独计数并排除，不做零值替换。比值为 1% 扰动经同记录固定模拟 action 的能量除以接收 IQ 减模拟 clean 的残差能量；残差含噪声和表示误差，不能称 SNR。10% 由固定实线性 action 的 +20 dB 缩放推算，不代表重跑完整模拟。

CPU 两线程、GPU 关闭，首批 2 条真实源 IQ 的冻结 DURA 检查后继续；无 F/TX/V/目标读取或模型更新。代码 8e6db2f6fd80c910e0d757c8e62fd4bb285296c1；独立审查发现服务器不兼容数组转换，已改用 DLPack 并通过定向复核；缩放、异常和真实环境转换三项测试通过。
