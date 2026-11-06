# C4 等变 IQ 恢复器，单次后验、从零 30 轮

- run_id：`20261107-phase1-Restore-CoordinateMAPDetail-E30-s2026092701-r01`
- group_id：`restoration-c4-operator-posterior-detail`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

共享物理和细节分支覆盖四个直角 I/Q 旋转。合并回原坐标的物理算子，单次 FP64 后验，恢复原坐标 IQ。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定比较

四个共享分支把估计的全实数物理算子变换回输入坐标再平均，同时平均正噪声方差。源 63,000 条 IQ 的高斯统计经四个直角旋转做矩匹配，固定零均值及旋转相容协方差。一次 FP64 后验求解后，共享 MSAT 细节分支校正四个坐标视图，输出映回原坐标再平均。该结构和先验共同改变。

训练仍用原 L_s/U_s，固定源模拟配对、连续 F0 特征和独立物理/细节梯度，全部权重从零单阶段 30 轮。真实 source2、梯度隔离、直角等变检查、初始零细节和严格 E0 重载通过后自动训练。最终轮次固定后自动预测原 Main 双总体七景，预测全部固定后独立 truth-last。逐景核对距同一 raw clean 3 个百分点，两总体分别保持 clean。
