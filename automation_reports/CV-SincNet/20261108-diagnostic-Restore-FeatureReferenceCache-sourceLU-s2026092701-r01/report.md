# 只缓存源参考：真实恢复器梯度一致性检查

- run_id：`20261108-diagnostic-Restore-FeatureReferenceCache-sourceLU-s2026092701-r01`
- group_id：`source-production-G-F0-reference-cache-only`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留原恢复信号的两次F0前向及先后顺序，只把no-grad源扰动参考替换为缓存；与原路径比较真实G的损失、梯度和AdamW一步。

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

## 固定缓存对照与停止条件

基于同一原 source-only JointPrior fixed E30 与冻结地面 Tri-SIG-TIME SSL F0，固定原 source L_s/U_s 索引 0/6301、odd 轮次 1/3、真实模拟器 trace 和全源先验。原恢复 IQ 的两次 F0 前向及原先后顺序均保留，只用预先计算的 no-grad 源扰动特征替换在线源参考。逐项记录原 rtol 1e-5 / atol 1e-6 的损失、两路 IQ 梯度、真实 G 参数梯度和新建 AdamW 单步结果，并报告最大绝对差；不修改失败标准。

CPU2 nice10/CUDA empty，代码 f1382427f84a1e61f553518fa46f8b0405b2900c；其他命令、路径、输出见 experiment.json。数据/模型来源或源角色错误、非有限值或后验残差 >1e-8 则停止；原容差有差异要如实报告。4 项本地针对性检查和一次独立 P0/P1 审查 PASS，远端首步加载真实源 F0/固定 G。完成只覆盖源两样本、CPU、全源先验及新建优化器，不声称 GPU、完整训练、速度或正式识别质量。
