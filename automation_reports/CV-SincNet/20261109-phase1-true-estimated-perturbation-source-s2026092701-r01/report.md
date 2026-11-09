# 源模拟信道与估计信道的微扰响应对照

- run_id：`20261109-phase1-true-estimated-perturbation-source-s2026092701-r01`
- group_id：`phase1-true-estimated-perturbation`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在合法源 IQ 上比较已固定 DURA 估计信道与匹配模拟器信道对相同合成扰动的响应，无 F、TX 标签或目标读取。

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

## 固定诊断与权限

本诊断对固定的 672 条已探索 Phase1 源 L_s/U_s 观测，分为 RX4/6 和 RX8 各七视图、每组 48 条。对 clean 构造四种相对 RMS 1% 的合成扰动；同一扰动分别通过当前观测的 DURA 估计作用和匹配模拟信道作用，报告响应相对平方误差、实内积方向一致度和模长比。两个作用的输出只在源诊断比较，目标部署不可读取模拟真值。

固定 DURA v1.0 和合法 source epoch19 缓存；不读取 TX 标签、F、V 或目标，不更新模型。人工方向不是实测设备身份扰动，源域结果不能代替七景身份准确率。真实 checkpoint 首批 smoke 后继续完成其余样本，不因诊断结果改变任何健康训练或冻结测试。

N607 GPU4 预检至少 8192 MiB 空闲、总计算进程少于 3 且本任务该卡最多一个 worker；输出目录唯一，遇源合同、非有限数或资源错误停止本诊断并保留其他 owner。命令、路径、计算预算及预期结果见 experiment.json；代码 commit 在发布前回填。

代码 commit：`895e0fd944ec70de48ce54778cf2e73c66e1b7b1`。独立 P0/P1 只读审查未发现阻断；同向、反向、倍增、异常形状和非有限值计算检查通过。

本次发布归档本地与 N607 SHA-256 均为 `d037e1ab390fce6e89ba59bfb3c12603b262a2f77630231760c0edb045548a00`；远端编译通过。远端无 pytest，使用实际发布模块执行等价针对性断言，同向、反向、倍增、错误形状／类型和非有限输入均通过；不据此声称真实权重 smoke 已通过。
