# 源模拟微扰相对观测残差的能量诊断

- run_id：`20261109-phase1-source-perturbation-visibility-s2026092701-r01`
- group_id：`phase1-source-perturbation-visibility`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅源模拟数据，计算四种 1% RMS 人工扰动的模拟输出能量与同记录 received 减模拟 clean 作用的残差能量之比。

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

## 预登记及检查

固定源域 epoch19 的 576 条受损观测，12 个 RX 组与场景组合各 48 条；四种 1% RMS 人工微扰通过同记录模拟信道，报告能量比的中位数、10% 与 90% 分位数及低于 −20 dB 数量。分母包含噪声和模拟表示误差，该比值不是信噪比或身份信息上限。仅用于源域机制诊断，不修改既定训练或测试。

N607 CPU、2 线程；首批两个真实源样本运行冻结 DURA 无查询检查。代码 66132b20ee12a90821277115cac92bc2a34ff927；独立 P0/P1 静态审查未发现阻断问题；远端两项能量计算和异常输入测试通过。
