# 冻结新恢复器源域识别一致性

- run_id：`20261109-phase1-frozen-models-proxy-s2026092701-r02`
- group_id：`phase1-frozen-models-proxy`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

对三个已经冻结的恢复模型比较源 clean 伪类别一致性，不是 TX 真值准确率。

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

## 固定方案与启动修复

672 条合法源观测、14 组，每组 48 条；固定比较 DURA 与累积量、累积量排序和独立多相盲恢复三个 final20 模型。固定 clean 识别器伪类别是参照，不是 TX 真值；每条重算 clean/raw/DURA 并核对固定源预测。F 仅用于冻结后源评估；结果不改变既定七景测试。

r01 在任何数据读取之前因 Python 同名常规包遮蔽失败；r02 沿用同一经校验发布代码，以直接脚本入口启动。远端已验证该入口正常导入，配对统计和异常输入检查通过，独立 P0/P1 静态审查无阻断问题；首次真实源样本检查作为本次运行首步。
