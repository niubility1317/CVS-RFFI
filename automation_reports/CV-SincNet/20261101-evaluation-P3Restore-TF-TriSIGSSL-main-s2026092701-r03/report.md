# 地面Tri-SIG-TIME SSL主验收：默认DURA与两版独立TF全量七景

- run_id：`20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03`
- group_id：`restoration-ground-TriSIGTIME-SSL-main-seven`；类别：`diagnostic`；阶段：`Phase3`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

按用户澄清，性能主线回到原Main split同一冻结Tri-SIG-TIME SSL。四路相同received/同F0/原六景加clean，两原总体完整预测后独立评分；0训练。

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

## 主验收恢复到正确基座

性能为主线，效率为支线。原地面训练Tri-SIG-TIME SSL/3ff8/50-11100/defaulthead，与原Main split两总体VALIDATED_ONCE观测。四路同一received→raw/DURAv1/TF_IQ/TF_F0→同一F0；原cross_rx126000与cross_both42000，每景均覆盖原总体，不以Phase3开发集、旧F50或均值代替。两TF权重保持b338原final30，不重训或选择。source2实际模型smoke先于新测试输入，两任务4x7全部预测后原独立scorer才读真值。

新adapter五项聚焦检查与唯一P0/P1审查PASS。实际方法代码 d4c995c1139fc62779a2a254c269006324b0415f，F0来源3ff8、G来源b338/defaultDURA47c明确分开。性能下降不能用更快、更小或更少阶段抵消；默认DURA与全部历史输出保持。
