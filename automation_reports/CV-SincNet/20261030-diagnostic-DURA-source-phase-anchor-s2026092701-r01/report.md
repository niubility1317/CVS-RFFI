# DURA 公开L-STF公共相位参考可行性诊断

- run_id：`20261030-diagnostic-DURA-source-phase-anchor-s2026092701-r01`
- group_id：`phase1-dura-source-public-phase-anchor`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

CPU2/0G/0更新；63000源IQ公开相关器，62488统计单一圆均值、留出512验证完整F保持；15RX/day逐一验收，无TX/V/target。

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


## 最小研究说明

相位均值只用原63000源L/U中排除固定512之后的62488条估计。512条只是统计留出，F50仍有原完整源训练知识，不能写成识别器未见域或新盲测。公开L-STF相关器沿用项目固定bank、80:160窗口、相对CFO修正和320延迟格；不含学习参数，不将复增益解释为TX硬件参数或已识别I/Q约定。上游参考见实验结果中的public_bank_reference，不改项目25Msps/20MHz解析模板。

单一全局圆均值µ必须同时满足：整体有效比例≥99.9%、合向量长度ρ≥0.99；15个RX/day分别有效比例≥99.9%、ρ≥0.95、组均值距µ≤5°。另在留出512上按µ−arg(gain)旋转clean，要求同F总体教师选择≥99.5%、每RX≥99%、平均Fcos/KL各≤0.001。F指标不用于重新拟合µ。任一失败就不据此发布或训练全局锚点，不改成RX专属均值或放宽阈值；相位无定义明确报告，不补默认角度。

这是CPU2/0GPU/0G/0神经训练的源诊断，源F参与资格判断，标F-informed。真实2Ls Fcheckpoint smoke首先执行，再统计与检查，F参数/状态/梯度保持。通过也只是另立研究资格，不能推出目标clean相位先验或直接改变部署。默认DURA及逐景3pp/clean/效率目标不变。


本地6项相关检查通过，唯一独立P0/P1审查PASS；N607预检核实身份/路径。使用既有CPU2环境与原VALIDATED_ONCE源句柄，源码/config以本次Git commit固定，真实F smoke在launcher首步直接继续。
