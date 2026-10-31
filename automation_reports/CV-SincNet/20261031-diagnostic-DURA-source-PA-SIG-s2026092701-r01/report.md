# PA与时序签名互补性：原头源域联合替换诊断

- run_id：`20261031-diagnostic-DURA-source-PA-SIG-s2026092701-r01`
- group_id：`phase1-dura-source-PA-SIG-complementarity`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定defaultDURA/source512×01/20，复用实际加权四分量，联合替换PA+SIG的clean源分量，接同一冻结原头；对照native/PA/SIG/allclean。0G/模型训练/TX-V-target。

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


## 固定 PA＋时序签名源互补假设

固定默认DURA/source512（128L/384U）×缓存01/20，复用此前实际原F50四个加权分量。native、只替换PA、只替换SIG、联合PA＋SIG、allclean五案例均使用同一原头；联合案例保持restored identity/response，按实际加法顺序重建。旧PA/源IQ资格FAIL保持，不用此前正式结果挑父模型；无新G、G权重读取或模型训练。

主联合案例须两视图均较native净增≥6/512、较SIG净增≥2；五源RX较native不负，每RX较SIG最多损失1且至少4/5不负；各非clean场景较native最多损失1、clean不损失。仅支持后续独立学习假设，不自动启动G；不放宽标准、换主或搜索阈值。源clean F选择不是真实TX，潜在混合特征可能没有对应IQ，pairedclean不可用于部署，allclean只是构造性正控制，不称恢复极限。

2项聚焦检查和实际入口通过，唯一P0/P1通过。首步真实2Ls核对实际完整F四加权分量、原source缓存与原头，F2/head2，PASS立即继续。主环完整F0/G0、仅原头5120输入；CPU2/空CUDA，0模型优化/TX/V/target。一次N607 preflight/归档SHA比较/远端编译，实际PID/CWD/argv/空CUDA/日志增长核实；全部1024源观测、5120头输入和分组独立复算。默认DURA与完整六景3pp/clean/效率目标保持。
