# 冻结特征条件恢复器源端路径诊断

- run_id：`20261101-diagnostic-Restore-FeatureCond-source-paths-s2026092701-r01`
- group_id：`restoration-source-feature-condition-paths`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原Ls128Us384×20draw，冻结FeatureCond/F0，检验正常、zero-feature、无FiLM、源clean特征oracle、投影前raw及连续latent。旧DURA/Full源控制精确复用。0更新/TX/V/OE/target。

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

## 固定源端路径诊断

冻结FeatureCond/c8659b15…原final30/7410和原Tri-SIG-TIME SSL/3ff8/defaulthead，原R07训练角色Ls128/Us384固定512包×20现成draw。五条IQ读出为正常预测特征条件、zero-feature（保留FiLM bias）、完全无FiLM、source clean连续特征oracle、正常路径单位功率投影前raw；另对预测连续latent使用原F0默认头及特征误差作诊断，不训练或替换分类头。先用实际Ls2无query smoke核actual/latent/raw逐值等于原G及全部输出有限，PASS立即执行完整矩阵。

共享encoder10240行、四decoder40960行，五IQ读出51200条F0特征及clean512参考，共51712条F0特征；latent另10240条默认头运算。源Ls2 smoke额外12条F0特征/10条头运算、4条G encoder和10条decoder。原source-style/ac801 identity的DURA/FullFGrad源记录精确复用，不重复旧G前向。全部20draw/七源景/五路径+latent保留，0更新、TX/V/OE/novel/target读取、统计拟合、checkpoint选择或Main新预测。

这是训练源内的机制诊断。source clean oracle不可部署，输入分布不同于训练时的预测latent，又与已学decoder耦合，不作可达性能界、理论极限或真实TX准确率。raw输出没有独立训练，关闭条件也不是同预算从零消融。结果只能指导合法源端机制研究，不改本版已完成Main负结果，不根据Main误分类调参。当前六景3pp+clean目标未达到，默认DURA保持。

实际运行代码、数据句柄、祖先、命令、CWD、停止规则及产物见experiment.json。3项针对性检查和一次独立P0/P1通过，N607/原CVS-RFFI/FP32-TF32off，GPU4/6 UUID准入剩余>=12GB、原compute<=2、只新增1条本任务NN/单卡总<=3，其他owner保持。部署G代码和历史权重不改，未新增训练或目标访问。
