# 源IQ频率分位复数与幅相理想修正诊断

- run_id：`20261031-diagnostic-DURA-source-frequency-oracle-s2026092701-r01`
- group_id：`phase1-dura-source-frequency-oracle`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用原paired source512×noVar/default×cache01/20，四等频点分位分别理想替换complex及Q4幅/相，原F保持性评估，0G/更新/TX/V/target。

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

旧source描述显示Q4仅约0.109%clean能量而相对误差大，不能据此认定是身份信息或直接改loss。本次只复用原phase-oracle的合法paired512 IQ，分别理想修正四个64频点分位的complex系数，以及Q4相位/幅度，检查原冻结F的clean选择保持和特征误差。按每包clean谱能量排序，不按标签、场景或RX选频点；所有干预只用于source机制诊断，不可部署、不作真实TX准确率或恢复极限。F读取14848分析+2Ls smoke，G/更新/TX/V/target均0。全部场景/组保留，无自动训练资格，不依据本次或健康noPriorAux目标结果调参。3项聚焦测试通过，唯一P0/P1审查PASS（无直接发布缺陷），发布后首步实际F smoke直接继续CPU2。原F50/数据契约/其他owner/默认DURA保持。
