# 冻结原 F 源特征的分类头保护 RX 方向证伪

- run_id：`20261031-diagnostic-DURA-source-F-RXgeometry-s2026092701-r01`
- group_id：`phase1-dura-source-head-protected-RX-geometry`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

源F缓存63000/fit62488/holdout512，soft6后验条件RX均值、原头rank5保护、留一坐标/留一RX几何；无训练/TX/V/target。

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

原F50/CROSS_RX/defaulthead保持，复用此前合法source63000条clean特征缓存。固定512（L128/U384）完全排除于62488条拟合子集；源RX/日期与原7/63/30角色保持。用原F头连续六维后验作加权坐标，计算各RX条件均值，不读真实TX、不硬化后验、不训练身份或RX分类头。新的F输入只有真实2Ls smoke；缓存teacher元数据、物理顺序和两行数值与实际F核对，之后仅源缓存代数，F参数/状态不变、G读取及神经更新0。

取共享RX均值差，再投影掉原归一化分类头中心rank5，得到最多4维源RX相关方向。固定条件：保护后共享能量占条件RX均值差至少5%；留一后验坐标平均cos>.1且正向比例>.5；拟合与512留出的RX方向平均cos>.1；留一RX子空间捕获均值至少20%、且至少4/5个RX各至少10%。零RX差异无定义，明确defined=False、capture0保守参与均值。每个留一RX子空间只用另外四RX，不能用留出RX拟合方向。

alpha.25的特征变换只检查原头代数保护；argmax保持是构造结果，不用作恢复器科学有效性依据，也不说明实际G的clean保持。通过仅支持下一独立源损失梯度校准，不在本run训练G；失败不放宽规则或修改soft温度/子空间秩/接收机权重。与Oct26不同F、真实L标签的历史几何区分。本结果只陈述源关联，不是因果硬件或未见RX的G泛化证明。
