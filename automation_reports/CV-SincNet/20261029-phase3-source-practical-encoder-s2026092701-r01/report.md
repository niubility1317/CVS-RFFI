# PAMF第三阶段：匹配预算的源端星地编码器联合增量

- run_id：`20261029-phase3-source-practical-encoder-s2026092701-r01`
- group_id：`pamf-phase3-source-practical-native-encoder-increment`；类别：`diagnostic`；阶段：`Phase3-source-practical-encoder-and-fixed-development`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Two independent222-update E51 children with original native six-view L supervision, matched extraCE0.68 and sourceOE0.1; clean vs practical+frozenG second view. Own clean-L geometry/four-view readout and calibration, then fixed C0/P1 development vs completed parent_E51 augmented-readout predictions.

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

## 实际冻结预登记

代码/config commit `b578c074ba68b8ff9aa539209b3410f1f103ba3d`；最小科学计划见`docs/PAMF_PHASE3_SOURCE_PRACTICAL_ENCODER_PLAN_20261029.md`，精确runtime为本run local_artifacts/runtime.json、发布后在release/runtime.json。两个独立222步E51子模型、native默认头/Sinc联合训练；primary practical_increment，control clean_increment。保留native原六路source监督，额外L CE0.68 clean vs star/G；OE21两个匹配view平均uniformKL0.1。两arm均保留历史native As，control不是完全无卫星训练。原11322＋222＝11544，optimizer重置，不按V/novel早停。

共享cache只L6300/OE21共6134，12434物理/49736计算view；训练不读V/U/novel/target。每arm自动train→source own-new-L geometry/四view读出/knownV校准/novel预测→独立source score→固定C0/P1预测和独立评分，15个stage，不等其他row或按分数改排。源校准30组FRR5%/原F总2pp；novel不拟合/校准。Q864 samecanonical原记录、parent_E51精确复用上轮augmented_readout，目标U/正式RX24-18为0。最新clean95/star90每主景与known保护保持；只开发单seed/RX，不称新TX或严格新日期盲测，无DA/聚类/注册声明。

聚焦32项唯一tests/68subtests已覆盖通过；dev原7tests/24subs与新增真实目录回归的定点邻近3tests/10subs分开保留，未重复无关全套。唯一P0/P1审查发现旧parent fit在/source/fit而非/fit的目录契约错误，已RED复现、定点修复、一次针对性确认PASS；其余无P0/P1。训练CachedL也保留原native tx_label_visible=True和正确cache_index，避免误用物理base_index。

N607一次preflight实际PASS：E51权重/源OE21 capsule与split/已完成父Q句柄匹配，父pipeline终态且owned0；所有卡有实际剩余容量。最多两row GPU worker、每卡最多新增两个，预约剩余显存4096MB/槽，不停止其他owner。单归档一次比较/远端compile、实际checkpoint smoke及live GPU/日志证据将在运行中补充；登记不代表运行或性能完成。
