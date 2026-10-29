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

## 实际两row完整分析（2026-10-29）

实际终态VERIFIED/ANALYZED_DEVELOPMENT_ONLY：共享cache和两row全部15stage COMPLETE，两个登记row ANALYZED、owned退出，foreign0。代码b578c074ba68b8ff9aa539209b3410f1f103ba3d，主practical_increment不变，科学NO_PROMOTION，不扩seed、不调目标阈值、不把control换成主结果。目标仍ACTIVE，正式RX24-18未访问。

两arm实际从同E51各222步，累计11544，defaulthead/Sinc均更新、weightdecay1e-4/CENTER_UNITS fresh，未另训身份head。额外view control全部clean；primary L三景9568/9582/9266、OE4839/4663/4706，均为固定独立scene流选择而非标签路由。训练不读sourceU/V/novel/target；source评测各自新cleanL几何、L/OE4四view读出200步、knownV30组校准后才novel预测，全部预测固定后独立truth。C0/P1 parent_E51精确复用上轮augmented_readout，原F0baseline同输入复用。

主MLP C0 parent→cleancontrol→practical：unknown96.8750→97.5694→98.6111%，knownaccept90.4514→90.2778→85.7639%，knownopen71.5278→70.1389→57.8125%，closed78.2986→75.6944→67.5347%。原F0closed79.6875%，primary总损伤21.8750pp，clean未知单项越高不能掩盖跨RX身份退化。

P1相同三状态unknown38.8889→37.1528→50.0000%，knownaccept93.0556→94.2708→89.0625%，knownopen68.5764→69.0972→62.8472%，closed71.8750→71.8750→69.9653%。原F0closed72.7431%，primary总损伤9.8958pp；相对matchedcontrol多拒未知37/288，却少开放正确36/576（−6.25pp）、AUROC87.2962→85.2515。primary逐景未知60.4167/46.8750/42.7083%均未过90，known保护均失败。

P1整体AUROC85.2515%已低于同时95%接收/90%拒识所需85.5%必要界；primary两条件闭集已分别低于原F0−2pp的77.6875%/70.7431%开放下限，Aopen≤Aclosed，故仅挪阈值无法达标。未扫描target oracle。源knownV小变化不能代表新RX身份保持；完整源/target所有方法与RX/TX/scene见结果报告，source两主目标也失败，不以源校准合格宣布泛化成功。

实际checkpoint cache F/G合成、每row native梯度、source/dev合成smoke通过；32unique tests/68subs覆盖，唯一review目录P1已RED/定点修复/一次targetedPASS。一次N607preflight/单归档SHA比较/compile VERIFIED；live driver2673776/cache2673864、rowdriver2675837/2675840、训练2675845/2675846、source2678276/2677698的PID/CWD/命令/CUDA和ownedGPU采样/日志增长已捕获，native训练GPU7/4各一次1540MB采样，不是峰值。训练101.4671/86.8424s；全pipeline712.9620s含缓存/并行/模型载入/完整源和开发处理，不是逐包延迟或因果加速证明。

完整444个native steps、800个readout loss、全部source和development评分及独立post_state在本run local_artifacts，实际权重/预测留N607不可覆盖root。完整报告为`docs/PAMF_PHASE3_SOURCE_PRACTICAL_ENCODER_RESULTS_20261029.md`。source/dev仍一模型seed、一个开发RX/三个star主景，novel3/source和开发已暴露；targetU/formal0、匿名注册0。本轮结果支持先保护父模型身份函数，进一步研究受约束星地特征映射及合法U无标签适应，不更新默认F/G、不回收开发Q训练，不把完整目标缩减为source拒识。
