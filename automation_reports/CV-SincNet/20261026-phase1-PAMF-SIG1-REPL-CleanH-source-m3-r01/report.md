# 单视图 REPL：仅 clean 和直接星地 IQ

三种方法各一个F seed2026092701，属于源域证伪，不是三种子目标结果。原完整PAMF保留一个canonical SIG_TIME（18976参数）和原joint160六类默认头联合训练。唯一矩阵、seed、数据/祖先、环境/CWD/命令、独占路径、GPU/owner、停止规则见[experiment.json](experiment.json)。代码身份为发布runtime的Git commit，不另建代码SHA或审批。

## 权限与输入

本run执行用户最新要求：仅clean IQ及直接叠加practical residual/post_sync/noEQ星地信道IQ；不使用恢复器代码根、权重、恢复IQ/特征/伪标签或恢复分数。新context/runtime仅F/AF/BASE/FOLD四份身份网络代码，信道模拟用本次release/code/leo_practical。该channel.py/residual.py相对原物理版本47c264的Git diff为空；不因新方法重验未改变的source数据。

原F0 L6300/U56700/V27000及物理角色互斥保持，sourceRX1/3/4/6/8、day1/2/3、六TX。U TX隐藏、RX/day可见，不能据RX推TX或负类。L全部真标签，不套用U置信门控；V只在冻结后验证，不更新参数/buffers/EMA等状态。默认分类头原样联合训练，没有后训练头或额外身份/RX分类器。

只加载已核实同契约m201/singleSIG_TIME ordinaryPL scratchE20（4440步、空祖先、无target/隐藏Utruth）；各行独立+10+20到累计E50/11100、13L+37LU。NoStyleNoRXE的完成config仅提供既有REPL配方元数据，不加载它的权重，更不继承被用户停止的任何恢复训练学生。新配置移除旧G元数据和G代码根，拒绝恢复/目标/新头句柄；继承核对复用既有实际source契约和E20证据。

## 三行独立差异

| 方法 | L | U | 新低置信关系 |
| --- | --- | --- | --- |
| REPL_H2_U | 保留原共享As两次真标签CE，As KL0 | 原mixed physical半替换为Ha/Hb，ordinary/raw保持 | 无 |
| REPL_H2_U_Pair | 同上 | 同上 | raw95拒绝physical半的同包Ha/Hb |
| REPL_H2_LU_Pair | 原As physical半也用独立Ha/Hb，全部真标签CE/As KL0不变 | 同上 | 同上 |

Ha/Hb使用相邻不同的三种source景及独立a/b随机实现，按物理ID、累计epoch、角色和a/b命名；source session接收机硬件保持。三景为low suburban/mid urban/low urban，不挑信道seed。原ordinary噪声、RDrop、L浅/中跨RX和U原ordinary浅/中/joint跨RX、raw对、LS.01、U clean95/权重6和原U/L cap1均保留，style和全特征RX外推关闭。已完成对照不训练、不重预测。

第一ordinary对和最终raw对不变；U只改变已有第三/第四次forward的physical半，L第三行只改变已有两次mixed forward的As半。L双H时不再生成会被覆盖的原共享H，防止无用信道计算。U每行30轮生成1701000个H输入；前两行L原生成426240个共享H，第三行L生成852480个独立H。F仍为原REPL每update L768包、U6次学生前向与原clean伪标签前向；没有新增F调用，但H生成成本增加。普通PL本身只有U1，不能把与原REPL相同的计算说成与普通PL计算相同。

低置信项为0.5×[KL(stop(pb)||pa)+KL(stop(pa)||pb)]，仅乘原clean置信<.95的physical mask，再按整个原physical半批平均；系数1.2，与原mixed损失尺度对应。两侧target各自detach，不熵锐化、不赋硬TX、不构造跨包正负类；共享错误均值没有自纠信号。新项加进原U pending_loss，一起接受原guard，不能绕过保护。仅复用已有H logits，0额外F调用。原G-G代理诊断不能证明H-H有效；本run必须由真实H smoke和源结果检验。初轮不额外加AWGN。

## 预先固定的源比较

复用既有PL-X3、REPL-SIG1、NoStyleNoRXE源预测，只投影raw和三个直接H部分，不读取恢复权重或生成恢复视图。PL-X3有三signature视图，单独说明它是历史source参照，不能冒充单视图严格目标对照。实际同架构Sup/普通PL三Fseed正式对照已经完成，保持原结果不重训。

三参照的raw最大26574/27000，H3最大60889/81000。本次通过条件固定为raw≥26547（最多−.1pp）且H3≥61132（至少+.3pp）。每新学生完整源V27000×4、按RX/TX/day报告，参数和全部注册buffers保持；三行全部完成后才选择。通过者按raw正确数优先、H3其次、并列偏好较少H/无新项的METHODS顺序。无通过不扩种子/目标；不得将源结果或同包数学性质宣称为目标改善。

最终完整目标仍为同一方法三Fseed和每seed原168000物理包（cross_rx126000+cross_both42000按正确数加权；cross_date30000另报），raw clean和六practical预测全部固定后独立truth-last。raw clean绝对下限85.98174603174603%、raw六景>72.0961970899471%不降低；仍需较监督和普通PL高1pp/1.5pp。DURA分数不进入新训练、源筛选或扩seed决策。

## 发布、执行与状态

10项focused数学/标签边界/mask/forward顺序/预算/配置/实际任务入口检查及编译通过，登记VALID，唯一P0/P1审查见[local_verification.json](local_verification.json)。实际原E20权重联合L128/U256 backward smoke为launcher第一步，检查真实双H差异、单视图原头、U总梯度及Pair项对sig/head/body非零有限梯度；PASS立即训练，无新审批token。

N607普通账户，GPU0..7，全球CUDA+pending每卡<2且free≥10000MiB才派发，计入其他owner。独占run/release root、单份archive传输比较、远端compile、独立PID/CWD/argv/GPU/step增长读回。任一系统技术失败停止新派发，保留本run健康已启动子进程和全部产物，不自动重跑，不操作其他owner。每行训练结束自动源冻结/4view预测/分析；候选通过后再登记目标三seed扩展，当前不是目标测试run。实际运行证据随后追加。


## 真实发布与训练启动

VERIFIED_RUNNING。实际执行commit `d860ee49a036c599bcb0d7848feed22cc7de8bd7`，发布时上游OID一致；archive传输比较VERIFIED、远端compile PASS。实际runtime四来源F/AF/BASE/FOLD，没有恢复器root/weight/IQ/分数句柄。

三真实E20联合source smoke全部PASS：每行L768/U256×6，原clean门控接受234/256（接受率不是正确率）；U双H256包，差异相对L2=1.074821；LU行L双H128包、相对差1.031408。每Pair行使用原physical半的10个拒绝U包，新项对signature/head/body梯度范数约0.102400/0.135390/0.235053，均有限非零；L全真标签、As KL0和ordinary跨RX保留。smoke无optimizer更新、额外F调用、Utruth/target读取。它证明执行，不证明性能收益。

三个真实学生在GPU0/3/4各一worker，PID/CWD/argv/CVD/onGPU及两次独立step增长均已核实。独立检查当时全部可读656条训练记录（218/223/215），包括部分批次后seen56700+256的边界、H/L/U/guard/门控计数和有限值；第二次进度读取234/236/215条。前两行已完成新增第1轮，实际耗时198.36/195.25秒；不是完整30轮耗时或收敛。0非有限损失/梯度跳步。旧停止任务没有恢复，已完成对照未训练/预测。

这些是三个变体各一个seed，非三seed成绩；当前只有源训练，无新目标预测/准确率，完整目标未达成。后续逐行按原固定计划自动冻结/源4view评测，全部完成后再判断。证据：[实际smoke、进程与完整可读记录检查](launch_evidence.json)。


## 运行中的机制观察：提高U权重不等于同比提高实际梯度

读取当时三个学生全部已生成的完整可读记录，分别1093/1136/1040步，共3269步；这是尚在运行的片段，未抽样，也不是完整30轮/收敛。没有新IQ/模型前向、V/target/U真值读取，不改变配方或候选顺序。源全矩阵的独立核算工具已准备，须等全部三个source完成后才运行，不能把准备好工具写成已核算。

| 方法 | 完整可读步数 | t2限幅步骤 | 分类头限幅步骤 | other限幅步骤 |
| --- | ---: | ---: | ---: | ---: |
| H2_U | 1093 | 86.18% | 87.74% | 83.90% |
| H2_U_Pair | 1136 | 87.76% | 88.29% | 84.15% |
| H2_LU_Pair | 1040 | 87.69% | 89.42% | 83.65% |

实际U/L原始梯度范数中位数约1.85至2.10，guard后保留的U范数/原U范数中位数约0.48至0.54。它说明既有weight6经常触及cap1；继续把统一weight加大，通常也会被限幅，不能把数值权重增幅当作真实作用同比增强。新H同包项仍可能改变梯度方向，因此这不是“半监督不起作用”的结论；当前先完成既有三行，不据此热调权重或选择变体。

这里统计的是合并后的U参数组梯度，包含原类别/跨RX/一致性等，不是新pair的单独贡献，也不是AdamW参数位移或样本利用率。other包含SIG_TIME及其他不在t2/cls_head组的参数，不能把other比例专门写成SIG_TIME的保留率。loss、接受率和梯度方向都不能替代最终clean/六景测试。本run仍在训练、目标未完成。见[九组完整可读记录汇总](ongoing_gradient_summary.csv)和[范围证据](ongoing_mechanism_evidence.json)。


## 三项均完成E30并接续最后20轮

三个学生均已完成从共享E20开始的新增10轮/2220步，形成各自实际E30检查点，累计每模型30轮/6660步；新增更新总数3×2220=6660，不能把共享父训练重复计算为本轮新增成本。独立CPU核算了三个实际checkpoint的run/row/seed、来源E20、source契约、原六类头6×160、唯一SIG_TIME分支、参数有限值，以及完整U/L/H2计数。没有新增模型前向、IQ/V/U真值/target读取。

每行该阶段U见567000包、学生视图3402000，双H输入567000；前两行原L共享As生成142080包，第三行L双H生成284160包。L全部真标签/As KL0、原跨RX关系、无恢复器权重/IQ保持。已观察到最后20轮的实际step记录从1开始，采集时分别1002/1100/771条；E50最终父来源和参数还须完成时核实。

当前不是最终模型或完整源比较，不按E30或中间训练loss选模。保持原计划到E50、逐行冻结并生成source4view，再按固定raw/H正确数门槛比较。证据：[E30实际检查点与接续核算](E30_transition_evidence.json)。


## 三项源实验完成后的独立核算

VERIFIED_COMPLETE。三项各为 seed2026092701，均完成累计 E50/11100 步（本批每项新增 30 轮/6660 步）。独立核算全部 19980 条新训练记录、三个实际最终 checkpoint、单 SIG_TIME 分支及原 6×160 默认头、E20→自身 E30→自身 E50 来源、完整 H/L/U 计数和有限值；没有恢复器权重或恢复 IQ。三项冻结后共生成 324000 个源域类别决策，180 个逐 TX/RX/day/overall 分层计数全部重算一致，另保存 12 个 macro-F1/混淆矩阵。此处独立核算没有新增 F 前向、IQ 或 U 真值/target 读取。

下面均为源 V，raw 分母 27000、三个 H 场景合计分母 81000。差值单位为百分点，参照为发布前固定的已有 source 最大值 raw26574、H360889；PL-X3 参照含三 SIG_TIME，不能冒充严格单视图目标对照。

| 方法 | raw | H3 均值 | raw 差值 | H3 差值 | 固定门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| REPL_H2_U | 98.329630% | 76.858025% | -0.092593 | +1.686420 | 通过 |
| REPL_H2_U_Pair | 98.322222% | 76.788889% | -0.100000 | +1.617284 | 通过 |
| REPL_H2_LU_Pair | 98.348148% | 76.777778% | -0.074074 | +1.606173 | 通过 |

门槛仍为 raw≥26547 且 H3≥61132，排序仍为 raw 优先、H3 其次、既定方法顺序破同分。固定源规则选中 `REPL_H2_LU_Pair`。这只是后续同一方法三 F 种子确认的条件成立；当前没有新增目标测试成绩，不能宣布优于普通 PL。下一步复用 seed01 成品，其他两个 seed 各从自身同契约普通 PL 的单视图 E20 分叉，仍按原 E50 总预算，不重训对照。

同一模型 H 与 raw 的错变对/对变错计数只描述信道损伤，不能解释成相对于普通 PL 的方法净收益。已完成单视图监督/普通 PL 三种子对照保留并复用；目标仍是同一新方法三 F 种子在原 168000 个主测试物理样本上的 raw clean≥85.98174603174603%、raw 六景>72.0961970899471%，另报 cross_date。源 V 已接近准确率上限，不替代跨接收机/跨日目标泛化证据。

证据：[固定规则与正确数](source_decisions.csv)、[逐层计数](source_strata.csv)、[独立检查及混淆矩阵](source_complete_evidence.json)。
