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
