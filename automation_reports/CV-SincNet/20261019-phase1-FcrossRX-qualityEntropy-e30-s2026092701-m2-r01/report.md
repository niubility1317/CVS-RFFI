# P77：物理新增噪声与 As 目标熵的匹配 E30

状态：RUNNING，两行已启动；尚无源域晋级或正式性能结论。起点 9507a915615683c960ed23ea7ac657dbce86da69。实际代码身份由发布前 Git commit 与 release_commit.txt 固定。

唯一假设：在固定 As 预算内，按原引擎实际新增噪声占比放松目标分布，比同一组平滑系数随机分配更有利于 low/mid urban 并保持 clean。不是宣称已过度自信或梯度反向；P40 默认头硬类间隔梯度并未反向，现有源错误多数不确定。固定 AsLS=.05 已失败，不能直接扩展同一全局参数。本轮变化是物理对应关系，待真实训练检验。

ε_i=.01+.19/(1+10^(s_i/10))，s_i 为 native signal_to_added_noise_db；ε 范围 [.01,.20]。该比值是新增噪声比，不是录音总 SNR，不使用白输入 lock 代理替代实际波形比。Q按对应样本使用，R用独立命名随机流在同批 As64 内置换完全相同 ε 多重集；两 Dropout pass 复用同 ε。Q−R检验物理条件关联；相对原C00的共同效应仍可能来自平均平滑，实际梯度幅度并不强行匹配。NaN或缺字段视为技术失败；原引擎合法None边界按实测preAGC新增噪声功率解释：pn>0为零信号，取ε=.20；pn=0取ε=.01，全部样本仍参与CE，不删除困难包。原始None及边界标签保留，不向JSON写Infinity。

新增 R/Q 两条同 SAT18 AS_CE20 student/ownEMA 父状态尾训，各追加 E10，总 E30/6660更新；重置原 AdamW lr.0002/wd.0001/cos10/CENTER_UNITS。复用 P76 C00 的同父、同group-scene独立H/原As ε=.01、As KL关闭/无As对比控制。原P43 NoAs30/As30只作同seed/总预算整配方参照，不冒称同As父NoAs尾段。五逻辑行中只有R/Q新训练，参考不重推理/重评分。

固定 F+CROSS_RX/SIG_TIME_X3/统一Sinc/joint160/defaultcos30，原骨干和头联合更新。完整 L6300/U56700隐藏TX/V27000、源RX1/3/4/6/8、day1/2/3、六TX、split392005；无训练RX留出。IQ2×256/fs25MHz，RMS center=false，原normalize_input后完整residual/post_sync/EQoff=>G1/ref30/reflect；quality adaptation/phase tracking/IQ compensation/AGC原样。实际pinned模块是r02的experiments/adv3b02_xuc/code/leo_practical；新runner固定P76代码32355ff0→P75ecdefc9→P68/SAT18原运行链，打印实际__file__。训练fc2.462GHz；正式固定IQ、fc2.45GHz与全部六practical保持。

每步128L，其中As64原生process一次、第二As64直接复制；6L768、As两pass各CE.125，总.25，L CE总系数1.25，全部有标签As CE保留。ordinary/raw CE/KL、ORD128 SupCon.1和原U batch_neighbor/ownEMA规则保持。只改变As CE内部平滑，不增加分支、头、诊断分类器、教师或NN，不另加一致性损失。ε和置换seed不按TX/目标数据定义，U真值不读取。场景按既有RX/day/eq group抽样但H独立；个人noise/phase/sync/tracking/EQ、session硬件与原接收状态/AGC依赖保留。

源判据先固定：唯一候选Q，全V clean相对匹配NoAs下降≤0.1绝对百分点；2draw×mid/low均值相对max(R,C00,原As)至少+0.5个百分点。失败仍完成两行各七环境预测和独立truth-last评分；固定方案关闭，不扫ε/置换seed或以正式分数救回R。源PASS才可另行执行已授权总E50且需匹配控制；本次不含E50或新种子。

预算：2×E10=20新model-epochs/4440更新；每行nativeNN3162962/source12600/formal1176002，共同smoke1540，总新8704668样本前向，本地NN0。共同父E20/4440成本复用一次；三参考的既有成本另列。每行process142080、As槽位284160；每step还构造64个人与每group一个未process环境，其成本独立记录。同NN不等于同墙钟，预算结束不等于收敛。

命令：在 PROJECT/releases/cvs_FcrossRX_qualityEntropy_E30_20261019_r01 运行 CVS-RFFI/python -u experiments/cvs_quality_entropy77/runner.py。环境N607普通账号现有Python3.10/Torch2.1/NumPy2.2，fullFP32/CPU2；唯一owner codex/root/quality-entropy77。不可覆盖输出 PROJECT/runs/20261019-phase1-FcrossRX-qualityEntropy-e30-s2026092701-m2-r01，逐rowsource/prediction/scoring独占。一次preflight、归档SHA比较、remotecompile和首步真实CPsmoke1540，PASS立即两行训练，无额外许可。每GPU最多2个全owner进程，保留其他健康任务。

停止：仅本run角色/parent/head/预算/非有限/输出碰撞/ε多重集或物理元数据错配、不可产生合法预测等技术失败；保留全部产物，不自动重试、不停止无关或健康peer。性能低不触发停机。全部预测固定后独立exec评分，源决定先保存并由owner观察再读新formal。

预期产物：每rowactual_modules/resolved_config/native_args/strictparent+ownEMA初始化、training_sequence/全部epoch与step日志、prop_contrast_steps与physics中的原/assignedε、真实AsCE旧新值和总L还原、final/frozen、固定源V1800×2draw分层、source_matrix_decision、all7predictions、独立scores、analysis_r01完整150行历史曲线/纠错/F1/worstRX/margin30/固定NoAs正确集合保持率/ε覆盖/实际成本与四组PNG/PDF。目标包无state/lock逐包标签时不得作正式物理因果归因。共同E20父正确集合如有既有合法缓存可复用，不新增训练或头。

去重时worker误检索到历史AsLS正式分层文本已披露并停止读取；本映射、上界与晋级只依据代码区别和源证据预先固定，没有据该正式数值选取。另一对话splitAdam的失败行属于技术失败，不接管、重试或重名启动。

补充：已定位精确SAT18 AS_CE20的合法源预测缓存（20261018-diagnostic-FcrossRX-ASCE20cohort-sourceV-s2026092701-m1-r01），固定父clean正确1767/1800，可在同ID源预测上0NN复算真实父集合保持率；不改变晋级，不新增网络读出。原NoAs正确集合仍分别保留。

实际代码 cf49ff220dc926bac704f15b7f34a79d1ef60c27，远端OID独立一致；139095byte单归档比较和25文件remote compile VERIFIED。真实父/ownEMA＋两图smoke1540NN PASS：128 native process、sourceIQ0/targetqueryfalse/optimizer0，epsilon同多重集且新CE实际生效，state不变/defaulthead与Sinc梯度有限非零。两worker的PID/CWD/argv/CUDA/实际GPU0/1已核实，来源EXACT_MATCH/scratchfalse/targetcontactfalse。运行期原始证据在local_artifacts，不发布私有运行细节。
