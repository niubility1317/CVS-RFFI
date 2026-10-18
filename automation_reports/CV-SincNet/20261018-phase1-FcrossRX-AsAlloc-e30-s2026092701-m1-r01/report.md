# 完整 practical：As64 的 TX×RX 分层分配 E30

ANALYZED（单种子）：固定TX×RX As64分配未通过源域两条件，关闭且不扩E50。全部clean＋六practical固定预测、独立truth-last评分和完整日志分析完成，无训练/评分技术缺项。

唯一假设：只在既有 L128 内均衡哪些样本进入 As64，能否在主要 clean 监督和每个物理位置权重稳定的条件下，提高完整 practical 识别。原位置交替选择不保证 TX×RX 配额。本项不增加信道损伤、视图、损失、门控、分类头或部署分支。

实际结论：newAs clean85.281548%，相对原NoAs30＋.033929个百分点，本seed满足clean下降≤.1pp阶段条件；相对原As30−.059524pp。mid/lowurban62.822024%/51.322619%，相对原As仅＋.015476/＋.078571pp，urban均值＋.031349pp，全部是n1微小变化。没有可保留的新星地收益或可重复双提升证据。

源决定先于正式读取保存：fullV clean−原NoAs−.114815pp、midlow−原As−.013889pp，两个固定条件均FAIL，因此不扩E50/不扫分组、父或seed。全部6300个L均As曝光，最少11次、最多37次；覆盖完整不等于机制成功。新增10modelE/2220updates、4352334NN含770smoke，native1057.828秒；selector1.662100秒包含于hook2.582119秒，不相加。

完整all7 accuracy/macroF1/最差RX、配对错误、suburban差距、实际曝光和成本见[最终分析](final_analysis.md)、[公开数值](results_public.json)、[最终验证](final_verification.json)。

## 唯一变化与归因

保持原随机 L128、位置与全部样本。根据合法 L 真标签及已知 RX 格，用私有 RNG、受当前可用数量约束的 waterfill 分配选满64，不放回、不补齐缺失格、不补样、换域或换 loader；所有 TX/RX 同一算法，没有具体类别专属参数。缺格或数量不足时，不承诺30格严格等额。selector 对 class-ID 只用作分组等价，不按具体类设置规则。

原 primary ordinary128、两pass IQ、ORD128 SupCon全部原anchors不变。mixed128的As64位置和ordinary64补集一起改变，每个原物理位置仍恰好一个mixed视图；没有将任意As重排到固定mask而重复物理权重。原6L/768LNN每step、L总尺度1.25/AsCE总体.25、raw控制、原symmetricKL及globalLS=.01保持。RDrop每pair复用同一个生成IQ。private selector 不消费旧全局NumPy/Python/Torch/CUDA RNG，但不据此宣称整个训练/U接受轨迹相同。实际有限样本As曝光改变，不能称只改变梯度方差。

原ordinary增强完整保留，As原本就对raw64作完整practical，不新增RMS/crop/shift/PA/DAC/CFO/量化/剪裁。scene仍逐PID IID，传播→N0/噪声→quality/lock/tracking→接收补偿/AGC依赖不变；NoExplicitCFO与必要内部补偿保持。U不使用本selector，TX真值仍隐藏，原weak/strong/teacher/EMA及lambda保持。

## 父模型、数据与匹配对照

仅新B01-s2026092701，逻辑ALLOC_AS30。原NOAS30、AS30冻结读出复用，共3logical×7views/3pairs；NoAs没有satellite分配算子，恒等控制无需重训。严格原P43 ownAs20 FINAL学生与ownEMA追加10/2220，总E30/6660；父run20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01，实际releasecommit19aedc3604fed9c193237dd92366164178af0983，own20从零4440/source固定轮次。不用AsLS/LRaw/NoU/AS_CE20等父模型。

源RX[1,3,4,6,8]/days[1,2,3]，完整L6300/U56700隐藏TX/V27000，split392005、eq1、256点/Fs25MHz；无RX留出或L_train/L_dev。沿用已核实相同角色及源缓存/VALIDATED_ONCE，只核对最小句柄和已验证父来源，不重新构建/重验数据。V不训练、不更新EMA或其他状态。

训练为原own20args/frozen13 ordinary调度、channel累计+20/4440与SupCon+20，原resetAdamW lr2e-4/WD1e-4/cos10/min1e-6，nativephase0/10、原U/EMA/.999保持；并非从零连续cos30。网络固定F三路SIG_TIME/统一Sinc/joint160与默认cos6×160 scale30联合训练。

## 历史去重与本次关键区别

整批batchStratify已有full5RX/4L训练，改变整个L批与主要clean/SupCon输入；本项只改同一原L128中As视图的分配，不更换loader，物理总mixed权重保持。sceneBalance只改变已选64的场景分配，本项不改场景概率。Asweight/AsLS只改监督权重/目标；均不更换As64。限定registry/源码检索未发现同项真实训练，不宣称全库穷尽。

共享传播propgroup20已实际heldRX6/8L E20源失败，不换fullsource接口重开。另一对话同NN双独立As已完成且源增量不足关闭，本侧不重复；不用其AS_CE20父。本项不依据正式结果选择候选、采样强度或种子。

## 冻结规则、评分与放弃条件

固定FINAL30，无中途选epoch。唯一source晋级条件：ALLOC_AS30 fullV27000 clean−原NOAS30≥−.1个绝对百分点；固定V1800两draw mid/low等权均值ALLOC_AS30−原AS30≥+.5个百分点。合取；任一失败关闭固定TX×RX分配，不扫分组粒度/采样强度/父/seed、不扩E50，全部正式仍完成。源通过只给后续总E50资格，本launch不自动追加。源健康仅真实标签/数值/明显执行损伤，不用特征、置信或代理门槛。

先保存源决定，再读取本轮正式分数。冻结后自动clean＋六practical，每view168000；targetRX[0,2,5,7,9,10,11]/days[0,1,2,3]。固定receivedIQ/SNR/信道/困难包全保留；每row所有7prediction固定后独立exec truth-last评分。正式不反馈选模/参数/seed/候选晋级或选择性重跑。已有暴露benchmark背景保留，不宣称新盲测。

## 指标与实际成本

报告all7 accuracy/macroF1/最差RX、mid/low与urban3均值、RX/day/TX、修复/新增/净错误、各suburban绝对准确率与自身clean差。newAs−NoAs是阶段clean/完整As效应，newAs−原As才是新增分配机制增量；n1或不足.1pp微增不称可重复双提升。不构造假四因子/交互。0NN独立重计预计378CM/63partition。

新增10modelEpoch/2220updates；预期native3162962NN=学生train2271960+eval324002+教师eval567000，source12600/formal1176002，合4351564；首步770NN=真实父2+actual6L768，总4352334NN，以实际计数为准。原20/30及旧读出单列复用，不重收费。selectorCPU耗时单列，不能由零额外NN推断零计算/提速，不把分析耗时当source预测wall。完整2220步allocation日志记录可用格、配额、mask/补集、旧mask计数及实际曝光，说明不均衡来源。old parity曝光只是同一新轨迹当前L批内的反事实位置掩码，不冒称旧As30真实曝光或逐物理信道实现已配平；不把配额统计改善当身份识别成功。

## 发布和技术停止

实际代码/config commit=93693c8e4250074777a1d638f1f78fa20e1082e4。一个不可变run/outputroot/launchowner codex/root/practical-AsAlloc；N607普通账号、现有CVS-RFFI/PyTorch fullFP32、CPU2，无安装/管理员操作。CWD=PROJECT/releases/cvs_FcrossRX_AsAlloc_E30_20261018_r01；output=PROJECT/runs/20261018-phase1-FcrossRX-AsAlloc-e30-s2026092701-m1-r01；发布命令 verified native Python -m experiments.cvs_satellite_allocation30.publish；worker CVS-RFFI/python -u experiments/cvs_satellite_allocation30/runner.py --mode worker --row B01-s2026092701。private runtime不入Git。

每GPU全部owner compute≤3/free≥12000MB，动态容量队列，不动健康旧任务。一次聚焦protocol/selector/RNG/实际调用检查、一次独立合并P0/P1；launcher首步真实CP no-query smoke PASS即训练；一次preflight、单archive本地远端SHA比较/compile、一次postPID/CWD/argv/GPU/loggrowth核验。仅owned系统性技术/角色/父/head/数值/输出/无法合法预测失败停止保留产物，科学负结果不阻止all7。无额外审批、签名或逐文件hash链。

预期source final_ssdg/completion/source_frozen/resolved_config/actual_native_args/initialization/EMA/optimizer/source_contract，完整reference/unit/shape/allocation日志，source_qualification与7prediction/scoring/complete；0NN analysis source_summary/training_logs/formal_summary/costs/complete。同row报告/登记/Git收尾，source5/final11汇总单archive取回。
