# 完整 practical 的 As 专属标签平滑 E30 预登记

TECHNICAL_RECOVERY：第一次发布在远端依赖路径检查失败；独立核验runroot未创建、无本run进程、0真实CP/NN/updates。保留r01发布，定点修复后用r02发布同一run及配方。没有本候选性能结果。

唯一假设：保留完整practical及旧ordinary训练，只将合法有标签As监督CE的epsilon由已有.01提高到固定.05，能否减小clean代价并提高low/mid urban。改变目标分布，不改IQ、softmax温度、分类头或CE总体权重。

## 配方与最小矩阵

仅一新B01-s2026092701，逻辑SATLS_AS30；原NOAS30/AS30复用冻结读出。NoAs没有As目标算子，变换恒等，无需无意义重训NoAs。新行严格从原P43 B01 ownFINAL20学生及ownEMA追加10轮/2220更新，总30/6660；父run20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01，冻结releasecommit19aedc3604fed9c193237dd92366164178af0983，stage20从零20/4440、固定FINAL20/source-only。不用LRaw/NoU/温度/权重变体或目标选出的父模型。

当前真实基线label_smoothing=.01，不能称hard/onehot训练。本次仅selected64 As两pass q=(1−epsilon)onehot+epsilon/6，new真类.9583333333/其他各.0083333333；old真类.9916666667/其他各.0016666667。所有LAs样本及六类同一公式，无质量、预测、置信、margin或类别筛选。

既有mixed128的两logits复用：按128位置均值，ordinary64 CE仍.01、As64 CE为.05；两pass CE再均值，并乘原mixed外.5。追加等价delta=.25*(sum_selected(CE05a−CE01a)+sum_selected(CE05b−CE01b))/128。As有效CE份额仍.25；common普通对.5、mixed对.5、raw对.25、总L尺度1.25保持。mixed128原对称KL、raw/common CE、SupCon和全局nativeLS=.01不改，禁止全局放宽oldterms限制。

保留6L/768LNN每step、SIG_TIME三路统一Sinc/joint160、默认cos6×160 scale30和正常联合训练。RDrop每pair两次使用同一已生成IQ；不会重抽第二个完整As。CROSS_RX sameTX/differentRX contrast仅原ordinary128浅/中层maps；不把sat/raw强制对齐，不新增网络、头或部署路径。原Uweak/strong/教师/EMA/.999/旧门控及lambda规则保持，native phase0/10，ownEMA严格恢复。

原ordinary实际配置和输入路径完全保留，parent20args/frozen13；原As已经完整practical(raw64)，未重复损伤ordinary。原3sceneIID、传播/噪声/质量/lock/tracking/AGC/内部补偿/noEQ/refSNR30关系保持，NoExplicitCFO不变。无新PA/DAC/剪裁/量化/CFO变换或额外RMS/crop/shift，不改固定receivedIQ。

训练沿原resetAdamW lr2e−4 WD1e−4 cos10/min1e−6，channel epoch+20/step+4440，SupCon累计+20；不是连续从零cos30。

## 依据与去重

通用依据：[When does label smoothing help? (NeurIPS 2019)](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html)。该研究提供限制过度置信的通用动机，也报告蒸馏可能受影响；不证明本任务有效，不把校准或CE下降当作身份识别收益。本次保持教师算法，由真实clean和practical准确率检验整体影响。

实际配置去重：satelliteCE-temperature的CE_T2/CE_HALF已完成全5RX/6L ownAs20+10匹配E30，固定sourceFAIL后关闭，二者仍LS=.01；本次改q而非温度/标量权重，p−q梯度并不等价。旧AsPairCE在heldRX1/8L E20移除AsKL已失败，本次不重开删KL。sceneBalance/weight.375/stepALT/MIXcontrast/NoU/LRaw等均保留历史不重开。另一对话当前检验同NN双独立As观测，本次不进入该路线。

## 数据与判定

源RX[1,3,4,6,8]/days[1,2,3]，完整L6300/U56700 TX真值隐藏/V27000，split392005、eq1、IQ256、Fs25MHz保持。无RX训练域留出或L5580/Ldev720。现有VALIDATED_ONCE数据及role_ids复用，仅核对最小句柄和父来源，不重复数据重验。V不训练/更新EMA或其他状态。

固定FINAL30，不挑epoch。唯一source晋级规则为：newAs fullV27000 clean−原NoAs30≥−.1个绝对百分点；固定V1800两draw mid/lowurban等权均值newAs−原As30≥＋.5个百分点。二者合取。任一失败固定.05关闭，不扫描epsilon/温度/权重/parent/seed，不加E50；全部all7正式仍自动完成。PASS仅获得后续总E50研究资格，不自动追加。源健康仅标签/角色/数值/明显执行损伤，无特征代理、置信或校准门槛。

每row冻结后自动source读出、clean＋六practical预测；该row所有7prediction固定后独立exec truth-side scorer才连接truth。主线程先保存source决定再读取本次正式分数。正式结果不回流选模、参数、seed、候选晋级或选择性重跑。已有暴露benchmark背景保留，不声明新的独立盲测。

正式每view168000；targetRX[0,2,5,7,9,10,11]/days[0,1,2,3]；clean、practical_high、practical_mid、practical_low_suburban、practical_high_urban、practical_mid_urban、practical_low_urban。固定IQ/SNR/信道/困难包全部保留，不删除或修改。

## 指标与实际成本

报告n1逐seed的all7 accuracy/macroF1/最差RX、low/midurban和urban3均值、RX/day/TX、修复/新增/净错误、suburban绝对准确率及自身clean差。newAs−oldNoAs为0.1pp阶段判定，newAs−oldAs为机制独立增量；<.1pp微增只作观察，n1不称可重复双提升。不要求额外Ag独立收益。三逻辑共3×7，0NN独立重计预期378CM/63partition；不虚构四条新训练。

新增10modelEpoch/2220updates。预期native3162962NN=学生train2271960＋eval324002＋教师eval567000，source12600/formal1176002，训练与读出4351564NN。首步真实单父noquerysmoke固定770NN=2realparent＋6L768，不增加第二原hook模型探针；新增总计4352334NN。纯synthetic CE/autograd检查不冒充NN。额外4×selected64 CE/step，共8880ops/568320logit包；原mixed widths[128,128]，实际[128,128,64,64,64,64]。新增CE标量及日志计算有真实CPU时间成本，不冒充CUDAevent或免费；旧原20/30及读出成本单列复用。

## 发布与停止

代码/config身份为一次Git commit，冻结后回填；一个不可变run/outputroot/owner codex/root/practical-AsLS。N607普通账号、现有CVS-RFFI/PyTorch fullFP32、CPU2；无安装或管理员操作。CWD=PROJECT/releases/cvs_FcrossRX_AsLS_E30_20261018_r02；输出=PROJECT/runs/20261018-phase1-FcrossRX-AsLS-e30-s2026092701-m1-r01；公开报告automation_reports/CV-SincNet/20261018-phase1-FcrossRX-AsLS-e30-s2026092701-m1-r01。

发布命令：verified native Python -m experiments.cvs_satellite_ls30.publish；worker：CVS-RFFI/python -u experiments/cvs_satellite_ls30/runner.py --mode worker --row B01-s2026092701。private RT沿原P43 reference_release/reference_commit，真实私有路径不入Git。

每GPU全部owner compute进程≤3、free≥12000MB；不动健康旧任务。一次聚焦协议负例/CE数学与梯度检查、一次独立合并P0/P1、一次realCP launcher首步smoke PASS直接训练；一次N607preflight/单releasearchive本地远端SHA比较/compile，以及一次PID/CWD/argv/GPU/loggrowth核验。只对明确owned系统性协议/角色/父/head/数值/输出/无法合法预测失败停止并保留产物，不自动换root重试；科学负结果不停止正式收尾。

预期产物：resolved_config/actual_native_args/initialization/EMA/optimizer/source_contract/source_frozen/final_ssdg/completion，reference/unit/shape/satls全2220step日志，source_qualification、7prediction/scoring/complete；0NNanalysis source_summary/training_logs/formal_summary/costs/complete、同row最终报告与登记/Git。source/final按5/11汇总文件单archive取回。不得把实现、smoke、sourcePASS或训练完成写成正式成功。
