# 固定 As 权重配方：02／03 的匹配 E30＋E50 复现

状态：VERIFIED_RUNNING。四条新增 weighted30 阶段已完成训练和七视图评分，均已严格加载自己的新 FINAL30，继续固定 weighted50；E50 和完整 24 条件配对重算尚未完成。run_id `20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01`。当前seed01源结果已通过固定资格：weightedAs50−ownweightedAs30 fullVclean＋.162963pp（要求≥0），两draw mid／low均值＋2.361111pp（要求≥＋.5）。root已在读取新E50正式指标前保存[source决定](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s2026092701-m2-r01/source_decision.json>)并固定02／03。analysis只给科学资格、不自动启动；本登记由root依既有表现好配方总E50授权执行，不新增种子筛选。

唯一假设与seed01相同：保持原完整practical及.375selected监督份额，延长至总E50能否在固定种子重复星地收益并减小clean代价。四轨迹B00/B01×2026092702/03，各复用自己的原FINAL20 student与EMA，再weighted10＋weighted20，总50轮／11100更新，轨迹20＋10＋20，非连续cos50。新增共120model-epochs／26640updates，不能把旧父20或中间30重复计费；seed01和所有旧30／50训练与预测不重跑。

固定矩阵：3modelseed×original/weighted×NoAs/As×E30/E50＝24逻辑条件。新增仅4轨迹的8冻结阶段；其余原12条件与seed01加权4条件复用。四行都在`{B00,B01}-s202609270{2,3}`，新阶段存row/weighted30与row/weighted50，各自含source／source_qualification／prediction／scoring。逻辑NOAS50等不是物理目录名。原02／03父仅`PROJECT/runs/20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02/{row}/stage20/final_ssdg.pth`；成功release `cvs_FcrossRX_As_replication_E30_s0203_20261018_r02`／实际commit63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47，P43基础19aedc3604fed9c193237dd92366164178af0983。failed r01和原FINAL30／50不作加权初始化。

旧四parent源审计实际本地20轮／4440更新、scratch／无来源继承／ownEMA／source roles EXACT_MATCH／无targetcontact；发布时一次最小preflight检查当前句柄和角色，首次实际CP烟测严格加载。原L6300／U56700隐藏TX／V27000、sourceRX[1,3,4,6,8]、day[1,2,3]、split392005／IQ256／eq1不变，不留训练域、不分Ldev。F＋CROSS_RX／三路SIG_TIME／统一Sinc／joint160／默认cos头继续与骨干联合训练，保留NoExplicitCFO与内部补偿，不新增头／部署分支。实际Ffactory、native和modelseed必须同行02／03，使用原P49工厂绑定；legacyseed01只代表原架构配方。

算法和IQ不变：.5primaryordinary128＋.125mixedordinary64＋.375selected64＋.25raw128，CE和对称KL用相同组均值，总L1.25，每step6L，ORD128CROSS_RX.1；NoAsselected实际raw.clone，Asselected为As(raw64)。后者每step64完整practical，uniform三场low_suburban／midurban／lowurban各1/3，无有标签置信过滤。所有旧ordinary／mask／RNG／U及ownEMA保持；新10／20段各resetAdamW lr2e−4 WD1e−4 cos10／cos20 min1e−6。ordinary固定自身原20参数epoch13；SupCon＋20／＋30，channel e21–30/step4441–6660与e31–50/6661–11100。train／val／receiver channel seeds20261016421／422／423保持。

每条weighted30源与全部七视图完成后，同PID只消费完成状态，无条件执行weighted50；中间源或正式数值不能决定续训／重排／挑行／重跑。只可按登记的系统技术失败规则停止，源科学负结果仍完成固定全部阶段。每新stage固定FINAL30／50后自动source、clean＋六practical固定预测，再独立exec truth-last评分。正式RX[0,2,5,7,9,10,11]、day[0,1,2,3]、168000/view、receivedIQ／SNR／通道／全部困难包保持。

全部24×7冻结pred与独立score齐备后才连接truth重算，报告每seed与配对均值±样本SD(ddof1)、全seed和均值两种clean阶段判定、全部七场accuracy／macroF1／最差RX／urban三场均值、RX/day/TX、修复／新增错误、suburban与自身clean差、weight×As×budget二／三因子交互。clean阶段为weightedAs50−weightedNoAs50≥−.1个绝对百分点，另报原As50／NoAs50和自身As30。不得把NoAs退化或不同实验收益相加当作As新增收益；n=1、微小<.1pp上涨、两source draw均不证明重复双提升。源fullV来自native，未保存全V逐包不伪造其F1／修复；V1800两draw fc2.45GHz与train/formal2.462GHz既有差异保留。

检查已完成：172聚焦seed／parent／stage／工厂／资格负例、8实际native参数绑定、8stage预测路径、19py和两remoteAST PASS；唯一新适配P0/P1审查0／0，不重审旧不变数学helper。分析9项synthetic checks及真实producer界面核对PASS，NN／IQ／CP／更新均0。launcher首步4old20真实checkpoint8NN＋4真实seed各6L3072NN共3080NN、0更新／无query，PASS即训练；不将old20合成探针冒称ownnew30的真实stage50烟测。50实际严格加载各自新30。

发布：普通N607既有CVS-RFFI/PyTorch FP32／CPU2，无admin／安装。CWD`PROJECT/releases/cvs_FcrossRX_AsWeight_E50_s0203_20261018_r01`，命令`CVS-RFFI/python -u experiments/cvs_practical_as_weight_replicate50/runner.py`。唯一owner`codex/root/practical-Asweight-replicate50`，不可变输出`PROJECT/runs/20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01`；唯一physicalrow日志`run/logs/{row}.log`，同PID跨阶段。GPU共享队列分配，每GPU全部owner总进程≤3、free≥12000MB；健康任务不动。一次preflight／一个release归档本远端SHA比较／remotecompile与启动后PID/CWD/argv/GPU/log增长；实际code/config身份由release_commit.txt和submit记录，发布后回填，不增设hash／receipt链。

停止仅限本run数据／父／头／数值／成本／不可变输出或不可启动系统技术错误，保留输出、不自动重试，不改变科学配方／预算／seed救跑。各stage全部真实step26640、L/U曝光、As／NoAs数、原始组损失与物理配置／工厂seed记录，parent/new/student/teacher/train/eval/source/formal/smoke/wall分别计费。source／prediction／score／零NN分析／报告登记与Git全部闭合。历史基准已暴露，结果不宣称新独立盲测；尚无本扩种结果，不保证60%或最终双提升。

完整矩阵与路径：[experiment.json](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-Asweight-e50-s0203-m4-r01/experiment.json>)。

实际执行代码`094a5e9e063b52c99b646c33c2b357a388ece5b8`，远端分支OID独立一致。唯一发布preflight核实普通账号、当前源资格及四original20父元信息/角色；singlearchive传输比较／19py编译VERIFIED，首次真实3080NN烟测PASS后立即训练。dispatcher1303723；B00-s2026092702 worker1303753／GPU0、B01-s2026092702 1303830／GPU1、B00-s2026092703 1304318／GPU2、B01-s2026092703 1304399／GPU3。actualPID／CWD／argv／CUDA／nvidia／日志增长VERIFIED，全GPU compute进程各1，符合总cap3。四行source_roles EXACT_MATCH、target_contact=false、ownEMA严格恢复，实际Ffactoryseed分别02／03与native同一；实际physics同各自原row，ordinary仍原20参数epoch13。每行首记录6L／As0或64／group.125/.375，总L误差<1e−7；未额外拟合头。0NN分析1305922已核实WAIT_SOURCE，truth未开；单archive比较／compile＋9checksPASS。新中间30与最终50尚未完成，当前快照不代表性能结论。


## E30 三种子部分读出与实际 E50 转段

四条新增中间段各完成 10 轮／2220 更新，累计 E30／6660，全部 clean＋六 practical 预测与独立 truth-last 评分完成。同一四个 PID 无条件进入各自 weighted50：真实 own30 student／EMA、factory seed02／03、reset AdamW 空状态／cos20、6L、As0或64、.125/.375 均已从实际元信息核实。该观察不加载 IQ／checkpoint、不产生模型前向或更新，不是额外发布门槛。

现有独立 scorer 的 12 个 E30 条件已覆盖固定三 seed。weightedAs−matchedweightedNoAs 的 clean 逐 seed 为 −.326786／−.505357／−1.227381 pp，均值 −.686508±.476843 pp，逐 seed 和均值均未达到 clean 代价不超过 .1 个绝对百分点。weightedAs−originalAs 的 clean −.388095±.089660 pp，mid／low urban ＋.531151±.121889／＋.703770±.105766 pp，urban 均值 ＋.490675±.130200 pp；这确认 E30 的取舍，不能冒称双提升。

这是已有独立 scorer 汇总，不替代完整 24 条件的独立 truth 重计数；逐包修复／新增错误及完整训练成本仍待最终流程。全部 50 阶段已在上述正式读出前固定并运行，本读出不反馈候选、种子、权重或续训安排。唯一待检验假设仍是 E50 能否保留 urban 收益并恢复 clean；若完整三 seed 的匹配 clean 仍不能达到 −.1 pp 阶段范围，就关闭本固定加权配方作为双提升候选，不扫权重／种子救跑。

[三种子 E30 部分报告](partial_E30.md)；[部分汇总 JSON](partial_E30.json)。E50 未结束，当前不报告 E50 扩种性能或重复双提升。
