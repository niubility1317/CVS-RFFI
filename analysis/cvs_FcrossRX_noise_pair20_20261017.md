# F＋跨 RX：正负噪声配对 As 的匹配 E20 源域训练

当前状态：三行均已自然完成并独立分析，固定正负噪声配对源域资格 FAIL。ANTI 相对独立噪声控制的 urban 均值为 −0.132 pp、low urban 为 −0.397 pp，关闭该固定方案，不扩 seed 或事后晋级 INDEP。

相对同预算 RAW10，ANTI 的源域 clean／urban 均值／low urban 分别为 +6.111／+10.317／+6.032 pp；这些收益未证明来自正负相关性。仅有 held source RX1／model seed2026092701，未作正式目标测试或跨 seed 确认，未达到 low>60% 和 suburban 接近 clean 的联合目标。实际 60 model-E／13320 更新／20879876 NN，并行墙钟约63.05分钟；整体 goal ACTIVE。

后续原预登记与逐行部分记录保留为执行历史，以完整独立源域结果及closure为最终状态。


本轮唯一问题是：在相同 practical 边际分布、监督权重和前向预算下，正负新增噪声的相关性是否让训练后的默认分类头获得 urban 收益，同时保护 clean。上轮冻结eval CE诊断两seed方差下降29.946%／31.790%，只支持检验采样理由，不是训练或身份提升。固定相干IQ插值及旧fixed additive As的跨seed失败均保留，不改其系数、SNR、seed、RX或epoch救援。

固定底座为F＋跨RX同TX SupCon、SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160、默认cosine30；无新部署分支／分类头／诊断头。三行RAW10、INDEP、ANTI都用model/globalordinary seed2026092701独立从零E20／4440更新；原13轮L＋7轮旧门控半监督、ownEMA、AdamW和cosine保持，U真值隐藏。旧ordinary原顺序／PA／shift及实际原配置三组共有，As只作用raw，不与ordinary噪声／多径叠加。

共同原三对监督绝对权重不变：raw总.75、ordinary.25，原SupCon只用ordinary两次前向的浅中层特征。额外两份IQ各用原同IQ RDrop：R=两次default头CE均值（LS.01）＋同IQ对称KL（alpha1），再平均两份R并乘.25。不同噪声IQ之间没有KL／特征对齐。总pair权重1.25、五对共10次L前向／包，全部合法有标签As直接CE，不经置信度筛选。

|行|额外两份IQ|新训练预算|L前向包数|
|---|---|---:|---:|
|RAW10|raw、raw|E20／4440|5683200|
|INDEP|同物理信道的独立noise0／noise1|E20／4440|5683200|
|ANTI|同物理信道的noise0／−noise0|E20／4440|5683200|

INDEP与ANTI共用每包／epoch／step的noise0，传播、遮挡、几何、CFO、相位、RX硬件、补偿和接收状态相同，仅noise1相关性不同。每一份从原primitive noise入口经过原RX IQ／补偿／G，再独立计算原AGC，不平均IQ、不缩小噪声、不调整reference30。训练用六种practical nominal均匀先验，25MHz／256IQ／fc2462MHz／residual／post_sync／EQoff G=1／reflect，质量仍为原白输入功率代理；新增噪声SNR不冒称总SNR，数值延迟／反射边界保持。非噪声latent逐包检查相等，正负组在处理后检查噪声仍互为负值。硬件按RX/day/eq固定，session不由TX标签定义。所有场景和状态的先验支持保留、0拒绝或困难样本删除。

fitRX3/4/6/8的有效L5040／U45360hidden／V21600，held source RX1原L1260只在固定FINAL20读出。原物理split392005和角色handles／VALIDATED_ONCE复用；不混用5580／720内部折，也不扩大可见标签。native源数据先核对原carrier，再按已验证fold过滤，训练／教师／校准／持久特征cache全部排除heldRX1。三行所有L/U物理顺序记录并完整比较。

不继承旧学生、EMA、optimizer或教师状态。dispatcher首步只用已核实相同seed、相同有效LUV契约的既有scratchE20 D_raw8L权重做2包合成no-query结构兼容smoke；随后丢弃，三条新model都从零。该旧权重来源7f398321cd2ffd2c2d2cb11f52f67bfd8c44987b、无祖先／目标选模／另训头，已核实来源结论复用。本轮RAW10是新10L匹配控制，旧raw8L不作为性能参照。

每行训练完成后固定末轮学生checkpoint，自动在原heldsourceRX1全部1260包上读出clean＋六种原practical，保存默认头预测、margin/30、物理ID和完整physics，再由离线独立reader重算accuracy／macroF1。源评价随机流20261016103、IQ／场景／SNR不改；训练channel20261017501、noise20261017502、硬件20261016102，训练按epoch／step刷新，和源评价／正式测试分离。此为最小source-only证伪矩阵，未通过固定源规则不扩seed或进入target，不把源成绩写成正式测试。

固定科学条件：ANTI相对RAW10的clean准确率和宏F1不降、urban三场景均值≥+3pp、low urban≥+5pp；相对INDEP的urban均值和low各≥+1pp，clean和宏F1下降≤.5pp。相对两参照的每个practical、fitV均下降≤.5pp，且完整预算／覆盖／default头联合更新、最终共同base≤.5。按全部条件共同判断，不以局部状态或某个positive对照替代。失败关闭固定采样训练理由，不调系数、SNR、seed、RX、抽样数或追加epoch救援；通过仅允许独立第二modelseed源确认，不证明Ag、四格交互、正式target或low>60。

报告所有row的clean／六practical绝对值、urban均值／low／mid、宏F1、每个suburban距自身clean、固定RAW10 clean正确集合的sat保持率、各RX/TX/day/state/lock/quality／signal_to_added_noise／CFO／残余核／AGC／边界及实际覆盖。目标仍要求clean正收益、low>60%、每个suburban距clean≤2pp（探索定义），不靠降低clean缩小差距；本轮不预告达成。

新成本60model-E／13320更新；每行5683200 LNN＋原U／ownEMA／V／8820held，总预计20879874 NN＋dispatcher2合成CPsmoke，实际由计数器重算。两As行各568320物理anchor呈现、1136640噪声观测，后者不是新增独立物理样本；额外4NN每anchor。无新U/teacher视图。没有训练父权重，兼容smoke的既有E20训练成本另列。墙钟待真实完成记录，E20耗尽不等于收敛。

本地40个配置／权限负例，3858合成真实网络前向／3更新通过：三组206个state键初态相同，共同loss与全部参数梯度精确一致，CPU/Python随机流末态相同，10前向输入和每对同IQ验证，noise0位相同，原默认头正常联合更新、export160精确。0真实IQ／CP／target访问。一次独立P0/P1静态审查PASS，未新增重复审查或报告审批。

一个不可覆盖run／release和唯一owner codex/root/FcrossRX-noisepair20-20261017；共享all-owner≤3/GPU、空闲显存≥12000MiB，CPU2／loader0。技术停止仅本run数据／来源／非有限／错预算／输出碰撞／无法合法预测等系统错误，保留全部partial，不自动重试，不误停其他健康实验；负成绩按科学规则完整结束并报告。

实际代码／配置 commit：ee305facac7874bfc1340291ee5360a3665621d9，已推送并独立核对远端OID相同；发布归档只比较一次本远端SHA、一次remote compile，首个真实CP no-query smoke PASS立即继续。启动后独立检查PID/CWD/argv/GPU/log增长与真实actual_modules／args，最终独立自然终态和成本。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noisepair_sourceRX1_E20_20261017_r01/experiments/cvs_f_noise_pair20/runner.py`。环境/CWD：`ordinary N607 CVS-RFFI Python/Torch originalfullFP32`／`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_noisepair_sourceRX1_E20_20261017_r01`。远端root：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-noisepair-sourceRX1-e20-s2026092701-m3-r01`，本地root：`E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017`；逐row config、六类seed、日志、输出、GPU政策、停止规则和预期artifact均在experiment.json。

REJECTED_EXTRA_GATE：不增加design批准、配置／报告hash、成员hash、smoke凭证、receipt链或额外review。当前整体目标ACTIVE；尚无合格新增Ag、Ag/As四格与交互或正式目标联合证明。

逐行部分完成记录（源域，非正式目标测试）

RAW10 已自然完成 scratchE20／4440 更新，固定FINAL20默认头联合训练；原进程已退出。独立读取七份既有预测，共8820个决策（每视图为相同1260物理样本），逐类混淆矩阵重算accuracy／macroF1均与原输出一致。重算0IQ读取／0NN／0更新；不加载checkpoint或拟合分类头。

|RAW10源held RX1视图|准确率%|宏F1%|
|---|---:|---:|
|clean|79.683|79.159|
|practical_high|37.857|27.086|
|practical_mid|38.175|28.078|
|practical_low_suburban|34.762|26.387|
|practical_high_urban|38.333|28.456|
|practical_mid_urban|31.190|25.681|
|practical_low_urban|27.698|24.137|

RAW10 urban三场景均值为 32.407%；这是等10L前向控制行的源域结果，不是本轮增强收益。实际NN前向包数 6959958，其中L前向 5683200；native记录1563.797秒，整矩阵并行墙钟仍待全部完成。全部训练步／序列与分层机制将在三行齐全后由同一独立reader完整重算。

两条As仍为同一健康进程正常训练，整体状态保持RUNNING。当前不能比较增强净收益、clean代价、跨seed重复性或判断固定资格；不改变其他行的系数、SNR、场景、seed、预算或最终选模规则。

一次非阻断的实际日志回读覆盖两As前3完整epoch／666步／85248个anchor，共340992条噪声记录：noise0与非噪声latent逐条相同，ANTI加噪功率逐包相同，INDEP各份功率不同；ANTI有85247个anchor的两份AGC增益不同，符合原AGC逐视图重算，不能把处理后整IQ当作严格相反。六种原场景均已出现。此为实际采样关系核验，不是身份识别提升。

现有梯度日志补充（0新IQ／NN／更新）：RAW10全部4440步均有有限裁剪前、总梯度和骨干范数，原limit固定5；裁剪前均值0.930822、最大10.650027，31步超过limit（0.698198%），原记录的裁剪后均值0.919881。离线reader将同样逐步统计三组并增加第7张梯度健康图，保留浮点误差，不新增科学晋级条件。这些范数不是跨噪声梯度协方差，不能把其变化解释为身份收益。新增4个无效梯度记录负例后reader共24个负例PASS，分析仍0模型／数据前向，训练发布ee305facac7874bfc1340291ee5360a3665621d9不变。

完整独立源域结果（固定 FINAL20，非正式目标测试）

三行均从零完成 E20／4440 更新，固定 F＋跨 RX 同 TX SupCon、SIG_TIME_X3、统一 Sinc CENTER_UNITS、NoExplicitCFO、joint160 与默认 cosine30，骨干及原头联合训练。fit RX3/4/6/8 的 L5040／U45360隐藏／V21600；held source RX1 原 L1260只作末轮读出。本轮无训练父权重，无额外拟合头，无目标访问。模型/global ordinary seed只有2026092701；三条arm不等于三个模型seed。

|视图|RAW10准确率%|INDEP准确率%|ANTI准确率%|RAW10宏F1%|INDEP宏F1%|ANTI宏F1%|
|---|---:|---:|---:|---:|---:|---:|
|clean|79.683|85.635|85.794|79.159|85.394|85.583|
|practical_high|37.857|57.937|58.492|27.086|56.544|57.106|
|practical_mid|38.175|53.889|54.127|28.078|52.678|52.884|
|practical_low_suburban|34.762|51.667|52.540|26.387|50.504|51.403|
|practical_high_urban|38.333|53.571|53.413|28.456|52.348|52.053|
|practical_mid_urban|31.190|40.873|41.032|25.681|40.504|40.620|
|practical_low_urban|27.698|34.127|33.730|24.137|33.636|33.133|

|行|urban三场景均值%|clean至high suburban差距pp|clean至mid suburban差距pp|clean至low suburban差距pp|
|---|---:|---:|---:|---:|
|RAW10|32.407|41.825|41.508|44.921|
|INDEP|42.857|27.698|31.746|33.968|
|ANTI|42.725|27.302|31.667|33.254|

|参照|视图|ANTI准确率变化pp|宏F1变化pp|修复数|新增错误数|
|---|---|---:|---:|---:|---:|
|RAW10|clean|+6.111|+6.424|90|13|
|RAW10|practical_high|+20.635|+30.020|374|114|
|RAW10|practical_mid|+15.952|+24.806|351|150|
|RAW10|practical_low_suburban|+17.778|+25.016|348|124|
|RAW10|practical_high_urban|+15.079|+23.597|334|144|
|RAW10|practical_mid_urban|+9.841|+14.939|261|137|
|RAW10|practical_low_urban|+6.032|+8.996|202|126|
|INDEP|clean|+0.159|+0.189|5|3|
|INDEP|practical_high|+0.556|+0.562|24|17|
|INDEP|practical_mid|+0.238|+0.207|22|19|
|INDEP|practical_low_suburban|+0.873|+0.899|23|12|
|INDEP|practical_high_urban|-0.159|-0.294|17|19|
|INDEP|practical_mid_urban|+0.159|+0.116|21|19|
|INDEP|practical_low_urban|-0.397|-0.503|11|16|

固定科学判定：FAIL。关闭本次固定噪声配对训练理由；不通过系数、SNR、seed、RX、抽样数或追加epoch救援，也不将 INDEP 事后改为晋级主方法。

|固定条件|结果|
|---|---|
|RAW10_clean|PASS|
|RAW10_clean_macroF1|PASS|
|RAW10_urban|PASS|
|RAW10_low|PASS|
|RAW10_all_practical|PASS|
|RAW10_seenV|PASS|
|INDEP_clean|PASS|
|INDEP_clean_macroF1|PASS|
|INDEP_urban|FAIL|
|INDEP_low|FAIL|
|INDEP_all_practical|PASS|
|INDEP_seenV|PASS|
|healthy|PASS|

ANTI相对RAW10：clean +6.111 pp，urban均值 +10.317 pp，mid urban +9.841 pp，low urban +6.032 pp。相对INDEP的urban均值为 -0.132 pp，low urban为 -0.397 pp。以上为源域同包配对结果，不能写成正式目标提升。

固定 RAW10 clean 正确集合：n=1004，所有行在相同物理集合上读出，不重新定义候选自己的正确集合。分类间隔为（真类logit−最大其他类logit）／30，只用 checkpoint 内默认头。

|视图|RAW10保持率%|INDEP保持率%|ANTI保持率%|RAW10间隔均值|INDEP间隔均值|ANTI间隔均值|
|---|---:|---:|---:|---:|---:|---:|
|clean|100.00000|98.70518|98.70518|0.16203|0.16673|0.17352|
|practical_high|37.64940|69.92032|70.41833|-0.03915|0.08621|0.08928|
|practical_mid|36.65339|64.44223|64.44223|-0.03905|0.07348|0.07650|
|practical_low_suburban|34.66135|61.85259|62.64940|-0.03914|0.06273|0.06535|
|practical_high_urban|37.74900|63.94422|63.94422|-0.04003|0.07240|0.07517|
|practical_mid_urban|30.47809|47.60956|47.90837|-0.04720|0.03488|0.03656|
|practical_low_urban|28.08765|39.44223|39.44223|-0.04606|0.01245|0.01332|

urban 状态分层（同一原信道观测；只作条件表现，不作移除某个物理损伤的因果归因）：

|视图|分层|状态|n|RAW10准确率%|INDEP准确率%|ANTI准确率%|ANTI−RAW10 pp|ANTI−INDEP pp|
|---|---|---|---:|---:|---:|---:|---:|---:|
|practical_high_urban|state|BLOCKED|130|29.231|23.846|23.077|-6.154|-0.769|
|practical_high_urban|state|LOS|1094|39.762|57.861|57.952|+18.190|+0.091|
|practical_high_urban|state|PARTIAL|36|27.778|30.556|25.000|-2.778|-5.556|
|practical_high_urban|lock|degraded|108|29.630|23.148|22.222|-7.407|-0.926|
|practical_high_urban|lock|locked|1128|39.450|57.092|57.004|+17.553|-0.089|
|practical_high_urban|lock|unlocked|24|25.000|25.000|25.000|+0.000|+0.000|
|practical_mid_urban|state|BLOCKED|500|19.800|19.600|19.600|-0.200|+0.000|
|practical_mid_urban|state|LOS|637|39.874|56.986|56.986|+17.111|+0.000|
|practical_mid_urban|state|PARTIAL|123|32.520|43.902|45.528|+13.008|+1.626|
|practical_mid_urban|lock|degraded|301|19.934|21.595|21.595|+1.661|+0.000|
|practical_mid_urban|lock|locked|735|39.048|55.510|55.918|+16.871|+0.408|
|practical_mid_urban|lock|unlocked|224|20.536|18.750|18.304|-2.232|-0.446|
|practical_low_urban|state|BLOCKED|714|19.468|19.888|18.908|-0.560|-0.980|
|practical_low_urban|state|LOS|384|41.146|55.990|56.250|+15.104|+0.260|
|practical_low_urban|state|PARTIAL|162|32.099|45.062|45.679|+13.580|+0.617|
|practical_low_urban|lock|degraded|228|21.930|27.193|26.754|+4.825|-0.439|
|practical_low_urban|lock|locked|467|40.899|55.460|55.889|+14.989|+0.428|
|practical_low_urban|lock|unlocked|565|19.115|19.292|18.230|-0.885|-1.062|

完整质量代理／新增噪声SNR／残余CFO／残余核／AGC／边界分层保留在 mechanism_analysis.json；全部 TX/day 配对变化在 source_strata.csv，未删除负分层。此处新增噪声 SNR 不是未知原录音的总 SNR。EQ关闭且G=1，无EQ噪声着色；25MHz、IQ256、residual/post_sync、reference30、原phase/IQ/quality适配、reflect与数值延迟24保持。原白输入功率代理与实际波形新增噪声SNR分箱差异只作描述，不另算补偿或状态。逐包门控／分支输出本轮未保存，不能据此声称门控改善。

训练执行与梯度健康（范数不是协方差，不能替代识别结果）：

|行|已见源V准确率%|已见源V最差RX／准确率%|最终共同L目标|裁剪前均值|裁剪后均值|超过原limit的步数／4440|零梯度步数|
|---|---:|---|---:|---:|---:|---:|---:|
|RAW10|98.023|RX3／95.426|0.308521|0.930822|0.919881|31|0|
|INDEP|98.065|RX3／95.722|0.316135|0.996684|0.984598|30|0|
|ANTI|98.074|RX3／95.759|0.315609|0.997147|0.986990|28|0|

source V 最差RX只指已见fit验证接收机；本次未见RX只有held RX1，不能用此表声称多未知RX最差性能。全部20轮／4440步逐项核对同IQ RDrop、两份额外RD均值、总pair权重1.25、原raw.75／ordinary.25及SupCon warm/ramp；没有跨噪声KL。原L/U全顺序和物理角色相同，U真值不可见，teacher/reader无训练噪声生成器；13L＋7U预算保持。

训练噪声覆盖与记录核对：

|As行|噪声观测数|TX×fitRX×六场景已覆盖格数／144|格内观测最少／最多|LOS|PARTIAL|BLOCKED|
|---|---:|---:|---|---:|---:|---:|
|INDEP|1136640|144／144|7520／8212|784614|132958|219068|
|ANTI|1136640|144／144|7520／8212|784614|132958|219068|

完整流式重算 2273280 条噪声记录；两As每包noise0记录完全相同，非噪声state完全相同，ANTI每对新增噪声功率／SNR相同；覆盖与原模型累计计数完全相同。生成器逐包检查primitive噪声在原RX/IQ/G处理后、AGC之前互为相反数；离线没有获取IQ，只核对已记录功率／状态／顺序，不声称独立重建波形。原AGC仍各视图独立计算，处理后整段IQ不要求互为相反。六场景和所有原状态先验均保留，所有合法有标签As直接CE，不做置信度筛选或难例删除。硬件session为实际RX/day/eq，不由TX定义，训练按epoch/step/物理包刷新，训练／源评价／正式测试流分离。

成本：

|行|新model-E|更新数|实际NN前向样本|L前向|额外L前向|native秒|进程至七视图结束墙钟秒（约）|
|---|---:|---:|---:|---:|---:|---:|---:|
|RAW10|20|4440|6959958|5683200|2273280|1563.797|1581.426|
|INDEP|20|4440|6959958|5683200|2273280|3757.877|3775.881|
|ANTI|20|4440|6959958|5683200|2273280|3359.411|3377.109|

三行合计新60 model-E／13320更新／20879876 NN（含dispatcher2包合成兼容smoke）。各行L有568320次anchor呈现，但unique L只有5040；两As各1136640次噪声观测不冒充独立物理样本。U各396368次原观测、unique45360；原U/EMA/V和8820 held读出计入成本。历史匹配D_raw8L E20／4440／5823318 NN只作兼容smoke来源，未向三条新模型继承任何权重，历史训练成本另列，不是本轮训练父权重。
整矩阵并行墙钟约 3782.731 秒，Linux btime约1秒精度。全部run-owned worker自然退出；分析／绘图0NN／0更新，未重启或修改其他健康任务。预算耗尽不代表收敛。

实际七组图（PNG＋PDF均保存）：

[source_accuracy](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017/final/figures/source_accuracy.png)
[training_curves](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017/final/figures/training_curves.png)
[urban_strata](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017/final/figures/urban_strata.png)
[fixed_clean_retention](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017/final/figures/fixed_clean_retention.png)
[error_redistribution](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017/final/figures/error_redistribution.png)
[noise_training_and_coverage](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017/final/figures/noise_training_and_coverage.png)
[gradient_health](E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017/final/figures/gradient_health.png)

执行命令：本记录原训练发布命令和代码ee305facac7874bfc1340291ee5360a3665621d9保持。独立读取在工作树运行 F:/App/miniconda3/python.exe -m experiments.cvs_f_noise_pair20.analyze --input E:/type10-7/local_artifacts/cvs_FcrossRX_noise_pair20_20261017/final，随后同环境运行 experiments.cvs_f_noise_pair20.mechanism；绘图仅将这些源域汇总和离线代码交给既有N607 matplotlib环境，无IQ／权重访问。完整命令／stdout／验证位于本地artifact根目录，0额外拟合头。

结论边界：本次只有一个源heldRX／一个model seed；未做正式目标测试或独立第二seed确认，未证明新增Ag、Ag/As四格及交互、跨seed联合收益、正式low>60%或每suburban距clean≤2pp。冻结CE的两seed方差下降只解释本次采样理由，本轮实际训练与识别结果必须单独判断。整体目标ACTIVE。


结果解释与下一步边界

同预算、同初始规则下，INDEP 与 ANTI 的源域 clean 和全部 practical 均优于重复raw控制；ANTI相对INDEP的额外urban／low收益未出现，不能把相对raw的增益归因于正负相关性。上轮冻结eval CE的方差下降29.946%／31.790%没有在本次完整RDrop／SupCon／U／AdamW训练中转化为预登记识别优势，关闭固定噪声配对方案，不扩第二seed，也不将INDEP事后改为选中方案。

在本次相同low urban观测上，ANTI−RAW10的LOS为+15.104 pp、PARTIAL为+13.580 pp、locked为+14.989 pp；BLOCKED为−0.560 pp、unlocked为−0.885 pp。低场景总计修复202个、破坏126个，净改善76/1260；相对INDEP则修复11个、破坏16个，净少5个正确。固定RAW10 clean正确1004包的low保持率由28.088%升至两种As相同的39.442%。这说明可恢复观测的识别保持改善，不证明某个物理损伤已被移除；最困难重遮挡／失锁部分仍未恢复。

整体正收益也不代表每个TX均改善：low urban的ANTI−RAW10在TX2、TX3、TX4分别为−15.238、−11.429、−2.381 pp。以下完整分组保留所有负结果，不用总分掩盖错误重新分配。

|TX|每视图包数|RAW10 low准确率%|INDEP low准确率%|ANTI low准确率%|
|---|---:|---:|---:|---:|
|0|210|12.857|27.143|27.619|
|1|210|4.762|42.381|40.476|
|2|210|33.333|18.095|18.095|
|3|210|44.286|33.810|32.857|
|4|210|61.905|59.048|59.524|
|5|210|9.048|24.286|23.810|

ANTI clean相对RAW10为+6.111 pp，未付出clean下降；但三个suburban仍距自身clean 27.302／31.667／33.254 pp，low仅33.730%，未达到联合目标。本次只完成单seed源域证伪；无正式目标分数或跨seed可重复性结论。

执行验证：本地实际40个训练配置／权限负例、3858合成NN／3更新PASS；唯一独立P0/P1审查PASS；发布真实CP no-query兼容smoke 2NN PASS。离线reader24个负例PASS，完整3×20轮／3×4440步／2273280噪声记录／21份预测重算PASS，原科学判定独立复现，7PNG＋7PDF已实际生成并逐张查看。分析／绘图0IQ／0NN／0更新；仅报告模板发生换行转义错误，执行前修复并通过语法检查，未改训练或重算分数。归档提取出现Python将来版本默认filter的非阻断弃用提示，产物102份完整且内容读取成功，没有重启或重测。

下一步唯一待检验假设：当前已训练As模型在BLOCKED／unlocked观测中仍有可用身份信息，但被新增噪声遮蔽；不是继续改变噪声配对。最小拟议实验为冻结本轮RAW10和ANTI各自默认头，仅用相同合法源L和同名low urban流，对比clean、完整residual、固定同一接收状态／补偿／AGC的移除新增噪声控制，记录现有分支／门控／默认头间隔，不拟合新头或更新模型。该诊断尚未启动，正式测试配置不变。预先以BLOCKED和unlocked均恢复至少一半full到clean错误差距作为“噪声侧有足够恢复空间”的证伪条件；不满足就放弃靠这类噪声侧方案突破，转查传播或现有参考证据，而不调强度／换seed救援。旧未增强模型的噪声移除已做过；本次若开展只回答新As训练后仍失败的归因问题，不重复旧参数搜索。

本轮closure为科学负结果，实验ANALYZED；整体Ag／As联合goal仍ACTIVE。
