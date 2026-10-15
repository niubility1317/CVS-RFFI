# 冻结 EMA：目标域 low urban HQ 配对验证

本次由侧对话用户明确要求：先查已有目标HQ实验，没有则启动验证。2026-10-15登记搜索HQ仅有源域L_dev720诊断；远端未发现target HQ run。本次独立工作树/branch为target_hq_diagnostic_20261015，不改动主线已有cvs_urban_bridge50代码或训练。完整配置、数据及逐row/seed/命令/输出见[experiment.json](experiment.json)。

## 固定模型与对比

使用源诊断64.86%→70.97%的同一A0 final50 checkpoint内`ema_model`和EMA自身默认cosine头。原训练为相同5580/720源内部折、旧门控33+17、E50/11100；本次0新增epoch、0 optimizer/EMA/head-fit更新。own_model严格核对实际source契约、source_frozen/completion/initialization/inner折及CP，再严格加载同CP EMA；不按目标成绩选择模型。

从原168000个固定target匿名ID以seed2026101513的固定随机排序选4200，完全不读TX/质量/损失来采样。覆盖范围由评分后实际RX/day/class计数披露；这是4200小试，非全168000结论。原full received缓存逐值保持只读；根据原manifest的low urban config、seed392005、receiver_seed2027、namespace/session重放full并逐样本比较（rtol/atol2e-6），不一致即技术失败。

四个输入条件预先固定：原full、同传播/同接收状态下独立同强度噪声、该同一独立噪声降低6dB的HQ、原共享噪声降低6dB的HQ对照。只改变模拟器新增噪声；原录音噪声保留、总SNR未知。传播/残余/EQ核、RX硬件、lock、quality代理、补偿参数、AGC增益/限幅、24点FIR及reflect边界都冻结为full动作。HQ不再做输出RMS，不增加可训练模块。EQ仍关闭G=1、refSNR30、post_sync残差路径不变。

## 预测、评分与解释

CPU2/CUDA_VISIBLE_DEVICES空；4200×4+真实CP smoke2，共16802前向。launcher先做真实EMA/原头无query smoke，通过立即继续；所有模型eval、requires_grad=False、no_grad，逐值确认state_dict不变。每个ID的四视图logits全部固定及complete后，以exec启动独立新进程scorer连接原truth；缺失/错序/非有限预测或更新标记会在truth打开前拒绝。

报告accuracy、Macro-F1、配对修复/新增错误，以及全部实际RX、day、LOS/PARTIAL/BLOCKED、lock和full新增噪声区间。同强度独立噪声对照用于区分换噪声与降噪效应；独立HQ与共享HQ分开报告。不按结果扩样、换seed或选择性重跑，不把本结果回流主线训练、候选或参数。

范围明确为`TARGET_HQ_COUNTERFACTUAL_NOT_FORMAL`。即使HQ目标准确率更高，也只证明较好输入下冻结EMA的表现，不是原full practical成绩或增强训练收益。正式clean＋六practical固定IQ不修改、旧产物不覆盖。单模型seed/单独立噪声实现不构成跨seed重现证据。

## 本地验证与发布

5项聚焦检查实际PASS（0.336秒）：标签盲稳定抽样、canonical full保持/独立噪声剂量、重放不匹配/弱化config拒绝、四预测完整后才允许评分及非有限/错序/更新负例、准确率与Macro-F1计数。独立scorer模块导入不加载torch/model/teacher。原coupled物理函数复用已完成源诊断的检查/审查结果。侧对话明确禁止subagents，因此本次不调用或声称完成独立Agent审查；新接线作一次手工P0/P1范围检查，并以协议负测和真实CP烟测验证。不启动额外审批或其他聊天。

代码与config Git提交后，一次release archive/传输SHA比较、远端compile与路径/普通账号/资源preflight；随后独立读回PID/CWD/argv/CPU/GPU/日志增长。精确命令为既有N607 Python `-u -m experiments.cvs_target_hq.predict`，CWD为新release。唯一owner为codex/side/target-hq-20261015，输出根/日志/源CP路径详见experiment.json。任何来源、handles、replay、非有限、预测、目录冲突或技术失败仅退出本owned CPU进程，保留产物，无自动重试，不动其他任务。

本次已自然完成预测、独立评分和同row分析，实际结果见下。实际发布commit、独立运行读回与完整结果将在同一报告补齐。原A0 E50成本与本次0训练/评价墙钟分开报告。

## 已完成目标域小试结果

VERIFIED：实际代码发布commit `28521bbc44dcad606c2dc62dc4443702c8534de0`。N607普通账号，CPU2/CUDA空，PID3138590的CWD/argv/环境独立读回匹配；真实冻结A0 EMA/默认头无query smoke PASS，运行中2970/4200时truth_read=False。日志2480→5669字节，末次完整4200预测/四视图complete、独立scoring complete、ANALYZED且进程自然退出。一次archive传输SHA比较和远端compile/preflight通过。

本次共16802个前向，预测/物理生成/检查墙钟33.890397秒；0新增epoch、0 optimizer/EMA/head-fit更新，全部模型状态逐值未变。父模型已有E50/11100训练成本不是本次新增训练。

目标域独立HQ确有收益：原full为2220/4200（52.8571%），独立HQ为2428/4200（57.8095%），增加4.9524个百分点。单纯换成同标称方差的独立噪声为53.3333%；在该同一独立噪声上降低6dB后再增加4.4762个百分点。这里的same_noise_power文件名指相同标称噪声方差／剂量，不指独立抽样后每条记录的实际噪声功率逐值相同。

|冻结EMA输入条件|准确率 %|Macro-F1 %|正确/总数|
|---|---:|---:|---:|
|原始 full practical low urban|52.8571|52.4575|2220/4200|
|独立噪声，同标称方差|53.3333|52.9247|2240/4200|
|独立噪声 HQ −6 dB|57.8095|58.0259|2428/4200|
|共享噪声 HQ −6 dB|57.7143|58.0161|2424/4200|

|配对比较|变化 pp|修复|新增错误|
|---|---:|---:|---:|
|独立噪声 HQ −6 dB − 原始 full practical low urban|+4.9524|549|341|
|独立噪声，同标称方差 − 原始 full practical low urban|+0.4762|471|451|
|独立噪声 HQ −6 dB − 独立噪声，同标称方差|+4.4762|371|183|
|共享噪声 HQ −6 dB − 原始 full practical low urban|+4.8571|365|161|

全部7个目标RX方向均为正，增幅3.2206至8.5217个百分点；这是跨RX一致的本次观测，不是跨模型seed重现。全部4个日期也均为正。模型seed仍只有2026092701、独立噪声实现每物理样本1个。

|目标RX|样本数|full %|独立HQ %|变化 pp|
|---|---:|---:|---:|---:|
|1-1|592|52.3649|56.4189|+4.0541|
|14-7|611|49.9182|53.5188|+3.6007|
|2-1|617|54.1329|60.1297|+5.9968|
|20-1|582|48.7973|52.4055|+3.6082|
|7-14|575|55.1304|63.6522|+8.5217|
|7-7|621|59.0982|62.3188|+3.2206|
|8-8|602|50.3322|56.1462|+5.8140|

### full原状态

|分层|样本数|full %|独立HQ %|变化 pp|
|---|---:|---:|---:|---:|
|BLOCKED|2408|34.3439|41.5698|+7.2259|
|LOS|1189|80.0673|80.2355|+0.1682|
|PARTIAL|603|73.1343|78.4411|+5.3068|

### full接收锁定

|分层|样本数|full %|独立HQ %|变化 pp|
|---|---:|---:|---:|---:|
|degraded|815|51.6564|58.8957|+7.2393|
|locked|1515|79.6700|80.3300|+0.6601|
|unlocked|1870|31.6578|39.0909|+7.4332|

### full新增噪声SNR区间

|分层|样本数|full %|独立HQ %|变化 pp|
|---|---:|---:|---:|---:|
|3-8|972|39.8148|47.5309|+7.7160|
|8-15|370|64.3243|69.1892|+4.8649|
|<3|1339|28.8275|36.5945|+7.7670|
|>=15|1519|79.5918|80.3160|+0.7242|

### 目标日期

|分层|样本数|full %|独立HQ %|变化 pp|
|---|---:|---:|---:|---:|
|2021_03_01|1092|50.8242|56.9597|+6.1355|
|2021_03_08|1024|55.3711|58.6914|+3.3203|
|2021_03_15|1075|53.3023|57.9535|+4.6512|
|2021_03_23|1009|52.0317|57.6809|+5.6492|

BLOCKED 2408包由34.3439%升至41.5698%（+7.2259pp，净多174正确）；LOS仅+.1682pp；PARTIAL+5.3068pp。unlocked组+7.4332pp；full新增噪声<3dB组+7.7670pp。主要净正确增量来自BLOCKED/低质量观测，分层重叠不相加作独立因果解释；接收动作全部冻结，这里是噪声剂量受控效应，不是重算接收状态后的总效应。

全部4200个full重放与原固定缓存最大绝对误差为0.0；实际baseline直接使用原缓存，没有改写正式IQ。独立0/−6dB的同一噪声实现逐样本剂量差平均/最小/最大均在浮点误差内为6dB；传播、接收状态及补偿/AGC检查通过。原录音噪声不被消除，新增噪声SNR仍不是总SNR。所有混淆矩阵准确率和Macro-F1已独立重算一致；每个分层维度的样本数均完整覆盖4200，无按困难程度删除样本。

![目标域HQ四条件与RX／状态／质量分层](E:/type10-7/local_artifacts/cvs_target_hq_20261015/target_HQ_counterfactual.png)

图从实际results.json和原channel_metadata生成，并已查看。原逐样本预测保存在唯一远端run/prediction，详细结果/derived_summary.json/metadata/resolved_config/读回证据和图在 E:/type10-7/local_artifacts/cvs_target_hq_20261015。未将IQ、模型、原预测/日志或生成图提交Git。

## 本次结论与边界

源域64.86%→70.97%的收益并非仅存在于源域：这次同EMA目标小试也出现52.86%→57.81%的收益。不过HQ目标准确率仍低于60%，不能外推为完整168000测试或跨seed结果，更不能把57.81%记为原full practical性能或增强训练成绩。本次使用EMA+4200子集，不直接与主线student默认头+168000的53.27%相减归因。

本次没有训练，所以不存在由本实验更新参数导致的clean性能代价；也未新测clean，不能据此声称任何HQ训练方法能保持clean。target结果不回流主线模型、损失、候选、超参数、种子或选择性重跑。所有条件和负面项（新增341个错误、未达60%、未跨seed、子集）均保留。

5项本地检查最新PASS0.265秒；独立scorer导入不含torch。独立Agent审查未执行，遵守侧对话禁止subagent约束，不声称审查已通过。输入/协议、真实checkpoint smoke、独立truth-last与不可覆盖输出保护均已实际执行。主线程既有代码/训练未修改。

复现入口：独立branch `codex/target-hq-diagnostic-20261015` 的 `experiments/cvs_target_hq/`；实际启动命令、CWD及来源在experiment.json。`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_target_hq.checks` 为本地聚焦验证；N607运行 `python -u -m experiments.cvs_target_hq.predict`，四视图预测完成后exec独立 `experiments.cvs_target_hq.score`。本次run已存在，不可再次提交同root。
