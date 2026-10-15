# HQ 教学：三个固定模型的源域冻结归因预登记

本分析为已授权 HQ E50 小试的冻结机制交付，不增加训练。三个角色在训练结果产生前固定：普通拼接 NOMINAL、HQ 标签监督 BRIDGE_CE、额外局部教学 BRIDGE_SIG。两条新模型已完成E50，本冻结源分析已发布并完成，实际结果和三图见下。既有 HQ 64.86%→70.97% 仍仅是旧冻结模型的源域受控诊断。

唯一矩阵、数据、路径/命令、环境/CWD、checkpoint来源、seed、stop及预期产物维护于 experiment.json 和 experiments/cvs_urban_bridge_mechanism/registration.json。代码身份是实际发布 Git HEAD，发布时写入 preflight/resolved；不增加报告或成员 SHA。

源角色复用 L_s6300 内部 L_train5580/L_dev720，fold2026101503，source RX1/3/4/6/8、day1/2/3、全部6TX，split/data392005。只前向物理互斥的 L_dev720；不打开 U，V 容器可能由既有源 cache 物化但前向为0，不调用任何目标数组或 truth。原冻结 final50/11100 默认头及 own EMA 严格加载，不拟合头、不更新 optimizer/EMA/normalization。新行还必须具备真实 VERIFIED 五视图预算及自己的 epoch5/5760 前向源校准；不匹配折、来源不明或目标接触的权重拒绝。

固定 single SIG_TIME、内部门控/参考响应、默认 cosine6×160 scale30。采样率25MHz、IQ256中心裁剪、原native RMS后模拟器unitRMS；真实 adv3b02_xuc/code/leo_practical residual/post_sync、EQfalse G1、reference_snr_db30及原质量适配/相位/IQ补偿/reflect/FIR24不改。实际 __file__ 写入resolved。正式七环境IQ从不读取或重新生成。

六种源 practical 场景全部保留，每个物理样本构造full和独立噪声HQ。两者共享传播/RX session/接收质量/lock/补偿/残余核/AGC/FIR/边界，只将模拟新增噪声幅度乘10^(-6/20)。原录音噪声与未知TX硬件不反演；新增噪声SNR不冒充总SNR。coupled实际 full逐值一致、状态/算子冻结证明须通过。源流2026101509独立于训练1401、验证1402、旧HQ1504、own-cal1508及正式固定流，RX/day/eq session不由TX定义。

比较固定 NOMINAL student clean 正确集合上的 full/HQ 默认头保持率、对 NOMINAL full 的错误修复/新增，以及 SIG 对真实 BRIDGE_CE full 的修复/新增；间隔为true logit−最强wrong logit后除30。NOMINAL先处理时没有CE预测，显式null/count0，不冒充已观察对照。各RX/LOS/PARTIAL/BLOCKED/lock/full新增噪声SNR区间分别汇总，三模型使用完全相同的720条源视图与分层。

实际 raw112 只取4窗的increment6、feature-area15、time-area6，排除dt。报告full→同H HQ ownEMA的有界组误差，同时报告同packet−最近wrongTX组余弦、既有各分支的身份关系/方差、实际gain贡献。跨模型共同误差使用BRIDGE_CE自己epoch5冻结的HQ RMS；各自训练尺度误差单列，不重新校准。几何相似增强与默认头间隔改善分别解释，不能互相替代。NOMINAL有不同训练CE/前向成本，局部增量主要看SIG−CE；随机优化仍限制唯一因果解释。

实际预算56882前向：真实NOMINAL checkpoint烟测2＋固定clean集合720＋3×（clean student/EMA1440＋6场景×full/HQ student/EMA2880）。新增epoch、optimizer/EMA/head-fit更新均0；原三个E50/11100及实际训练秒数保留，源分析另记墙钟。不执行或声称新训练收敛、跨seed重现或正式测试提升。训练梯度/门控/曲线读取原主实验完整日志，源分析本身没有回放optimizer更新。

执行在N607普通账号，CPU2、CUDA_VISIBLE_DEVICES为空。精确命令为既有解释器 `-u -m experiments.cvs_urban_bridge_mechanism.probe`，CWD为本run的独占release，路径见experiment.json。首步真实checkpoint默认头无query烟测PASS后继续；全套实际final50就绪才发布。一次archive比较/远端compile/preflight，独立PID/CWD/argv/log和最终results/complete读回，不以退出码单独证明完成。源/CP/物理算子/非有限/状态/预算/输出碰撞/技术错误只退出本owned源行，保留所有产物、不自动重试、不触碰健康训练或其他owner。

本地5个唯一检查分批执行：首4项中实际组/统计/真实冻结模型3项PASS，源资格fixture因Windows路径字符串比较出现StopIteration；改为Path比较后定向PASS.010s，并补入真实calibration status检查。发布接线定向PASS.006s，证明无需本地读取远端checkpoint且临时运输模块恢复。已知torch/NumPy ABI警告保持原环境，检查exit0。compileall通过。未用合成模型结果冒充源或目标准确率。

独立源候选一次有界P0/P1审查PASS：已核对三模型严格加载、同H/full状态、L_dev边界、默认头/ownEMA、统计/56882预算及发布/输出保护；三个final50实际冻结后，已完成一次源发布、真实56882预算/状态保护、分层结果、三图及独立终态读回。追踪表为analysis/cvs_bridge_teaching_mechanism_traceability_20261015.md，当前verified6/pending0/deferred0/rejected0/blocked0；原主训练全日志审计已完成。最高风险：局部误差下降但默认头间隔不增、正确证据没有恢复；HQ仍可能错，单seed源代理不足以证明正式跨RX收益。

## 实际发布、完成与证据边界

VERIFIED：实际源分析代码 `6c2933bb4089695181057837914fd7e426d98902`，一次 archive 传输比较通过、远端 compile PASS、普通账号 preflight VERIFIED；真实 NOMINAL/default-head 首步 smoke PASS。登记 release/run 独占，提交 PID3220414。首次独立读回时该进程已经自然结束，未观察到存活的 PID/CWD/cmdline 窗口；完成结论来自 resolved 实际模块路径、smoke、三模型进度、完整 results/complete 及终止日志，不伪称读到了存活进程，也不为补该窗口重跑。

实际 56882 个前向、50.256831 s、0 optimizer/EMA/head-fit 更新，student/ownEMA 状态均未变，V前向0、U未打开、query/target未读。全部六场景×720条同H/full/HQ算子核对通过；标量sample trace实际25,920条，36个model×scene×view单元均720条。只涉及一个模型seed，不构成正式结果或跨seed复现。

## 默认头与固定集合

|冻结模型|clean %|mid full %|low full 正确/720|low full %|low HQ %|low fixed703保持率 %|low head margin/30|相对NOMINAL修复/新增|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|NOMINAL|97.6389|79.7222|463/720|64.3056|69.5833|65.7183|0.089640|0/0|
|BRIDGE_CE|97.0833|80.5556|470/720|65.2778|70.1389|66.7141|0.082212|26/19|
|BRIDGE_SIG|97.5000|79.5833|474/720|65.8333|71.2500|67.2831|0.081173|30/19|

同一固定703集合上，SIG 相对 CE 修复 11 条、又新增 7 条错误；净增4条，与全部720条的470→474一致。clean为699→702，增加3条。默认头平均间隔从 0.082212 降至 0.081173，不能将小幅纠错等同于判别关系整体增强。

本流2026101509与旧HQ控制1504、训练、常规源panel及own-cal流独立；本次NOMINAL low full64.3056%不与旧流EMA64.8611%或训练中E45的69.0278%直接相减。共享的是本次三模型的实际观测与固定集合。HQ分数只代表降低模拟新增独立噪声的输入干预，正式测试保持原强度。

|low full状态|n|NOMINAL正确|BRIDGE_CE正确|BRIDGE_SIG正确|
|---|---:|---:|---:|---:|
|LOS|230|219|221|221|
|PARTIAL|88|76|77|77|
|BLOCKED|402|168|172|176|

SIG相对CE增加的4条正确均来自BLOCKED（172/402→176/402，增加0.9950个百分点）；LOS/PARTIAL正确数相同。该改进规模小，且mid urban的源表现未同步提高；不能归因为广泛恢复所有urban退化或单一分支的可靠性提升。

## 局部教学误差、身份区分与融合

|raw112分组|CE误差|SIG误差|CE配对−最近wrongTX余弦|SIG配对−最近wrongTX余弦|
|---|---:|---:|---:|---:|
|increment|0.191077|0.102984|-0.108816|-0.100393|
|feature_area|0.223052|0.065632|-0.024318|-0.036545|
|time_area|0.175004|0.090806|-0.072988|-0.070346|

共同尺度固定为CE自己E5的HQ RMS，未为本分析重校准。三个有界组误差都降低，但相对最近错误TX的gap仍均为负，feature-area gap还变差。实际SIG贡献/joint norm从 0.132706 增至 0.138526，不构成可靠身份信息增加的单独证据。全部分支关系、方差、gain及各自尺度误差保留在results.json；没有拟合诊断头。

原模型训练成本：

|模型|已完成原epoch|原更新数|原训练墙钟 s|
|---|---:|---:|---:|
|NOMINAL|50|11100|4593.410037|
|BRIDGE_CE|50|11100|7082.839002|
|BRIDGE_SIG|50|11100|7157.628982|

本源分析不新增训练成本。NOMINAL的原训练CE/前向预算与新五视图模型不同，只作历史参考；SIG−CE匹配五视图及CE预算，单seed随机优化仍限制唯一因果解释。原主实验全step日志/训练曲线审计另由full_log_audit处理，本次不把metadata当成已读完整主日志。

[完整源结果](E:/type10-7/local_artifacts/cvs_bridge_teaching_mechanism_20261015/results.json)、[分层摘要](E:/type10-7/local_artifacts/cvs_bridge_teaching_mechanism_20261015/source_summary.json)、[独立终态读回](E:/type10-7/local_artifacts/cvs_bridge_teaching_mechanism_20261015/readback_initial.json)。

## 已生成并检查的三图

![source_default_head_repairs.png](E:/type10-7/local_artifacts/cvs_bridge_teaching_mechanism_20261015/figures/source_default_head_repairs.png)

![source_SIG_geometry_usage.png](E:/type10-7/local_artifacts/cvs_bridge_teaching_mechanism_20261015/figures/source_SIG_geometry_usage.png)

![source_state_quality_strata.png](E:/type10-7/local_artifacts/cvs_bridge_teaching_mechanism_20261015/figures/source_state_quality_strata.png)

后续候选依据仅来自这里的源机制：关闭联合/原始特征对齐，独立比较无额外约束、clean参考和同H HQ参考的单侧默认头间隔。该候选尚未实现或发布，见analysis/cvs_hq_head_relation_source_design_20261015.md；不改本已冻结方案或判定，不预称优于KL。

完整原训练审计现已完成：[新22200步＋复用33300步](E:/type10-7/local_artifacts/cvs_bridge_teaching_e50_20261015/final/training_audit.json)，M05 verified。此前待完成表述为当时状态，最终计数verified6/pending0。没有新增训练或前向，也没有改变源分析结论。
