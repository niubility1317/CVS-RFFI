# Phase1 DualGuard16完整结果分析

时间：2026-07-12。实验：`phase1_dgleo_dualguard16_20260712`。

## 最终结论

`DualGuard16`已经修复上一批P1实验的训练塌缩：16/16候选完成120 epoch，final source-val保持98.45%-98.68%，冻结heldout的overall为88.89%-89.76%、strict UDU为84.11%-85.51%。因此，上一批63/64候选降至16.67%的问题确实来自source-episode结构损失失控，而不是测试缓存或checkpoint错载。

但本批仍没有得到“可拒识的跨域泛化表征”。相对同seed的C0，所有机制单元在两个seed上都提高了overall、strict UDU、rx11 floor和独立satellite stress，说明DG方向有小幅稳定正增益；open-set侧只有fixed proxy_vaccept、tail accept及部分ratio出现小幅改善，p95/p99/CVaR、source overflow、bridge、low-density和overflow accept没有跨seed一致改善。更关键的是，source-episode overflow仍为0.987-0.989，legacy proxy_vaccept仍为0.616-0.623，legacy bridge仍为1.000，U_s direct和U_s invariance均未实际生效，16/16候选均被fail-closed阻断。

因此，本批的科学定位是：**训练稳定性修复成功、闭集DG小幅改善、独立星地评估暴露新短板、open-set机制仍未闭环**。当前没有候选可以宣称真实unknown拒识改善或直接推进Stage2/Phase3 unknown成功结论。

## 协议边界

- 本批是Phase1 source-only弱标签地面域泛化训练，使用`ManySig.pkl`。数据切分为labeled/unlabeled/source-val=`0.08/0.72/0.20`，按项目定义`rho_label=0.08/(0.08+0.72)=0.10`。
- 本轮优化步骤和final-only checkpoint selection没有直接使用真实unknown、目标receiver/day或heldout测试反馈。
- 当前只能评价闭集DG、独立合成星地stress、known几何、proxy风险、prototype/endpoint导出质量。
- 不能声明真实unknown FAR、FPR95、Stage2 old_acc、seen_new_acc、H_old_new或真实星地部署成功。
- `endpoint_accept_v1`与prototype均未导出，因此动态DM软门控不能被解释为最终拒识边界。
- 但是本批初始化和teacher均来自`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；该父checkpoint历史上使用`best_metric=joint_safe`并周期性读取同一heldout receiver/day和弱satellite评估。因此，本批是“source-only数据训练、final-only本轮选模”，但不是完全test-blind的模型lineage。绝对heldout结果只能作为heldout-aware ancestor上的诊断；同teacher、同seed的C1-C7相对C0配对差值仍可用于本轮机制比较。

## 训练健康与证据完整性

|检查项|结果|判断|
|---|---|---|
|scheduler|`TERMINAL`，16候选，实际4.05小时；16个子进程均以guard exit code 5结束|全部跑完，未触发10小时deadline；`TERMINAL`只表示队列终结，不表示promotion成功|
|epoch与结构化日志|16/16均有120行CSV和120行JSONL，epoch连续1-120|完整|
|stdout|16个stdout各5382行，完整扫描无Traceback、RuntimeError、CUDA OOM、Killed、FATAL或AssertionError|无fatal|
|checkpoint一致性|16/16的`final_ssdg.pth`、terminal SHA和frozen heldout SHA完全一致|没有错载checkpoint|
|冻结heldout|16/16状态`COMPLETE`，只对final权重评估一次|符合final-only与test不参与选模协议|
|非有限梯度|每条跳过8-9个batch，16条合计135步；分布在约5-8个epoch|不构成塌缩，但属于系统性训练/预算控制缺陷，需继续定位|
|source结构loss|所有启用候选均满足`loss<=upper_bound`；实际最大0.208-0.377，远低于上界10.07-12.17|无界小半径问题已修复|
|terminal|16/16均为`NON_PROMOTABLE_GUARD_BLOCKED`|模型完成不等于可promotion|
|tail reference|16/16的reference-to-final状态`FAILED`，`p99_delta`为空|绝对p99超限且没有可保存的安全reference|
|prototype/endpoint|16/16均`SKIPPED_FAIL_CLOSED`，`endpoint_export_ready=false`|导出阻断正确，但拒识闭环未完成|

训练期日志中的`[TEST] overall_tx=nan% (0/0)`不是测试缓存故障，而是`stage_test_eval_ran=0`的占位输出。`protected_overall_tx/strict_udu`在训练CSV中为空，冻结heldout仅在E120 final权重后执行。16条final结果实际不同；上一批大量相同16.67%是模型真实塌缩，本批不存在该现象。该协议可以避免test泄漏，但无法从本批计算heldout strict UDU的best-final曲线。

## 泛化能力主表

下表为每个机制单元两个paired seeds的均值；Delta为相对同seed C0后再求均值。当前satellite指标来自与训练`leo_*_weak`无scenario、family、config hash和implementation重叠的六个`legacy_full`压力场景。该六场景floor包含`storm_mp`、`geo_clear`和`mixed_orbit`；按`项目.md`只能视为legacy/diagnostic stress，不能作为deployment-primary LEO成功门槛。

|cell|overall|strict|rx floor|sat mean|sat floor|sat strict floor|Delta overall|Delta strict|Delta rx|Delta sat mean/floor|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`C0_DG_ANCHOR`|88.93|84.32|70.55|43.52|38.67|35.65|+0.00|+0.00|+0.00|+0.00/+0.00|
|`C1_LEAVE_DOMAIN_ONLY`|89.39|85.13|72.81|43.93|38.95|36.00|+0.46|+0.81|+2.27|+0.41/+0.27|
|`C2_LOCAL_NO_DENSITY`|89.38|85.14|72.75|44.04|38.95|36.00|+0.45|+0.82|+2.21|+0.52/+0.27|
|`C3_LOCAL_BALANCED`|89.46|85.28|72.92|44.06|39.00|36.08|+0.53|+0.96|+2.38|+0.53/+0.33|
|`C4_LOCAL_STRONG_PROTECT`|89.45|85.23|72.40|44.05|38.96|35.99|+0.52|+0.91|+1.85|+0.52/+0.28|
|`C5_SAT_GEOM_STRONG`|89.40|85.14|72.57|44.01|38.94|35.98|+0.47|+0.82|+2.03|+0.49/+0.26|
|`C6_U_DOMAIN_QUAR`|89.47|85.28|72.78|43.99|38.94|35.99|+0.54|+0.96|+2.24|+0.47/+0.26|
|`C7_FULL_JOINT`|89.29|84.97|72.19|44.13|39.03|36.08|+0.37|+0.65|+1.64|+0.60/+0.35|

泛化结论：

1. 所有C1-C7单元在两个seed上均同时提高overall、strict UDU、rx11 floor、sat mean和sat floor，方向稳定，但只有两个seed，仍是探索性证据。
2. `C3_LOCAL_BALANCED`是最均衡DG单元：overall 89.46、strict 85.28、rx11 floor 72.92、sat mean/floor 44.06/39.00。它相对C0分别提高0.53、0.96、2.38、0.53/0.33pp。
3. `C6_U_DOMAIN_QUAR`的overall/strict均值最高0.01pp以内，`C7_FULL_JOINT`的satellite mean/floor最高，但C7的overall/strict/rx floor反而低于C3，完整联合没有形成额外协同收益。
4. 最弱receiver始终是unseen-day rx11，范围69.00%-75.09%。S1普遍约69.0%-71.0%，S2约72.1%-75.1%，seed效应约3-5pp，弱receiver没有被稳定修复。
5. 独立`legacy_full`星地stress的sat mean只有42.85%-45.00%，aggregate floor 38.27%-39.56%，strict floor 35.24%-36.65%。历史74%-78%的sat floor多数使用弱增强同族口径，不能直接比较；本批反而证明跨增强族星地泛化仍很弱。由于本批没有独立的简化LEO residual heldout族，不能给出合规的deployment-primary LEO结论。

## Open-set代理主表

`fixed`指标来自final权重上的固定source-val、多视图、receiver-aware local component评估；`legacy`来自原`proxy_unknown_*`路径。C0关闭source-episode损失，因此其CSV中的source-episode overflow=0是inactive占位，不是优良结果。

|cell|p95|p99|CVaR|source ovf|fixed proxy|fixed bridge|low-density|tail|overflow|ratio|src-ep ovf|legacy proxy/bridge|open grad share|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
|`C0_DG_ANCHOR`|54.64|82.83|69.07|0.453|0.411|0.219|0.092|0.826|0.582|2.976|N/A|0.621/1.000|0.011|
|`C1_LEAVE_DOMAIN_ONLY`|53.79|82.95|68.43|0.460|0.394|0.200|0.094|0.818|0.554|2.921|0.988|0.621/1.000|0.026|
|`C2_LOCAL_NO_DENSITY`|53.57|82.75|68.46|0.463|0.411|0.249|0.094|0.819|0.569|3.055|0.988|0.622/1.000|0.026|
|`C3_LOCAL_BALANCED`|53.58|83.07|68.60|0.459|0.395|0.203|0.094|0.816|0.559|2.953|0.988|0.623/1.000|0.026|
|`C4_LOCAL_STRONG_PROTECT`|53.60|82.94|68.43|0.462|0.409|0.239|0.094|0.821|0.554|2.904|0.989|0.620/1.000|0.033|
|`C5_SAT_GEOM_STRONG`|53.72|82.76|68.46|0.460|0.402|0.223|0.094|0.811|0.552|2.899|0.988|0.622/1.000|0.025|
|`C6_U_DOMAIN_QUAR`|53.35|82.78|68.44|0.455|0.403|0.230|0.096|0.812|0.547|2.939|0.987|0.621/1.000|0.028|
|`C7_FULL_JOINT`|53.45|82.81|68.35|0.461|0.412|0.248|0.095|0.820|0.554|2.919|0.988|0.618/1.000|0.036|

Open-set结论：

1. p95均值相对C0下降0.84-1.29度，但该改善不跨seed稳定：S1普遍恶化约1.19-1.69度，S2改善约3.15-3.77度。p99均值仅变化-0.08至+0.24度，所有单元仍约82.75-83.07度，不能认为tail被压短。
2. source overflow没有改善。除C6近似持平外，机制单元均比C0高0.002-0.009；source-episode overflow长期0.987-0.989，远高于0.90门槛，也是14/16候选的终局阻断原因。
3. `C1/C3/C6`的fixed proxy在两个seed上都小幅下降，但均值改善只有0.007-0.017，未达到预设0.05；legacy proxy仍0.616-0.623，legacy bridge全部1.000。
4. fixed tail accept下降0.005-0.015且两个seed方向一致，是本批最稳定的open代理小改进；但绝对值仍为0.811-0.821，satellite tail accept仍约0.81-0.83，接收风险仍很高。
5. bridge、low-density、overflow accept和p95/p99/CVaR的变化均存在seed反转。C7 fixed bridge相对C0还恶化0.029，完整联合不能作为open-set主候选。
6. dynamic DM在E100-E120可达到local p99约71.7-72.7度、proxy约0.239-0.243、bridge约0.018、radius/inter约1.46-1.49；fixed source-val则约82.8度、0.39-0.41、0.20-0.25、2.90-3.05，legacy proxy/bridge约0.62/1.00。三层风险没有同步下降，再次证明动态batch软门控不等于最终拒识边界。

## Clean与星地视图几何

|视图|p50|p95|p99|CVaR|proxy|bridge|tail accept|overflow accept|radius/inter|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clean，跨单元范围|1.51-1.62|6.06-6.32|82.75-83.07|37.00-37.32|0.107-0.115|约0|0.411-0.423|0.259-0.310|1.088-1.101|
|sat weak训练视图，跨单元范围|5.70-6.25|53.35-54.64|78.85-79.33|68.35-69.07|0.394-0.412|0.200-0.249|0.811-0.826|0.547-0.582|2.899-3.055|

clean p95约6度说明95%的clean known样本已有紧核心；clean p99却约83度，说明仍有约1%的极端异常尾。更严重的是，satellite视图p95约53-55度、clean-sat pair p95约82-83度，表明`concat_sa`主要依靠CE/KD让分类器容纳大幅星地位移，而没有把星地样本映射回同一RFF invariant core。这正是“泛化依赖宽接收域”几何矛盾在当前实现中的直接证据。

## 梯度、U_s与解耦机制

- E100-E120的teacher clean KL约5.2、teacher sat KL约9.3-10.4、sat CE约2.8；direct metric加权约0.34或0.51、proxy约0.037-0.047、source episode仅约0.0015-0.0057。
- 在梯度有限的E100-E120，open控制器达到最大约4倍scale后，post shared-z_id gradient share仍只有0.011-0.036，16/16低于0.06下限。所有候选因此出现`B_os_eff_below_min`。
- 梯度预算实现还有P0数值缺陷：被跳过的非有限batch中closed norm为空/非有限，但`pre_budget/post_budget`仍用`max(1e-12, open+closed)`计算，产生`5.77e7-2.77e10`的大于1伪比例。终局低于0.06的结论来自有限final batch，仍成立；但全程controller命中率、均值和“预算已受控”证据不可直接相信。
- global grad clip几乎每个epoch均激活，pre-clip常在约250以上，clip后固定约5。该设置防止塌缩，但没有解决closed梯度压过open几何梯度的问题。
- U_s direct weighted loss和selected count在16/16均为0。要求U direct的12个候选全部报`US_DIRECT_LOSS_IDLE`。
- U_s三态平均每112个query只有约3个trusted core、2.4个ambiguous tail、约106个outside；outside loss仍为0。该分支只记录/隔离，没有产生有效direct open-set梯度。由于当前`U_s`来自已知类source池，把outside直接当unknown负样本会与DG目标冲突；正确动作应是隔离后进行receiver/channel不变性修复或延迟重路由，而不是强制unknown reject。
- `train_u_zid_invariance_active=0`覆盖16/16，终局全部报`UNLABELED_ZID_INVARIANCE_RUNTIME_INACTIVE`。因此C6的闭集提升不能归因于U_s身份/域解耦已经生效。
- receiver/day/channel泄漏probe excess分别约0.620-0.653、0.152-0.183、0.350-0.365；阈值为0.20/0.15/0.15。receiver和channel泄漏严重，day也全部超限，16/16均失败。

## 每候选同row结果

|candidate|seed|overall|strict|rx11 floor|sat mean/floor/strict floor|p95/p99/CVaR|source ovf/src-ep ovf|fixed/legacy proxy|fixed/legacy bridge|low-density|tail/overflow|ratio|leak rx/day/channel|
|---|---:|---:|---:|---:|---|---|---|---|---|---:|---|---:|---|
|`C0_DG_ANCHOR_S1`|712101|88.97|84.53|69.00|44.20/39.08/36.05|54.86/82.92/68.49|0.455/N/A|0.443/0.619|0.305/1.000|0.094|0.830/0.577|2.907|0.649/0.152/0.364|
|`C0_DG_ANCHOR_S2`|712211|88.89|84.11|72.09|42.85/38.27/35.24|54.42/82.74/69.66|0.452/N/A|0.379/0.623|0.134/1.000|0.090|0.822/0.587|3.045|0.652/0.183/0.365|
|`C1_LEAVE_DOMAIN_ONLY_S1`|712101|89.64|85.31|70.90|44.69/39.45/36.56|56.32/83.53/69.33|0.458/0.989|0.409/0.622|0.213/1.000|0.098|0.830/0.586|2.927|0.623/0.172/0.354|
|`C1_LEAVE_DOMAIN_ONLY_S2`|712211|89.15|84.95|74.72|43.18/38.44/35.45|51.26/82.38/67.53|0.462/0.988|0.378/0.621|0.187/1.000|0.091|0.806/0.521|2.915|0.620/0.181/0.359|
|`C2_LOCAL_NO_DENSITY_S1`|712101|89.60|85.28|70.47|44.86/39.44/36.56|56.43/83.29/69.72|0.463/0.989|0.444/0.623|0.321/1.000|0.097|0.826/0.595|2.913|0.642/0.174/0.355|
|`C2_LOCAL_NO_DENSITY_S2`|712211|89.16|84.99|75.04|43.23/38.45/35.44|50.71/82.22/67.20|0.463/0.988|0.378/0.620|0.177/1.000|0.091|0.812/0.544|3.197|0.627/0.182/0.360|
|`C3_LOCAL_BALANCED_S1`|712101|89.72|85.49|71.03|44.81/39.46/36.59|56.25/83.61/69.67|0.458/0.989|0.415/0.623|0.231/1.000|0.097|0.829/0.598|2.943|0.639/0.170/0.358|
|`C3_LOCAL_BALANCED_S2`|712211|89.20|85.07|74.82|43.31/38.55/35.58|50.91/82.53/67.53|0.459/0.988|0.375/0.623|0.175/1.000|0.091|0.804/0.520|2.963|0.623/0.174/0.356|
|`C4_LOCAL_STRONG_PROTECT_S1`|712101|89.60|85.27|70.16|44.90/39.46/36.53|56.33/83.48/69.43|0.461/0.990|0.448/0.621|0.331/1.000|0.097|0.827/0.577|2.853|0.639/0.173/0.359|
|`C4_LOCAL_STRONG_PROTECT_S2`|712211|89.30|85.19|74.63|43.19/38.45/35.46|50.87/82.40/67.43|0.463/0.988|0.369/0.618|0.148/1.000|0.092|0.815/0.531|2.955|0.639/0.178/0.350|
|`C5_SAT_GEOM_STRONG_S1`|712101|89.65|85.30|70.24|44.85/39.44/36.53|56.55/83.19/69.70|0.459/0.989|0.436/0.622|0.299/1.000|0.098|0.818/0.581|2.873|0.636/0.169/0.358|
|`C5_SAT_GEOM_STRONG_S2`|712211|89.14|84.97|74.91|43.17/38.44/35.44|50.89/82.33/67.23|0.461/0.988|0.368/0.621|0.148/1.000|0.091|0.804/0.523|2.925|0.634/0.177/0.362|
|`C6_U_DOMAIN_QUAR_S1`|712101|89.76|85.51|70.47|44.69/39.34/36.43|56.05/83.13/69.63|0.455/0.988|0.434/0.622|0.292/1.000|0.100|0.821/0.584|2.924|0.631/0.178/0.350|
|`C6_U_DOMAIN_QUAR_S2`|712211|89.18|85.05|75.09|43.29/38.54/35.55|50.64/82.42/67.25|0.455/0.987|0.373/0.621|0.168/1.000|0.091|0.804/0.510|2.954|0.620/0.176/0.359|
|`C7_FULL_JOINT_S1`|712101|89.48|85.03|69.58|45.00/39.56/36.65|56.18/83.45/69.64|0.459/0.989|0.450/0.620|0.335/1.000|0.098|0.828/0.583|2.880|0.622/0.174/0.357|
|`C7_FULL_JOINT_S2`|712211|89.11|84.91|74.80|43.26/38.50/35.51|50.73/82.18/67.05|0.462/0.988|0.373/0.616|0.161/1.000|0.092|0.812/0.526|2.959|0.631/0.175/0.353|

## 候选决策

|candidate cell|泛化结论|拒识潜力|主要风险|可否进入真实unknown评估|下一步动作|
|---|---|---|---|---|---|
|`C0_DG_ANCHOR`|恢复历史DG量级，但rx11和独立sat floor低|仅作同seed控制|无source/direct结构；legacy bridge=1|否|保留为DG控制|
|`C1_LEAVE_DOMAIN_ONLY`|两个seed均改善全部DG维度|fixed proxy/tail小幅改善|source overflow变差，p99与bridge不稳定，src-ep=0.988|否|保留为最简单安全正例|
|`C2_LOCAL_NO_DENSITY`|DG稳定改善|tail略降|proxy/bridge/ratio不优，source overflow变差|否|作为density消融负例|
|`C3_LOCAL_BALANCED`|本批最均衡DG单元|fixed proxy/tail小幅、跨seed同向改善|p99/source overflow/src-ep和legacy边界未改善|否|下一轮Phase1结构修复主基线|
|`C4_LOCAL_STRONG_PROTECT`|强teacher下DG保住|ratio与tail有小幅信号|更大权重仍未达到梯度下限，fixed proxy/bridge不稳定|否|证明单纯加权不够|
|`C5_SAT_GEOM_STRONG`|没有优于C3/C7的独立sat收益|tail/ratio小幅改善|sat p95仍约54、pair p95约82，channel泄漏高|否|星地几何诊断负例|
|`C6_U_DOMAIN_QUAR`|overall/strict与C3近似，source/tail/overflow略优|可作U_s安全利用诊断|U_s direct/invariance均为0，不能把收益归因于无标签解耦|否|修复U_s路由后再复验|
|`C7_FULL_JOINT`|sat mean/floor最高，但DG弱于C3|没有联合open收益|fixed proxy/bridge/source更差，U direct空转|否|不作为主推进候选|

## 历史比较

- 相对上一批P1：从63/64塌缩到16/16保持约89% overall和84%-85.5% strict，训练稳定性修复是明确成功。
- 相对ADV3B02：当前overall/strict相近或小幅更高，但receiver floor均值仍低于ADV3B02报告的75.55；当前p99约82.8，高于其约79.2，legacy proxy也没有优势。
- 相对OSFIX_PROXY_A：当前均值overall/strict低于其90.19/86.40；OSFIX的弱sat口径与当前独立full-physics口径不可直接比较；source-episode overflow仍未解决。
- 相对V2FIX8：当前overall/strict恢复到相近量级，rx floor略高于V2FIX8中位；但当前独立satellite指标不能与V2FIX同族弱增强sat floor约77直接比较。
- 指标实现和评估视图已变化，因此历史绝对值只能用于量级审查。最可信结论仍来自本批同seed C0配对差值。

## P0问题

1. **最终边界未闭环。**dynamic DM proxy/bridge很好看，但fixed和legacy风险不同步；16/16没有`endpoint_accept_v1`或prototype。
2. **source-only核心/尾部矛盾未解。**source-episode overflow约0.99，local component只是动态batch结构，仍没有稳定跨receiver/day invariant core memory。
3. **open梯度预算未达到。**最大scale下post share只有0.011-0.036，CE/KD/satellite目标仍决定主方向；单纯继续加lambda不会解决参数路径和冲突分配问题。
4. **星地分类与几何脱节。**clean p95约6度，sat p95约54度，pair p95约82度；模型仍通过宽域容纳星地视图，而非提取不变RFF。
5. **U_s没有进入目标路径。**U direct、U invariance和outside loss为0，三态几乎全部落入outside，C6/C7没有验证预期机制。
6. **域泄漏严重。**receiver/channel probe远超阈值，身份/域解耦失败。
7. **p99极端尾未消除。**clean p95很低但clean p99约83度，说明少量极端样本决定最终接收安全。
8. **梯度预算数值链不闭合。**非有限batch产生大于1的伪`B_os_eff`，scheduler又把所有子进程exit code 5统一汇总为`TERMINAL`；上游若只看scheduler状态会误判实验成功。
9. **绝对heldout存在teacher lineage污染。**父checkpoint由同一heldout-aware joint-safe路径选出，当前绝对DG值不能作为全新盲测证据。

## P1问题

1. tail reference需要5次post-U heavy observation，到E115才ready；绝对p99超限后不保存reference，导致所有`p99_delta`为空。应分离“诊断best reference”和“可导出safe reference”。
2. 当前独立satellite stress与历史弱增强sat口径不同，但矩阵成功标准仍写sat floor>=74；同时六场景floor混入只能作为diagnostic的storm/GEO/mixed，目标合同与`项目.md`不一致。
3. `source_overflow`、`source_episode_overflow`、dynamic DM、fixed endpoint和legacy proxy使用不同几何/阈值合同；必须版本化并明确同一拒识入口的主指标。
4. 只有两个paired seeds，且S1/S2的p95相差约5度、rx11 floor相差约3-5pp；open代理改善尚不具统计稳定性。
5. final-only保护了test协议，但没有独立DG validation strict曲线，不能在本批判断训练后期strict UDU是否回落。
6. 所有run都出现少量相同位置的非有限梯度batch跳步，虽未影响完成，仍需定位数据batch或loss分支的共同触发源。
7. 日志每epoch打印跳过测试的`NaN`占位，容易与真实NaN混淆，应改为`SKIPPED_NOT_EVALUATED`。
8. `protected_metric_snapshot.receiver_floor`实现为所有named test项的最小值，不是显式只在逐receiver项中取最小。本批恰好16/16都等于unseen-day rx11，但字段语义脆弱，后续应拆成`receiver_floor`与`split_floor`。
9. C0等消融单元故意关闭部分机制，却和主候选共享统一promotion readiness。应显式区分`DIAGNOSTIC_COMPLETE`、`TRAINING_HEALTH_PASS`和`PROMOTION_READY`，避免把预期消融阻断与训练失败混为一谈。
10. 16条全程只记录常数LR=`4.3e-5`，checkpoint没有scheduler state。若常数LR是设计选择，应写入协议；否则当前缺少后期几何收敛/冻结所需的可审计调度证据。

## 最终判断

当前实验对Phase1的贡献是：修复source-episode无界梯度导致的训练塌缩；所有机制单元相对同seed C0稳定小幅提高闭集DG、rx11和独立satellite stress；确认fixed tail accept存在小幅下降信号；用独立星地评估、泄漏probe和fail-closed export暴露了更真实的问题。

当前不能声明的是：真实unknown FAR/FPR95改善、最终拒识边界改善、U_s direct/invariance成功、receiver-aware local prototype成功、星地部署成功或Stage2/Phase3成功。

最主要风险是：source-episode overflow约0.99、legacy bridge=1、legacy proxy约0.62、p99约83度、satellite tail/overflow高、open梯度预算不足、U_s路径空转、receiver/channel泄漏严重、独立satellite floor仅约39%。

最值得保留的Phase1修复基线是`C3_LOCAL_BALANCED`；`C6_U_DOMAIN_QUAR`仅作为U_s路由修复后的诊断基线。当前不存在可直接推进的Stage2真实unknown候选。
