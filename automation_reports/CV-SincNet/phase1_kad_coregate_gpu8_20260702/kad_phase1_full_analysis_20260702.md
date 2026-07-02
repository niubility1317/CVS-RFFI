# KAD Phase1地面训练全量分析

分析时间:2026-07-02 23:44-23:58 CST  
分析对象:`phase1_kad_coregate_gpu8_20260702`和`phase1_kad_hardening_secondlane_gpu8_20260702`  
远端根目录:`/home/szu2070436088/2510044040/CV-SincNet`  
本地证据目录:`E:\type10-7\automation_reports\CV-SincNet\phase1_kad_coregate_gpu8_20260702\analysis_20260702_2344`

## 一、协议边界

KAD是Phase1 source-only地面训练实验。当前证据只能支持以下判断:闭集DG能力、strict UDU/receiver floor/星地压力鲁棒性、known特征几何、proxy/virtual unknown风险、prototype导出质量。不能声明真实`unknown_FAR`、`FPR95`、Stage2 `old_acc`、`seen_new_acc`或`H_old_new`已改善。

KAD8和KAD16H均使用source训练数据与source派生satellite stress，不使用target receiver support/query，不构成Stage2-A/B/C评估。proxy unknown和virtual unknown只能作为Phase1拒识代理风险，不是真实unknown。

## 二、训练健康与证据完整性

|证据项|结论|
|---|---|
|候选数|KAD8 8个候选，KAD16H 8个候选，共16个。|
|完成度|16/16均完成`E200/200`，每个`metrics_epoch.csv`均为200行。|
|stdout完整扫描|每个候选stdout约8717行，`[EPOCH-END]`出现200次，fatal scan为0。|
|NaN分类|每个stdout有858条NaN/Inf相关行；样例集中在早期未激活或零样本字段，如`proxy active=0`、`TEST overall_tx=nan% (0/0)`、`sat_cos=nan`、`grad=nan`。final `nonfinite_test_metric_count=0`。|
|metrics完整性|每个候选均有closed-set、strict UDU、per-receiver、satellite、known几何、proxy unknown、source episode和prototype相关字段。|
|prototype导出|16/16均导出`phase2_zid_prototypes.json/.pt`；JSON含`fusion_components`、`fusion_accept_policy=local_component`、`global_fused_radius_is_accept_region=False`。|
|best/final|多数候选final不是best row；KAD16H1 final触发`one_epoch_drop` guard，best checkpoint强但final退化。|
|SSH状态|只读巡检与全量解析后，本地无残留`ssh.exe`，无N607/bridge的`ESTABLISHED:22`连接。|

本次生成的机器可复查表:

- `kad_full_log_metrics_summary.json`:全量CSV+stdout+prototype摘要。
- `kad_generalization_table.csv`:泛化主表。
- `kad_rejection_table.csv`:拒识代理主表。
- `kad_pair_deltas.csv`:KAD16H相对同GPU KAD8的差值。
- `kad_health_table.csv`:训练健康表。

## 三、泛化能力主表

数值为final epoch同row。`receiver_floor`取rx7-rx11 floor；`UDU_floor`取unseen-day per-rx floor；`sat_floor`取`leo_clear_weak/low_elev_weak/rain_weak`三视图aggregate floor。

|candidate|overall|strict UDU|receiver_floor|UDU_floor|sat_mean|sat_floor|best-final风险|
|---|---:|---:|---:|---:|---:|---:|---|
|KAD8G0_COREGATE_ANCHOR|88.04|82.14|74.73|73.50|72.30|71.23|低，best-final gap小。|
|KAD8G1_HOLDOUT_STRESS|84.65|75.85|72.45|68.47|47.63|46.52|低，但sat floor弱。|
|KAD8G2_BRIDGE_CVAR|81.77|72.44|58.49|48.91|49.69|48.78|receiver gap 5.03，弱receiver未修复。|
|KAD8G3_SOURCE_OVERFLOW|86.29|81.00|77.27|71.22|70.40|69.34|best-final strict gap 2.15，有回落。|
|KAD8G4_LOW_DENSITY_GATE|86.71|81.17|70.54|64.57|68.72|67.45|低到中。|
|KAD8G5_ENERGY_MARGIN_Q05|86.80|80.64|70.12|70.74|68.50|67.41|低到中。|
|KAD8G6_RADIUS_INTER_BUDGET|85.90|78.36|73.38|68.33|46.97|45.85|sat floor弱。|
|KAD8G7_COMBINED_SAT_REPAIR|84.73|78.21|68.83|65.28|52.44|51.17|receiver gap 5.35，sat仍弱。|
|KAD16H0_HARDENED_DEFAULT|85.69|79.79|73.83|68.05|66.95|65.87|低；相对KAD8G0全面下降。|
|KAD16H1_THREESIGMA_NEGCTRL|86.02|80.16|73.89|66.08|73.72|72.58|高；best-final overall gap 2.40、strict gap 3.99，guard失败。|
|KAD16H2_BRIDGE_COREQ75|85.64|79.38|78.18|69.21|48.83|48.07|closed/floor强，sat floor弱。|
|KAD16H3_SOURCE_COREQ75_QUAR|86.69|78.23|77.32|70.71|70.56|69.39|低；overall/floor稳，strict低于KAD8G3。|
|KAD16H4_TAIL_SENTINEL_GUARD|87.01|80.51|74.79|69.30|67.38|66.38|中；receiver gap 2.56。|
|KAD16H5_PROXY_ONLY_BOUNDARY|84.00|80.08|70.18|65.35|62.67|61.83|中；整体和sat下降。|
|KAD16H6_P80_RADIUS_BUDGET|83.02|75.08|67.39|55.67|46.17|45.32|低但整体弱。|
|KAD16H7_HARDENED_COMBINED_SAT|83.53|73.38|65.86|57.07|45.96|45.21|高；strict gap 3.36，sat弱。|

泛化结论:

- 最强闭集DG仍是KAD8G0:overall、strict UDU、satellite mean/floor最均衡。
- KAD16H没有带来稳定广义提升。只有KAD16H1在satellite mean/floor上显著提升，但它是three-sigma负控且final guard失败；不能作为主推进证据。
- receiver floor修复发生在KAD8G3/KAD16H2/KAD16H3/KAD16H4，但不是全维度修复。KAD16H2 receiver floor高达78.18，却sat floor只有48.07，说明receiver floor和星地压力没有同步受保护。
- satellite增强没有稳定保护弱receiver。G1/H1改善的是sat平均/地板，但H1的UDU_floor降到66.08；H6/H7继续低sat floor，combined sat repair未成功。

## 四、拒识潜力主表

数值为final epoch同row。`bridge_accept=1.0`代表same-class bridge代理仍全部被接收，是严重失败信号。

|candidate|p95|p99|tail_frac|source_overflow|proxy_vaccept|proxy_accept|bridge_accept|component_p95|radius/inter|判断|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|KAD8G0|49.04|59.14|0.028|0.516|0.671|0.361|1.000|49.00|1.23|闭集强但accept域宽。|
|KAD8G1|67.19|78.89|0.053|0.725|0.602|0.066|1.000|66.51|1.38|tail/overflow过大。|
|KAD8G2|66.08|78.84|0.060|0.721|0.546|0.099|1.000|65.79|1.41|proxy稍好但geometry很宽。|
|KAD8G3|44.73|50.65|0.023|0.501|0.636|0.524|1.000|44.50|1.49|p95/p99好，但proxy接收极高。|
|KAD8G4|43.32|49.17|0.012|0.460|0.628|0.435|1.000|43.31|1.33|tail最短之一，但open-set仍危险。|
|KAD8G5|49.82|59.89|0.035|0.514|0.663|0.301|1.000|49.76|1.35|energy不够，proxy仍宽。|
|KAD8G6|66.24|78.28|0.048|0.698|0.600|0.117|1.000|65.20|1.64|半径/inter过高。|
|KAD8G7|70.29|80.91|0.060|0.805|0.608|0.115|1.000|69.71|1.45|combined导致tail/overflow最大。|
|KAD16H0|53.91|72.99|0.044|0.593|0.534|0.060|1.000|54.13|1.06|proxy改善但p99/overflow扩大。|
|KAD16H1|58.22|79.84|0.050|0.501|0.413|0.003|1.000|57.73|0.99|proxy最佳，但three-sigma负控和final退化。|
|KAD16H2|65.60|78.46|0.051|0.711|0.593|0.130|1.000|65.41|1.52|receiver强但accept域仍宽。|
|KAD16H3|59.83|78.87|0.053|0.666|0.588|0.026|1.000|59.58|1.05|proxy_accept低，但tail/overflow恶化。|
|KAD16H4|47.45|55.68|0.037|0.464|0.655|0.421|1.000|47.18|1.37|tail一般，proxy失败。|
|KAD16H5|49.72|60.82|0.033|0.541|0.646|0.269|1.000|49.63|1.35|proxy-only没有收紧accept域。|
|KAD16H6|66.83|79.52|0.050|0.704|0.591|0.076|1.000|66.61|1.43|radius budget未降半径。|
|KAD16H7|67.44|80.31|0.057|0.782|0.614|0.089|1.000|66.65|1.38|combined仍是tail/overflow负例。|

拒识结论:

- 没有候选达到“known域安全收紧”。所有候选`bridge_accept_rate=1.0`，说明same-class bridge代理仍全部被当作known接收。
- proxy_vaccept最低的是KAD16H1(0.413)，但仍远高于可拒识表征应有的风险水平，而且它依赖three-sigma负控并出现final退化。
- p95下降不能说明安全。KAD8G4 p95=43.32、p99=49.17、tail_frac=0.012是最紧的一组，但proxy_vaccept=0.628、proxy_accept=0.435，open-set代理仍失败。
- min_inter高不等于拒识成功。KAD16H1 min_inter=81.96、KAD16H3=78.29，但bridge_accept仍为1.0。
- source_episode_overflow普遍偏高，范围约0.460-0.805。source episode仍在把困难source样本合法化为known包络，未形成core/tail/outside治理。

## 五、双目标冲突四象限

判定以同GPU KAD16H对KAD8差值为主，KAD8内部以G0为anchor。

|候选|象限|理由|
|---|---|---|
|KAD8G0|闭集强但open-set危险|当前闭集anchor最强，但proxy_vaccept=0.671、source_overflow=0.516。|
|KAD8G1|两者都差|satellite floor弱，tail/overflow高。|
|KAD8G2|泛化下降但部分proxy改善|proxy_vaccept低于G0，但overall/strict/receiver/sat均弱。|
|KAD8G3|泛化部分修复但open-set危险|receiver floor好，p95/p99短，但proxy_accept=0.524。|
|KAD8G4|拒识几何改善但open-set危险|tail最短，闭集还可，但proxy_vaccept和bridge失败。|
|KAD8G5|中等闭集但open-set危险|energy margin未明显降低accept风险。|
|KAD8G6|两者都差|satellite floor与半径/inter均弱。|
|KAD8G7|两者都差|combined sat repair未修sat，tail/overflow最大。|
|KAD16H0|泛化下降但proxy代理改善|proxy_accept和vaccept下降，但p99、tail、source_overflow上升。|
|KAD16H1|泛化提升但协议/稳定风险大|satellite和proxy最佳，但three-sigma负控、best-final退化、bridge仍失败。|
|KAD16H2|receiver泛化提升但open-set危险|overall/strict/receiver对G2大幅提升，sat弱且proxy_vaccept上升。|
|KAD16H3|闭集稳但tail扩大|overall/floor小升，strict下降，p99和source_overflow显著恶化。|
|KAD16H4|弱receiver修复但拒识风险上升|overall/floor小升，proxy_vaccept和tail均变差。|
|KAD16H5|泛化下降且拒识未解决|proxy_accept略降但vaccept仍0.646，sat下降。|
|KAD16H6|两者都差|闭集、sat和tail均弱。|
|KAD16H7|两者都差|combined强化没有修sat，source_overflow仍0.782。|

没有候选落入“泛化提升且拒识风险下降”的主推进象限。最接近的是KAD16H1，但它的机制是three-sigma负控，且final guard失败，只能作为诊断性对照。

## 六、机制归因

1. source episode仍偏包容尾部。KAD16H0从KAD8G0的source_overflow 0.516升到0.593；H3虽启用core quantile quarantine，但source_overflow仍升到0.666。说明仅把半径模式改成core相关不足以把tail排除出accept治理。
2. vacuum/proxy unknown能降低部分proxy_accept，但不能解决bridge。H1 proxy_accept降到0.003、proxy_vaccept降到0.413，但bridge_accept仍为1.0。proxy被推走不等于真实known tail被治理。
3. fusion/local component字段已导出，但只证明字段存在，不证明拒识成功。16个prototype JSON均为`fusion_accept_policy=local_component`、`global_fused_radius_is_accept_region=False`、6个components；但component半径和bridge_accept仍显示宽边界风险。
4. longer epoch出现过拟合/回落。H1 best epoch 198，final overall下降2.40pp、strict下降3.99pp并触发one-epoch drop guard；H7 strict gap 3.36pp。不能只看best行。
5. satellite增强只改善局部平均，不保护所有弱receiver。H1 sat_floor=72.58最高，但UDU_floor=66.08低于G1，且final guard失败；H7 combined sat floor只有45.21。
6. pseudo-label未见明显崩溃，但不是当前主要矛盾。final test nonfinite=0，训练完成；主要瓶颈在known accept域治理，而非训练崩溃。

## 七、失败模式清单

- 闭集增强但open-set风险上升:H2对G2 overall/strict/receiver提升明显，但proxy_vaccept从0.546升到0.593，radius/inter从1.406升到1.518。
- p95下降但p99/overflow仍高:G4 p95=43.32但proxy_vaccept=0.628；H1 p95下降到58.22但p99=79.84、bridge_accept=1.0。
- min_inter高但proxy_vaccept仍高:H3 min_inter=78.29但proxy_vaccept=0.588；H1 min_inter=81.96但bridge_accept=1.0。
- source_episode_overflow升高:H0、H3、H6相对对应KAD8均升高；G7/H7绝对值最高。
- best checkpoint强但final退化:H1 final guard失败；H7 strict gap 3.36；G3 strict gap 2.15。
- fusion flag存在但不能等价于融合成功:JSON字段齐全，但没有拒识代理指标支撑。
- satellite平均提升但弱receiver未修复:H1 sat_mean高，但UDU_floor仅66.08；H2 receiver floor高但sat_floor仅48.07。

## 八、候选决策

|candidate|泛化结论|拒识潜力|主要风险|可否Stage2真实unknown评估|下一步动作|
|---|---|---|---|---|---|
|KAD8G0|当前闭集DG anchor最稳。|弱，accept域宽。|proxy_vaccept高、source_overflow高。|可作为Stage2诊断baseline，不可声明推进。|跑真实unknown query基线，验证Phase1风险是否转化为FAR。|
|KAD8G1|sat弱，非主线。|弱，tail/overflow高。|sat floor 46.52。|低优先。|保留为holdout stress负例。|
|KAD8G2|receiver/sat均弱。|proxy略好但geometry宽。|receiver floor 58.49。|低优先。|只用于bridge loss负例分析。|
|KAD8G3|receiver floor强，strict尚可。|p95/p99较好但proxy失败。|proxy_accept 0.524、bridge=1。|可做诊断Stage2。|加shell/inter-class/bridge negative重新跑。|
|KAD8G4|tail最短之一。|geometry好但proxy失败。|proxy_vaccept 0.628。|可做诊断Stage2。|验证真实unknown是否仍被低密度区接收。|
|KAD8G5|闭集中等。|energy margin未明显收紧。|proxy_vaccept 0.663。|低优先。|重设计energy margin与最终gate同口径。|
|KAD8G6|sat弱。|半径/inter高。|radius/inter 1.64。|不建议。|淘汰或只作radius budget负例。|
|KAD8G7|combined未生效。|tail/overflow最大。|source_overflow 0.805。|不建议。|停止沿该combined配置扩大。|
|KAD16H0|泛化较G0下降。|proxy_accept/vaccept下降但tail扩大。|p99和source_overflow恶化。|可作硬化默认诊断。|加入hard gate防止tail扩大。|
|KAD16H1|satellite和proxy代理最好。|proxy最佳但仍不安全。|three-sigma负控、final guard失败、bridge=1。|可作诊断，不可主推。|用core gate复刻其sat收益，禁用three-sigma accept。|
|KAD16H2|receiver floor显著提升。|拒识风险上升。|sat弱、proxy_vaccept上升。|可作弱receiver诊断。|将bridge loss改为直接accept惩罚。|
|KAD16H3|overall/floor稳。|tail/overflow恶化。|source_overflow 0.666。|低优先诊断。|强化source episode density gate。|
|KAD16H4|弱receiver有改善。|proxy失败。|tail与proxy_vaccept升高。|低优先诊断。|tail sentinel改成quarantine而非guard名义项。|
|KAD16H5|泛化下降。|proxy-only无实质收紧。|vaccept 0.646。|不建议。|淘汰proxy-only路线。|
|KAD16H6|泛化与sat均弱。|radius budget未降风险。|sat floor 45.32。|不建议。|淘汰或重写radius budget定义。|
|KAD16H7|combined失败。|tail/overflow仍高。|sat floor 45.21、source_overflow 0.782。|不建议。|停止该combined sweep。|

## 九、下一轮实验设计

|实验组|目标|变量|指标|成功标准|失败判据|
|---|---|---|---|---|---|
|E1 hard gate/local component dry-run|验证最终accept gate是否真的由core component控制。|component radius只用core q70/q75/q80；禁用p95/p99/three-sigma accept；global radius永不作为accept。|component_accept_radius、proxy_vaccept、bridge_accept、true Stage2 unknown FAR诊断。|bridge_accept<0.3、proxy_vaccept<0.2，overall下降<1pp。|p95变小但bridge或true unknown仍高。|
|E2真实Stage2 unknown query诊断|检验Phase1代理风险是否转化为真实unknown拒识风险。|候选:G0、G4、H0、H1、H2、H3；固定prototype，不做target调阈。|unknown FAR、FPR95、AUROC、old target accepted/full acc。|G4/H1若proxy低但真实FAR仍高，判proxy不足。|unknown query被用于阈值拟合或声明Stage2成功。|
|E3 shell/inter-class/same-class bridge negative|直接打类间低密度与same-class bridge。|shell negative采样、inter-class convex negative、same-class cross-domain bridge negative；CVaR accept惩罚。|bridge_accept、low_density_accept、radius/inter、p99。|bridge_accept明显下降且strict UDU不降>1pp。|min_inter上升但bridge_accept仍=1。|
|E4 core/tail/outside quarantine|把known治理分层，而不是全部合法化。|core用于accept半径；tail只分类保真；overflow进入quarantine loss。|tail_frac、source_overflow、p99、component_radius_max。|source_overflow<0.35、p99下降，receiver floor不降>1pp。|tail被导出为local known component。|
|E5 source episode density gate|修复source episode包容尾部。|leave-domain query按density/core过滤；tail query只保分类不扩半径；overflow直接惩罚accept。|source_episode_overflow、source_episode_tail_query_rate、strict UDU。|overflow下降且strict UDU维持。|泛化靠扩大shell维持，overflow不降。|
|E6弱receiver satellite stress修复|解决sat平均与receiver floor脱钩。|针对rx弱点的sat stress mix、per-rx quota、sat floor guard。|receiver_floor、UDU_floor、sat_floor、best-final gap。|sat_floor和UDU_floor同步提升，final guard通过。|sat_mean升但UDU_floor或弱receiver继续低。|

## 最终判断

当前实验对Phase1的贡献是:KAD8提供了强闭集DG anchor，KAD16H证明了部分硬化项可以降低proxy_accept/proxy_vaccept或提升receiver floor，但没有形成同时满足跨域泛化与known accept域收紧的主推进候选。

当前不能声明的是:真实`unknown_FAR`改善、`FPR95`改善、Stage2成功、新类注册成功、old/new校准改善、fusion成功。local component字段存在只说明导出包具备字段，不说明open-set gate已经安全。

最主要风险是:所有候选`bridge_accept=1.0`，proxy_vaccept仍偏高，source_episode_overflow普遍高，p99/tail和closed-set收益经常冲突，KAD16H1存在best-final gap和three-sigma负控风险。

最值得推进的候选是:没有正式主推进候选。诊断优先级为KAD8G0作为闭集baseline、KAD8G4作为紧tail但proxy失败样本、KAD16H1作为three-sigma/proxy负控、KAD16H2作为receiver floor修复负控、KAD16H0作为硬化默认对照。

下一步最小验证是:E2真实Stage2 unknown query诊断+E1 hard gate dry-run。先确认真实unknown是否被当前wide accept域接收，再决定是否扩大E3/E4/E5训练矩阵。
