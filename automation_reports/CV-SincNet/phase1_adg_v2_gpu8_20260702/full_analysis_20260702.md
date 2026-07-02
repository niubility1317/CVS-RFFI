# Phase1 ADG-V2完整实验分析

分析时间：2026-07-02 15:15-15:35 Asia/Hong_Kong。

## 结论先行

当前ADG-V2八卡实验不能证明Phase1正在朝“可拒识的跨域泛化表征”稳定前进。最强闭集候选是`ADG8G7_STRONG_ALL_SAT_E200`，final strict UDU为84.42、receiver floor为77.03，接近旧`ADV3B02`；但它的proxy_vaccept升到0.520、source_episode_overflow仍有0.552、bridge_accept仍为1.0，且satellite strict floor只有45.44。也就是说，闭集DG局部保住了，拒识代理和星地压力鲁棒性没有过关。

本批最有价值的科学信息是负例：ADG新增bridge/shell/low-density/energy/radius/tail项没有打掉`bridge_accept=1.0`，tail/overflow治理反而把source_episode_overflow推高到0.55-0.69区间；G6/G7虽然把p99压到约60度，但proxy_vaccept与low-density accept同步变差。

## 协议边界

本批是Phase1 source-only地面训练。依据`AGENTS.md`与`项目.md`，本报告只能评价闭集DG能力、星地压力鲁棒性、known特征几何、proxy/virtual unknown风险和prototype导出质量。不得声明真实unknown_FAR、FPR95、Stage2 old_acc、seen_new_acc或H_old_new已经改善。proxy/virtual unknown只是训练期代理，不是真实unknown。

## 证据完整性与训练健康

远端状态文件显示8个候选均`exit=0`，每个候选均有200行`metrics_epoch.csv`与`phase2_zid_prototypes.json/.pt`。本地已回收8份metrics、8份prototype JSON、scheduler/status和7份stdout；`ADG8G7` stdout远端清单显示存在且约1.0MB，但因N607临时拒绝新SSH连接未成功拉回本地，因此G7 stdout全文扫描标为缺口。G7仍有完整CSV、prototype、scheduler finish和status exit=0。

|candidate|epoch/status|exit0|stdout|EPOCH-BEGIN|CONFIG-ADG|PROXY-ADG|Traceback/Runtime/Fatal|final nonfinite metric count|备注|
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
|ADG8G0_B02_ANCHOR_E200|200/200|True|本地完整|200|1|200|0/0/0|2.0|见CSV|
|ADG8G1_BRIDGE_CVAR_E200|200/200|True|本地完整|200|1|200|0/0/0|2.0|见CSV|
|ADG8G2_SHELL_LOW_DENS_E200|200/200|True|本地完整|200|1|200|0/0/0|4.0|见CSV|
|ADG8G3_ENERGY_Q10_E200|200/200|True|本地完整|200|1|200|0/0/0|4.0|见CSV|
|ADG8G4_RADIUS_RATIO_E200|200/200|True|本地完整|200|1|200|0/0/0|4.0|见CSV|
|ADG8G5_TAIL_OVERFLOW_E200|200/200|True|本地完整|200|1|200|0/0/0|4.0|见CSV|
|ADG8G6_CONSERVATIVE_ALL_E200|200/200|True|本地完整|200|1|200|0/0/0|2.0|见CSV|
|ADG8G7_STRONG_ALL_SAT_E200|200/200|True|远端存在但本地未拉回|0|0|0|0/0/0|4.0|见CSV|

stdout中的`nan`主要来自早期inactive proxy/sat telemetry和未运行test占位，不等同于训练崩溃；CSV层面`train_skipped_nonfinite_loss=0`，G2/G3/G4/G5/G7有少量`train_skipped_nonfinite_grad`，但未触发drop guard或PAIC guard。

## 泛化能力主表

参照旧`ADV3B02_CORE90_SOFT_E200`：overall=89.18，strict=84.89，receiver_floor=75.55，sat_strict_floor=68.77。参照旧`ADV3B30_SAT_STRONG_E200`：sat_strict_floor=69.84。

|candidate|overall|strict UDU|receiver_floor|sat_strict_floor|weak receiver|best-final strict gap|strict delta vs B02|sat floor delta vs B02|
|---|---:|---:|---:|---:|---|---:|---:|---:|
|ADG8G0_B02_ANCHOR_E200|84.84|77.28|68.74|43.63|rx8/68.74|0.33|-7.61|-25.14|
|ADG8G1_BRIDGE_CVAR_E200|86.41|78.56|82.10|48.83|rx7/82.10|-0.06|-6.33|-19.95|
|ADG8G2_SHELL_LOW_DENS_E200|86.39|80.72|74.36|49.36|rx7/74.36|0.22|-4.17|-19.41|
|ADG8G3_ENERGY_Q10_E200|85.87|79.59|66.78|44.63|rx8/66.78|1.59|-5.30|-24.14|
|ADG8G4_RADIUS_RATIO_E200|87.58|81.55|83.29|46.19|rx7/83.29|0.02|-3.34|-22.58|
|ADG8G5_TAIL_OVERFLOW_E200|87.01|81.20|76.13|43.86|rx8/76.13|0.10|-3.69|-24.91|
|ADG8G6_CONSERVATIVE_ALL_E200|86.18|77.94|70.72|43.95|rx8/70.72|0.16|-6.95|-24.82|
|ADG8G7_STRONG_ALL_SAT_E200|88.28|84.42|77.03|45.44|rx7/77.03|0.21|-0.47|-23.33|

泛化判断：`ADG8G7`是唯一接近旧B02 strict的候选，但satellite floor比旧B02低23.33pp、比旧B30低24.40pp；`ADG8G4`receiver floor最高但strict和satellite不足；G0/G1/G2/G3/G5/G6均不能说稳定提升。best-final gap普遍小，说明问题不是final偶然回落，而是训练全程没有形成高satellite floor。

## 拒识潜力主表

参照旧`ADV3B02`：proxy_vaccept=0.407，bridge=1.0，source_overflow=0.459，p99=79.16，component accept radius p95=7.38。

|candidate|p95|p99|r3sigma|tail_frac|min_inter|source_episode_overflow|ow_vac|proxy_auc|proxy_vaccept|hard_accept|shell|bridge|low_density|components|proto_r95|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ADG8G0_B02_ANCHOR_E200|59.92|75.44|54.94|0.048|67.32|0.623|0.111|0.727|0.408|0.289|0.075|1.000|0.008|33|15.00|
|ADG8G1_BRIDGE_CVAR_E200|60.50|76.92|57.05|0.043|67.09|0.611|0.141|0.728|0.398|0.292|0.084|1.000|0.009|33|14.36|
|ADG8G2_SHELL_LOW_DENS_E200|54.46|68.27|51.51|0.048|58.21|0.605|0.135|0.710|0.425|0.327|0.148|1.000|0.009|33|15.00|
|ADG8G3_ENERGY_Q10_E200|57.86|72.48|55.07|0.047|62.17|0.635|0.140|0.728|0.417|0.289|0.079|1.000|0.008|33|15.00|
|ADG8G4_RADIUS_RATIO_E200|58.18|74.99|55.95|0.047|66.64|0.610|0.129|0.720|0.403|0.305|0.085|1.000|0.008|33|12.93|
|ADG8G5_TAIL_OVERFLOW_E200|59.76|76.93|56.36|0.048|68.53|0.692|0.137|0.723|0.415|0.291|0.059|1.000|0.007|30|12.46|
|ADG8G6_CONSERVATIVE_ALL_E200|50.21|60.65|50.36|0.038|51.78|0.620|0.162|0.694|0.488|0.351|0.181|1.000|0.013|32|14.62|
|ADG8G7_STRONG_ALL_SAT_E200|50.91|60.82|51.30|0.029|48.72|0.552|0.180|0.673|0.520|0.378|0.191|1.000|0.013|33|15.00|

拒识判断：p99在G6/G7降到约60度、G2降到68度，但p95下降或p99下降不等于接收域安全。所有候选`bridge_accept=1.0`，source_episode_overflow全部高于旧B02，G6/G7的proxy_vaccept还明显高于旧B02。min_inter保持较高不能抵消bridge和tail风险，因为风险发生在tail、低密度区和same-class bridge，而不是全局类间中心距离。

prototype导出字段齐全，`fusion_accept_policy=local_component`且`global_ball_accept=false`，但训练态`component_radius_p95`仍达约50-60度，导出包通过15度radius cap截断；G0/G2/G7有18%-36%组件命中cap。结论是fusion字段存在，不等于fusion成功或known域真实收紧。

## 双目标四象限

|candidate|象限|依据|
|---|---|---|
|ADG8G0_B02_ANCHOR_E200|两者都不足|strict不足;receiver floor弱;satellite floor塌陷;bridge全接收;overflow高|
|ADG8G1_BRIDGE_CVAR_E200|两者都不足|strict不足;receiver floor可用;satellite floor塌陷;bridge全接收;overflow高|
|ADG8G2_SHELL_LOW_DENS_E200|两者都不足|strict不足;receiver floor弱;satellite floor塌陷;bridge全接收;overflow高|
|ADG8G3_ENERGY_Q10_E200|两者都不足|strict不足;receiver floor弱;satellite floor塌陷;bridge全接收;overflow高|
|ADG8G4_RADIUS_RATIO_E200|两者都不足|strict不足;receiver floor可用;satellite floor塌陷;bridge全接收;overflow高|
|ADG8G5_TAIL_OVERFLOW_E200|两者都不足|strict不足;receiver floor可用;satellite floor塌陷;bridge全接收;overflow高|
|ADG8G6_CONSERVATIVE_ALL_E200|两者都不足|strict不足;receiver floor弱;satellite floor塌陷;bridge全接收;overflow高;proxy_vaccept上升|
|ADG8G7_STRONG_ALL_SAT_E200|泛化提升但拒识风险未降/上升|strict可用;receiver floor可用;satellite floor塌陷;bridge全接收;overflow高;proxy_vaccept上升|

没有候选落入“泛化提升且拒识风险下降”的主推进象限。`ADG8G7`落入“泛化提升但open-set危险”象限；其余多为两者都不足或机制负例。

## 机制归因

|机制|是否生效|证据|
|---|---|---|
|source episode|未生效且方向相反|overflow从旧B02的0.459升到0.552-0.692；G5本应治理tail/overflow，却最高0.692。|
|bridge治理|未生效|G1/G6/G7开启bridge权重后`bridge_accept`仍为1.0，bridge loss还更大。|
|shell/low-density|只给出局部诊断|G2 p99降到68.27，但bridge、overflow、proxy_vaccept没有改善，satellite仍低。|
|energy q10|负例|G3 strict、receiver floor、satellite均弱，energy低分位没有带来proxy安全。|
|radius/inter-ratio|导出半径可压，但训练态风险未消失|G4 prototype_r95较低且receiver floor高，但bridge=1、overflow0.610、satellite46.19。|
|组合项|产生冲突|G6/G7压低p99，却提升proxy_vaccept和low_density accept；G7闭集强但open-set更危险。|
|satellite增强|失败|G7名义上有strong satellite guard，但satellite strict floor只有45.44，远低于旧B30的69.84。|
|longer/final稳定性|不是主因|多数best-final gap小，低satellite floor和高overflow是全程问题，不是final单点回落。|

## 失败模式

- 闭集增强但open-set风险上升：`ADG8G7` strict=84.42、receiver floor=77.03，但proxy_vaccept=0.520、bridge=1.0、overflow=0.552。
- p95/p99下降但overflow仍高：`ADG8G6/G7` p99约60度，但overflow仍0.620/0.552。
- min_inter高但proxy/bridge仍高：所有候选bridge=1.0，说明类中心间隔不是accept安全保证。
- source_episode_overflow升高：本批所有候选均高于旧B02的0.459，G5最高0.692。
- best checkpoint强但final退化：不是主要问题，最大strict gap为G3的1.59pp；更大的问题是satellite floor全程低。
- fusion flag存在但导出包不等于成功：local component字段齐全，但训练态component半径仍大，多个组件命中15度cap。
- satellite平均/强配置未修复弱receiver：G7闭集最强但satellite floor仍45.44，G4 receiver floor高但satellite46.19。

## 候选决策

|candidate|泛化结论|拒识潜力|主要风险|可否Stage2真实unknown评估|下一步动作|
|---|---|---|---|---|---|
|ADG8G0_B02_ANCHOR_E200|泛化整体弱于旧B02，strict仅77.28，satellite floor43.63|proxy_vaccept约等于旧B02，p99略降，但overflow升至0.623，bridge=1|不是旧B02严格复现；satellite崩；overflow扩大|否|先修复anchor一致性：holdout TX恢复1，确认代码改动/seed/ADG telemetry是否引入偏移|
|ADG8G1_BRIDGE_CVAR_E200|receiver floor高但strict78.57，satellite48.83|bridge权重开启但bridge仍1.0，overflow0.611|bridge治理未命中真实accept边界|否|审计bridge样本构造和loss尺度，改为显式same-class/inter-class bridge negative dry-run|
|ADG8G2_SHELL_LOW_DENS_E200|strict80.72、receiver floor74.36，仍低于主线|p99降到68.27，但proxy_vaccept升、overflow0.605、bridge=1|只压尾部角度，不压bridge/overflow；satellite49.36|否，可做机制诊断|保留shell/density作局部组件gate实验，不进入主推|
|ADG8G3_ENERGY_Q10_E200|strict79.60、receiver floor66.78，明显弱|energy q10损失未带来proxy改善，overflow0.635|弱receiver和satellite双塌|否|淘汰当前energy-q10单项；若保留，只作为能量边界负例|
|ADG8G4_RADIUS_RATIO_E200|receiver floor83.29最高，strict81.55中等，satellite46.19|prototype半径较小，但bridge=1、overflow0.610，训练态component半径仍大|半径收缩没有转化为拒识；satellite崩|否，可做hard-gate干跑|用冻结G4做local component gate dry-run，验证半径包是否只是导出cap|
|ADG8G5_TAIL_OVERFLOW_E200|strict81.20、receiver floor76.13，但satellite43.86|tail_accept_loss归零但source_episode_overflow最高0.692|tail/overflow机制方向相反|否|淘汰当前tail/overflow权重；重做core/tail/outside quarantine|
|ADG8G6_CONSERVATIVE_ALL_E200|strict77.94、receiver floor70.72，泛化不足|p99降至60.65，但proxy_vaccept0.489、low_density更高、bridge=1|闭集与proxy接收风险同步恶化|否|只保留“p99可压低”这个诊断，不作为候选|
|ADG8G7_STRONG_ALL_SAT_E200|本批闭集最强：strict84.42，receiver floor77.03；但satellite45.44|p99降至60.82，但proxy_vaccept0.520、overflow0.552、bridge=1|闭集强但open-set危险且satellite guard失败|仅限Stage2-A真实unknown诊断，不可主推|用G7做proxy-real gap评估，同时回滚satellite schedule/holdout设置修复|

## 下一轮实验设计

|实验组|目标|变量|指标|成功标准|失败判据|
|---|---|---|---|---|---|
|真实Stage2-A unknown query评估|检验Phase1 proxy风险是否映射到真实unknown|冻结`ADG8G7`、`ADG8G4`、旧`ADV3B02`；target receivers=`20-1,3-19,7-14,7-7,8-8`；unknown pairs=`10-1,10-10`与`1-16,4-10`；不训练、不用support调阈值|unknown_FAR、FPR95、AUROC、old closed acc、accepted acc、coverage/defer|`unknown_FAR<=0.05`且old drop<=3pp，或明确定位proxy-real gap|unknown低FAR只能靠拒绝old，或unknown仍高|
|hard gate/local component dry-run|验证prototype导出cap是否真能形成安全known域|对G7/G4/B02跑`local_component`、hard radius、density gate；core q=`0.75/0.80/0.85`，radius cap=`9/12/15deg`，tail auto-accept关闭|component radius p95/max、coverage、old drop、unknown FAR、per-receiver coverage|低FAR同时old coverage可接受，且弱receiver不过度掉线|半径cap只是压coverage，old被大面积拒绝|
|bridge negative训练小矩阵|专门打掉`bridge_accept=1.0`|same-class bridge、inter-class slerp midpoint、component shell negative；bridge权重`0.004/0.008`，holdout TX恢复1|bridge_accept、hard_proxy_accept、proxy_vaccept、strict、receiver floor|bridge<0.5，proxy_vaccept不高于0.40，strict下降<1pp|bridge仍>0.9或strict/receiver塌陷|
|core/tail/outside quarantine|阻止tail继续撑宽known接收域|core-only auto accept；tail只review/defer；outside进入source safe；overflow target=`0.18/0.25/0.35`|source_episode_overflow、p99、tail_frac、coverage、old closed acc|overflow<0.35，p99<70，strict>=83|overflow仍>0.45或old/receiver floor下降>2pp|
|source episode density gate|修复ADG把overflow升高的问题|`proxy_unknown_holdout_tx_per_batch=1/2/3`对照；density temp=`2/3/4deg`；source episode mix=`0.5/0.75`|source_episode_overflow、low_density_accept、p99、proxy_vaccept、strict|恢复到旧B02 overflow附近或更低，同时strict不掉|holdout增大继续扩大tail或proxy_vaccept升高|
|弱receiver与satellite stress修复|恢复B30/旧B02的satellite floor|回滚ADG强sat schedule，复用B30 schedule；rx7/rx8 reweight；sat late decay；固定reject项为G7低p99配置|sat strict floor、sat aggregate floor、rx7/rx8、strict UDU|sat strict floor>=68.5且receiver floor>=75|satellite平均升但弱receiver未修复，或拒识代理继续恶化|

## 最终判断

当前实验对Phase1的贡献是：给出了ADG-V2的强负例和局部诊断证据。G7说明闭集strict/receiver floor仍可维持，G6/G7说明p99可以被压低；但全批没有打掉bridge全接收，也没有压低source_episode_overflow，更没有保护satellite floor。

当前不能声明的是：真实unknown_FAR/FPR95改善、Stage2成功、Stage2 old_acc/seen_new_acc/H_old_new改善、fusion成功、ADG-V2主线成功。

最主要风险是：bridge_accept=1.0、source_episode_overflow=0.55-0.69、G6/G7 proxy_vaccept上升、satellite strict floor约43-49、prototype radius cap掩盖真实半径、G7 stdout未本地全文扫描。

最值得推进的候选是：没有主推进候选。`ADG8G7_STRONG_ALL_SAT_E200`只值得作为诊断候选进入真实Stage2-A unknown评估，因为它最接近旧B02闭集表现并压低p99；但它不同时满足泛化与拒识潜力，不能promote。

下一步最小验证是：冻结`ADG8G7/ADG8G4/ADV3B02`做真实Stage2-A unknown query dry-run；并行开bridge negative、core/tail/outside quarantine、source episode density gate和satellite weak receiver修复四个小矩阵。只有`bridge<0.5`、`source_overflow<0.35`、satellite floor恢复到约68+，且真实unknown FAR与old retention同时过关，才允许进入Stage2候选推进。

## 产物路径

- 原始回收小文件：`E:\type10-7\automation_reports\CV-SincNet\phase1_adg_v2_gpu8_20260702\artifacts\remote_complete_20260702_1515`
- 派生摘要CSV：`E:\type10-7\automation_reports\CV-SincNet\phase1_adg_v2_gpu8_20260702\artifacts\analysis_20260702_1515\adg_v2_candidate_summary.csv`
- 派生摘要JSON：`E:\type10-7\automation_reports\CV-SincNet\phase1_adg_v2_gpu8_20260702\artifacts\analysis_20260702_1515\adg_v2_analysis_summary.json`
