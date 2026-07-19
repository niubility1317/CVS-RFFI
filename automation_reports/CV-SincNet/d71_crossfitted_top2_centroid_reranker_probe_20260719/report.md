# D71交叉拟合top-2局部中心重排探针

## 1.执行前登记

- 实验ID：`d71_crossfitted_top2_centroid_reranker_probe_20260719`；operator：Codex；状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 比较目标D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D70最终与D62全部汇总/floor持平，但旧→新多2次且额外计算显著；提交`e9549c7e`。旧类行替换路线停止。
- 根目录`E:\type10-7`非Git；本报告镜像、代码、测试和追踪进入`E:\type10-7\github_publish\CVS-RFFI-repo`。只暂存D71拥有路径，不覆盖其他工作树改动。

## 2.方法锁与假设

D71始终保留D62全类joint分数，只允许经过两折support-held pair非劣门和全类TP/FP原子门的最近中心pair，在D62当前top-2内部交换次序。它不能引入第三类，不改D62单类行，不做全pair投票或全局score融合。pair公式统一、标签置换等变；K1与空mask精确D62。

假设：D62的主要损失来自少数局部碰撞，而不是全局类别几何整体错误。低方差pair中心只处理D62已认为最相近的两个候选，可能修复old4/old5/new1/new3，同时避免D64全pair锦标赛对全部决策的系统性改写。

## 3.数据、协议与资源

固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；复用D18`VALIDATED_ONCE/p2_min_v1` enrollment-only capsule，不重验数据。query只测试，no clean/source/query truth/role/quota/count/global assignment。ground输入锁0，因为D22仍`formal_phase2_eligible=false`。

部署目标为D62 int8/FP16 head加稀疏int8 pair方向；每query最多增加一个288D pair dot及top-2排序，状态上限256KiB，dense query graph为0。所有额外fit、MAC、pair状态、INT8/FP32差异和实测时延都必须单列。

## 4.验证、运行与停止门

先完成core的partition、pair方向、类置换、top-2第三类不变、原子门、K1/空mask、INT8/FP32和非法输入测试；再接入锁定D62、运行D42–D71回归链并在干净worktree复验。只有这些通过才登记真实105行命令。

真实完成后必须报告7候选、3场景、11类、15fold、接受pair、held TP/FP、训练20epoch、量化、资源、artifact及D62/D65–D70同row对照。相对D62若A/N/H/J/min-A/min-N或场景floor发生交换，或没有至少一项严格改善，则状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不跑第二seed或125，不扫描pair阈值/权重/温度。

## 5.实现与本地验证

- 新增`code/cvsrffi/stage2_d71_top2_centroid_reranker.py`：exact-once两折、最近中心pair方向、逐pair两类非劣门、全类TP/FP原子门、INT8/FP32稀疏pair状态和top-2交换器。
- 新增`code/scripts/probe_d71_crossfitted_top2_centroid_reranker.py`：接入锁定D62，保留D62 base state；对before/final分别登记pair gate/state，并显式增加适配MAC、query额外计算、pair状态和非单affine口径。
- 两个D71测试文件共12项，覆盖partition、pair registry、第三类不变、空门、K1、INT8/FP32、active gate、非法support、D62 audit包装、state identity、协议与调用闭包；12/12通过。
- D42–D71共36个测试文件、357/357通过，用时82.8s。pytest退出后仅出现Windows临时`pytest-current`清理`PermissionError`，命令exit0且全部测试已完成，判为已知包装清理噪声。
- 当前尚无真实outer性能；下一步提交精确D71文件，建立干净worktree并复跑357项。干净链通过后才登记105行命令。

## 6.干净版本与真实运行锁

- 实现提交`8599f5a4`；干净worktree`E:\type10-7\code\snapshots\d71wt`为detached HEAD，`git status -sb`仅`## HEAD (no branch)`。
- 干净D42–D71全链357/357通过，用时83.0s；同样只有exit0后的临时pytest目录清理噪声。
- 实际checkout SHA：probe`ed4323551f91ed2652cac1ae7969d96c43e85bc504f75dadbd8cdca6e6435986`、core`79e4f192d55673a6ca9b140ead4e881f45a611f07cc1233610937525112d7c8e`、D62 helper`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。
- 本地运行，不使用N607。输出目录已验证不存在；禁止覆盖或原目录重跑。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d71wt\code\scripts\probe_d71_crossfitted_top2_centroid_reranker.py' `
  --d71-arm crossfitted_top2_centroid_reranker `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d71wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d71_crossfitted_top2_centroid_reranker_probe_20260719\crossfitted_top2_centroid_reranker' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期闭包：105行、30个目标row、30次top fit、120次inner D62、2280条component fit；before/final各30个pair audit，所有partition exact-once。pair只能交换D62 top-2，ground/query-fit/clean/source/role/quota访问0。

## 7.真实运行完成状态

- 本地runner完成105/105行、exit0；Runner用时211.208s，含shell总用时218.705s；未使用N607。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d71_crossfitted_top2_centroid_reranker_probe_20260719\crossfitted_top2_centroid_reranker`。
- receipt状态`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，selection为`Z0_SUPPORT_ONLY`，query0；30次top fit、120次inner D62、2280条component fit、210次pair-aware score call闭合。
- INT8与matched FP32目标各15fold；二者outer指标、混淆和预测完全一致。

## 8.七候选同一行完整性能

B/A/N/H/F/J与min-B/A/N均为百分数；混淆依次为旧→新/新→旧/新→新。

|候选|B|A|N|H|F|J|min-B|min-A|min-N|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D42-USLDA-INT8，即D71|91.11|82.22|84.00|82.33|8.89|26.67|83.33|53.33|73.33|23/9/15|F下降来自B先下降；非改进|
|D42-USLDA-FP32-MATCHED|91.11|82.22|84.00|82.33|8.89|26.67|83.33|53.33|73.33|23/9/15|与INT8完全一致|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78|75.56|72.67|73.35|12.22|23.33|80.00|60.00|40.00|33/22/19|诊断基线|
|D42-D40-HNBR-INT8-NEGATIVE|85.56|85.00|15.33|25.16|0.56|0.00|66.67|63.33|0.00|2/0/0|低F但新类失效|
|D42-D41-BEC-INT8-NEGATIVE|86.11|20.56|78.67|31.50|65.56|0.00|76.67|0.00|36.67|142/0/32|旧类崩溃|
|D42-PROTOnet-CDA-ZID160|71.11|48.33|52.67|48.97|22.78|0.00|33.33|13.33|3.33|0/0/0|诊断基线|
|Z0_SUPPORT_ONLY|71.11|48.33|52.67|48.97|22.78|0.00|33.33|13.33|3.33|0/0/0|selection fallback|

## 9.场景、类别与十五fold

|场景|B|A|N|H|F|J|min-B|min-A|min-N|row-floor B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|95.00|91.67|96.00|93.57|3.33|50.00|80.00|70.00|90.00|70.00/60.00/80.00|2/2/0|
|leo_low_elev_weak|88.33|78.33|76.00|75.98|10.00|20.00|70.00|60.00|50.00|50.00/60.00/20.00|8/5/7|
|leo_rain_weak|90.00|76.67|80.00|77.45|13.33|10.00|70.00|30.00|70.00|60.00/30.00/30.00|13/2/8|

|类别|B或N-before|A或N-final|相对D62|观察|
|---|---:|---:|---:|---|
|old1|96.67|90.00|B/A均0|稳定|
|old2|96.67|90.00|B/A均0|稳定|
|old3|83.33|93.33|B−13.33pp，A0|before被pair重排伤害|
|old4|83.33|53.33|B+3.33pp，A0|终态瓶颈未修复|
|old5|93.33|73.33|0|第二旧类瓶颈未修复|
|old6|93.33|93.33|0|无均值遗忘|
|new1|—|73.33|0|全局min-N未修复|
|new2|—|93.33|0|稳定|
|new3|—|76.67|0|弱场景瓶颈未修复|
|new4|—|90.00|0|稳定|
|new5|—|86.67|−3.33pp|clear/fold1丢1个样本|

|场景/fold|B|A|N|H|F|J|row-floor B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear/0|100.00|100.00|90.00|94.74|0.00|50.00|100/100/50|0/1/0|
|clear/1|100.00|83.33|90.00|86.54|16.67|0.00|100/0/50|0/1/0|
|clear/2|91.67|83.33|100.00|90.91|8.33|50.00|50/50/100|1/0/0|
|clear/3|100.00|100.00|100.00|100.00|0.00|100.00|100/100/100|0/0/0|
|clear/4|83.33|91.67|100.00|95.65|-8.33|50.00|0/50/100|1/0/0|
|low/0|83.33|66.67|80.00|72.73|16.67|50.00|0/50/50|4/1/1|
|low/1|83.33|58.33|70.00|63.64|25.00|0.00|50/50/0|1/0/3|
|low/2|83.33|91.67|70.00|79.38|-8.33|0.00|50/50/0|0/2/1|
|low/3|100.00|100.00|70.00|82.35|0.00|0.00|100/100/0|0/1/2|
|low/4|91.67|75.00|90.00|81.82|16.67|50.00|50/50/50|3/1/0|
|rain/0|83.33|83.33|60.00|69.77|0.00|0.00|50/50/0|2/0/4|
|rain/1|100.00|66.67|90.00|76.60|33.33|0.00|100/0/50|4/1/0|
|rain/2|91.67|83.33|80.00|81.63|8.33|50.00|50/50/50|1/0/2|
|rain/3|83.33|75.00|90.00|81.82|8.33|0.00|50/0/50|3/0/1|
|rain/4|91.67|75.00|80.00|77.42|16.67|0.00|50/50/0|3/1/1|

## 10.pair行为、训练、量化与资源

- before：9/15折active、4折联合原子失败回退、2折无pair回退；共接受28个pair，单折min/mean/max=0/1.87/5。before的广泛重排使B下降1.67pp、row-before-floor下降13.33pp。
- final：12/15折联合原子失败精确回退，只有clear/fold1、2、3 active，共接受34个pair；low/rain终态15折中的10折全部精确D62。说明该机制没有触达真正弱场景。
- 唯一改变D62 final prediction hash的是clear/fold1：A保持83.33%，N从100%降到90%，H下降4.37pp，J保持0；旧→新从2降到0，但新→旧从0增到1。这是错误类型交换，不是联合改善。
- 30个before partition和30个final partition全部exact-once；outer-held/query不参与pair fit或gate。D71全进程记录38次reranked prediction，但多数发生在support审计或before，未形成终态收益。
- 20epoch训练与D62一致：epoch1 loss mean/min/max=1.0320/0.9732/1.1174、support acc95.14%、gradient norm1.0838；epoch20为0.1027/0.0756/0.1274、100%、0.1354；query rows始终0。
- 量化：INT8/FP32 before/final outer argmax变化0，support变化0，margin sign flip0；base score最大绝对误差min/mean/max=0.000377/0.000882/0.001915。pair方向量化误差final最大0.001022，bias最大0.0000295，未改变预测。
- 每target row总adaptation MAC=41,386,008,354，其中D71额外16,494,784,384；query由D62的6,624增至8,064，额外1,440。pair状态extra min/mean/max=0/698/4,312B；组合INT8状态8,583/9,281/12,895B，均低于256KiB。参数2,016、20step、峰值CUDA22,886,912B，dense query graph0。
- pair公式不使用class ID、old/new角色、scene/receiver、query truth、quota、batch count或global assignment；ground输入0，clean/source访问false。

## 11.与D62及近期版本比较

|版本|B|A|N|H|F|J|min-A|min-N|混淆|D71相对表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D62|92.78|82.22|84.67|82.62|10.56|26.67|53.33|73.33|23/8/15|D71 B−1.67、N−0.67、H−0.29、new→old+1；A/J/floor不变|
|D65|92.22|86.11|59.33|67.12|6.11|16.67|70.00|46.67|16/28/33|D71保新类明显，但A低3.89pp|
|D66|93.33|83.33|83.33|82.59|10.00|23.33|53.33|66.67|20/9/16|D71 N+0.67、J+3.33，但B−2.22、A−1.11、H−0.26|
|D67|92.78|82.78|83.33|82.16|10.00|26.67|53.33|73.33|22/11/14|D71 N+0.67、H+0.18，但B−1.67、A−0.56|
|D68|58.89|51.67|14.00|18.66|7.22|0.00|43.33|0.00|20/118/11|D68低F来自整体塌缩，不是可取基线|
|D69|92.78|81.67|74.67|77.39|11.11|30.00|53.33|53.33|27/23/15|D71 A/N/H/F更好，但J−3.33、B−1.67|
|D70|92.78|82.22|84.67|82.62|10.56|26.67|53.33|73.33|25/8/15|同D62主指标；D71 N/H更低|

D71的F改善1.67pp完全等于B下降1.67pp，而A没有增加。这种“缩短before与after差距”的方式不构成抗遗忘成功。D62仍是当前联合最强。

## 12.目标差距、ground边界与缺陷

- K10/new5门要求A>=92%、min-A>=88%、N>=92%；D71为82.22%、53.33%、84.00%，分别差9.78pp、34.67pp、8.00pp。未过development gate，不运行第二seed或125。
- D71 ground input严格为0，没有使用地面压缩旧类原型。D66仍是唯一实际消费84个int8 domain-class cell的近期版本，但其manifest无正式资格且产生负交换；不能绕过协议把它接入D71。
- 核心缺陷：pair gate对before较宽松，破坏注册前旧类；对final在low/rain全部原子回退，只在最容易的clear激活，并把一个new5样本改错为旧类。support pair受限正确数不能预测弱场景outer收益。
- 停止D71的top-2 pair中心、pair阈值/权重/温度及pairwise kNN变体；不把“旧→新减少2次”脱离同一行的新→旧+1、N−10pp来宣传。

## 13.artifact封存

|artifact|bytes|SHA256|
|---|---:|---|
|training_log.jsonl|14,833,131|`b9cccc24a58bf134fdcb839e06b40577569ac70bd348b6df85ed3d906d4c2003`|
|support_audit.json|313,676|`a9ab8153ff5cf7552bb9e34fbed69aea37051b17591cb78bb99c66e89e628067`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|selection.json|2,990|`fe4dfb86ce06102c3ebe277c94083b4ae426169562016b42aae773eab530f99f`|
|RECEIPT.json|5,030|`18a04e6eaa65af777d21020ef2e985052c8f9dcc572bc91c7b70d09768ebbb93`|
|D71_PROBE_METADATA.json|2,404|`41b543a7c977aae7fa4684ded74749238eecd3221b5a0c23c98c115528bf8723`|
|d71_full_performance_summary.json|132,319|`7181b45081774b46eee11262fd6faf818a44f032e094129032e2fdcc664ea05d`|

## 14.最终判定与下一步

状态`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D71确实改变了决策，并证明top-2局部重排能在support gate下减少部分旧→新错误；但它未提高after-old，降低了before-old、seen-new和H，且没有触达low/rain终态。停止pair重排路线。D62继续是当前最强：92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
