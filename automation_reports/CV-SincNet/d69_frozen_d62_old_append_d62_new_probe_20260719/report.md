# D69冻结D62旧行并追加同族新行探针

## 1.执行前登记

- 实验ID：`d69_frozen_d62_old_append_d62_new_probe_20260719`；operator：Codex；最终状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 目标：保留当前联合最强D62的绝对跨类尺度，检验D65式Stage2-B旧行冻结能否减少注册遗忘，同时由D62同族final head提供新类行。
- 当前最强D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D68已完成105/105行并以B/A/N/H=58.89/51.67/14.00/18.66否决；其低F=7.22是注册前B先塌陷形成的伪改善。D68最终证据提交为`19c4603b`。
- 根目录`E:\type10-7`非Git；代码、追踪和本报告镜像进入`E:\type10-7\github_publish\CVS-RFFI-repo`。工作树中其他大量改动与D69无关，提交只暂存D69拥有路径。

## 2.唯一方法锁

Stage2-B执行完整D62并冻结6个旧类行`(W_B,b_B)`。Stage2-C在11类support上执行同一D62得到`(W_C,b_C)`，只追加其中5个新类行：

```text
W_final=concat(W_B[old],W_C[new])
b_final=concat(b_B[old],b_C[new])
```

不做逐行标准化、符号翻转、alpha融合、温度、offset、角色门、class名单、scene/receiver分支或超参数扫描。K1沿用D62自身精确D46 fallback。最终仍为一个全注册类affine head。

## 3.假设、可观察结果与停止条件

- 假设：D62的绝对行尺度已经包含有效joint竞争信息；只冻结旧行而让新行来自同族D62 final，可能比D65的异族block-LDA追加更兼容。
- before state、预测和全部指标必须与D62匹配；final旧FP32行与before逐bit相同，final新FP32行与D62 final逐bit相同。
- 相对D62必须无A/N/H/J/min-A/min-N交换，并至少严格改善A、F、J或floor之一；否则首seed即停止。
- INT8相对matched FP32的before/final argmax变化及margin sign flip必须为0；资源须保持正式上限。
- 真实105行完成后详细报告全部候选、场景、类、fold、混淆、训练、量化、资源、artifact和同排历史对照。失败不做第二seed或125。

## 4.数据与协议

- 固定development cell：receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8。
- 复用D18`VALIDATED_ONCE/p2_min_v1`enrollment-only capsule；方法变化不触发数据重验。
- query只评分一次且不参与拟合；每query独立面对全部已注册类。clean/source、role Oracle、quota、batch assignment和dense query graph均禁止。
- ground实际输入锁为0。D22尚未达到正式Phase2资格；D66读取84个int8 cell仍为负交换，D69不以协议无效依赖换取旧类指标。

## 5.实施计划

新增独立D69 lifecycle wrapper、probe和专项测试，不修改D62历史实现或artifact。先验证：对称support、before精确D62、旧行bitwise冻结、新行精确D62 final、类置换等变、K1 fallback、调用配对、量化state旧行不变、禁止分支和资源闭包；随后运行D42–D69完整链。代码验证、提交和干净worktree复跑后，才登记并执行真实105行命令。

## 6.实现与本地验证

- `code/cvsrffi/stage2_d69_frozen_d62_append.py`：纯生命周期core，Stage2-B缓存D62旧行，Stage2-C只追加joint D62新行，记录旧/新行hash、支持准确率和禁止访问审计。
- `code/scripts/probe_d69_frozen_d62_old_append_d62_new.py`：复用锁定D62数学实现与D42 runner，增加编译后INT8/FP32旧state逐bit检查、资源闭包、source closure和D69 metadata。
- `tests/test_stage2_d69_frozen_d62_append.py`、`tests/test_probe_d69_frozen_d62_old_append_d62_new.py`：10项专项，覆盖before精确同D62、K1、append identity、新support不能改旧行、类置换、非法support、生命周期、state全部字段及禁止分支。
- 首次测试命令因本机Conda实际位于`F:\App\miniconda3`而不是旧路径失败，未进入pytest；改用正确hook并显式`conda activate ssr-gpu`。随后类置换测试的验证索引把“原类→新类”误当成逆置换，修正测试索引后专项10/10通过；算法实现未为此改动或放宽。
- D42–D69完整链335/335通过，用时81.1s；包含D42集成测试20项和D43–D69全部相关专项。测试运行目录为`local_artifacts/d69_full_chain_335`。
- source SHA256：core`bb59c3828ce63cdd168c00fe26a2ca82a2d7a37fade8105dc57f1dc9e6ec3bbf`；probe`f999f02523d150eb11e3cb872acd3df35271b44a4e9df8fc21cbce325c2f37a1`；未修改的D62 helper`c685e60402b5b172a0b2ed77e647e3aa506048b759f637fd88dcfb31ca114bcd`。

当前只有代码和合成验证，尚无真实outer性能。下一步提交精确文件，建立干净worktree并复跑335项；真实运行前继续保持状态`PREREGISTERED_IMPLEMENTATION_VALIDATED_PERFORMANCE_PENDING`。

## 7.干净版本与真实运行命令

- 实现提交：`ca1f0336e32eed9768cabc861d8981890a5ae5be`；干净worktree：`E:\type10-7\code\snapshots\d69wt`，detached HEAD为该提交且建立时`git status -sb`仅`## HEAD (no branch)`。
- 干净worktree中D42–D69完整链335/335再次通过，用时83.0s；运行目录`E:\type10-7\local_artifacts\d69_clean_full_chain_335`。
- 本轮真实实验在本地执行，不使用SSH/SCP/N607；Python为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。输出目录在登记时必须不存在，禁止覆盖或失败后原目录重跑。
- 实际执行的干净checkout source SHA：probe`8a6582820e8715806e9bc7284a9d348242f301a290c4f265ae7cb04484317c69`、D69 core`739b52a0b404dea61ba1b92322347899976acf4c04e1092e6505992e4621d100`、D62 helper`38ae1114a06d135bca806f470417cd28a634fec0da449888665c6843615d4a20`。第6节的SHA是主工作树LF内容hash；Git checkout按本仓库行尾规则生成CRLF，receipt/source closure以本条实际执行hash为准。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d69wt\code\scripts\probe_d69_frozen_d62_old_append_d62_new.py' `
  --d69-arm frozen_d62_old_append_d62_new `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d69wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d69_frozen_d62_old_append_d62_new_probe_20260719\frozen_d62_old_append_d62_new' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期闭包（后按第8节修正precision计数）：105行、30条目标candidate row、60个D69 fit audit、30对before/final、1080个D62 component fit记录且无pending。before精确D62，final旧行INT8/FP32 state逐bit不变，新FP32行精确joint D62；ground实际输入0，query/clean/source/role/quota/global assignment访问0。任何断言失败均停止并保留原目录。

## 8.真实运行与post-run verifier恢复

- runner完成105/105行，receipt为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，runner耗时119.840s，端到端126.760s；query未向predictor开放，选择仍为`Z0_SUPPORT_ONLY`。
- 初版probe在runner完成后把precision-specific生命周期调用误计为15对，触发`D69 lifecycle pair closure drift`。真实正确闭包是INT8与matched FP32分别拟合：60个fit audit、30对before/final；D62 component fit仍为1080。该错误只在post-run计数断言，不影响已密封training log、receipt或预测。
- 为避免因验证器错误重复评分同一development cell，没有重跑runner。`recover_d69_postverify_metadata.py`只读验证既有sealed artifact，确认105行、60个D69 audit、30对生命周期、旧INT8/FP32 state逐bit不变、新FP32行精确joint D62、ground输入0，并仅新增此前缺失的`D69_PROBE_METADATA.json`；receipt和training log hash保持不变。
- 完整摘要：`d69_full_performance_summary.json`，95614B，SHA256=`6211aba416044c1d09b39cb7e9b7d4fdbb20f48510264501993401e379324566`。

## 9.七候选同排性能

|候选|B|A|N|H|F|J|min-B/A/N|混淆旧→新/新→旧/新→新|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|33/22/19|
|D42-D40-HNBR-INT8-NEGATIVE|85.56|85.00|15.33|25.16|0.56|0.00|66.67/63.33/0.00|2/0/0|
|D42-D41-BEC-INT8-NEGATIVE|86.11|20.56|78.67|31.50|65.56|0.00|76.67/0.00/36.67|142/0/32|
|D42-PROTOnet-CDA-ZID160|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|0/0/0|
|D42-USLDA-FP32-MATCHED|92.78|81.67|74.67|77.39|11.11|30.00|80.00/53.33/53.33|27/23/15|
|**D42-USLDA-INT8（D69）**|**92.78**|**81.67**|**74.67**|**77.39**|**11.11**|**30.00**|**80.00/53.33/53.33**|**27/23/15**|
|Z0_SUPPORT_ONLY|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|0/0/0|

D69精确保持D62的B=92.78%，但相对D62的A/N/H分别下降0.56/10.00/5.24pp，F恶化0.56pp；J提高3.33pp不能抵消min-N从73.33%降到53.33%以及新→旧从8增到23的交换伤害。

## 10.三场景、逐类与逐fold表现

|场景|B|A|N|H|F|J|min-B/A/N|row-floor B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|98.33|91.67|92.00|91.16|6.67|50.00|90.00/70.00/80.00|90.00/70.00/70.00|4/4/0|
|leo_low_elev_weak|91.67|73.33|64.00|67.99|18.33|10.00|80.00/40.00/30.00|70.00/30.00/10.00|13/11/7|
|leo_rain_weak|88.33|80.00|68.00|73.01|8.33|30.00|60.00/50.00/50.00|60.00/50.00/30.00|10/8/8|

|匿名类|角色|B|A或N|变化|
|---|---|---:|---:|---:|
|cls_1f3344|旧|96.67|90.00|-6.67|
|cls_33bbd1|旧|96.67|93.33|-3.33|
|cls_75aa6d|旧|96.67|93.33|-3.33|
|cls_8b02d9|旧|80.00|53.33|-26.67|
|cls_a53ca1|旧|93.33|66.67|-26.67|
|cls_f8dfc2|旧|93.33|93.33|0.00|
|cls_09f800|新|—|53.33|—|
|cls_1c2ad8|新|—|83.33|—|
|cls_b8fbac|新|—|66.67|—|
|cls_d3afb5|新|—|90.00|—|
|cls_f608a3|新|—|80.00|—|

|场景-fold|B|A|N|H|F|J|旧→新/新→旧/新→新|
|---|---:|---:|---:|---:|---:|---:|---:|
|clear-0|100.00|100.00|80.00|88.89|0.00|50.00|0/2/0|
|clear-1|100.00|91.67|90.00|90.83|8.33|50.00|1/1/0|
|clear-2|91.67|75.00|100.00|85.71|16.67|50.00|2/0/0|
|clear-3|100.00|100.00|90.00|94.74|0.00|50.00|0/1/0|
|clear-4|100.00|91.67|100.00|95.65|8.33|50.00|1/0/0|
|low-0|100.00|75.00|70.00|72.41|25.00|50.00|3/2/1|
|low-1|83.33|66.67|70.00|68.29|16.67|0.00|3/0/3|
|low-2|83.33|83.33|70.00|76.09|0.00|0.00|1/2/1|
|low-3|100.00|75.00|50.00|60.00|25.00|0.00|3/3/2|
|low-4|91.67|66.67|60.00|63.16|25.00|0.00|3/4/0|
|rain-0|83.33|83.33|50.00|62.50|0.00|0.00|2/1/4|
|rain-1|100.00|75.00|70.00|72.41|25.00|50.00|3/2/1|
|rain-2|91.67|83.33|80.00|81.63|8.33|50.00|1/1/1|
|rain-3|83.33|83.33|70.00|76.09|0.00|50.00|2/2/1|
|rain-4|83.33|75.00|70.00|72.41|8.33|0.00|2/2/1|

clear场景接近目标，但low/rain显著失败；不能用clear的N=92%掩盖low的N=64%、min-N=30%和F=18.33%。旧类`cls_8b02d9`、`cls_a53ca1`各下降26.67pp，新类`cls_09f800`仅53.33%，说明冻结并未形成通用floor保护。

## 11.机制、量化、训练与资源

- before的D69实际行与D62 joint行max/L2/bias差均为0，compiled support acc为100%，因此B精确复现不是偶然。
- final的6个冻结旧行相对D62 joint-old行发生大几何差异：系数max绝对差均值8.8916、L2差均值27.3819、bias max差均值1.3796；5个新行则逐bit等于D62 joint-new行。拼接support acc仍达99.32%，outer却N下降10pp，说明support拟合不能证明跨head尺度可交换。
- D62 final support gate在15fold中仅3fold active、12fold因atomic gate回退；冻结旧行后，新行所处的final joint坐标系与before旧行坐标系不同，导致23个新样本被旧行吸收，同时旧→新也增加到27。
- INT8与matched FP32全部指标完全一致：before/final outer argmax变化0、support argmax变化0、margin sign flip0；最大score误差均值0.000776、最大0.001769。失败不是量化造成。

|epoch|loss|support acc|grad norm|
|---:|---:|---:|---:|
|1|1.031996|95.14%|1.083757|
|2|0.801388|95.97%|0.870572|
|3|0.623484|97.78%|0.690893|
|4|0.500504|97.50%|0.540671|
|5|0.415989|97.78%|0.436324|
|6|0.353962|98.19%|0.369829|
|7|0.299062|98.61%|0.315457|
|8|0.260996|98.89%|0.301407|
|9|0.233931|99.03%|0.256953|
|10|0.216143|99.03%|0.235860|
|11|0.190273|99.58%|0.220582|
|12|0.174391|99.31%|0.202662|
|13|0.160626|99.72%|0.185954|
|14|0.152731|99.86%|0.205840|
|15|0.142408|99.72%|0.173981|
|16|0.131352|100.00%|0.166464|
|17|0.126780|99.72%|0.170467|
|18|0.115133|99.72%|0.147418|
|19|0.109940|99.86%|0.131373|
|20|0.102685|100.00%|0.135354|

|资源项|每个target row|
|---|---:|
|closed-form LDA fit|72|
|D62 additional component fit|36|
|总adaptation MAC|24,891,223,970|
|D62 additional LDA MAC|16,934,178,816|
|Fisher dense上界|6,879,707,136|
|每query MAC|6,624|
|trainable parameters|2,016|
|persistent/registry state|8,583B/941B|
|峰值CUDA显存|22,886,912B|
|epoch/optimizer step|20/20|

资源硬上限、持久状态和query独立性均通过，但适配MAC极高且性能未达标。D69额外query MAC、持久状态、optimizer step和ground输入均为0。

## 12.与D62/D65/D66/D67/D68同排比较

|版本|ground输入|B|A|N|H|F|J|min-B/A/N|混淆|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D62|0|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|23/8/15|当前联合最强|
|D65|0|92.22|86.11|59.33|67.12|6.11|16.67|80.00/70.00/46.67|16/28/33|冻结旧行但新类不足|
|D66|84个int8 cell|93.33|83.33|83.33|82.59|10.00|23.33|80.00/53.33/66.67|20/9/16|ground负交换|
|D67|0|92.78|82.78|83.33|82.16|10.00|26.67|80.00/53.33/73.33|22/11/14|stacking轻微负交换|
|D68|0|58.89|51.67|14.00|18.66|7.22|0.00|50.00/43.33/0.00|20/118/11|逐行标定灾难失败|
|D69|0|92.78|81.67|74.67|77.39|11.11|30.00|80.00/53.33/53.33|27/23/15|冻结拼接负交换|

D69相对D65恢复N+15.33pp、H+10.27pp，但A-4.44pp、F+5.00pp且min-A-16.67pp；相对D67的A/N/H分别-1.11/-8.67/-4.77pp。它没有吸收D65的低遗忘成功信号，反而同时加剧旧→新和新→旧混淆。

K10门缺口：A距92%差10.33pp、min-A距88%差34.67pp、N距92%差17.33pp。停止D62-before旧行＋D62-final新行直接拼接路线，不做第二seed、125或任何角色offset/温度修补。D62继续保持当前最强。

## 13.ground实际利用结论

D69的metadata、fit audit和resource均为`ground_component_input_count=0`，没有利用地面压缩旧类原型。原因仍是D22当前`formal_phase2_eligible=false`且provenance未按现协议验证；不能为满足“看起来用了ground”而引入不具正式资格的依赖。

D66才是真实ground实验：84个int8 domain-class cell、每类14个，但相对D62以N-1.33pp、min-N-6.67pp、J-3.33pp换取A+1.11pp。D67–D69进一步说明当前主要瓶颈是Stage2-B/Stage2-C跨注册阶段的joint坐标系与floor，而不是缺少一个更强ground权重。

## 14.artifact与最终决定

|artifact|大小|SHA256|
|---|---:|---|
|D69_PROBE_METADATA.json|2,920B|`c1b432dd20e02db684ede1eaab85862162ff858fc6be39075f38e488185c3e07`|
|geometry_audit.json|5,132B|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|RECEIPT.json|5,030B|`565d8d3cef86cc2b8949a23b6854efa99544e8a2356011ae192a4962d6bd4d23`|
|resource_audit.json|6,498B|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|selection.json|2,992B|`bfb82c1a51482db470b9cab787a72c7269ca167524d56618fa22f3bd232d8ac1`|
|support_audit.json|313,666B|`758aa5f9b4bf13d9fe8e010b9a8e971da30a86b8054bbe2aefcd61aad5f9fa64`|
|training_log.jsonl|21,209,061B|`a0e32644e7f265d9232b2dfaad3935cfab8d037f07f9c6f593b8b9018ab13aa6`|

最终判定：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D69是D67–D69第三轮探索；按强制节奏，启动D70前必须完成目标/协议/历史/完整日志复盘，并在本报告记录保留信号、淘汰路线和下一机制。

## 15.D67–D69三轮正式技术复盘

### 15.1复盘输入与协议重核

- 本轮重新完整读取活动目标与`项目.md`；刷新`conversation_index`得到1005条项目记录，并搜索`D62/D65/D66/D67/D68/D69/ground/原型/遗忘/floor/注册`及`crossfitted atomic gate/row replacement/freeze/append`。历史主张仍强调int8 ground中心＋偏移/半径，但当前实时manifest与D66证据优先：D22未具正式资格，D66真实接入仍为负交换。
- 已复核D62、D65、D66、D67、D68、D69同cell完整105行日志、逐类、逐fold、混淆、量化和资源，而非只看报告摘要。
- 下一候选仍必须同时产生同一run的B、A、N、H、逐类old/new、F、min-floor和三类混淆；`LEO_weak-only`、单物理样本单观测、support-only更新、no clean/source、no query truth/role/quota/global assignment、class-agnostic floor和int8正式状态全部保持。

### 15.2三轮结论

|轮次|机制|正信号|决定性缺陷|结论|
|---|---|---|---|---|
|D67|D62/D65交叉拟合连续stacking|A+0.56pp、F-0.56pp|N-1.33pp、H-0.47pp，新→旧+3；alpha均值仅2.906%|支持风险不能稳定决定连续专家权重|
|D68|D65逐行方向＋尺度标定、旧行冻结|符号后support风险约降10%|B先塌至58.89%，N=14%；逐行等幅化删除joint绝对尺度|永久停止per-row standardization|
|D69|冻结D62 before旧行＋追加D62 final新行|B精确92.78%，INT8/FP32零差，J+3.33pp|N-10pp、min-N-20pp，新→旧+15，F反而恶化|永久停止跨stage直接行拼接|

### 15.3保留经验与淘汰路线

- 保留：D62 final joint head仍是唯一同时保持B、N和floor的强基线；D65说明“旧行生命周期稳定性”确实能降低F，但必须在final joint坐标系内实现；D69的before精确复现和INT8零变化证明实现/量化不是当前瓶颈；clear场景D69达到A91.67/N92.00，说明主要困难集中在low/rain下尾而非所有样本不可分。
- 淘汰：连续alpha stacking、D65 signed/per-row归一化、D62 before旧行与final新行盲拼、未验证ground强接入、角色offset/temperature补丁和任何按难类ID定向规则。
- 重复失败模式：support acc接近100%仍不能预测outer；跨stage独立拟合head的绝对坐标系不可直接交换；只保护旧行会压新类，只增强新行会侵入旧类；总体或J单项改善经常掩盖min-class交换。

### 15.4下一轮唯一高信息量假设

D70锁为`crossfitted_atomic_lifecycle_row_replacement`：始终以D62 final joint head作为完整base，绝不盲拼。每个inner leave-one-rank-out fold同时拟合D62 before-old和D62 final-joint；在held support上逐个检验“仅把某个旧类final行替换成before行”的候选。候选旧行只有在自身TP不降、FP不增且严格改善至少一项时进入初选；把全部初选行同时替换后，必须对当前全部11类满足TP逐类不降、FP逐类不增，否则整组回退精确D62。full support只按该binary mask替换旧行，新行始终来自final joint D62。

该机制吸收D62的atomic gate成功经验和D65的生命周期信号，但与D67不同，不学习连续权重；与D69不同，不替换全部旧行；与D68不同，不改变任何行的center/scale。无超参数、无class ID、无query/scene/receiver/角色分支、无ground。若mask为空则精确D62；若真实outer出现任何A/N/H/J/min-A/min-N交换，首seed否决。它仍同时评价旧类适应和新类注册，不允许只以F下降晋级。

复盘完成，允许进入D70预注册与实现；125继续禁止，直到统一候选通过开发门及第二development seed。
