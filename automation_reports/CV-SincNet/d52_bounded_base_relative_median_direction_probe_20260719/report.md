# D52有界base-relative median方向开发报告

## 1.状态

- run ID：`d52_bounded_base_relative_median_direction_probe_20260719`
- operator：Codex
- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`
- 范围：本地receiver20-1、seed713101、K10/new5开发单元；不访问N607、不运行125。
- 当前最强合法开发点：D46，但不promotable。

## 2.目标、假设与比较对象

D51证明coordinate-median相对mean的稳健方向能改善rain old，但全局小RMS把部分修正放大到2.51，造成low-elev/new交换伤害。D52只检验三轮回顾预注册的假设：方向有效，尺度失败。唯一公式为：

```text
u_c = coordinate_median_r(x_rc) - mean_r(x_rc)
v_c = u_c / max(||u_c||_2, eps)
gamma_c = 1 - ||mean_r(x_rc / ||x_rc||_2)||_2
s_c = ||W_D45,c - mean_j(W_D45,j)||_2
DeltaW_c = gamma_c * s_c * v_c
W_D52,c = W_D45,c + DeltaW_c
b_D52,c = b_D45,c
```

直接比较D45、D46、D51；不得仅按rain单项或独立极值晋级。

## 3.协议与数据锁

- `protocol_schema=p2_min_v1`，复用已`VALIDATED_ONCE`胶囊；方法变化不触发数据重验证。
- 固定`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`×5 outer folds；标称K10，实际每outer fit K8；new5。
- 只用support；query及其视图test-only，禁止truth/role/count/quota/global reassignment/query-dependent optimization。
- 禁止clean/source、dense query graph、class-ID/场景/receiver/handle分支。
- before/final使用同一公式；K1/K2精确D45 fallback；无alpha、threshold、clip或scan。

## 4.本地文件和验证

|文件|用途|
|---|---|
|`code/scripts/probe_d52_bounded_base_relative_median_direction.py`|探针、closure verifier、资源账|
|`tests/test_probe_d52_bounded_base_relative_median_direction.py`|公式、边界、对称性、资源测试|
|`analysis/d52_bounded_base_relative_median_direction_traceability_20260719.md`|设计–验证追踪|
|本报告|运行锁与完整性能账本|

验证结果：`py_compile`通过；D52定向10/10通过；D45–D52联合116/116通过；D40–D52执行闭包256/256通过。pytest退出后的临时目录清理出现一次`PermissionError`提示，但主进程exit0，不影响项目断言。更宽的历史测试面另有3个与D52无关的既有断言漂移：1项候选列表仍只锁到D35，2项仍要求D25 schema literal位于`run()`函数体；当前runner已经扩展到D42且schema构造位置已变化，因此不为D52改写。执行前还需clean worktree、输入hash与输出不存在检查。

## 5.预注册成功/停止标准

|指标|最低要求|
|---|---|
|预测实质性|相对D45至少1/15 outer预测SHA变化|
|联合表现|总体和各场景after/new/H/joint及逐类floor不得出现不可解释退化|
|对当前最强D46|至少保持new84.67%、min-new73.33%，同时改善old侧或遗忘|
|rain修复|after至少78.33%，forget不高于10pp，且不能以low-elev/new伤害换取|
|协议与量化|query/role/quota/count/global/clean/source为0/false；FP32/int8 argmax和margin翻转为0|

任一联合门失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`：不扫尺度、不clip、不加门控、不跑第二seed、不formalize、不运行125。

## 6.计划运行和产物

- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`；本地`device=auto`；串行单进程。
- runtime root：`E:\type10-7\code\snapshots\d41wt`（只读bootstrap）。
- probe root：待实现提交后创建`E:\type10-7\code\snapshots\d52wt`clean detached worktree。
- output：`E:\type10-7\automation_reports\CV-SincNet\d52_bounded_base_relative_median_direction_probe_20260719\bounded_base_relative_median_direction`。
- 预期：`training_log.jsonl`、`selection.json`、`support_audit.json`、`geometry_audit.json`、`resource_audit.json`、`RECEIPT.json`、`D52_PROBE_METADATA.json`，完成后追加`full_performance_summary.json`。
- 风险：bounded scale可能过小而不改变决策；base范数可能仍把某些类推过边界；raw coordinate median方向不具旋转等变性。三者均须用完整同row指标判断。

## 7.性能报告承诺

实验完成后，本报告必须补充：7候选总体表、3场景表、old/new逐类表、15个outer行、相对D45/D46/D51的同row差值与预测变化、混淆、完整20epoch训练轨迹摘要、几何修正范数、int8/FP32误差、资源与全部artifact SHA。不得只写缺陷或只报单项最好值。

## 8.执行锁与exact command

- Git承载仓库分支`codex/cvs-rffi-release-20260626`；实现提交`422dfbd9`。根目录`E:\type10-7`不是Git仓库，本报告同步保留Git版与根目录镜像。
- clean detached worktree：`E:\type10-7\code\snapshots\d52wt`，状态`## HEAD (no branch)`；探针SHA256`32a37c732ce54bc236b999defb755f2f1c466d04a8c807241db0d169cff7846e`。
- clean worktree内`py_compile`和D52定向10/10再次通过。
- before/after seal SHA为`53ace286…d9f75`/`c70aedf3…b50ff`；签名授权envelope为`31a2ad99…ceb0e`/`a2483d6e…be76`；int8 manifest为`15b5e144…629c`；class binding为`bb89a1db…c901f`。全部与既有锁一致。
- runtime root存在；输出目录启动前不存在。本地串行`device=auto`，无N607连接。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d52wt\code\scripts\probe_d52_bounded_base_relative_median_direction.py' `
  --d52-arm bounded_base_relative_median_direction `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d52wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' `
  --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' `
  --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' `
  --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' `
  --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' `
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d52_bounded_base_relative_median_direction_probe_20260719\bounded_base_relative_median_direction' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 9.完成状态与证据闭包

- 原启动包装在工具5秒超时后退出，但Python PID25700继续运行；只读监控后进程自行退出，未重启、未覆盖。完整落地105/105行、7候选×15 folds，receipt elapsed`74.372s`。
- metadata verifier确认D43/D52各30条目标行、总105行、source closure和forced-nonpromotable均通过；query未打开。
- receipt状态`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；selected为Z0且`selected_positive_route=false`。这只是runner防晋级选择，不替代下述D52完整性能分析。

## 10.七候选总体性能

H为15个同row H均值；`min-*`为逐类跨15行均值的最小值；混淆为`old→new/new→old/new→new`。

|Candidate|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Z0_SUPPORT_ONLY|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|identity fallback|
|D42-PROTOnet-CDA-ZID160|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|与Z0同指标|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|诊断比较器|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|保旧、新类崩溃|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|旧类灾难遗忘|
|D52-INT8|90.56%|81.67%|80.00%|79.96%|8.89pp|26.67%|83.33%|66.67%|66.67%|19/15/15|旧类floor和遗忘改善，new明显受损|
|D52-FP32-MATCHED|90.56%|81.67%|80.00%|79.96%|8.89pp|26.67%|83.33%|66.67%|66.67%|19/15/15|与int8完全一致|

## 11.分场景性能

|场景|before|after|new|H|forget|joint|min-after|min-new|混淆|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|leo_clear_weak|96.67%|90.00%|96.00%|92.70%|6.67pp|40.00%|70.00%|80.00%|4/1/1|整体高，但new较D46低2pp|
|leo_low_elev_weak|90.00%|76.67%|74.00%|73.68%|13.33pp|20.00%|60.00%|40.00%|9/5/8|H、forget、new floor失败|
|leo_rain_weak|85.00%|78.33%|70.00%|73.51%|6.67pp|20.00%|60.00%|60.00%|6/9/6|old floor/遗忘显著改善，但new较D46低10pp|
|总体|90.56%|81.67%|80.00%|79.96%|8.89pp|26.67%|66.67%|66.67%|19/15/15|旧类稳定性换取新类伤害，不晋级|

## 12.逐类性能

总体old为before→after；场景old为after；new为注册后准确率。

|角色|类|总体|clear|low-elev|rain|表现|
|---|---|---:|---:|---:|---:|---|
|old|O0/`1f33441e`|83.33→86.67%|100%|80%|80%|注册后恢复|
|old|O1/`33bbd165`|96.67→93.33%|90%|90%|100%|稳定|
|old|O2/`75aa6d50`|93.33→90.00%|90%|90%|90%|稳定|
|old|O3/`8b02d999`|86.67→66.67%|70%|60%|70%|较D45/D51显著修复old floor|
|old|O4/`a53ca128`|96.67→70.00%|90%|60%|60%|主要遗忘类|
|old|O5/`f8dfc2ed`|86.67→83.33%|100%|80%|70%|rain偏低|
|new|N0/`09f80039`|80.00%|80%|90%|70%|总体可用|
|new|N1/`1c2ad882`|80.00%|100%|60%|80%|low-elev伤害|
|new|N2/`b8fbace5`|66.67%|100%|40%|60%|new floor失败类|
|new|N3/`d3afb5d1`|86.67%|100%|90%|70%|rain下降|
|new|N4/`f608a348`|86.67%|100%|90%|70%|rain下降|

## 13.十五个outer行

floor为`before/after/new`，混淆为`old→new/new→old/new→new`。

|场景|fold|before|after|new|H|forget|joint|floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|91.67%|100.00%|90.00%|94.74%|-8.33pp|50%|50/100/50%|0/1/0|
|clear|1|100.00%|83.33%|90.00%|86.54%|16.67pp|0%|100/0/50%|2/0/1|
|clear|2|91.67%|83.33%|100.00%|90.91%|8.33pp|50%|50/50/100%|1/0/0|
|clear|3|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100%|0/0/0|
|clear|4|100.00%|91.67%|100.00%|95.65%|8.33pp|50%|100/50/100%|1/0/0|
|low-elev|0|91.67%|75.00%|70.00%|72.41%|16.67pp|0%|50/50/0%|3/1/2|
|low-elev|1|75.00%|58.33%|80.00%|67.47%|16.67pp|50%|50/50/50%|1/0/2|
|low-elev|2|91.67%|91.67%|70.00%|79.38%|0.00pp|0%|50/50/0%|0/2/1|
|low-elev|3|100.00%|91.67%|60.00%|72.53%|8.33pp|0%|100/50/0%|1/2/2|
|low-elev|4|91.67%|66.67%|90.00%|76.60%|25.00pp|50%|50/50/50%|4/0/1|
|rain|0|75.00%|75.00%|50.00%|60.00%|0.00pp|0%|50/50/0%|0/2/3|
|rain|1|91.67%|75.00%|70.00%|72.41%|16.67pp|0%|50/50/0%|2/3/0|
|rain|2|91.67%|83.33%|80.00%|81.63%|8.33pp|50%|50/50/50%|1/0/2|
|rain|3|83.33%|75.00%|80.00%|77.42%|8.33pp|50%|50/50/50%|2/2/0|
|rain|4|83.33%|83.33%|70.00%|76.09%|0.00pp|0%|50/50/0%|1/2/1|

## 14.相对D45/D46/D51

|版本|before|after|new|H|forget|joint|min-after|min-new|混淆|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D45|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|matched底座|
|D46|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|当前最强合法开发点|
|D51|92.22%|82.22%|82.00%|81.16%|10.00pp|26.67%|46.67%|70.00%|23/12/15|RMS尺度几何残差|
|D52|90.56%|81.67%|80.00%|79.96%|8.89pp|26.67%|66.67%|66.67%|19/15/15|old floor最佳，但new最差|

D52相对D45改变13/15个预测SHA：forget`-1.11pp`、joint`+3.33pp`、min-after`+13.33pp`、old→new`-5`，但before`-1.67pp`、after`-0.56pp`、new`-4.00pp`、H`-2.19pp`、min-new`-3.33pp`、new→old`+7`。相对D46同样改变13/15行：after持平、forget`-1.67pp`、min-after`+13.33pp`，但new`-4.67pp`、H`-2.37pp`、min-new`-6.67pp`、new→old`+7`。相对D51，min-after`+20pp`、forget`-1.11pp`，但new`-2pp`、H`-1.20pp`、min-new`-3.33pp`。

场景上，rain相对D46 after`+1.67pp`、forget`-6.67pp`、min-after`+30pp`，却同时new`-10pp`、H`-3.94pp`、min-new`-10pp`。这说明D52把边界系统性推向old：old→new减少而new→old增加，不是old/new联合提升。

## 15.机制行为

|阶段|量|min|mean|max|
|---|---|---:|---:|---:|
|before|resultant rho|0.8895|0.9386|0.9731|
|before|gamma|0.0269|0.0614|0.1105|
|before|raw direction L2|0.0618|0.1181|0.1784|
|before|base discriminant L2|9.985|15.554|23.783|
|before|correction L2|0.435|0.959|2.409|
|final|resultant rho|0.8793|0.9399|0.9813|
|final|gamma|0.0187|0.0601|0.1207|
|final|raw direction L2|0.0435|0.1153|0.2339|
|final|base discriminant L2|10.965|18.753|30.365|
|final|correction L2|0.276|1.149|3.128|

逐类`||correction||=gamma*||W_c-mean(W)||`最大绝对误差仅`1.33e-15`，证明实现严格满足预注册边界。但这个边界并不保守：D45判别范数很大，使final平均/最大修正`1.149/3.128`，反而超过D51的`0.736/2.508`。因此D52修复了D51的“小RMS除法”缺陷，却没有解决“修正相对决策尺度过大”的根因。

## 16.训练、量化与资源

B20沿用D45冻结轨迹，20个epoch完整，query rows始终0。

|epoch|loss mean|support acc|gradient norm|anchor loss|
|---:|---:|---:|---:|---:|
|1|1.031996|95.14%|1.08376|0.000000|
|5|0.415989|97.78%|0.43632|0.000839|
|10|0.216143|99.03%|0.23586|0.002265|
|15|0.142408|99.72%|0.17398|0.003222|
|20|0.102685|100.00%|0.13535|0.003828|

|项目|结果|判定|
|---|---:|---|
|FP32/int8 before/final argmax变化|0/0|通过|
|margin符号翻转|0|通过|
|support argmax变化|0/0|通过|
|int8最大score误差|min`3.946e-4`、mean`9.070e-4`、max`1.690e-3`|未改变决策|
|LDA fit/MAC|36/1,065,830,400|D45闭合|
|D52额外适配|227,520 MAC-equivalent|闭合|
|coordinate median比较上界|117,504|单列|
|总适配/query MAC|1,071,034,560/6,624|闭合|
|参数/state|2,016/8,583B|通过|
|epoch/step|20/20|通过|
|CUDA peak|22,886,912B|实测|
|query/role/quota/count/global assignment|全部0/false|通过|
|clean/source/dense graph|false/false/0B|通过|

## 17.Artifact清单

|文件|大小/B|SHA256|
|---|---:|---|
|D52_PROBE_METADATA.json|1,828|`690ee1d396c9e5997f2402b603789c7c88062af9d84231ae3031055c60369fbb`|
|RECEIPT.json|4,845|`4eff36f582cfc31909d9d468d199d828304d9b3bfb5e324142ddc94e0cb56986`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|selection.json|2,990|`1dea292170b541eb29e0d3e31847c789b6db0546da29797cbca412b072dbe3f3`|
|support_audit.json|313,487|`4cc48efb12eddfe07d8b8b606464768f8b207a9205efbeb57cd1722f4e6f9cc0`|
|training_log.jsonl|37,699,677|`403c574f55569cd9d237544cc995f10d5f487ea7b7a7dcca555563f06c060b51`|
|full_performance_summary.json|74,221|`a200a70fa191a9965ec66a8462d316fe8484d8c03765b16c7e87363779cf691e`|

summary完整读取D52/D45/D46/D51各105行，生成器为`code/scripts/summarize_d52_performance.py`。

## 18.晋级判定与下一步

|门|结果|判定|
|---|---|---|
|相对D45改变≥1预测|13/15行|通过|
|总体/各场景after/new/H/joint/floor不退化|new/H/min-new退化|失败|
|保持D46 new/min-new|80.00/66.67%<84.67/73.33%|失败|
|rain after≥78.33%、forget≤10pp|78.33%、6.67pp|通过|
|rain不得伤害new|70.00%，较D46-10pp|失败|
|量化0/0/0|0/0/0|通过|

D52详细结论：旧类逐类下界、rain遗忘和old→new混淆均明显改善，证明D51的median方向确实包含有效旧类稳定信号；但base-relative范数仍过大，将决策系统性推向old，导致新类注册性能和new→old混淆恶化。停止D52公式，不扫描系数、不做clip/角色门控、不跑第二seed、不formalize、不运行125。当前最强仍为D46，且仍未满足项目要求。
