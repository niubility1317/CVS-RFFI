# D49严格nested全局余弦原型融合探针报告

## 1.实验身份与状态

- 实验ID：`d49_nested_global_cosine_fusion_probe_20260719`。
- 时间：2026-07-19T01:19:49+08:00。
- 操作者：Codex主代理。
- 状态：`IMPLEMENTED_VERIFIED_NOT_RUN`。
- 目标：在不改变D42固定received-IQ表征、query路径和数据协议的前提下，检验严格nested support-LOO选择的全局cosine prototype head能否补足D45的rain旧类尾部或low-elev新类尾部。
- 比较对象：matched D45、D46当前最强合法development点、D42及已锁基线。

## 2.协议与数据锁

- `protocol_schema=p2_min_v1`，复用`phase2_data_status=VALIDATED_ONCE`的固定capsule/split；本轮仅改变方法，不重验数据。
- development cell固定为receiver`20-1`、seed`713101`、K10/new5、clear/low-elev/rain×5 outer折，共105行，实际每个outer fit为K8。
- before state必须在首次读取new support前物化并保持不可变；query及其view只测试，query读取、更新、真值、角色、配额、全局重分配均为0。
- 不访问clean/source、未批准derived state、receiver/scene/handle、class ID分支；old/new prototype按完全同式分别由本类support生成。
- 本轮不访问N607，不运行125，不新增data admission/hash/governance机制。

## 3.唯一方法定义

D42已产生全局单位球特征`x=normalize(exp(logdiag)⊙f)∈R^288`。D49的cosine head为：

`p_c=normalize(mean_{i∈S_c}x_i)`，`z_cos,c(x)=x^T p_c`。

因此`W_cos,c=p_c,b_cos,c=0`，无需新增非线性query view，仍可编译为一个`C×288`系数和`C`个intercept。

当K>1时，顶层每个held rank仅用其余`K−1`个rank：

1. 完整重拟合D45 base learner，包括其自身full/block inner-LOO可靠性权重；
2. 由同一inner-train生成global cosine prototypes；
3. 两head各自用inner-train class-centered logit RMS归一；
4. 每个held support恰评分一次，计算两head的macro class-balanced CE；
5. 以FP64稳定`softmax(-C×CE)`给出严格正且和为1的单一global权重。

完整support再独立拟合D45和cosine、重算各自RMS，并直接从量化前FP32 affine合成：

`W=w_D×W_D/s_D+w_C×P/s_C`，`b=w_D×b_D/s_D`。

合成后只做一次class-common canonical centering，再进入既有int8 coefficient/FP16 intercept lifecycle。K1不建立顶层cosine权重，系数、intercept及audit必须逐位回退D45。K2的nested D45 inner-train为K1，但D45/cosine顶层权重不强制1:1。CE完全相等时权重必须精确0.5/0.5。

## 4.禁止项与fail-close

- 不得再次做三块query L2归一化；不增加temperature、block-cosine、固定1:1 arm、一维stacking、clip、median、阈值或权重扫描。
- `C<2`、prototype resultant norm≤既有`ENERGY_EPSILON`、任一head RMS退化、非有限CE、权重为0/1或和偏离`1±1e-12`均停止。
- support rank顺序/class label置换必须等价；FP32或int8出现精确top tie必须停止，不能借registry顺序宣称对称。
- 完整support只融合和量化一次；不得先量化D45再解码融合。

## 5.资源预注册

一次D45 before或final fit的LDA次数为`F(1)=2`，`F(n>1)=2+2n`。D49 before+final严格nested总数为`2[F(K)+K×F(K−1)]`；K≥3时为`4+4K+4K²`，当前K8必须为292次，而不是D45的36次。LDA MAC必须按before/final及outer-main/nested-inner分组逐项闭合。

cosine额外适配成本必须分别审计support类均值、prototype归一、support RMS和FP32融合；query仍只有一套affine state，参数量、state字节和query MAC按实际artifact报告。CUDA峰值、host FP64峰值若未测量必须明确标为未测，不能用估算替代实测。

## 6.首次outer评分前验证门

- global-cosine reference与单affine score逐元素等价；query view仍为`full_288d_only`。
- 每折D45 state、prototype和RMS只读inner-train，held/train无交集且所有support恰held一次。
- K1逐位D45回退；K2、CE tie、零范数、RMS退化、权重端点及FP32/int8 top-tie覆盖测试。
- rank顺序和class-label置换测试；完整support只融合/量化一次。
- nested LDA fit/MAC闭合，K8精确292次；最终只有一套`C×288+C`state。
- D42–D48回归通过；独立代码复审P0=0、P1=0。

## 7.开发晋级与停止门

相对matched D45必须至少改变1条final outer prediction，否则立即拒绝。aggregate和clear/low-elev/rain各自的after-old、seen-new、H、joint、min-old/min-new不得退化，forgetting不得增加；至少严格改善一个D45已知失败项，优先rain after-old或rain forgetting。before/final/margin的int8相对FP32翻转必须为`0/0/0`。

任一门失败即记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，仍完整报告每候选、每场景、每类、15个outer行、混淆、训练轨迹、量化、资源、artifact和相对D45变化；不追加温度、权重变体、第二development seed或125。只有全部门通过才另行formalize，之后才可讨论125 screen。

## 8.版本、文件、命令与输出占位

- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`；当前分支`codex/cvs-rffi-release-20260626`。工作树存在用户的无关改动，本轮只stage本报告、D48复盘修正、D49 trace/script/tests。
- 根目录`E:\type10-7`含不可用`.git`目录但`git status`判定不是Git仓库；根报告仅为运行报告镜像，不宣称根目录已版本化。
- 已新增：`code/scripts/probe_d49_nested_global_cosine_fusion.py`、`tests/test_probe_d49_nested_global_cosine_fusion.py`、`analysis/d49_nested_global_cosine_fusion_traceability_20260719.md`。
- 本地环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。
- 计划输出：`E:\type10-7\automation_reports\CV-SincNet\d49_nested_global_cosine_fusion_probe_20260719\nested_global_cosine_fusion`。
- exact launch command、commit/SHA、PID/GPU、log和最终artifact在首次运行前补锁；当前没有PID、GPU分配或结果。

## 9.完成后详细性能表占位

|Candidate|机制|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|最终判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|待运行|strict nested D45/global-cosine fusion|—|—|—|—|—|—|—|—|—|—|—|

完成后还必须写入：三场景表、11类before/after/new表、15个outer行、相对D45 correct→wrong/wrong→correct变化、两head CE/RMS/权重分布、prototype resultant norm、B20每epoch/step轨迹、FP32/int8误差与top-tie、292次LDA资源闭包、state/MAC/显存、全部artifact大小/SHA和缺陷机制解释。

## 10.运行前实现与审计闭包

D49实现没有改动D42/D45文件。每个顶层held fold确实调用完整D45 fit；K1逐位直返D45，K2真实locked路径和非强制1:1权重已测试。cosine从D42全局单位球support按实际target向量重算；support transformed rows和targets在top-level wrapper中分别与正式support输入＋frozen log-diag、正式labels/classes逐元素绑定。verifier从绑定support重算full/nested prototype、resultant norm、两head RMS、held logits、macro/per-class CE和global权重，并将一次FP32融合绑定D42 matched state，再独立重编译核对双残差int8系数、FP16尺度及intercept。runner全局D42 score入口在所有outer FP32/int8 `argmax`前执行exact top-tie fail-close。

K8 before/final的8组LDA库存闭合为292次。D49额外适配MAC预注册为prototype`445,536`、两head RMS`11,635,584`、nested held scoring`1,446,912`、一次FP32融合`19,618`，合计`13,547,650`；最终总MAC以真实artifact为准。CPU不得误报CUDA峰值已测；host FP64峰值未测必须继续标为未测。

测试演化：初版D49为11项；独立复审先后发现outer tie、held证据、资源账、before生命周期、显存语义和自报字段闭环等P1，均在运行前修复。最终D49`13 passed`，D45＋D48＋D49`51 passed`，D43–D49`100 passed`，D42–D49全链`144 passed`，`py_compile`和`git diff --check`通过。pytest退出码为0；结束后的`pytest-current`清理出现Windows`WinError 5`，属于临时目录清理噪声，不是测试断言失败。

最终独立设计复核为P0=0、P1=0。仅保留声明边界：strict nested只描述冻结outer-B20后的head层，B20本身仍由outer-fit support训练，不能宣称全链路nested或无泄漏泛化；292次是与锁定调用结构逐项一致的精确理论库存，不是独立函数调用计数器；当前development为K8，K1正式证据应使用明确D45 fallback artifact。

## 11.执行预检与锁定命令

- D49代码提交：`0ed6a9cb61e74fecac171d71af5c5de53abac8af`；detached worktree`E:\type10-7\code\snapshots\d49wt`HEAD一致、`git status --porcelain`为空；探针SHA256为`b26b2c330178a960a88059d0f0c9d8ee675945b9149736d728c3e84d4a530b60`。
- runtime继续使用历史锁定面`E:\type10-7\code\snapshots\d41wt`。该worktree当前显示3个既有status项，其中2个是EOL工作树差异，`stage2_predictor_bundle.py`有历史锁定内容差异；本轮不修改、不提交、不覆盖。D49 bootstrap对全部runtime模块的内置SHA source closure只读验证通过，三个关键实际SHA分别为`49a05c6f…f48`、`4b1dee1d…f9fc`、`bb27beaa…69fd`。另建的clean-HEAD复本因不匹配这些内置锁而明确拒绝，故不用于实验。
- before/after seal、before/after envelope、component manifest和class binding的实际SHA逐项匹配`53ace286…d9f75`、`c70aedf3…b50ff`、`31a2ad99…ceb0e`、`a2483d6e…be76`、`15b5e144…629c`、`bb89a1db…c901f`。
- 输出`E:\type10-7\automation_reports\CV-SincNet\d49_nested_global_cosine_fusion_probe_20260719\nested_global_cosine_fusion`启动前不存在。运行在本地串行`device=auto`；不访问N607，不生成125。

锁定执行命令为：

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d49wt\code\scripts\probe_d49_nested_global_cosine_fusion.py' `
  --d49-arm nested_global_cosine_fusion `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d49wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d49_nested_global_cosine_fusion_probe_20260719\nested_global_cosine_fusion' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 12.执行完成状态

- 本地开发实验已完成，wall time为`396.998s`，receipt记录算法段耗时`380.749s`；完整读取`105/105`行，每个7候选均为3场景×5折共15行。
- 实验单元为receiver`20-1`、seed`713101`、K10/new5；outer评分实际fit K8。未打开query，未访问clean/source，未访问N607，未运行125。
- D49 int8与matched FP32逐预测一致，但D49相对D45的15/15个outer预测SHA均改变，满足“确有机制作用”的必要条件。
- 结果状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。失败原因不是代码、量化或资源门，而是准确率、场景稳健性、遗忘和新类floor同时违反晋级门。

## 13.七候选完整总体性能

以下H均为15个matched row的`H_old_new`均值；`min-*`先按类跨15行取均值，再在类间取最小值；混淆顺序为`old→new/new→old/new→new`。

|Candidate|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|表现与判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|Z0_SUPPORT_ONLY|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|identity fallback，旧/新两端均低|
|D42-PROTOnet-CDA-ZID160|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|与Z0同指标，不晋级|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|诊断比较器，整体弱于D45|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|保旧但新类注册崩溃|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|偏新导致旧类灾难性遗忘|
|D49-USLDA-INT8|91.11%|76.67%|72.67%|73.90%|14.44pp|20.00%|76.67%|63.33%|40.00%|29/26/15|本轮候选；量化正确但性能负向|
|D49-USLDA-FP32-MATCHED|91.11%|76.67%|72.67%|73.90%|14.44pp|20.00%|76.67%|63.33%|40.00%|29/26/15|与int8完全一致|

## 14.D49分场景性能与行为

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|主要表现|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|leo_clear_weak|91.67%|86.67%|96.00%|91.06%|5.00pp|50.00%|70.00%|70.00%|80.00%|7/1/1|新类总体好，但旧类仍未到92%，且一个旧类仅70%|
|leo_low_elev_weak|88.33%|68.33%|64.00%|65.51%|20.00pp|0.00%|80.00%|50.00%|10.00%|12/11/7|双端同时失稳，fold0/4旧类各仅50%|
|leo_rain_weak|93.33%|75.00%|58.00%|65.14%|18.33pp|10.00%|80.00%|50.00%|30.00%|10/14/7|before强但注册后旧类掉18.33pp，新类仅58%|
|三场景总体|91.11%|76.67%|72.67%|73.90%|14.44pp|20.00%|76.67%|63.33%|40.00%|29/26/15|不满足任一主要准确率目标|

场景行为呈明显分裂：clear下cosine信息可用，但low-elev/rain中同一全局权重把支持集上的较低CE误当成query可泛化证据，导致旧类被新类侵入和新类回落到旧类同时增加。该现象不是单一方向的bias，而是两类边界都被扭曲。

## 15.逐类性能

类名使用完整handle的前8个十六进制字符；完整handle和三场景逐类浮点值保存在`full_performance_summary.json`。old类给出总体before→after及三场景before→after；new类给出总体和三场景注册准确率。

|角色|类handle|总体|clear|low-elev|rain|表现|
|---|---|---:|---:|---:|---:|---|
|old|1f33441e|100.00→96.67%|100→100%|100→90%|100→100%|最稳健旧类|
|old|33bbd165|93.33→90.00%|90→90%|90→80%|100→100%|总体达到90%，仍低于88%正式floor的独立确认要求所需裕量|
|old|75aa6d50|93.33→70.00%|100→90%|80→60%|100→60%|low-elev/rain共同塌陷|
|old|8b02d999|93.33→70.00%|90→70%|100→70%|90→70%|三场景注册后均为70%|
|old|a53ca128|90.00→63.33%|100→90%|80→50%|90→50%|最差old均值，恶劣场景仅50%|
|old|f8dfc2ed|76.67→70.00%|70→80%|80→60%|80→70%|before本身最弱，clear略有修复|
|new|09f80039|40.00%|80%|10%|30%|主导new floor失败，强场景依赖|
|new|1c2ad882|90.00%|100%|90%|80%|最稳健新类|
|new|b8fbace5|80.00%|100%|60%|80%|low-elev下降|
|new|d3afb5d1|86.67%|100%|100%|60%|rain下降|
|new|f608a348|66.67%|100%|60%|40%|rain明显下降|

逐类证据说明均值不是由单一异常折造成：old`75aa6d50/8b02d999/a53ca128`和new`09f80039/f608a348`构成稳定的困难类集合；下一版应直接针对这些类的support几何可靠性，而不能继续提升全局cosine占比。

## 16.十五个outer行

混淆仍按`old→new/new→old/new→new`，floor按`before/after/new`。

|场景|fold|before|after|new|H|forget|joint|floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|100.00%|83.33%|90.00%|86.54%|16.67pp|50%|100/50/50%|2/0/1|
|clear|1|91.67%|91.67%|100.00%|95.65%|0.00pp|50%|50/50/100%|1/0/0|
|clear|2|83.33%|83.33%|90.00%|86.54%|0.00pp|50%|50/50/50%|1/1/0|
|clear|3|83.33%|91.67%|100.00%|95.65%|-8.33pp|50%|50/50/100%|1/0/0|
|clear|4|100.00%|83.33%|100.00%|90.91%|16.67pp|50%|100/50/100%|2/0/0|
|low-elev|0|83.33%|50.00%|60.00%|54.55%|33.33pp|0%|50/0/0%|5/2/2|
|low-elev|1|83.33%|75.00%|70.00%|72.41%|8.33pp|0%|50/0/0%|1/1/2|
|low-elev|2|91.67%|91.67%|70.00%|79.38%|0.00pp|0%|50/50/0%|0/3/0|
|low-elev|3|91.67%|75.00%|60.00%|66.67%|16.67pp|0%|50/50/0%|2/2/2|
|low-elev|4|91.67%|50.00%|60.00%|54.55%|41.67pp|0%|50/0/0%|4/3/1|
|rain|0|100.00%|75.00%|60.00%|66.67%|25.00pp|0%|100/50/0%|2/2/2|
|rain|1|100.00%|91.67%|80.00%|85.44%|8.33pp|50%|100/50/50%|1/2/0|
|rain|2|83.33%|58.33%|50.00%|53.85%|25.00pp|0%|50/0/0%|3/3/2|
|rain|3|91.67%|66.67%|40.00%|50.00%|25.00pp|0%|50/0/0%|3/4/2|
|rain|4|91.67%|83.33%|60.00%|69.77%|8.33pp|0%|50/50/0%|1/3/1|

最差联合行为出现在low-elev fold0/4与rain fold2/3：它们同时出现高遗忘、zero joint floor和双向混淆，不允许用clear的高new准确率掩盖。

## 17.相对D45及当前最强D46

|版本|before|after|new|H|forget|joint|min-after|min-new|混淆|相对结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D45 matched|92.22%|82.22%|84.00%|82.16%|10.00pp|23.33%|53.33%|70.00%|24/8/16|直接matched基线|
|D46当前最强|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|53.33%|73.33%|25/8/15|此前最强合法开发点|
|D49|91.11%|76.67%|72.67%|73.90%|14.44pp|20.00%|63.33%|40.00%|29/26/15|明显低于D45/D46|
|D49−D45|-1.11pp|-5.56pp|-11.33pp|-8.25pp|+4.44pp|-3.33pp|+10.00pp|-30.00pp|+5/+18/-1|仅min-old提高，但以new floor和整体性能大幅下降为代价|

分场景matched差值：

|场景|Δbefore|Δafter|Δnew|ΔH|Δforget|Δjoint|Δmin-after|Δmin-new|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|-6.67pp|-3.33pp|-2.00pp|-2.51pp|-3.33pp|+10.00pp|0.00pp|-10.00pp|
|low-elev|0.00pp|-11.67pp|-10.00pp|-9.94pp|+11.67pp|-20.00pp|-10.00pp|-30.00pp|
|rain|+3.33pp|-1.67pp|-22.00pp|-12.30pp|+5.00pp|0.00pp|+20.00pp|-40.00pp|

15/15个outer prediction SHA与D45不同，证明D49确实改变了决策。当前日志未持久化逐query预测数组，因此不能诚实重构exact correct→wrong/wrong→correct样本转移；报告保留了每折SHA、同折指标差和可核验的汇总混淆差，不用边际推断冒充逐样本证据。

## 18.两head可靠度与几何

|阶段|量|min|mean|max|
|---|---|---:|---:|---:|
|before|D45权重|0.2146|0.3368|0.4522|
|before|cosine权重|0.5478|0.6632|0.7854|
|before|D45 nested macro CE|0.7792|0.9247|1.0097|
|before|cosine nested macro CE|0.7473|0.8090|0.8967|
|before|D45 full-support RMS|0.9958|0.9978|0.9988|
|before|cosine full-support RMS|0.04378|0.04975|0.05334|
|before|prototype resultant norm|0.8895|0.9386|0.9731|
|final|D45权重|0.0843|0.3303|0.7184|
|final|cosine权重|0.2816|0.6697|0.9157|
|final|D45 nested macro CE|1.1799|1.3938|1.5067|
|final|cosine nested macro CE|1.2512|1.3157|1.3991|
|final|D45 full-support RMS|0.9980|0.9984|0.9989|
|final|cosine full-support RMS|0.04138|0.04454|0.04683|
|final|prototype resultant norm|0.8793|0.9399|0.9813|

final的cosine平均权重在clear/low-elev/rain分别为`0.4057/0.7919/0.8116`；对应D45/cosine nested CE均值分别为`1.2367/1.2728`、`1.4695/1.3407`、`1.4753/1.3337`。恶劣场景正是cosine被赋予约80%权重的场景，但真实outer性能下降最大。这说明support nested CE与query泛化排序失配，而不是prototype坍缩：resultant norm始终≥0.8793。

## 19.B20完整训练轨迹

下表为15个D49 int8 outer行在每个epoch的均值；`query rows`为该epoch跨15行总计，始终为0。

|epoch|support acc|loss|CE|anchor loss|grad norm|query rows|
|---:|---:|---:|---:|---:|---:|---:|
|1|95.14%|1.031996|1.031996|0.000000|1.083757|0|
|2|95.97%|0.801388|0.801384|0.000099|0.870572|0|
|3|97.78%|0.623484|0.623470|0.000293|0.690893|0|
|4|97.50%|0.500504|0.500477|0.000548|0.540671|0|
|5|97.78%|0.415989|0.415947|0.000839|0.436324|0|
|6|98.19%|0.353962|0.353905|0.001145|0.369829|0|
|7|98.61%|0.299062|0.298990|0.001451|0.315457|0|
|8|98.89%|0.260996|0.260908|0.001745|0.301407|0|
|9|99.03%|0.233931|0.233830|0.002017|0.256953|0|
|10|99.03%|0.216143|0.216030|0.002265|0.235860|0|
|11|99.58%|0.190273|0.190148|0.002492|0.220582|0|
|12|99.31%|0.174391|0.174256|0.002698|0.202662|0|
|13|99.72%|0.160626|0.160481|0.002888|0.185954|0|
|14|99.86%|0.152731|0.152578|0.003062|0.205840|0|
|15|99.72%|0.142408|0.142246|0.003222|0.173981|0|
|16|100.00%|0.131352|0.131183|0.003368|0.166464|0|
|17|99.72%|0.126780|0.126605|0.003501|0.170467|0|
|18|99.72%|0.115133|0.114952|0.003621|0.147418|0|
|19|99.86%|0.109940|0.109754|0.003730|0.131373|0|
|20|100.00%|0.102685|0.102493|0.003828|0.135354|0|

训练本身平滑收敛，support准确率最终100%，但outer性能显著恶化。这是“支持集过拟合/可靠度代理失配”的直接证据，不能通过增加epoch解释或修复；继续训练反而不会解决query泛化问题。

## 20.量化、资源与协议审计

|项目|结果|门槛/解释|
|---|---:|---|
|matched FP32 before argmax变化|0|通过|
|matched FP32 final argmax变化|0|通过|
|FP32/int8 margin符号翻转|0|通过|
|before/final support argmax变化|0/0|通过|
|FP32/int8 exact top tie|0/0|通过|
|int8最大score绝对误差|min`1.106e-4`、mean`2.884e-4`、max`7.186e-4`|数值误差未改变决策|
|LDA闭式fit|292|K8精确库存闭合|
|LDA MAC|8,467,089,408|主要适配开销|
|D49额外适配MAC|13,547,650|prototype/RMS/held scoring/fusion|
|总适配MAC|8,485,613,698|固定15行一致|
|query MAC|6,624|单一affine＋argmax|
|trainable parameters|2,016|≤80,000|
|persistent state|8,583B|≤262,144B|
|registry state|941B|包含在持久状态审计内|
|epochs/optimizer steps|20/20|≤30/≤50|
|CUDA peak|22,886,912B|`cuda:0`实测|
|host FP64 peak|未测|未用估算冒充实测|
|query fit/truth/role/quota/count/global assignment|0/false|全部通过|
|clean/source access|false/false|通过|
|dense query graph|0B|通过|

资源与协议门均通过，但这只能证明“合法且可部署地失败”，不能把负性能提升为可晋级结果。

## 21.Artifact清单

|文件|大小/B|SHA256|
|---|---:|---|
|D49_PROBE_METADATA.json|2,240|`ab3e63fe203b0814e7f42da41fc14a336eaf4e031f37de60f6fc17f9d086876f`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|RECEIPT.json|4,845|`acf60cfaac690bc4a71d4e2076ad816540b3eab9ae8bf801ca53f09f9e468e6c`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|selection.json|2,992|`6e0205bc40c0ad72062e7a6dd2620067919f4337310a8523b6a928fa81412a31`|
|support_audit.json|313,476|`77a11814ae9da03b741c2a30ffeffa92c06e22050aa579c33405aa70db9d9c3d`|
|training_log.jsonl|82,167,397|`5d16f247b4c437519f50b0dcd3f58188d2823bc0febc2e707ed549699ad06440`|
|full_performance_summary.json|53,296|`4f7af422fa7cc711dd981813de0f73314b6f8926e924f811c89117b930a8dd63`|

`full_performance_summary.json`由完整D49/D45各105行日志生成，包含全部候选、场景、逐类、15折、matched差值、机制、20步训练、量化和资源结构。生成器为`code/scripts/summarize_d49_performance.py`。

## 22.缺陷、停止门与下一轮研发结论

|晋级门|D49结果|判定|
|---|---|---|
|至少改变1条D45预测|15/15行SHA变化|通过|
|总体after/new/H/joint不降|分别-5.56/-11.33/-8.25/-3.33pp|失败|
|三场景上述指标均不降|clear、low-elev、rain均有下降|失败|
|forgetting不增|总体+4.44pp，low-elev+11.67pp，rain+5.00pp|失败|
|min-old/min-new不降|min-old+10.00pp，但min-new-30.00pp|失败|
|严格改善已知失败项且无交换伤害|没有|失败|
|量化翻转0/0/0|0/0/0|通过|

D49的核心缺陷已定位为“support nested CE对cosine head过度乐观”：final阶段low-elev/rain的cosine权重约0.79/0.81，而同场景outer H相对D45下降9.94/12.30pp。prototype resultant norm正常、B20训练收敛、量化零翻转，因此不能把问题归咎于prototype坍缩、优化未收敛或int8误差。

停止动作：不运行D49权重/温度变体，不增加第二development seed，不formalize，不运行125。当前最强合法开发版本仍为D46，而非D49。下一版应回到D46/D45稳定主干，研究能识别困难类和场景失配的support-only类级保守机制；不得继续采用仅凭全局nested CE提高cosine占比的路线。
