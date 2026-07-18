# D57交叉拟合双向混淆流门报告

## 1.状态与问题

- 状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_EXACT_D46_FALLBACK_NOT_PROMOTABLE`；operator Codex；105/105行完成，exit0，Runner elapsed112.812s；不运行125。
- 固定receiver20-1、seed713101、K10/new5、3场景×5fold development cell；复用`VALIDATED_ONCE p2_min_v1`。
- D56把after从D46的81.67%提高到83.33%、forget从10.56pp降到8.33pp，却把new从84.67%降到80.67%、min-new从73.33%降到60.00%。D57只修复这一可观测交换，不修改D46的B20、full/block head、RMS、classwise权重、量化或query路径。

## 2.预注册机制

对D56已经合法生成的每折support inner-held D46分数，折`r`的流修正只能由其余`K−1`折构造：

`Delta b_c^(-r)=(out_c^(-r)-in_c^(-r))/((K-1)*C)`。

对每个匿名类`c`，分别统计基础D46与“只加入坐标`Delta b_c^(-r)`”后的：

- `positive_correct_c`：真实类为`c`的held样本预测正确数；
- `false_positive_c`：真实类不为`c`但被预测为`c`的held样本数。

仅当`positive_correct_adjusted>=positive_correct_base`、`false_positive_adjusted<=false_positive_base`且至少一项严格改善时，`accept_c=1`；否则为0。然后把同一mask应用到全support D56流：`delta_c=accept_c*Delta b_c`，删除类公共常数后一次性加入D46截距。

为避免多个坐标联合后产生交互伤害，再以同一cross-fit方式同时应用所有accepted坐标；若任一类的positive correct下降或false positive增加，则`atomic_fallback=true`并精确返回D46。K1/K2也精确D46 fallback。

## 3.协议、对称性与禁止项

全部证据来自support inner-held分数和support标签；outer-held/query/clean/source不可达。公式对类标签置换等变，不使用class ID、old/new角色、scene、receiver或handle。无alpha、temperature、clip、threshold、坐标顺序、贪心迭代、第二arm或参数扫描。before/final分别按同一公式独立拟合；这不是按角色分支。最终只保存一套int8/FP16 affine state，逐query独立全类argmax，dense query graph为0。

## 4.成功门与停止门

D57必须至少保持D46的before92.22%、after81.67%、new84.67%、H82.33%、forget≤10.56pp、min-after53.33%、min-new73.33%、joint23.33%，并严格改善after/forget/floor至少一项；三场景不得交换伤害；INT8/FP32翻转为0/0/0；至少1个final prediction改变。失败即停止，不放宽门、不扫描，不跑第二seed/formal/125。

完成后必须详细报告7候选、3场景、11类、15fold、D46/D56同折变化、每类base/adjusted positive与false-positive计数、accept mask、atomic fallback率、补偿分布、20epoch、量化、资源和全部artifact SHA。D57完成后执行D55—D57三轮技术复盘。

## 5.实施计划

1. 复用D56一次额外inner-score refit，不增加第二套head或query state。
2. 添加单类双向门、联合原子门、rank/class置换、K1/K2、无坐标顺序、资源和tamper测试。
3. `ssr-gpu`窄验证、Git提交、clean detached worktree锁定后，只运行一次105行本地development矩阵。
4. 当前不访问N607；若后续候选通过开发门，远端动作须先执行规定preflight。

## 6.本地实现与验证

- 方法脚本：`code/scripts/probe_d57_crossfitted_bidirectional_flow_gate.py`，SHA256=`e91a4c4cbe20483493aa7846ce4c789be8022b7bb757ef13159591436440bb09`。
- 测试脚本：`tests/test_probe_d57_crossfitted_bidirectional_flow_gate.py`，SHA256=`074a8134b9ed367ce2d620dd38406a03ea28863e4a3213bab7b9c3727c221d67`。
- 资源闭包：复用D56的68次LDA拟合库存；D57新增LDA拟合数、优化步数、query state均为0，只新增cross-fit计数、逐坐标门和联合门的标量运算/比较。
- 安全闭包：每坐标分别验证positive不降与false-positive不增；联合交互不安全时清空全部mask并精确返回D46；K1/K2无条件精确返回D46。
- 验证命令：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q tests\test_probe_d57_crossfitted_bidirectional_flow_gate.py tests\test_probe_d56_loo_confusion_flow_intercept.py tests\test_probe_d46_classwise_loo_reliability_fusion.py`。
- 验证结果：31/31通过；覆盖安全坐标生效、联合交互原子回退、K1/K2回退、类置换等变、坏证据闭锁、D56/D46全回归链。

## 7.执行锁

- 实现提交：`05864827f5e9444ec89649d33f1abfa644934092`；clean detached worktree：`E:\type10-7\code\snapshots\d57wt`，执行前状态为`HEAD (no branch)`。
- clean探针SHA256：`39efa88e4012bc742c972d19b1b714adc33632c661a9209ddc3c74d7d462d745`（Git checkout后的CRLF字节）；clean环境D57＋D56＋D46测试31/31通过。
- runtime只读复用`E:\type10-7\code\snapshots\d41wt`；数据、seal、authorization envelope和int8组件均不重建、不修改。
- 本地前台串行执行，Conda环境`ssr-gpu`，`--device auto`；不访问N607。launcher PID在启动时记录；Runner日志为输出目录下`training_log.jsonl`，预期还包括`metrics.jsonl`、`support_audit.json`、receipt和`D57_PROBE_METADATA.json`。
- 2026-07-19首次启动PID12088在组件加载前被`ADV3B02 class binding SHA256 drift`拒绝，exit1；原因是最初命令误用clean checkout中的CRLF字节副本。失败后目标输出目录不存在、训练行0、无D57 Python残留，因此不是一次候选实验。唯一修复是把`class-binding`改回哈希已验证的Git承载面原字节文件；其SHA256=`bb89a1db…c901f`，其余参数完全不变。
- 输出`E:\type10-7\automation_reports\CV-SincNet\d57_crossfitted_bidirectional_flow_gate_probe_20260719\crossfitted_bidirectional_flow_gate`启动前必须不存在。只允许以下105行development命令执行一次：

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d57wt\code\scripts\probe_d57_crossfitted_bidirectional_flow_gate.py' `
  --d57-arm crossfitted_bidirectional_flow_gate `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d57wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d57_crossfitted_bidirectional_flow_gate_probe_20260719\crossfitted_bidirectional_flow_gate' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

## 8.结论先行

D57的安全门在before15个fit和final15个fit上全部拒绝，没有一个类坐标同时满足“positive correct不降、false positive不增且至少一项严格改善”。因此30/30个fit均触发`no_coordinate_accepted_exact_d46_fallback`，补偿、系数变化和截距变化全部为0；D57的15折预测哈希、总体、场景、逐类、floor和混淆与D46完全一致。

具体性能为before-old92.22%、after-old81.67%、seen-new84.67%、同排H82.33%、forgetting10.56pp、joint23.33%、min-before80.00%、min-after53.33%、min-new73.33%，混淆old→new/new→old/new→new为25/8/15。D57没有退化，但也没有产生预注册要求的至少1个预测变化或任何严格性能改善，距离K10确认门after92%、min-old88%、new5 92%仍分别差10.33pp、34.67pp、7.33pp，故不晋级、不跑第二seed、不formalize、不运行125。当前最强合法版本仍是D46，D57只是D46的安全等价回退证据。

## 9.七候选完整同排性能

unknown、coverage、rollback、defer不属于本闭集support-only Runner，均为N/A。每行指标均来自同一候选的15个outer rows。

|候选|机制|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆old→new/new→old/new→new|判定|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|B3_SINGLE_IQ_DIAG_FFTRF|B3单IQ对角FFTRF|87.78%|75.56%|72.67%|73.35%|12.22pp|23.33%|80.00%|60.00%|40.00%|33/22/19|低于D46|
|D42-D40-HNBR-INT8-NEGATIVE|HNBR旧负路线|85.56%|85.00%|15.33%|25.16%|0.56pp|0.00%|66.67%|63.33%|0.00%|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|BEC旧负路线|86.11%|20.56%|78.67%|31.50%|65.56pp|0.00%|76.67%|0.00%|36.67%|142/0/32|旧类崩塌|
|D42-PROTOnet-CDA-ZID160|ProtoNet-CDA|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|
|D42-USLDA-FP32-MATCHED|D57 matched FP32|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|80.00%|53.33%|73.33%|25/8/15|与INT8一致，精确回退|
|D42-USLDA-INT8|D57双向安全门|92.22%|81.67%|84.67%|82.33%|10.56pp|23.33%|80.00%|53.33%|73.33%|25/8/15|主候选，等价D46|
|Z0_SUPPORT_ONLY|support-only原型|71.11%|48.33%|52.67%|48.97%|22.78pp|0.00%|33.33%|13.33%|3.33%|0/0/0|弱基线|

## 10.三场景性能与表现

|场景|before|after|new|H|forget|joint|min-before|min-after|min-new|混淆|行为|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|leo_clear_weak|98.33%|90.00%|98.00%|93.57%|8.33pp|40.00%|90.00%|70.00%|90.00%|4/1/0|最佳场景；5/5fit回退D46|
|leo_low_elev_weak|88.33%|78.33%|76.00%|75.98%|10.00pp|20.00%|80.00%|60.00%|50.00%|8/5/7|新类与旧类均低于门；5/5fit回退D46|
|leo_rain_weak|90.00%|76.67%|80.00%|77.45%|13.33pp|10.00%|60.00%|30.00%|70.00%|13/2/8|after和旧类floor最差；5/5fit回退D46|

场景行为不是平均退化：clear已接近new门但after仍差2pp；low-elev同时存在旧新不足；rain的after仅76.67%、min-after30%，说明主要瓶颈仍是困难场景下的旧类适应和下尾，而不是量化或新类整体均值单一问题。

## 11.逐类别性能

|旧类|哈希前缀|before→after|变化|
|---|---|---:|---:|
|O0|cls_1f33|90.00→90.00%|0.00pp|
|O1|cls_33bb|96.67→90.00%|−6.67pp|
|O2|cls_75aa|96.67→90.00%|−6.67pp|
|O3|cls_8b02|80.00→53.33%|−26.67pp，全局旧类floor|
|O4|cls_a53c|100.00→73.33%|−26.67pp|
|O5|cls_f8df|90.00→93.33%|+3.33pp|

|新类|哈希前缀|seen-new|表现|
|---|---|---:|---|
|N0|cls_09f8|73.33%|全局新类floor|
|N1|cls_1c2a|93.33%|唯一超过92%|
|N2|cls_b8fb|76.67%|次低|
|N3|cls_d3af|90.00%|接近门|
|N4|cls_f608|90.00%|接近门|

场景下尾：low-elev的N0/N2均50%，rain旧类O3=30%且O4=60%。D57未用具体类ID做保护，全部类使用同一门控公式；上述类名只用于事后解释。

## 12.十五折完整表现

|场景|fold|before|after|new|H|forget|joint|before/after/new floor|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|clear|0|100.00%|100.00%|90.00%|94.74%|0.00pp|50.00%|100/100/50%|0/1/0|
|clear|1|100.00%|83.33%|100.00%|90.91%|16.67pp|0.00%|100/0/100%|1/0/0|
|clear|2|91.67%|83.33%|100.00%|90.91%|8.33pp|50.00%|50/50/100%|1/0/0|
|clear|3|100.00%|91.67%|100.00%|95.65%|8.33pp|50.00%|100/50/100%|1/0/0|
|clear|4|100.00%|91.67%|100.00%|95.65%|8.33pp|50.00%|100/50/100%|1/0/0|
|low|0|91.67%|66.67%|80.00%|72.73%|25.00pp|50.00%|50/50/50%|4/1/1|
|low|1|66.67%|58.33%|70.00%|63.64%|8.33pp|0.00%|50/50/0%|1/0/3|
|low|2|91.67%|91.67%|70.00%|79.38%|0.00pp|0.00%|50/50/0%|0/2/1|
|low|3|100.00%|100.00%|70.00%|82.35%|0.00pp|0.00%|100/100/0%|0/1/2|
|low|4|91.67%|75.00%|90.00%|81.82%|16.67pp|50.00%|50/50/50%|3/1/0|
|rain|0|83.33%|83.33%|60.00%|69.77%|0.00pp|0.00%|50/50/0%|2/0/4|
|rain|1|100.00%|66.67%|90.00%|76.60%|33.33pp|0.00%|100/0/50%|4/1/0|
|rain|2|91.67%|83.33%|80.00%|81.63%|8.33pp|50.00%|50/50/50%|1/0/2|
|rain|3|91.67%|75.00%|90.00%|81.82%|16.67pp|0.00%|50/0/50%|3/0/1|
|rain|4|83.33%|75.00%|80.00%|77.42%|8.33pp|0.00%|50/50/0%|3/1/1|

## 13.与D46和D56比较

|指标|D46|D56|D57|D57−D46|D57−D56|
|---|---:|---:|---:|---:|---:|
|before|92.22%|91.67%|92.22%|0.00pp|+0.56pp|
|after|81.67%|83.33%|81.67%|0.00pp|−1.67pp|
|seen-new|84.67%|80.67%|84.67%|0.00pp|+4.00pp|
|H|82.33%|80.95%|82.33%|0.00pp|+1.39pp|
|forgetting|10.56pp|8.33pp|10.56pp|0.00pp|+2.22pp，变差|
|joint|23.33%|23.33%|23.33%|0.00pp|0.00pp|
|min-after|53.33%|56.67%|53.33%|0.00pp|−3.33pp|
|min-new|73.33%|60.00%|73.33%|0.00pp|+13.33pp|
|混淆|25/8/15|21/12/17|25/8/15|0/0/0|+4/−4/−2|
|改变预测折数|—|5/15相对D46|0/15相对D46|0|5/15相对D56|

D57完整撤销了D56在5折上的old/new交换：恢复4pp新类与13.33pp新类floor，同时也放弃D56的1.67pp after、2.22pp forgetting和3.33pp min-after改善。安全门做到了不伤害，但没有将D56信号分解成可联合改善的坐标。

## 14.双向门机制审计

|阶段|类坐标总数|positive受损|FP增加|两者同时受损|positive严格改善|FP严格改善|双向非劣但全相等|accepted|active fit|atomic fallback|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|before|90|27|34|0|13|12|29|0|0/15|0/15|
|final|165|45|80|3|31|25|43|0|0/15|0/15|

before cross-fit流绝对值mean/max为0.02136/0.11905，final为0.01686/0.12987；信号不是全零。问题是严格改善总伴随另一侧伤害，剩余双向非劣坐标又全部没有严格变化。最终补偿L1/L2/max、系数变化L2、截距变化L2全部精确为0。

下表给出每个final fit的逐匿名类向量，顺序固定为该row注册类顺序；`base+`/`coord+`是positive correct，`baseFP`/`coordFP`是false positive，mask全为0。完整before与final向量同时保存在`full_performance_summary.json`引用的training log审计中；before也为15/15全零mask。

|场景/fold|base+|coord+|baseFP|coordFP|mask|
|---|---|---|---|---|---|
|clear/0|7/5/8/8/7/7/7/7/8/8/7|7/8/8/4/7/7/7/7/8/0/7|1/1/0/1/0/0/1/1/0/3/1|3/42/0/1/3/1/2/1/0/0/2|00000000000|
|clear/1|8/7/7/8/7/7/8/6/8/7/6|7/0/7/8/7/7/3/8/8/1/8|1/4/0/0/1/0/1/0/0/2/0|1/0/0/0/1/0/1/28/0/1/26|00000000000|
|clear/2|7/6/8/8/8/8/8/6/8/8/7|7/6/8/8/8/8/8/7/8/0/7|1/2/0/0/0/0/0/0/0/3/0|3/3/0/0/0/0/0/35/0/0/6|00000000000|
|clear/3|8/6/8/8/7/7/8/7/8/7/7|4/6/8/8/7/7/2/7/8/2/7|1/1/0/0/1/0/1/0/0/2/1|1/27/0/0/1/0/1/9/0/2/2|00000000000|
|clear/4|7/5/8/8/8/7/8/6/8/7/7|5/5/8/8/8/7/6/8/8/1/7|2/2/0/0/0/1/1/0/0/3/0|2/19/0/0/0/1/1/36/0/0/5|00000000000|
|low/0|7/3/5/7/6/6/7/3/6/6/7|7/7/5/2/6/6/5/4/8/0/7|1/2/1/3/1/1/2/5/0/8/1|1/40/5/3/6/1/2/10/1/0/2|00000000000|
|low/1|7/4/7/6/7/6/7/4/5/7/7|7/7/7/5/7/6/2/7/7/0/7|1/2/1/3/0/1/3/2/1/7/0|1/21/1/3/5/1/0/28/2/0/2|00000000000|
|low/2|6/5/5/8/6/8/7/6/8/6/5|8/6/6/1/6/8/7/0/8/1/7|0/3/0/3/1/0/1/5/0/4/1|16/7/19/0/6/0/1/0/0/1/13|00000000000|
|low/3|6/3/4/8/6/6/5/2/6/6/6|6/3/7/5/6/6/5/2/6/1/6|1/5/0/1/2/1/3/6/1/8/2|3/8/14/1/3/1/3/9/1/1/3|00000000000|
|low/4|7/5/6/8/7/8/6/3/6/6/6|7/6/6/0/7/7/6/3/6/0/7|1/2/1/4/0/1/2/4/0/5/0|1/11/3/0/3/1/3/18/2/1/9|00000000000|
|rain/0|7/3/8/6/6/8/6/6/7/7/5|5/7/8/8/2/7/6/4/7/0/8|2/2/0/0/4/1/1/3/0/6/0|2/40/0/6/0/1/9/5/1/0/39|00000000000|
|rain/1|7/3/7/7/6/8/7/5/5/7/6|7/7/7/7/6/8/3/0/6/1/6|1/1/0/0/1/0/2/9/1/4/1|1/60/0/3/5/0/2/0/11/3/6|00000000000|
|rain/2|6/3/7/7/4/8/6/4/4/6/6|6/8/7/7/4/8/2/0/8/0/7|1/1/0/0/4/0/3/9/0/9/0|16/66/2/4/7/0/2/0/51/0/21|00000000000|
|rain/3|7/2/7/6/5/8/6/4/6/6/6|2/6/7/6/5/8/5/0/6/3/7|3/3/0/1/2/0/3/9/1/3/0|4/41/1/3/6/0/3/0/5/1/13|00000000000|
|rain/4|7/2/6/7/4/8/5/6/5/5/6|5/7/6/7/8/8/5/0/6/0/6|2/3/1/0/0/0/2/11/1/7/0|4/50/7/2/56/0/15/0/13/0/33|00000000000|

## 15.训练表现

20epoch全部为support-only，所有epoch的`query_rows_used_sum=0`。主训练与D46/D56相同，loss由1.0320降至0.1027，support accuracy由95.14%升至100%，门控本身无优化步。

|epoch|loss mean|support acc|grad norm|CE|anchor|
|---:|---:|---:|---:|---:|---:|
|1|1.0320|95.14%|1.0838|1.0320|0.000000|
|2|0.8014|95.97%|0.8706|0.8014|0.000099|
|3|0.6235|97.78%|0.6909|0.6235|0.000293|
|4|0.5005|97.50%|0.5407|0.5005|0.000548|
|5|0.4160|97.78%|0.4363|0.4159|0.000839|
|6|0.3540|98.19%|0.3698|0.3539|0.001145|
|7|0.2991|98.61%|0.3155|0.2990|0.001451|
|8|0.2610|98.89%|0.3014|0.2609|0.001745|
|9|0.2339|99.03%|0.2570|0.2338|0.002017|
|10|0.2161|99.03%|0.2359|0.2160|0.002265|
|11|0.1903|99.58%|0.2206|0.1901|0.002492|
|12|0.1744|99.31%|0.2027|0.1743|0.002698|
|13|0.1606|99.72%|0.1860|0.1605|0.002888|
|14|0.1527|99.86%|0.2058|0.1526|0.003062|
|15|0.1424|99.72%|0.1740|0.1422|0.003222|
|16|0.1314|100.00%|0.1665|0.1312|0.003368|
|17|0.1268|99.72%|0.1705|0.1266|0.003501|
|18|0.1151|99.72%|0.1474|0.1150|0.003621|
|19|0.1099|99.86%|0.1314|0.1098|0.003730|
|20|0.1027|100.00%|0.1354|0.1025|0.003828|

support收敛不代表outer成功；本轮门控正确拒绝了support-held上存在双向冲突的所有非零修正。

## 16.量化、资源与协议

- INT8与matched FP32完全一致：before outer、final outer、before support、final support argmax变化均0，margin sign flip0；最大score绝对误差min/mean/max为0.000377/0.000946/0.001915。
- 资源：68次LDA fit、2,010,728,448 LDA MAC；其中D56基础附加32次fit、944,898,048 LDA MAC。D57新增fit0、优化步0、标量MAC-equivalent11,136、比较5,024；总适配2,022,245,234 MAC。query为6,624 MAC，参数2,016，persistent state8,583B，registry941B，CUDA峰值22,886,912B，20epoch/20step。
- D57只增加support adaptation时的门控运算，不增加query state；最终仍为int8 coefficient＋float16 intercept单affine逐query全类argmax。
- query rows/features/labels/role/quota/true-count/global assignment/dependent optimization均0/false；clean/source访问false；dense query graph0B；`query_opened=false`。
- 本development Runner没有独立记录单query平均/P95 wall-clock latency或backbone/FFT前向次数，因此这两项为N/A，不能由Runner总耗时112.812s推断部署延迟。

## 17.产物与完整性

|artifact|bytes|SHA256|
|---|---:|---|
|D57_PROBE_METADATA.json|2,019|`2fd3b38bd6521aecb61271b8cb9154029cc81c3e03350a394bd25237fa63f0ed`|
|full_performance_summary.json|126,630|`782416ea5bfba64a7e40a4e15a85871268834a26ca1ed427608b3965556a1efb`|
|geometry_audit.json|5,132|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|RECEIPT.json|5,036|`ed996f83e63530f604d8342b07957f9da1d57f0fd66368e1759301b9e371318b`|
|resource_audit.json|6,498|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|selection.json|2,990|`1bb7708d18d83f2afcf1e2238637d00e0f3842cc80cf330721b4e0b3a6073fe9`|
|support_audit.json|313,681|`8c865f4d61ad416c654fa9d2067b8a020b64535a7791f8831320ab46c77a7de7`|
|training_log.jsonl|19,561,925|`84ea8085a8021535c707a27cc07f28aec13a94c66a84a189a4060feb7c1d5bdd`|

输出目录：`E:\type10-7\automation_reports\CV-SincNet\d57_crossfitted_bidirectional_flow_gate_probe_20260719\crossfitted_bidirectional_flow_gate`。parser为`code/scripts/summarize_d57_performance.py`。运行前预期的独立`metrics.jsonl`未由该Runner生成；完整性能来自105行`training_log.jsonl`并由summary闭包保存，未伪造缺失产物。

## 18.成功门与判定

|门|要求|D57|判定|
|---|---|---|---|
|保持D46|before/after/new/H/floor/forget/joint均不差|全部精确相同|通过|
|严格改善|after、forget或floor至少一项严格改善|0项改善|失败|
|预测变化|至少1个final prediction变化|0/15折|失败|
|三场景无交换伤害|不得以旧换新或以新换旧|全部精确D46|通过但无收益|
|量化|INT8/FP32翻转0/0/0|全部0|通过|
|协议|query/source/clean/Oracle/quota不可达|全部闭合|通过|
|K10项目门|after≥92%、min-old≥88%、new5≥92%|81.67/53.33/84.67%|失败|

最终判定：`COMPLETED_DIAGNOSTIC_NEGATIVE_EXACT_D46_FALLBACK_NOT_PROMOTABLE`。

## 19.D55—D57三轮技术复盘

复盘已重读active objective与`项目.md`，查询项目conversation index中LOO、混淆流、截距和D46路线，并核对D55、D56、D57完整105行日志与报告。协议再确认：三轮均只读固定LEO_weak received IQ及support，query0、clean/source0、无角色Oracle、无class quota、无全局重分配；旧域适应与新类注册始终按同排指标等权判断。

|轮次|机制|总体性能|关键行为|结论|
|---|---|---|---|---|
|D55|raw classwise LOO-CE直接作截距|83.33/70.56/69.33/H68.46，forget12.78pp|15/15折改变，双侧全面恶化|连续CE量纲与部署logit不匹配；永久停止raw CE截距|
|D56|一次性无角色混淆流|91.67/83.33/80.67/H80.95，forget8.33pp|old→new−4但new→old+4；5/15折改变|含旧类修复信号，但以新类和floor为代价；停止强度/clip扫描|
|D57|cross-fit双向类门＋原子回退|92.22/81.67/84.67/H82.33，forget10.56pp|30/30fit、255/255坐标无一可严格双向接受；0/15折改变|安全有效但学习能力为零；停止基于离散混淆计数的截距修正族|

保留的成功经验是：D46的classwise LOO full/block可靠性融合仍是唯一同时保持较高before、after和new的合法底座；D56说明support内部“吸收/被吸收”拓扑能定位rain遗忘，但D57证明单坐标截距无法把该信号转化为双向改善。下一轮不得继续CE、degree、流强度、clip、场景门、old/new门或特定类保护。

下一研发方向转向D46底座上的统一连续几何，而不是离散预测计数：以每类support inner-held正确类margin下尾和非目标吸收margin为双向证据，在full/block两个已归一化head内构造类置换等变的低秩/对角收缩；目标是直接改善正margin下界与负吸收上界，并用同一公式覆盖全部注册类。必须预注册无超参闭式形式、K1/K2回退、support-only闭包和D46严格保持门；若不能在窄单测后产生非零且双向安全的support-held更新，则不读取outer结果、不运行125。
