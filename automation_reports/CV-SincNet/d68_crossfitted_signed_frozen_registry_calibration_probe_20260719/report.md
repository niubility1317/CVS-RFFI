# D68交叉拟合有向冻结registry标定探针

## 1.执行前登记

- 实验ID：`d68_crossfitted_signed_frozen_registry_calibration_probe_20260719`；operator：Codex；最终状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 目标：把D65冻结Stage2-B决策几何的低遗忘信号转化为旧/新全注册类可比的单一affine head，同时避免D67连续堆叠的支持代理错配。
- 当前联合最强仍为D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D65信号为A86.11、F6.11、min-A70.00，但N59.33、H67.12；D67连续堆叠为A82.78、N83.33、H82.16、F10.00，未晋级。
- cell固定receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；复用同一`VALIDATED_ONCE/p2_min_v1`D18 enrollment-only support，不因方法变化重验数据。
- 根目录`E:\type10-7`非Git；代码、追踪和本报告镜像进入`E:\type10-7\github_publish\CVS-RFFI-repo`。D67最终证据提交为`9365099e`，其余工作树改动均不属于D68。

## 2.实证根因

D67完整日志显示，D62仿射行在before90行、final165行中的正类均值始终高于负类均值，方向反转数为0。D65则在before90个旧类行中有12个反转，final165行中有19个反转，其中12个旧类、7个新类。D67标准化使用`abs(mean_pos-mean_neg)`确定gap尺度，却保留原始行方向；因此D65 final支持风险4.139319，是D62的0.532406的7.78倍，闭式`alpha`只能压到均值2.906%。

D68检验一个可证伪假设：D65的主要失败之一不是冻结几何本身，而是冻结/追加行之间存在符号和尺度不一致；用相同support-only公式先校正每个匿名行的方向，再统一标定，可能保留D65旧类稳定性并恢复新类竞争力。

## 3.唯一机制锁

对每个stage、每个已注册匿名类`c`，按physical rank执行leave-one-rank-out交叉拟合。K8时为8折：每折held一个rank/类、train七个rank/类；每个support physical row恰好held一次，held不得参与对应D65 expert、方向或标定统计的训练。

每折在train support构造D65冻结Stage2-B covariance/Stage2-C append-only expert，并对held support产生原始score。聚合全部inner-held score后，对每个类计算：

```text
delta_cv,c = mean_positive_cv,c - mean_negative_cv,c
orientation_c = +1, if delta_cv,c >= 0; otherwise -1
```

方向只由交叉拟合held score确定；没有方向阈值、置信门或class名单。随后在full support构造一个D65 expert，并对其full-support原始score计算：

```text
center_c = (mean_positive_full,c + mean_negative_full,c) / 2
within_c = sqrt((var_positive_full,c + var_negative_full,c) / 2)
gap_c = abs(mean_positive_full,c - mean_negative_full,c) / 2
scale_c = max(within_c, gap_c, float32_eps)
h_c(x) = orientation_c * (g65,c(x) - center_c) / scale_c
```

全部`h_c`删除类公共affine项后编译为一个全注册类head。before/final、旧/新类均用同一公式；Stage2-C只沿用D65合法生命周期中的冻结covariance和追加行，不在query读取注册阶段或角色。K1因无法交叉拟合而精确回退D62；K≥2使用leave-one-rank-out，不设K专属参数。

## 4.与历史路线的区别及ground边界

- 不同于D67：不混合D62/D65，不求`alpha`，不把support平方风险当作连续专家权重；D67整条连续堆叠路线保持关闭。
- 不同于D65：最终用于argmax的不是原始冻结行，而是交叉拟合方向锁＋全support统一尺度的有向行。
- 不同于D62：没有TP/FP离散行替换、Fisher residual或atomic gate。
- 不使用旧/新角色offset、class ID规则、scene/receiver分支、outer-held/query拟合、threshold/temperature/ridge/fold扫描。
- D68不读取地面组件。当前D22 manifest明确为`formal_phase2_eligible=false`、`UNVERIFIED_UNDER_CURRENT_PROTOCOL`；把它加入候选会使方法无法满足最新正式目标。D66已真实读取84个ground int8 cell，但只得到A+1.11pp、N−1.33pp及floor交换，不能把“已读取”误写成“已有效利用”。

## 5.判门、停止条件与完整报告要求

- 相对D62，总体B/A/N/H/J、三项全局class floor、三场景同类指标、遗忘和三类混淆不得交换伤害，并至少严格改善A、F、J或任一floor。
- INT8相对matched FP32的before/final support与outer argmax变化、margin sign flip必须为0；全部分数有限。
- leave-one-rank-out必须exact-once且held/train交集0；`orientation∈{-1,+1}`，类置换等变；最终只保留单一affine state，query额外MAC/state为0。
- 若支持内有向D65风险仍显著高于D62，或真实outer不满足无交换门，状态即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，停止方向标定路线；不扫描方向阈值、scale、fold、温度或按角色修补。
- 即使通过也先运行第二development seed，不直接启动125。
- 真实105行完成后必须报告7候选、3场景、11类、15fold、方向反转/稳定性、support风险、量化、20epoch训练、资源、artifact、D62/D65/D66/D67同排对照和目标缺口，不得只报告缺陷。

## 6.待实施与验证

新增独立D68数学core、probe、专项测试和摘要，不修改D62/D65/D67历史实现或artifact。测试至少覆盖leave-one-rank-out exact-once、符号解析例、类置换、K1 D62回退、D65 lifecycle、共同affine中心化、INT8编译等价、禁止分支和资源闭包。

本轮先本地实现与验证，不访问N607。代码验证后提交、建立干净worktree、复跑D42–D68完整链，再补精确105行命令和输出目录。

## 7.R1真实运行前生命周期修订

首版实现和D42–D68全链325/325通过后，在真实运行前复核发现：若Stage2-C用11类full support重新标定6个旧行，会改变旧行的center/scale/orientation并破坏D65最有价值的“注册后旧行冻结”性质。R1因此在任何真实性能计算前修订为：

1. Stage2-B完成6个旧行的交叉拟合方向锁和full-old support统一标定，删除旧类共同affine项后冻结全部旧行字节、方向和共同项。
2. Stage2-C仍对11类执行leave-one-rank-out，以同一匿名公式为5个新行确定方向和full-support尺度；只把新行减去Stage2-B冻结的同一个共同affine项后追加。
3. 6个旧行在Stage2-C输出中必须FP32逐bit不变；新行与旧行均为有向标准分数，最终仍是一个全注册类affine head。query没有old/new角色输入、分支、offset或quota。

此修订替代第3节中“final阶段重新编译全部`h_c`”的含义；full support统计在Stage2-C只决定新追加行，旧行只做只读诊断。它不新增超参数，也不改变cell、数据、判门或停止条件。专项测试必须新增旧行bitwise冻结断言，然后重跑完整链。

## 8.实现与本地验证

- `code/cvsrffi/stage2_d68_signed_calibration.py`：对称support验证、leave-one-rank-out exact-once、行标准化、class-balanced风险、方向解析解和单affine编译。
- `code/scripts/probe_d68_crossfitted_signed_frozen_registry_calibration.py`：D65生命周期、8折inner expert、Stage2-B共同affine与旧行冻结、Stage2-C有向新行追加、资源与runner闭包。
- `tests/test_stage2_d68_signed_calibration.py`与`tests/test_probe_d68_crossfitted_signed_frozen_registry_calibration.py`：10项专项，覆盖partition、解析翻转、风险下降、类置换、FP32编译、K1 D62回退、旧行bitwise冻结、禁止分支和source closure。
- 初版专项10项中唯一失败来自合成样本精确并列时FP64/FP32 tie-break不同；中心化分数误差仍满足阈值。测试修正为严格检查中心化误差，只对非并列样本要求argmax一致；真实INT8/FP32零变化门未放宽。
- R1专项10/10通过；显式激活`ssr-gpu`后的D42–D68完整链325/325通过，用时81.1s。pytest exit0后仍有既知Windows`pytest-current`临时链接清理权限告警，不属于测试失败。

当前仅有合成/代码验证，没有outer性能结论。下一步提交实现、建立干净worktree并复跑完整链，然后登记精确真实105行命令。

## 9.干净版本、真实运行命令与预期闭包

- 实现提交：`8b5644d656863506da04bc8e46d0dc8c8ac3292c`；干净worktree：`E:\type10-7\code\snapshots\d68wt`，detached HEAD为该提交且`git status -sb`仅`## HEAD (no branch)`。
- 干净D42–D68完整链325/325通过，用时82.8s；pytest exit0后的Windows临时链接清理告警同前，不影响验证。
- 执行source SHA：probe`dd2e4fcb...89257`、D68 core`7348a07c...04c41`、D67 helper`8140bb36...6d128`、D65 helper`bc0c6e14...e4acb`、D62 helper`38ae1114...d4a20`、D67 core`643cd83b...5763f`。
- 本轮本地执行，不使用SSH/SCP/N607；Python为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`。输出目录在登记时不存在，禁止覆盖或失败后原目录重跑。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d68wt\code\scripts\probe_d68_crossfitted_signed_frozen_registry_calibration.py' `
  --d68-arm crossfitted_signed_frozen_registry_calibration `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' --probe-root 'E:\type10-7\code\snapshots\d68wt' `
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
  --output 'E:\type10-7\automation_reports\CV-SincNet\d68_crossfitted_signed_frozen_registry_calibration_probe_20260719\crossfitted_signed_frozen_registry_calibration' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```

预期闭包：105行、30条目标candidate row、60个D68 before/final fit audit、480个leave-one-rank-out partition；每个目标row资源记录每stage8折、16个inner D65 covariance fit＋1个full Stage2-B covariance fit。Stage2-C旧行FP32逐bit不变，ground实际拟合输入0，query/clean/source/role/quota/global assignment访问0。任何生命周期、partition、方向、量化、资源或artifact断言失败均停止并保留原目录。

## 10.真实运行完成状态

- 真实运行完成105/105行，进程exit0；runner耗时51.800s，端到端shell约61s。receipt状态为`DEVELOPMENT_SUPPORT_ONLY_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，选择结果为`Z0_SUPPORT_ONLY`，不得晋级或启动125矩阵。
- D68目标行共15个INT8 outer row和15个matched FP32 row；fit记录60条、目标验证行30条、fit audit60条、leave-one-rank-out partition480条。旧行Stage2-C FP32逐bit不变断言全部通过。
- 结果摘要：`d68_full_performance_summary.json`，129650B，SHA256=`c5f932f893ef59c835aa4d0b7c3693ce3d4af94f0fe07bcb74528995911d98a4`。

## 11.七候选同排性能

百分比均为15个outer row同排均值；`J`为joint floor，`min-B/A/N`为跨全部row聚合后的最差类准确率；混淆顺序为旧→新/新→旧/新→新。

|候选|B|A|N|H|F|J|min-B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78|75.56|72.67|73.35|12.22|23.33|80.00/60.00/40.00|33/22/19|
|D42-D40-HNBR-INT8-NEGATIVE|85.56|85.00|15.33|25.16|0.56|0.00|66.67/63.33/0.00|2/0/0|
|D42-D41-BEC-INT8-NEGATIVE|86.11|20.56|78.67|31.50|65.56|0.00|76.67/0.00/36.67|142/0/32|
|D42-PROTOnet-CDA-ZID160|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|0/0/0|
|D42-USLDA-FP32-MATCHED|59.44|52.78|14.00|18.70|6.67|0.00|50.00/43.33/0.00|19/118/11|
|**D42-USLDA-INT8（D68）**|**58.89**|**51.67**|**14.00**|**18.66**|**7.22**|**0.00**|**50.00/43.33/0.00**|**20/118/11**|
|Z0_SUPPORT_ONLY|71.11|48.33|52.67|48.97|22.78|0.00|33.33/13.33/3.33|0/0/0|

D68的`F=7.22%`不能解释为旧类保护成功：遗忘定义是`B-A`，而B已经从D62的92.78%塌至58.89%。相对D62虽然F数值减少3.33pp，但A下降30.56pp、N下降70.67pp、H下降63.97pp、J下降26.67pp；这是低起点造成的伪改善。

## 12.三场景、逐类与逐fold表现

|场景|B|A|N|H|F|J|min-B/A/N|row-floor B/A/N|混淆|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|60.00|51.67|18.00|21.54|8.33|0.00|40.00/40.00/0.00|20.00/20.00/0.00|7/39/2|
|leo_low_elev_weak|63.33|51.67|6.00|8.33|11.67|0.00|50.00/40.00/0.00|30.00/10.00/0.00|10/40/7|
|leo_rain_weak|53.33|51.67|18.00|26.10|1.67|0.00|30.00/30.00/0.00|0.00/0.00/0.00|3/39/2|

|匿名类|角色|B或N前|A或N后|变化|
|---|---|---:|---:|---:|
|cls_1f3344|旧|50.00|50.00|0.00|
|cls_33bbd1|旧|66.67|43.33|-23.33|
|cls_75aa6d|旧|60.00|56.67|-3.33|
|cls_8b02d9|旧|56.67|50.00|-6.67|
|cls_a53ca1|旧|63.33|53.33|-10.00|
|cls_f8dfc2|旧|56.67|56.67|0.00|
|cls_09f800|新|—|0.00|—|
|cls_1c2ad8|新|—|30.00|—|
|cls_b8fbac|新|—|3.33|—|
|cls_d3afb5|新|—|30.00|—|
|cls_f608a3|新|—|6.67|—|

|场景-fold|B|A|N|H|F|J|旧→新/新→旧/新→新|
|---|---:|---:|---:|---:|---:|---:|---:|
|clear-0|83.33|83.33|10.00|17.86|0.00|0.00|1/7/2|
|clear-1|33.33|25.00|40.00|30.77|8.33|0.00|1/6/0|
|clear-2|41.67|33.33|20.00|25.00|8.33|0.00|1/8/0|
|clear-3|83.33|66.67|10.00|17.39|16.67|0.00|2/9/0|
|clear-4|58.33|50.00|10.00|16.67|8.33|0.00|2/9/0|
|low-0|75.00|58.33|0.00|0.00|16.67|0.00|2/10/0|
|low-1|50.00|33.33|20.00|25.00|16.67|0.00|4/5/3|
|low-2|66.67|50.00|0.00|0.00|16.67|0.00|2/8/2|
|low-3|75.00|66.67|0.00|0.00|8.33|0.00|2/8/2|
|low-4|50.00|50.00|10.00|16.67|0.00|0.00|0/9/0|
|rain-0|66.67|58.33|30.00|39.62|8.33|0.00|1/6/1|
|rain-1|41.67|41.67|10.00|16.13|0.00|0.00|0/9/0|
|rain-2|41.67|41.67|10.00|16.13|0.00|0.00|2/9/0|
|rain-3|75.00|75.00|20.00|31.58|0.00|0.00|0/8/0|
|rain-4|41.67|41.67|20.00|27.03|0.00|0.00|0/7/1|

15/15fold的joint floor均为0；低仰角场景5fold中3fold的新类准确率为0。新→旧累计118，说明主要故障不是旧类被新类覆盖，而是行标准化后的旧类分数系统性压过新类。

## 13.机制审计与根因

|阶段|INT8负方向行|交叉拟合方向差均值|fold同向数均值|原始风险均值|有向风险均值|编译support acc均值|
|---|---:|---:|---:|---:|---:|---:|
|Stage2-B before|14/90|0.822705|6.556/8|9.042585|8.151211|83.47%|
|Stage2-C final|28/165（重算29）|0.559746|6.261/8|3.693592|3.334150|50.23%|

INT8目标行共42个负方向行，matched FP32重复审计后总计84；符号校正把平均support风险约降低9.86%（before）和9.73%（final），却仍处于极高绝对水平。更关键的是，每行按自己的`center/scale`等幅化，删除了D65原始冻结几何中用于跨类argmax的绝对尺度。Stage2-B尚未注册新类时B已降到58.89%，所以根因发生在注册前标定；Stage2-C冻结旧行只是逐bit保留了一个已经损坏的head。最终118/150个新类held样本判成旧类，证明旧/新行尺度未对齐。结论是“修符号”不是D65的主解，D65低F主要来自生命周期冻结，而不是有向标准化，也不是地面原型。

## 14.量化、训练与资源

- matched FP32为B/A/N/H/F/J=59.44/52.78/14.00/18.70/6.67/0.00；INT8相对FP32有before outer argmax变化1行、final outer变化3行、margin sign flip1次、before/final support argmax变化2/8次，未通过零变化门。INT8/FP32最大score绝对误差均值0.017435、最大0.030243。
- 三种最小margin均为负：old-new均值-5.6297、new-old均值-7.1029、new-new均值-3.1925，进一步证明joint竞争不可用。

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

20epoch训练loss稳定下降且support acc达到100%，但outer性能灾难，属于明显支持集过拟合/几何编译错配；全程query训练行0。

|资源项|每个target row|
|---|---:|
|closed-form LDA fit|17|
|inner D65 fit MAC|457,539,976|
|D68 calibration scalar MAC|392,448|
|D68新增adaptation MAC|426,262,221|
|总adaptation MAC|491,705,037|
|每query MAC|6,624|
|trainable parameters|2,016|
|persistent/registry state|8,583B/941B|
|峰值CUDA显存|22,886,912B|
|epoch/optimizer step|20/20|

持久状态上限通过；D68额外持久状态、query额外MAC、额外optimizer step均为0。资源合规不改变性能否决。

## 15.与D62/D65/D66/D67同排比较

|版本|ground实际输入|B|A|N|H|F|J|min-B/A/N|旧→新/新→旧/新→新|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|D62|0|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|23/8/15|当前联合最强|
|D65|0|92.22|86.11|59.33|67.12|6.11|16.67|80.00/70.00/46.67|16/28/33|冻结降低F但新类不足|
|D66|84个int8 cell|93.33|83.33|83.33|82.59|10.00|23.33|80.00/53.33/66.67|20/9/16|ground真实接入但负交换|
|D67|0|92.78|82.78|83.33|82.16|10.00|26.67|80.00/53.33/73.33|22/11/14|轻微旧类改善但新类交换|
|D68|0|58.89|51.67|14.00|18.66|7.22|0.00|50.00/43.33/0.00|20/118/11|灾难性负向|

相对D62，D68的K10门缺口为A距92%差40.33pp、min-A距88%差44.67pp、N距92%差78.00pp。相对D65，D68的F反而恶化1.11pp，同时A/N/H分别下降34.44/45.33/48.46pp。停止整条per-row signed calibration路线，不做第二seed、不做scale/温度/角色offset扫描、不运行125。

## 16.地面压缩旧类原型的实际利用结论

本版本没有利用：metadata和resource audit均记录`ground_component_input_count=0`。这是主动遵守最新项目要求，因为D22当前虽具有允许的int8聚合schema，但`formal_phase2_eligible=false`且当前协议下provenance未验证，不能成为正式候选依赖。

历史上真正读取地面压缩旧类特征的是D66：84个int8 domain-class cell、每类14个；但相对D62仅B+0.56pp、A+1.11pp，同时N-1.33pp、min-N-6.67pp、J-3.33pp，属于负交换，不是“有效利用成功”。D65的低遗忘也不来自ground，而来自冻结Stage2-B几何。后续不能仅为了声称使用ground而重新接入；必须先把D22 provenance正式闭合，并要求ground在同一候选行上同时改善旧域适配、新类注册和floor。

## 17.artifact与最终决定

|artifact|大小|SHA256|
|---|---:|---|
|D68_PROBE_METADATA.json|2,275B|`02f82f66f484cc27ff881db4854c360d68b72a9fcb040765eb56ab5ad4c36081`|
|geometry_audit.json|5,132B|`ae4b735a45fdae38eaf8bb6bfd23e57df9d60560144027a1e3e33937556300dc`|
|RECEIPT.json|5,027B|`d9492bef9aeeb371825c9c5320c1c45b2a291ba34e8e12fe20dfbdb72f4d2fdc`|
|resource_audit.json|6,498B|`00f364e567e0462feb955321e6d414c0dc95493dab35d3ed00dfacd71a8d6e2b`|
|selection.json|2,995B|`668ec33c8a0574b4ce9669b4627746b8f8d6e2e88d91f7d2613fb67f5f6f7f8d`|
|support_audit.json|313,685B|`9b871c337aec58e827e151529da924a35a872b6d482feb02e59da75cf6df043f`|
|training_log.jsonl|10,699,705B|`c815398eac96a15978d866b222a92eadac64b1ccd254f1e0b8ab7cf349ef2e59`|

最终判定：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。D62继续保持当前最强。D69只允许检验“保持D62绝对联合尺度、冻结Stage2-B旧行、仅追加D62同族新行”的单一机制；不再做per-row标准化，不读取未正式闭合的ground组件。
