# ERBT-IDR M2.9 TASR48 Phase2实验v2预登记

- run ID：`erbt_idr_m29_tasr48_screen_20260825_v2`
- 当前状态：`ANALYZED / SINGLE_SEED_NEGATIVE_NO_PROMOTION`
- protocol：`p2_min_v1`，只复用原`VALIDATED_ONCE`数据及匹配的`capsule_id/split_id`
- 方法与矩阵：完全复用v1冻结的TASR48实现、5个arm、2个receiver、3个K/new条件和seed `7282101`；不改变任何科学参数
- 新run原因：v1在prediction写出前因确定性`KeyError: 'quantization'`技术失败，0个合法prediction，必须保留现场并使用不可覆盖新输出
- 代码分支：`codex/m29-tasr48-20260825`
- 修复commit：`c80f0f577ce3fd53928c7ef16724cb15ac384eff`

## 一、定点修复

1. 按真实接口读取`state.audit["compiler"]`中的量化审计，不再错误访问不存在的下一层`["quantization"]`；
2. 并行执行器只维持最多`max-workers`个已派发row；任一row失败后停止继续派发，取消尚未运行的future并保留已产生现场；
3. 不改TASR48、FFT96、identity160、D92、数据、评分器、矩阵或晋级阈值。

v1状态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；v2只有在30个prediction receipt与`matrix_index.json`齐全后才启动truth-last scorer。

## 二、冻结输入与输出

- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m29_tasr48_screen_20260825_v2_r3`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v2`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m29_tasr48_screen_20260825_v2`
- 复用只读Phase1 bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/control/m29_tasr_bundle.npz`
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`
- supplemental feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`
- scoring root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`
- supplemental scoring root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3`
- CWD：release内`code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 设备：CPU，`max-workers=2`

## 三、正式命令

prediction：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m29_tasr48_matrix.py \
  --run-id erbt_idr_m29_tasr48_screen_20260825_v2 \
  --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features \
  --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features \
  --tasr-bundle /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v1/control/m29_tasr_bundle.npz \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v2/predictions \
  --device cpu --max-workers 2
```

truth-last评分：

```bash
PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m29_tasr48_matrix.py \
  --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v2/predictions/matrix_index.json \
  --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars \
  --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 \
  --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m29_tasr48_screen_20260825_v2/scores \
  --bootstrap-repeats 2000
```

## 四、停止规则、artifact与晋级门槛

仅在协议/query泄漏、错误receiver/seed/K/new/scene/split、错误checkout、输出碰撞、prediction不闭合、scorer连错truth，或同一确定性pre-prediction异常至少出现两行时技术停止。不得因性能低停止。

预期30个`predictions.cvspred`、30个`row_execution_receipt.json`、`matrix_index.json`、`scored_matrix_index.json`、30个same-row/four-state分数、24个paired比较及`results_summary.json`。

TASR48相对评分后最优FFT96对照必须同时满足：`Delta H>=0.002`、`N_help>N_harm`、old/new floor下降均不超过0.005、deployment state bytes更低，才允许进入多seed或完整125；否则以单seed负结果闭环。

## 五、本地验证

- 量化审计真实行回归先复现`KeyError: 'quantization'`，修复后通过；
- fail-fast回归先因缺少按需派发实现失败，修复后确认首行失败时只执行首行；
- M29及M2.4/M2.5/M2.8相关完整回归共40项通过；
- 6个M29模块/脚本Python编译通过；
- 本次修复不改变数据、方法、矩阵和评分合同。

## 六、v2 prediction与首次scorer结果

- r3 release HEAD：`29fb336d061f65d8f15335053e068140e6a842cd`；归档本地/远端SHA-256均为`16f44de512b4001f2d5a2ccbffd873186cea1e5210947513ebf64363c24df694`；远端编译通过。
- prediction主PID `3012439`及2个CPU worker的CWD、cmdline、父子关系和日志增长均通过启动后检查。
- prediction最终闭合：30个`predictions.cvspred`、30个`row_execution_receipt.json`和`matrix_index.json`齐全；独立读回为`row_count=30`、30个receipt均为`PREDICTIONS_COMPLETE_TRUTH_UNOPENED`、`fit_query_rows_used=0`、`query_truth_opened=false`。
- 首次truth-last scorer仅创建空`scores`根，0个分数文件，随后因M29行为receipt中的`full_block_weights={full:1,block3:1}`不满足旧评分合同“和为1”而停止；prediction已经先完整闭合，scorer结果没有反馈预测器。
- 根因是行为receipt兼容字段的固定值笔误：M29使用全空间D92，合法旧合同表示应为`{full:1,block3:0}`。这不影响已冻结预测值、量化结果、资源值或数据协议。
- 定点修复同时覆盖两条路径：新prediction写合法权重；scorer只对已闭合M29旧receipt的唯一已知`1/1`指纹映射为`1/0`，已合法`1/0`原样保留，其他权重fail-closed。原prediction和空`scores`现场不修改，新评分使用不可覆盖`scores_v2`。
- 红→绿回归已覆盖新receipt与旧receipt评分适配；完整相关回归41项通过。下一release使用不可覆盖`erbt_idr_m29_tasr48_screen_20260825_v2_r4`，不重跑prediction。

## 七、r4评分兼容层补充

- r4 release HEAD：`002e788ced666d895ff0bf613c8384f5cea14feb`；归档本地/远端SHA-256均为`ab113ef46706043b3c7d6a9451b737c3d09ccb36eeae68fd5d1b9463435a72aa`；远端编译通过。
- r4 scorer成功通过行为receipt校验并写出前4个row的8个分数文件，随后在首个TASR48 row因`resource scope declaration drift`停止；`scores_v2`现场保留。
- 对照确认TASR48原resource把两个auxiliary计费标志都设为true，表示Phase1 bundle状态与TASR48 query计算已纳入候选资源；FFT/identity为false。旧scorer只接受false，但原数值资源字段已经包含对应成本。
- 完整修复现统一为M29兼容投影层：原resource真值不改并留给最终资源分析；只向旧scorer投影两个false scope标志；仅接受两个标志均为布尔且彼此相等，其他组合fail-closed。
- 兼容层红→绿回归与完整相关回归42项通过。新评分使用不可覆盖r5 release和`scores_v3`，继续复用已闭合30份prediction。

## 八、最终结果与晋级判断

最终状态为`ANALYZED / SINGLE_SEED_NEGATIVE_NO_PROMOTION`。本轮证据覆盖1个seed、6个paired input identity、30个方法row和90个scene unit。它足以否决当前TASR48进入多seed或完整125，但不能外推为所有频谱压缩方法均无效。

| arm | 平均H | old-class post | new-class | forgetting | 平均old floor | 平均new floor | 平均部署状态字节 |
|---|---:|---:|---:|---:|---:|---:|---:|
|`M29-FFT96-A4`|0.52235|0.58102|0.49278|0.12963|0.21944|0.21667|13577|
|`M29-FFT96-A1`|0.49688|0.55556|0.46611|0.13611|0.22500|0.20556|13577|
|`M29-FFT96-A05`|0.48936|0.54630|0.46208|0.14815|0.20278|0.21389|13577|
|`M29-TASR48-A1`|0.39320|0.45602|0.36153|0.17269|0.12778|0.15000|12777|
|`M29-IDENTITY160`|0.21559|0.35370|0.16167|0.19907|0.06944|0.00556|9077|

冻结FFT96权重中，`M29-FFT96-A4`的平均H最高，因此它是预登记规则下的正式对照。TASR48相对该对照的平均H差为-0.12915，低于+0.002门槛；7560个同query比较中，`N_help=472`、`N_harm=1420`，加权accuracy差为-0.12540。6个paired identity的H差全部为负，范围为-0.07590至-0.16163；6个McNemar双侧p值均小于1.3e-7，方向一致，不是由单一receiver或单一K条件造成。

floor门槛同样失败。TASR48相对FFT96-A4的平均old floor下降0.09167，平均new floor下降0.06667，均超过0.005上限。两种方法的跨场景全局最低floor都出现0，因此这里采用预登记同row汇总的平均class floor判断，而不是用共同为0的极小值掩盖退化。

资源收益不稳定。K1/new20和K5/new20的4个paired identity中，TASR48部署状态为15247字节，低于FFT96的16527字节；K10/new5的2个identity中，TASR48为7837字节，高于FFT96的7677字节，原因是992字节Phase1 bundle在较小分类头上抵消了48维压缩收益。因此“所有同identity部署状态更小”门槛失败。

四状态结果说明TASR48并非完全无效。相对identity160，TASR48在注册前把old accuracy平均提高0.07593；注册后把old accuracy提高0.10231、new accuracy提高0.19986、H提高0.17760。注册本身使old accuracy下降：无DA时为-0.19907，有DA时为-0.17269，old accuracy的difference-in-differences为+0.02639。也就是说，TASR48保留了部分频谱判别信息，并缓和了部分注册损失，但其48维残差摘要丢失的信息远多于receiver扰动校准带来的收益，仍显著弱于完整FFT96。

晋级门槛5项中，`Delta H`、help/harm、old floor、new floor和全条件资源优势全部未通过。本轮不启动多seed或完整125。下一候选若继续压缩频谱，应保留FFT96主分支并把TASR类表示作为附加残差或门控信号，而不是直接替代FFT96；这属于下一轮方法假设，不改变本轮负结论。

最终scorer使用r5 release，HEAD为`19124781919ad1b60abb62850552d2bfbe57a4e4`，归档本地/远端SHA-256均为`e2bc50f8b2df931e9cf2ed4c06259d489b8040e1f15b120dc8be22c24a4aba56`。最终评分根为`runs/erbt_idr_m29_tasr48_screen_20260825_v2/scores_v3`，包含30个same-row、30个four-state、24个paired比较和`scored_matrix_index.json`；`results_summary.json`已独立读回为`ANALYZED`且`promotion_gate.passed=false`。
