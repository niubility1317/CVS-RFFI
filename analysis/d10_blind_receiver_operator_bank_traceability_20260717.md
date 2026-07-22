# D10盲接收侧operator bank追踪

日期：2026-07-17

范围：在单一sealed LEO_weak received IQ上预登记`base`、widely-linear二阶统计IQ imbalance circularization、温和FFT谱包络去趋势/均衡三个逐样本接收后view，并复用D9的逐类最多2-view稀疏融合、非退化门、注册锁、K10选择锁和资源累加。本轮只打开D8b strict K10-only enrollment support，不打开query、truth、prediction、score或scorer。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D10-01|`项目.md`§7.1、§7.1.1|每个物理样本只使用唯一已叠加LEO_weak的sealed received IQ；三个view共享同一父IQ，不生成额外LEO状态且不增加K|`code/cvsrffi/stage2_blind_receiver_operator_bank.py`、测试、runner|verified|operator provenance、resource audit与真实artifact复核通过|禁止clean/source入口|
|D10-02|任务要求|三view bank严格为`base`、widely-linear circularization、温和FFT envelope EQ|同上|verified|operator registry与runner metadata测试通过|最多3个view|
|D10-03|任务要求|widely-linear operator只用单样本二阶统计，具有数值稳定、退化输入回退和固定强度上限|同上、测试|verified|batch/singleton一致、零能量回退、circularity下降、RMS gain cap测试通过|不得跨样本估计|
|D10-04|任务要求|FFT operator使用固定小shrink做平滑谱包络去趋势/均衡，保持FFT相位与残余CFO，不进行CFO估计、移频或去旋|同上、测试|verified|逐bin phase、主峰bin、残余谱形与gain边界测试通过|保留细粒度TX指纹|
|D10-05|任务要求|复用D9逐类最多2-view稀疏凸融合、K10 support删除选择、逐类/总体/floor非退化门|同上|verified|D10+D9联合回归及真实selection audit通过|D9 engine为唯一选择内核|
|D10-06|任务要求|若D10最终support总体、每类和floor未全部非退化，或相对base没有任何准确率严格改善，则自动回退base并报告|同上、测试、runner|verified|三operator feature完全相同时强制base回退反例通过；真实6个状态均有严格改善|保守晋级门|
|D10-07|Stage2-C注册边界|After旧类operator、weight、prototype、calibration和lineage锁定，仅追加新类；旧类support intrusion guard不退化|同上、测试|verified|旧tensor/calibration bitwise测试与真实artifact复核通过|复用D9 extension|
|D10-08|K-shot统一工作点|只在严格K10选择；K1/K5只用K10有序lineage前缀重建prototype，不得重选operator、weight或calibration|同上、测试、runner|verified|12个真实nested证明及K1/K5单测通过|保存嵌套证明|
|D10-09|`项目.md`§7.2|query接口逐样本面对全部注册类，必须使用samplewise sealed extractor，无label/role/quota/global assignment/query fit|同上、测试|blocked|合成callback的batch局部性与plain extractor反例通过，但真实runner使用`forward_zid160(...,batch_size=64)`，未把正式callback绑定为samplewise sealed extractor|未打开query，但不能据此授权candidate-bound query|
|D10-10|资源约束|0参数、0epoch、最多3个去重backbone前向、无dense query图、D10状态与base资源合计不超过256KB|同上、测试、runner|blocked|状态与head/operator估算通过；未报告端到端backbone MAC、singleton/P95时延和峰值显存|资源证据不完整|
|D10-11|反例测试|拒绝非float32/非有限IQ、错误operator、跨样本统计、错误父IQ/operator provenance、非K10选择、非嵌套K1/K5、资源超限|测试|verified|聚焦测试24项通过|fail closed|
|D10-12|真实support-only锁定|在D8b strict K10-only before/after enrollment包运行三个场景，输出不可变state/audit/report/COMMIT|`code/scripts/run_d10_support_only_enrollment.py`、artifact|rejected|artifact内部哈希与12个nested证明通过，但COMMIT未绑定代码SHA，旧support复用只锁既有feature数组而未保存正式feature fingerprint；单测还包含random feature mapping|仅保留support diagnostic，禁止candidate-bound query|
|D10-13|证据边界|只报告support删除验证，不声明query性能或正式晋级|本文、artifact|verified|query相关五个flag均为false；floor 0.1–0.5明确不晋级|需要独立未评分query后才能判断性能|

## 预登记operator

1. `base`：原始sealed received IQ。
2. `wl_iq_circularize`：对每个received-IQ样本独立去均值计算`C=E|x|²`与`P=E[x²]`，对`rho=P/C`执行固定模长裁剪后使用稳定widely-linear系数`beta=-rho/(1+sqrt(1-|rho|²))`；校正后恢复原均值与中心RMS。固定`|rho|`上限防止过校正，退化能量输入直接回退base。
3. `fft_envelope_eq`：逐样本FFT，使用固定奇数窗口在log-magnitude上估计圆周平滑谱包络，以固定小shrink和固定gain clip做温和去趋势；FFT相位逐bin原样保留，不估计CFO、不移频、不乘相位斜坡，IFFT后仅恢复输入RMS。

## D9复用与D10保守回退

D10把三个真实operator逐样本feature和真实provenance校验后映射到D9固定三槽选择内核。D9继续负责物理support删除、12个稀疏候选、逐类非退化、组合总体/floor/每类门、After旧类锁与K1/K5原型重建。D10额外要求最终准确率向量相对全base至少一项严格改善；若只有margin变化而总体、floor和所有逐类准确率均相同，则强制全base回退并在audit中记录。

## 验证

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile \
  code/cvsrffi/stage2_blind_receiver_operator_bank.py \
  code/scripts/run_d10_support_only_enrollment.py \
  tests/test_stage2_blind_receiver_operator_bank.py \
  tests/test_run_d10_support_only_enrollment.py
PASS

C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q \
  tests/test_stage2_blind_receiver_operator_bank.py \
  tests/test_run_d10_support_only_enrollment.py \
  tests/test_stage2_floor_sparse_operator_fusion.py \
  tests/test_run_d9_support_only_enrollment.py
........................                                                 [100%]
24 passed
```

首次尝试按`conda activate ssr-gpu`运行时，PowerShell非交互激活仍落到`F:\App\miniconda3\python.exe`且缺少pytest；随后使用已验证的`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`串行重跑通过。这是Conda包装噪声，不是项目失败。

## D8b strict K10 support-only结果

artifact：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d10_support_only_strict_k10_v1`

COMMIT SHA256：

`c43e8a1acf79a9777090903a9a640204d3796be29443c053d40bc405d9c3ca6a`

状态：

`SUPPORT_ONLY_D10_LOCKED_NO_QUERY_OPEN`

|场景|状态|support overall base→final|support floor base→final|strict改善门|最终operator|状态|
|---|---|---:|---:|---|---|---:|
|`leo_clear_weak`|before|0.7167→0.7667|0.1000→0.3000|PASS|3种|26,316B|
|`leo_clear_weak`|after|0.4200→0.5000|0.1000→0.2000|PASS|3种|48,226B|
|`leo_low_elev_weak`|before|0.7333→0.7667|0.3000→0.3000|PASS|`base+fft_envelope_eq`|26,316B|
|`leo_low_elev_weak`|after|0.3800→0.4600|0.1000→0.1000|PASS|3种|48,226B|
|`leo_rain_weak`|before|0.7333→0.7500|0.3000→0.3000|PASS|3种|26,316B|
|`leo_rain_weak`|after|0.6800→0.7000|0.5000→0.5000|PASS|3种|48,226B|

这六行都满足总体、floor和每类不退化，并至少有一项准确率严格改善，因此没有触发D10保守base回退。改善仍只是注册support删除验证；floor仅0.1–0.5，不能据此晋级，更不能声明query性能。

反向证据：

- 39个artifact文件全部只读。
- COMMIT引用的audit、report和18个state文件哈希全部重算一致。
- 12个K1/K5证明全部保持operator indices、weights、calibrations和K10 lineage prefix锁定。
- `query_package_opened`、`query_truth_opened`、`query_prediction_opened`、`query_score_opened`和`scorer_opened`全部为false。
- 无clean/source访问，无额外LEO状态，无view增加K。

反向审计：10项verified，0项deferred，1项rejected，2项blocked。Widely-linear与FFT operator机制、D9稀疏选择复用和support侧K10/K1/K5锁定属于已验证的机制证据；正式feature callback的samplewise authority、旧feature fingerprint、代码身份和端到端资源绑定不完整。当前artifact只能保留为support diagnostic，不能进入candidate-bound query。

## 最终候选选择决定

D10已写入不可变不选择marker：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d10_support_only_strict_k10_v1\SUPPORT_FLOOR_INSUFFICIENT_NOT_SELECTED.json`

虽然六个support状态相对base均有非退化改善，但before floor仅0.3，after-new floor为0.1–0.5，不足以授权candidate-bound query。marker明确：

- `candidate_bound_query_generation_authorized=false`
- `candidate_bound_query_package_created=false`
- 所有query/truth/prediction/score/scorer打开标志均为false
- `performance_claim_authorized=false`

此外已新增正式绑定NO-GO marker：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d10_support_only_strict_k10_v1\SUPPORT_PROTOCOL_BINDING_INCOMPLETE_NOT_SELECTED.json`

该marker明确记录random feature mapping测试证据不能替代正式feature authority、旧feature fingerprint未绑定、正式callback未证明samplewise、runner使用batch64、COMMIT缺少代码SHA，以及端到端MAC/时延/峰值显存缺失。

因此D10最终状态为operator机制已验证但正式绑定被阻断的support-only诊断证据；未被选择、未创建candidate-bound query，也不得进入性能声明。
