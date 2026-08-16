# Task 8报告：source-only阈值校准、same-row评分与Gate

状态：IMPLEMENTED_AND_VERIFIED

## 范围与边界

- 新增code/cvsrffi/phase1_mirage/calibration.py和code/cvsrffi/phase1_mirage/scoring.py，以及对应聚焦行为测试；未改动Task1—7实现，未实现CLI、bundle或target predictor。
- score table为冻结dataclass，显式记录V_cal/P_cal或V_select/P_select角色、物理ID与query ID、quality、unknown risk、support内标记、预测类和fold；known row额外记录true class、receiver、day、scene。
- 校准只接受V_cal+P_cal且update_count=0。role错配、target角色、重复/跨known-proxy ID、非有限/越界分数、列长度错配或多fold表均fail closed。
- 阈值冻结后，score_same_row只读取V_select/P_select的冻结decision table；不重新搜索阈值、不读取target，也不把defer算作proxy显式拒识。

## Python环境

所有Python验证串行使用conda run -n ssr-gpu：

~~~text
sys.executable=C:\Users\lh594\.conda\envs\ssr-gpu\python.exe
CONDA_PREFIX=C:\Users\lh594\.conda\envs\ssr-gpu
torch=2.10.0+cu128
~~~

## TDD记录

### RED

- 初始聚焦命令：conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_calibration.py tests/phase1_mirage/test_scoring.py -q。
- 结果：10项均因缺少cvsrffi.phase1_mirage.calibration失败，证明测试针对尚不存在的实现。
- 自审新增scene等权RED：clear中3个正确样本、rain中1个错误样本时旧same-row macro为.75，错误地按样本量加权；预期fold内scene等权为.50。

### GREEN

- calibration.py使用head.DecisionThresholds/decide生成不可变三态decision table；以P_cal显式拒识→V_cal registered coverage→低defer的字典序搜索经验唯一值/分位点有限grid，并以known FRR≤.10作为硬约束；无可行解抛NO_DEPLOYABLE_SEPARATION。
- scoring.py从单一冻结decision table计算known macro、per-class/min-class、receiver/day/scene、worst-scene、FRR、proxy显式拒识/false accept/defer/coverage和AUROC。fold内scene等权，六fold摘要强制6个唯一fold且不能重用同一decision table。
- Gate1读取显式协议/训练收据；Gate2执行.02/.01/非降/5-of-6全条件；Gate3执行AUROC、delta、FRR和零proxy update全条件；promoted=Gate1&Gate2&Gate3，不存在补偿。
- Gate4仅接受两个已封存target summary并执行精确的macro、min/worst、四scene显式拒识和FRR规则。唯一arm选择仅消费已promote的source receipt，按最弱Gate余量→source macro→proxy AUROC→bundle bytes→稳定arm ID，bundle_bytes缺失按无穷大确定性处理，不读取target。

## 验证

~~~text
conda run -n ssr-gpu python -m py_compile code/cvsrffi/phase1_mirage/calibration.py code/cvsrffi/phase1_mirage/scoring.py
PASS

conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_calibration.py tests/phase1_mirage/test_scoring.py -q
11 passed

conda run -n ssr-gpu python -m pytest -p no:cacheprovider tests/phase1_mirage/test_protocol_policy.py tests/phase1_mirage/test_data.py tests/phase1_mirage/test_proxy.py tests/phase1_mirage/test_model.py tests/phase1_mirage/test_head.py tests/phase1_mirage/test_losses.py tests/phase1_mirage/test_trainer.py tests/phase1_mirage/test_calibration.py tests/phase1_mirage/test_scoring.py -q
125 passed
~~~

## 自审与限制

- reviewed P0/P1：role分离、update_count=0、跨表ID互斥、known FRR定义、defer语义、scene/fold等权、5/6边界、Gate不可补偿和Gate4不反馈。未发现阻断项。
- 本实现只提供评分与收据纯函数，不构成source性能Gate通过或target性能结论；完整矩阵和一次性target artifacts仍由后续任务生成。

## Fix round1：I1/I2/I3与C1权威裁决

- C1裁决：批准规格§3.4只注册leo_clear_weak、leo_low_elev_weak、leo_rain_weak三个target scene；Gate4精确要求global加这三个scene。因此现有global、clear、low_elev、rain四项检查正确，不新增第四个scene。
- I1：正式calibrate_thresholds现在只接受0≤max_known_frr≤.10；.1000001 fail closed，较宽松的阈值不能冒充正式Gate。
- I2：唯一arm的首要比较改为预注册的无量纲Gate2/3连续slack最小值：macro delta=(d-.02)/.02、min delta=(d-.01)/.01、worst-scene=d/.005、fold=(n-5)/1、AUROC=(a-.85)/.15、AUROC delta=(d-.05)/.05、FRR=(.10-frr)/.10。Gate1和proxy update_count=0为布尔闭合，不将所有promoted arm压为同一零余量。
- I3：FrozenDecisionTable只能由使用模块私有seal的校验工厂瞬时建立，实例不存储或暴露seal，并携带DecisionSourceReceipt（role、fold、行数、hash、proxy update count）。score_same_row再次校验receipt、唯一/跨known-proxy ID、role、fold和零update，伪造或篡改表fail closed。
- RED：3项预期失败，分别覆盖放宽FRR、直接/伪seal或交叉ID表、混合原始单位导致的arm排序反转。
- GREEN：同3项通过；Task8聚焦14 passed；Task1—8限定回归128 passed（125项基线加3项新增）；git diff --check通过。

## Fix round2：I3 seal复用闭合

- RED：从合法FrozenDecisionTable读取_factory_seal后，公开构造器仍可接受该token并生成行收据一致的伪表；新测试以该真实复用路径预期拒绝，得到DID NOT RAISE。
- GREEN：FrozenDecisionTable保持init=False且不定义公开构造器；模块私有_create_frozen_decision_table以sentinel调用并使用object.__new__写入已校验字段，但不把sentinel写入实例。普通构造或传入_factory_seal均为TypeError/CalibrationProtocolError，合法factory表仍可same-row评分。
- 保留score_same_row对来源receipt、role、fold、proxy update count、重复ID和跨known-proxy ID的复验；本轮不试图防御object.__new__、反射或模块globals修改等同进程恶意代码路径。
- 验证：I3聚焦RED 1项预期失败→GREEN 1项通过；Task8聚焦14 passed；Task1—8限定回归128 passed；git diff --check通过。
