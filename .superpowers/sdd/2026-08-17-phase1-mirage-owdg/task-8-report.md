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
