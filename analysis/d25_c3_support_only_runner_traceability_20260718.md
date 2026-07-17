# D25-C3 support-only runner追溯表

日期：2026-07-18

状态：本地实现与40项相关回归完成，待N607 support-only实验

边界：在既有D25 runner中新增显式`--candidate-set c3_v1`分支，默认`d25_v4`行为和75行锁保持回归不变；历史v4仍由其Git提交与原SHA固定，新分支产生独立source closure。两条分支都只消费同一物理LEO_weak IQ一次提取的`z160/FFT96/RF32`三块并拼接成一行288D表征。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|C3R-01|用户与主任务分工|候选严格锁为`Z0/B3/C0/C3A/C3B/C3C`|`code/scripts/run_d25_support_only_concat.py`、focused tests|verified|candidate-lock test|显式`c3_v1`；C1/C2不重跑，B3仅诊断|
|C3R-02|矩阵定义|6候选×3个LEO_weak场景×5个L2O fold=90行|同上|verified|cardinality test|每类K10中K8 fit、K2 held support诊断|
|C3R-03|单观测协议|每个物理IQ仅生成一行；FFT96/RF32各提取一次并复用|同上|verified|沿用D25 lineage与默认分支回归|不得生成额外LEO状态或派生support行|
|C3R-04|query隔离|CLI、`run`和fit路径无query/truth/role/quota/global assignment/source/clean入口|同上|verified|parser/signature测试|held support只用于开发support-only诊断，不是正式query|
|C3R-05|C3阶段语义|C3A=`20+0`；C3B=`20+10`；C3C strong-floor=`15+15`|同上|verified|config lock test|C3 shared adapter仅288个gamma；三者总adaptation epoch≤30正式档|
|C3R-06|完整训练日志|每个fold内逐step保存完整loss、逐类loss、grad、gamma/state hash|同上|verified|真实3-step fold测试|训练trace嵌入90行日志|
|C3R-07|防遗忘|Stage2-C后shared gamma、旧prototype prefix及旧score列bitwise冻结，并对注册前后fit-old support逐类预测/floor执行非退化门|同上|verified|真实fold hash/metric测试|raw score冻结不等于预测无遗忘；失败候选不得晋级|
|C3R-08|资源与几何|输出全K10资源审计、batch1延迟、MAC、状态及support-only prototype碰撞几何|同上|verified|full-state代码路径与核心资源测试|相对identity-only单qKNN报告比率|
|C3R-09|source closure|锁C3 core、D25 core、D24/CIAF、D19 helper及当前runner SHA|同上|verified|candidate-lock source closure测试|support打开前后重算并完全相等|
|C3R-10|固定artifact面|只写`training_log/support_audit/selection/resource_audit/geometry_audit/RECEIPT`|同上|verified|沿用D25 artifact回归|不得持久化feature/prototype/IQ/logit|
|C3R-11|本地验证|`py_compile`、focused pytest和历史D25默认分支回归全部通过|模块与tests|verified|40项PASS|待launcher/N607运行后补远端证据|
|C3R-12|full-K10终门|fold选择后必须用完整K10、3场景old-support逐类非退化终门复核；失败撤销C3并回退C0|runner、tests|verified|模拟rain失败回归|避免fold正向但full-K10退化仍被标为selected positive|

## 本地验证

- `python -m py_compile code\scripts\run_d25_support_only_concat.py`：PASS。
- `python -m pytest -q tests\test_run_d25_c3_support_only_diag_floor.py tests\test_run_d25_support_only_concat.py tests\test_stage2_multimodal_diag_floor_adapter.py tests\test_stage2_multimodal_concat_fusion.py`：40项PASS。
- `d25_v4`默认候选仍为5个、期望75行、candidate-lock schema v1；`c3_v1`为6个、期望90行、schema v2。
- C3 candidate lock、selector和`selection.json`统一以C0为安全基线；历史v1 lock不增加`candidate_set`字段。source closure覆盖本轮直接执行文件与既有D25/D19控制面，D19更深传递依赖范围沿用历史密封控制，属于本次support-only筛选的已知复现边界，不能外推为formal bundle闭包。
