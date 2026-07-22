# D4a diag scenario-atomic最小集成Traceability

日期：2026-07-17
范围：仅修改`stage2_diag_cosine_exploration.py`及其直接测试；复用独立D4a原语；不改offline package或runner脚本

|ID|Source|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D4I-01|新单观测协议|新增正式候选名并禁止公共共享head API静默处理|`code/cvsrffi/stage2_diag_cosine_exploration.py`|verified|focused pytest|必须走scenario-atomic orchestration|
|D4I-02|用户要求|每scenario只用本scenario K个support独立拟合，不执行三场景support concat|同上|verified|共享fit失败桩+scenario state shape test|三场景仅共享算法和超参数|
|D4I-03|用户要求|从support/query NPZ中的post-channel IQ SHA生成view lineage|同上|verified|`view_lineage.json`直接测试|physical ID暂以opaque sample token承载|
|D4I-04|用户要求|跨scenario support/query token与post-channel IQ SHA集合两两互斥|同上|verified|独立token fixture、loader回归和真实dev row|两个bundle loader均已改为两两互斥|
|D4I-05|用户要求|support-only拟合，query inference-only且不更新状态|同上|verified|独立原语immutability测试+diag resource断言|复用独立D4a原语|
|D4I-06|用户要求|轻量cosine/prototype head并输出LOO floor与资源审计|同上|verified|state/resource/lineage artifact test|不实现完整新类FloorLock训练head|
|D4I-07|最小修改|只修改diag模块及直接测试，必要小adapter模块为已新增原语|`tests/test_stage2_diag_cosine_exploration.py`|verified|git status|不改offline package与runner|
|D4I-08|交付纪律|聚焦回归、py_compile、diff检查且不提交Git|上述文件|verified|`19 passed`、py_compile、diff/status|保留其它工作树改动|

## 外部阻塞解除与真实开发结果

`somph_predictor_bundle`和通用`stage2_predictor_bundle`已由主线改为跨scenario token两两互斥；offline package按scenario独立选择support/query。D4a随后完成真实`rx20-1/seed713101/K10/new5/10/20`开发行。

D4a合法但性能失败：注册前old_acc为76.39%，new5注册后old_acc/seen-new为62.22%/68.67%，且旧类遗忘14.17pp。该结果证明pipeline已可运行，不证明方法达标；D4a不得晋升，后续改为old-head lock和support-only floor guard。
