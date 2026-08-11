# D92 E0 FULL MAXMIN FLOORBOOST Hard11实施计划

> **执行要求：**按本计划完成TDD、最小本地门、独立P0/P1复核和唯一N607发布；不得用候选query结果修改公式、行集、权重或门槛。

**目标：**在保留E0_FULL_ONLY低计算量的前提下，显著提高最弱旧类floor，并把旧类遗忘从“不得变差”升级为明确、大幅改善目标。

**结构：**在同一DA1_REG1 support上各拟合一次FULL和BLOCK；固定0.25强度注入RMS对齐的BLOCK旧类contrast；按每个旧类对全部注册类的20%低分位support margin与同态新类竞争漂移构造retention分数，再执行0.35倍FULL旧类RMS上界的零和max-min bias校正；旧类校正严格零和，新类行逐字节保持FULL，异常数值校准回退E0_FULL_ONLY。K1/K2仍为原D92 FULL exact alias。

**冻结标量：**contrast_lambda=0.25，margin_quantile=0.20，retention_bias_kappa=0.35。分位算法固定为NumPy method=lower，使K5取最小样本、K10取第二小样本；它是算法身份而非第四个可调标量。新候选query结果返回前不得调整。

**技术栈：**Python3.10、NumPy、pytest、p2_min_v1、8-shard N607 RTX3090。

---

## Task 1：冻结科学公式和可追溯身份

**文件：**

- 修改：code/scripts/probe_d92_registration_balanced_covariance.py
- 修改：code/cvsrffi/stage2_d92_e0d_slim.py
- 修改：code/cvsrffi/stage2_d92_e0d_query_evaluation.py
- 测试：tests/test_probe_d92_registration_balanced_covariance.py
- 测试：tests/test_stage2_d92_e0d_slim.py
- 测试：tests/test_stage2_d92_e0d_query_evaluation.py

**步骤：**

1. 先写失败测试，覆盖新类行不变、旧类均值不变、校正零和/有界、类置换等变、K1/K2 alias、query零访问、fit/MAC/state receipt。
2. 新增独立floorboost mode，不修改已冻结OCF25/50语义。
3. 对DA1_REG1旧support计算正确类相对全部其他注册类的低分位margin，并扣除正确类相对旧类margin与相对全类margin之差的均值，得到同态新类竞争漂移惩罚。
4. 对retention分数经tanh、组内去均值和最大绝对值归一，分配上界0.35倍FULL旧类RMS的6维旧类bias修正；只更新旧类bias，不更新新类行。
5. 结构/registry/P0漂移直接报错；仅数值校准退化回退原FULL。
6. 在ssr-gpu中运行三份聚焦测试及py_compile。

## Task 2：冻结Hard11单臂矩阵

**文件：**

- 新增：configs/stage2_d92_full_maxmin_floorboost_hard11_v1.json
- 新增：code/cvsrffi/stage2_d92_floorboost_hard11.py
- 新增：tests/test_stage2_d92_floorboost_hard11.py

**步骤：**

1. 冻结目标中的10个performance outer、1个K1 liveness outer和三个LEO weak场景。
2. 固定claim_scope=DEVELOPMENT_ONLY_FLOOR_HARD_SCREEN、单arm、11job、33scene-arm、8shard。
3. 仅对接既有VALIDATED_ONCE context；不重复IQ、物理ID、receiver、TX或场景验证。
4. method lock写入三个标量、历史对照路径和全部性能/资源门。
5. 测试行集、角色、计数、selection identity和K1 smoke identity。

## Task 3：实现最小runner与分析器

**文件：**

- 新增：code/scripts/run_d92_floorboost_hard11.py
- 新增：code/cvsrffi/stage2_d92_floorboost_hard11_analysis.py
- 新增：code/scripts/analyze_d92_floorboost_hard11.py
- 新增：tests/test_run_d92_floorboost_hard11.py
- 新增：tests/test_stage2_d92_floorboost_hard11_analysis.py

**步骤：**

1. 复用E0OCF已验证的prediction closure、truth-side scorer隔离、共享distinct-outer技术停派和不可覆盖写入。
2. K1真实checkpoint truth-free smoke必须先于8shard。
3. 分析器只读取新候选11job与冻结历史paired_rows.csv，不重跑D92/E0_FULL_ONLY。
4. 输出10行三方法同排CSV、三场景分解、资源表及三个明确裁决分支。
5. liveness行只核别名，不进入性能均值。

## Task 4：本地发布门与Git

1. 在ssr-gpu中运行新增测试及相关E0D/OCF回归。
2. 运行py_compile、JSON解析、两个CLI --help和git diff --check。
3. 完成一次独立P0/P1复核；P2不得阻塞。
4. 更新本run报告、核对Git状态并提交本轮完整实现。
5. 封存runtime closure、config和launch，记录文件SHA与精确N607映射。

## Task 5：唯一N607 Hard11发布

1. 普通账号直连preflight，确认项目根、GPU和四个不可覆盖路径。
2. 唯一runner同步三件套，核对远端SHA、入口、Python/CUDA。
3. 一次detached launch；按PID/CWD/cmdline/GPU/manifest/smoke/首波证据监控。
4. 只因协议或系统性技术故障停止，禁止按性能停止。
5. 完成后完整取回source/logs/smoke/output并核对11/11、22个state prediction/COMMIT/fit/resource、11 score、8 shard summary。

## Task 6：真实数据裁决

**显著遗忘硬门：**

- 相对D92：Hard10 mean delta forgetting不高于-0.5pp；
- 至少8/10行遗忘不增加；
- 最差单行delta forgetting不高于+0.5pp；
- 相对E0_FULL_ONLY：mean forgetting至少改善1.8pp；
- 进取目标：相对D92 mean forgetting不高于-1.0pp且9/10行不增加。

其余floor、H、旧类均值、新类准确率和资源门沿目标文件执行。最终只能给出ADVANCE_TO_FULL125、REVISE_ONCE_FLOORBOOST或REJECT_FLOORBOOST；即使通过也不得自动启动full125。
