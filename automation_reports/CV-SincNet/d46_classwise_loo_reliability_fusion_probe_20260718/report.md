# D46类级head-only LOO可靠度融合探针报告

## 1.身份与目标

- 实验ID：`d46_classwise_loo_reliability_fusion_probe_20260718`
- 操作者：Codex`/root`
- 当前状态：`PREREGISTERED_LOCAL_PROBE_NOT_RUN`
- development cell：receiver`20-1`、seed`713101`、K10/new5、3个LEO弱场景、5个outer physical-rank held折。
- query sealed；不访问confirmation seeds，不生成125结果；当前不访问N607。

D45的全局support-LOO权重修复了量化翻转，却与D44在15/15个outer held预测上完全相同，且仍未修复rain旧类遗忘。D46只改变融合粒度：每个注册类以相同、类标签置换等变的公式获得full/block权重，从而允许不同类别跨过不同决策边界。B20、full与3-block LDA、support RMS、量化器、输入数据、outer folds和比较门保持不变。

## 2.机制与数据协议

对组件`g∈{full,block}`和匿名类`c`，在每个outer fit的合法support内部按physical-row rank做head-only leave-one-out。B20只在outer support训练一次并冻结；inner仅重拟合LDA/RMS。所有组件在RMS、CE和融合前先进入canonical affine gauge：每个特征的系数在类维均值为0，截距在类维均值为0。该规范消除不影响单组件argmax、却会污染类级异权融合的任意类公共仿射项。

逐类inner-held CE记为`CE_g,c`，锁定：

`log_evidence_g,c=-K×CE_g,c`，`w_g,c=softmax_g(log_evidence_g,c)`。

这里`K`必须是当前outer fit/inner分区的实际K，本development fit为K8，而不是capsule名义K10。无temperature、clip、阈值、权重扫描或class ID表。最终完整support fit的类`c`分数为：

`δ_c=w_full,c×δ_full,c/s_full+w_block,c×δ_block,c/s_block`。

公式不读取receiver、TX、old/new角色、handle、场景、outer-held或query。support label仅用于合法的support监督拟合与inner可靠度；outer-held仅在完整state冻结后评价。K1固定1:1等价回退；K2若两组件同CE证据不能闭合到1:1则fail closed。所有query仍独立在全部注册类上argmax，无truth、role Oracle、quota或global reassignment。

## 3.资源口径

LDA fit inventory沿用D45精确四组：before/final各2个main fit，K>1时before/final各`2K`个inner head fit，总数`4K+4`。另计：

- 可靠度打分MAC：K1仅有完整support RMS评分，为`2KD(C_old²+C_all²)`；K>1为`2K(K+1)D(C_old²+C_all²)`；
- 类级仿射融合MAC：`2(D+1)(C_old+C_all)`。

metric B20仍只训练一次、20 epoch/20 optimizer steps；最终只持久化一个融合int8/FP16 query state。host FP64 covariance峰值继续标记未实测，不能由CUDA峰值替代。

## 4.预注册性能门

相对D42 original固定基准必须同时满足：协议/lifecycle/source/ground/state/resource/artifact闭包；inner train-held互斥且support row exact-once；canonical gauge与类标签置换闭环；权重有限、严格为正且逐类和为1；before在首次new support读取前物化且不可变；聚合before-old、after-old、seen-new、H、最低before/after/new和joint均不退化，forgetting不增加，并至少一个final floor严格改善；clear、low-elev、rain各自before/after/new/H/joint不退化且forgetting不增加；before/final int8-FP32 argmax变化与margin翻转均为0；final old→new/new→old/new-new不超过D42的26/10/18。

报告必须保留全部匿名类×场景同row准确率和混淆，不能按类ID调参。目标协议没有要求每个单类准确率相对D42逐项不退化，因此晋级按预锁的通用minimum/lower-tail与逐场景门判断，不事后增加或删除单类门。

此外，D46的15个final held预测必须至少有1个与D45不同；若全部相同，即使汇总指标相同也判为`rejected`，不继续扫描温度或权重。探针强制identity并禁用full-K10；即使所有门通过，也只能进入另行正式候选实现和封闭开发验证，不能直接宣称正式性能或启动125。

## 5.文件、版本与执行计划

- 探针：`code/scripts/probe_d46_classwise_loo_reliability_fusion.py`。
- 单测：`tests/test_probe_d46_classwise_loo_reliability_fusion.py`及D42–D45回归。
- 追溯：`analysis/d46_classwise_loo_reliability_fusion_traceability_20260718.md`。
- 预期输出：`E:\type10-7\automation_reports\CV-SincNet\d46_classwise_loo_reliability_fusion_probe_20260718\classwise_inner_loo_likelihood`。
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，本地串行，device=`auto`。
- Git：根目录`E:\type10-7`非Git；代码、测试、追溯与正式报告进入`github_publish/CVS-RFFI-repo`，只暂存本轮精确文件；根目录只保留报告镜像。

本地预运行验证：独立代码复核无P0；其发现的K1资源P1已用分段MAC和K1无likelihood指数语义修复，两项P2以逐fold class-major held索引重算和真实full＋block K2等证据测试加固。D42–D46定向回归`82 passed`，pytest退出码0；退出后本机临时目录`pytest-current`出现既知`WinError 5`清理噪声，不影响测试结论。真实运行前还需精确暂存提交、detached clean worktree和source hash closure。

## 6.运行与结果

待运行后更新同一报告，包括完整105行解析、同row候选表、逐场景与逐类表、D42/D45 prediction差异、量化一致性、资源、artifact哈希、异常、判定和下一轮研发决策。
