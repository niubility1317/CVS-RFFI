# Tweak配置可移植性复现实验V5：平方L2 triplet公式修复预登记

- run_id：`tweak_config2_portability_20260909_v5`
- 当前状态：`LOCAL_VERIFIED_RELEASE_PENDING`
- 唯一launch owner：Codex主Agent

## 确定性根因与单变量修复

- 论文第6页Eq.(1)视觉复核为`max(||f(A)-f(P)||² - ||f(A)-f(N)||² + α,0)`，其中两项均是**平方L2距离**。V1—V4以`torch.linalg.vector_norm`直接作为loss距离，省略平方；这不是未公开默认值，而是公式实现失配。
- V5只把triplet损失中的正/负距离替换为各embedding差的平方和。严格hard筛选仍按原文`dAN<dAP`，其比较因平方单调性保持等价；校准与闭集决策继续使用论文Algorithm1/2的非平方Euclidean距离；模型、数据、切分、batch、SGD、LR probe、N/M、矩阵和seed均不变。
- 先写失败测试锁定平方损失数值，再作最小实现和本地真实数据验证。V4完整产物保留；只有测试、Git交付、远端编译/CUDA验证全部通过后才创建新的V5 release/run/log/output根。

## TDD与本地验证

- RED：把batch-hard、margin-violating和strict-hard三个真实损失数值的期望改为平方L2后，旧实现分别产出1.6/0.2/1.1而不是应有的7.1/0.09/3.1，三项均按预期失败。
- GREEN：只将`triplet.py`三个loss路径的正/负距离改为`(embedding difference).square().sum(dim=1)`；严格筛选语义、采样、优化器控制和校准/决策代码未改。29项相关测试、三个模块编译通过。
- 真实官方Config2只读首批为`[64,2,128]→[64,12]`，平方strict-hard loss=`0.4830861688`有限，16组参数梯度存在；尚未启动V5。
