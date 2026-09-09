# Tweak配置可移植性复现实验V5：平方L2 triplet公式修复预登记

- run_id：`tweak_config2_portability_20260909_v5`
- 当前状态：`RUNNING_HEALTHY_THROUGH_LATEST_PROBE`
- 唯一launch owner：Codex主Agent
- Git代码提交：`8e87a6cc94b1f18b63bb1cf8f2d61c8cc0f3f69f`，已push且远端OID独立一致。

## 确定性根因与单变量修复

- 论文第6页Eq.(1)视觉复核为`max(||f(A)-f(P)||² - ||f(A)-f(N)||² + α,0)`，其中两项均是**平方L2距离**。V1—V4以`torch.linalg.vector_norm`直接作为loss距离，省略平方；这不是未公开默认值，而是公式实现失配。
- V5只把triplet损失中的正/负距离替换为各embedding差的平方和。严格hard筛选仍按原文`dAN<dAP`，其比较因平方单调性保持等价；校准与闭集决策继续使用论文Algorithm1/2的非平方Euclidean距离；模型、数据、切分、batch、SGD、LR probe、N/M、矩阵和seed均不变。
- 先写失败测试锁定平方损失数值，再作最小实现和本地真实数据验证。V4完整产物保留；只有测试、Git交付、远端编译/CUDA验证全部通过后才创建新的V5 release/run/log/output根。

## TDD与本地验证

- RED：把batch-hard、margin-violating和strict-hard三个真实损失数值的期望改为平方L2后，旧实现分别产出1.6/0.2/1.1而不是应有的7.1/0.09/3.1，三项均按预期失败。
- GREEN：只将`triplet.py`三个loss路径的正/负距离改为`(embedding difference).square().sum(dim=1)`；严格筛选语义、采样、优化器控制和校准/决策代码未改。29项相关测试、三个模块编译通过。
- 真实官方Config2只读首批为`[64,2,128]→[64,12]`，平方strict-hard loss=`0.4830861688`有限，16组参数梯度存在；尚未启动V5。

## 已预登记的N607落地

- 远端项目：`/home/szu2070436088/2510044040/CV-SincNet`；数据根80个输入文件已读回。新的release=`releases/tweak_config2_portability_20260909_v5/source`、run=`runs/tweak_config2_portability_20260909_v5/official_config2_full`、log=`logs/tweak_config2_portability_20260909_v5/train.log`在启动前均不存在。
- 启动前GPU0空闲（约1MiB）；完整命令以`CUDA_VISIBLE_DEVICES=0`绑定物理GPU0、runner内`--device cuda:0`。不传smoke限制，使用原有100epoch与1+100 LR方案。
- 启动后将独立核验PID/PPID/CWD/cmdline/GPU/log；低性能保留为科学结果，只有确定性技术故障触发有界修复。

## 已验证发布与启动

- 源归档从提交`f11cf0bab5c9fc65ebf7e3e71a1654008fe34157`导出；本地和远端SHA-256均为`6e5c1a97aab4e51969d184aa795ad24dd22420e99954fd240e1068a020f41138`，解包和三个改动模块编译通过。解包时仅出现远端时钟较本机归档时间略早的tar时间戳warning，不影响SHA、文件或编译核验。
- 远端真实CUDA验证：`[64,2,128]→[64,12]`，平方strict-hard loss=`0.3845960796`有限，16组参数梯度存在，启动前唯一V5输出根仍为空。
- 2026-09-09 22:35 CST在物理GPU0以完整默认命令启动PID=`523456`。15秒独立probe确认PPID=1、CWD与cmdline为本V5 release/run、GPU进程占466MiB；log尚为空符合每epoch打印，无异常或最终产物。此为初始健康证据，不是完成或数值结论。
