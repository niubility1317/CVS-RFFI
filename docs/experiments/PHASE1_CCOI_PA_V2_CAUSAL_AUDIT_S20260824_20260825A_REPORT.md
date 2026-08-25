# Phase1 CCOI-PA-V2因果审计实验报告

## 0.当前结论与状态

- Run ID：`PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825A`
- 当前状态：`LANDED`
- 候选：冻结CCOI-PA-V2 C4的source-only因果审计，不重复C0–C4训练
- 实现提交：`6134e9c5fe11b3cbd01ea906eaab2fe1ed64f2a3`
- 协议：Phase1，固定`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 数据使用：仅在`L_s`拟合小探针/小头，在`V_select`审计；目标域和query访问为0
- GPU：N607 GPU1
- 预登记结论规则：任一固定停止条件失败即`STOP_PA_M2`；全部通过才允许设计residual V3
- 尚无远端结果。本节只记录已完成的本地实现与预登记，不能写成性能结论。

## 1.为什么不直接发布V3

提交`26ac49e0`的深度复盘指出，现有V2负结果不能简单归因于“hard挑战码塌缩”。当前证据仍缺少三段因果链：

1. `q`是否是跨TX共享的激励代理，而不是TX、receiver、day或帧位置捷径；
2. `theta`是否包含跨记录、跨receiver稳定的TX特异增量，而不是公共包络响应；
3. sidecar是否能纠正Core90错误，而不只是复述Core90已经使用的PA map。

因此本轮不扩大分类矩阵，不重新训练C0–C4，也不先加入Soft-DTW、OT、强制码本均衡、低秩解冻或quality gate。本轮先用冻结状态做最小可证伪因果审计。

## 2.吸收、修正与否定

### 2.1吸收的内容

- 接受“内容对齐只能控制激励，不能自动消除信道和接收机”的前提，把`q`降格为received-waveform excitation proxy。
- 接受全局`V_select`配对审计，替代旧的逐batch关系计数。
- 接受token hard occupancy、位置占用、每包code数、转移矩阵和clean/LEO assignment consistency的双口径审计。
- 接受TX/RX/day/位置泄漏探针，并使用线性、MLP、kNN和token序列多视角。
- 接受H0–H6同容量holdout因子分解及TX×receiver×day分组配对bootstrap。
- 接受base/sidecar四格互补性、rescue、harm和oracle ceiling分析。
- 接受公共响应残差化，但实现为TX×receiver×day两折out-of-fold cross-fitting，防止同样本公共预测泄漏。
- 接受先审计、后决定是否进入residual V3的路线。

### 2.2明确否定或收窄的内容

- 否定“packet dominant code为4/48就证明token级码本塌缩”。两者统计对象不同，且hard code不进入下游判决。
- 否定“C4相对C1的92.5% NMSE下降证明设备算子成立”。两者训练监督不同；公平的C4内部真实配对相对shuffle仅改善2.815%。
- 否定把`1-NMSE`称为标准`R²`，统一称“归一化能量拟合分数”。
- 否定把`0.70`阈值描述为有效筛选。同TX跨receiver匹配率为99.904%，旧阈值近似无筛选。
- 不把完整分类矩阵、多seed、完整125、Soft-DTW、OT、多机制融合和quality gate作为本轮前置条件。
- 不追溯伪造旧run的负样本或anchor历史；只为未来训练补充可观测字段。

## 3.方法设计

### 3.1E0-Q：q泄漏与码语义审计

输入为冻结C4 challenge encoder在`L_s`和`V_select`产生的连续token表示及code概率。

输出：

- packet mean和token sequence上的TX、receiver、day线性/MLP/kNN probe；
- token位置probe；
- token-level hard occupancy；
- 每个token位置occupancy；
- 每包不同code数量；
- code transition matrix；
- soft有效码数和packet dominant统计；
- source clean与三个`leo_*_weak`场景的code assignment consistency。

随机基线按类别数计算，泄漏判断使用归一化增益：
`(accuracy-chance)/(1-chance)`。

### 3.2E0-G：全局pair geometry

先收集完整`V_select`的`q`，再分块计算全局余弦关系，不再逐batch汇总。固定扫描阈值：
`0.50,0.70,0.80,0.90,0.95,0.98,0.99`。

每个阈值报告：

- 同TX跨receiver匹配/不匹配；
- 跨TX同receiver匹配；
- positive anchor coverage；
- negative anchor coverage；
- 对应余弦距离统计。

训练脚本另外为未来run记录`positive_pairs/negative_pairs/anchor_count/anchor_fraction`，但不修改旧C4权重和历史。

### 3.3E0-H：H0–H6与HR因子分解

所有row使用相同隐藏宽度、优化器、训练步数和归一化目标：

| Row | 输入 | 识别问题 |
|---|---|---|
| H0 | `q_holdout` | 仅激励代理能解释多少PA map |
| H1 | `theta_support` | support本身是否有设备稳定信息 |
| H2 | 正确`theta+q` | 完整条件预测 |
| H3 | shuffle `theta+q` | 正确support配对是否必要 |
| H4 | 另一TX的`theta+q` | support是否真正TX特异 |
| H5 | 同TX跨receiver的`theta+q` | operator是否跨receiver稳定 |
| H6 | 同TX跨day的`theta+q` | operator是否跨记录稳定 |
| HR | cross-fit公共响应残差 | TX增量是否在去除公共内容/域响应后保留 |

H4/H5/H6均在对应关系集合内选择与当前`q`最近的候选，不用随机替换掩盖关系质量。比较使用TX×receiver×day分组配对bootstrap，固定1000次重采样。

### 3.4E0-C：互补性审计

对source clean和三个`leo_*_weak`场景计算：

- base正确且sidecar正确；
- base正确且sidecar错误（harm）；
- base错误且sidecar正确（rescue）；
- 两者都错误；
- oracle accuracy和oracle gain；
- rescue-harm。

该分析只回答冻结sidecar是否存在互补证据，不以source synthetic LEO替代真实目标域结论。

## 4.已落地实现

| 文件 | 落地内容 |
|---|---|
| `code/cvsrffi/ccoi_causal_audit.py` | 归一化能量拟合分数、token/packet码审计、全局pair扫描、H2–H6索引、分组bootstrap、互补四格表 |
| `code/audit_phase1_ccoi_pa_v2.py` | 冻结Core90+C4 sidecar，source-only特征提取、探针、H0–H6/HR、互补审计、固定停止判定和七个JSON artifact |
| `code/train_phase1_ccoi_pa.py` | 未来训练历史新增负样本数、anchor数和anchor覆盖率；不改变损失行为 |
| `code/scripts/launch_phase1_ccoi_pa_v2_causal_audit_20260825.sh` | 不可覆盖预检、真实checkpoint无query smoke、单次审计启动及artifact闭合 |
| `code/tests/test_ccoi_causal_audit.py` | 纯统计与停止规则回归测试 |
| `code/tests/test_phase1_ccoi_pa_causal_audit_runner.py` | 协议、sidecar、不可覆盖、探针、cross-fit HR和synthetic smoke测试 |
| `code/tests/test_phase1_ccoi_pa_runner.py` | 未来pair观测字段回归测试 |

独立P0/P1审查发现并修复两项直接问题：

1. launcher未拒绝已有日志，存在覆盖风险；现已在创建目录前检查主日志和smoke日志。
2. 初版HR公共预测器使用同样本拟合和推断，存在残差泄漏；现改为按TX×receiver×day分组的两折out-of-fold cross-fitting。

## 5.本地验证

- Conda环境：`ssr-gpu`
- Python：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`
- PyTorch：`2.10.0+cu128`
- 聚焦及相关CCOI回归：`54 passed`
- 新runner和相关生产模块：`py_compile PASS`
- synthetic smoke：产生7个JSON artifact，均为source-only且UTF-8无替换字符
- synthetic smoke路径：`E:\type10-7\local_artifacts\ccoi_causal_audit_synthetic_smoke_20260825B`
- 本机Git Bash路由探针被替换为`/bin/bash`，因此本地`bash -n`状态为`FAILED`；没有执行错误shell payload。该项不阻断，发布前在N607真实Bash完成语法检查。
- `ruff`未安装，因此不声明lint通过；这不是实验阻断条件。

## 6.固定停止规则

以下任一条件出现，审计结论即为`STOP_PA_M2`，不进入Soft-DTW、OT、多机制、多seed或更大分类矩阵：

1. q的TX或receiver probe归一化增益大于`0.10`；
2. `0.70`阈值下跨TX同挑战negative anchor coverage小于`0.80`；
3. H2相对H0、H3或H4任一NMSE相对下降小于`5%`；
4. H2相对H0、H3或H4任一分组bootstrap 95%CI下界不大于0；
5. H5相对H4未达到至少`5%`且95%CI下界大于0；
6. H6相对H4未达到至少`5%`且95%CI下界大于0；
7. source三个LEO场景平均oracle gain小于`0.30`个百分点；
8. source三个LEO场景汇总`rescue-harm<=0`。

只有八项全部通过，下一步才是`DESIGN_RESIDUAL_V3`。低性能本身不停止健康运行，只影响科学晋级。

## 7.最小实验登记

### 7.1输入

- Core90 checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- C4 sidecar：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth`
- WiSig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

### 7.2环境与路径

- N607 CWD/release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_6134e9c5`
- GPU：`1`
- Output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_causal_audit_20260825`
- Run output：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_causal_audit_20260825/PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825A`
- Smoke output：同上追加`_REAL_CKPT_NO_QUERY_SMOKE`
- Log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_causal_audit_20260825`

### 7.3精确启动命令

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_6134e9c5
ROOT="$PWD" GPU=1 RUN_ID=PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825A \
CHECKPOINT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
SIDECAR=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth \
RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_causal_audit_20260825 \
LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_causal_audit_20260825 \
bash code/scripts/launch_phase1_ccoi_pa_v2_causal_audit_20260825.sh
```

launcher第一步是真实checkpoint+C4 sidecar的无query smoke；通过后在同一唯一owner进程中直接继续正式审计，不创建smoke许可artifact。

### 7.4预期artifact

- `protocol_and_smoke.json`
- `feature_audit.json`
- `probe_audit.json`
- `pair_geometry.json`
- `holdout_factorization.json`
- `complementarity.json`
- `audit_manifest.json`
- 主日志和smoke日志

### 7.5技术停止规则

只在以下直接技术事实出现时停止唯一run进程树并保留现有产物：source/query协议越界、错误checkpoint/sidecar/schema、错误CWD或输出碰撞、真实smoke失败、无法产生七个非空artifact、确定性异常或无日志增长。低性能、停止规则不通过或负科学结果不终止健康运行。

## 8.N607预检证据

2026-08-25 11:37 CST完成直接N607只读预检：

- 普通账户：`szu2070436088`
- 项目根可见；
- GPU0有一个既有训练进程，约622MiB；
- GPU1–7均约1MiB且利用率0；
- 本run output、主日志、smoke路径均不存在；
- 同run ID进程不存在；
- 选择GPU1，不干预GPU0既有任务。

2026-08-25完成一次性release发布：

- release内容提交：`d0ce0677a6e951db76b7bdd298636abf72720ba1`；
- 本地归档：`E:\type10-7\local_artifacts\releases\phase1_ccoi_pa_v2_causal_audit_20260825_6134e9c5.tar.gz`；
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_6134e9c5.tar.gz`；
- 本地/远端唯一归档SHA256均为`420cbb60fcb237ddfffab24c48966c9997f13675eed4908055f257fb14893dd7`，状态`VERIFIED`；
- 解压目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_6134e9c5`；
- N607真实`bash -n`和三个生产Python模块编译均通过，状态`REMOTE_COMPILE_PASS`。

## 9.实验结果

尚未启动。发布、真实smoke、运行健康、完整artifact和最终科学判定将在本节追加，不提前宣称。
