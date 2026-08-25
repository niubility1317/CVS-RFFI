# Phase1 CCOI-PA-V2因果审计B实验报告

## 0.状态与证据边界

- Run ID：`PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825B`
- 当前状态：`LOCAL_VERIFIED`
- 修复提交：`83d4453bdd4e54927ffc86e32a2cd5d12f976f6b`
- 候选：冻结Core90和既有C4 sidecar的source-only因果审计
- 协议：`L_s`拟合、`V_select`审计，target/query访问为0，不重复C0–C4
- GPU：N607 GPU1
- A状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，B是新的不可覆盖run，不是A重启
- 当前尚无B远端结果，不能提前宣称性能、数据健康或科学晋级。

## 1.A暴露的问题与B的定点修复

A在真实checkpoint无query smoke中发现：C4 state_dict包含TX/RX辅助probe权重，但初版runner用无probe的默认sidecar结构执行`strict=True`回载，因unexpected keys失败，未进入正式审计。

B不删除权重、不用`strict=False`绕过。修复从真实state_dict重建：

- q维度和challenge hidden维度；
- codebook大小；
- TX/RX辅助probe及其类别数；
- response维度和operator维度。

随后仍用`strict=True`回载。TX类别数或receiver域数与实际数据不一致时直接失败。新增真实含辅助probe state_dict的回归测试并按TDD先红后绿。

## 2.审计方法

### 2.1E0-Q：挑战语义与泄漏

- TX、receiver、day的线性、MLP、kNN和token sequence probe；
- token位置probe；
- token-level hard occupancy、逐位置occupancy、每包code数、转移矩阵；
- soft有效码、packet dominant code；
- source clean及`leo_clear_weak/leo_low_elev_weak/leo_rain_weak` assignment consistency。

`q`只称received-waveform excitation proxy，不假定它已经等于干净基带内容。

### 2.2E0-G：全局pair geometry

完整收集`V_select`后分块做全局余弦扫描，阈值固定为：
`0.50,0.70,0.80,0.90,0.95,0.98,0.99`。

报告同TX跨receiver正关系、跨TX同receiver困难负关系及positive/negative anchor coverage，解决旧版逐batch统计中`d3=0`无法归因的问题。

### 2.3E0-H：同容量因果分解

| Row | 输入 | 问题 |
|---|---|---|
| H0 | q-only | 公共激励代理能解释多少PA map |
| H1 | theta-only | support是否含稳定设备信息 |
| H2 | 正确theta+q | 完整模型 |
| H3 | shuffled theta+q | 精确support配对价值 |
| H4 | other-TX theta+q | TX特异性 |
| H5 | same-TX cross-RX theta+q | 跨receiver稳定性 |
| H6 | same-TX cross-day theta+q | 跨记录稳定性 |
| HR | cross-fit公共响应残差 | 去公共响应后是否保留TX增量 |

所有row使用同容量、同优化器、同训练步数和同归一化目标。H4/H5/H6在各自关系集合内选择q最近候选。差异使用TX×receiver×day分组配对bootstrap，固定1000次。

### 2.4E0-C：互补性

在source clean和三个synthetic LEO场景计算base/sidecar四格表、rescue、harm、oracle accuracy和oracle gain，判断sidecar是否具有Core90未使用的纠错证据。source synthetic LEO只作机制诊断，不替代真实目标域性能结论。

## 3.固定科学停止规则

任一条件失败即`STOP_PA_M2`：

1. q的TX或receiver probe归一化增益大于`0.10`；
2. `0.70`阈值negative anchor coverage小于`0.80`；
3. H2相对H0、H3、H4任一NMSE相对下降小于`5%`；
4. 上述三项任一分组bootstrap 95%CI下界不大于0；
5. H5相对H4未达到至少`5%`且95%CI下界大于0；
6. H6相对H4未达到至少`5%`且95%CI下界大于0；
7. 三个source LEO场景平均oracle gain小于`0.30`个百分点；
8. 三个source LEO场景汇总`rescue-harm<=0`。

只有全部通过才标记`DESIGN_RESIDUAL_V3`。负科学结果不停止健康运行，不扩大为多seed或完整分类矩阵。

## 4.实现与本地验证

实现文件：

- `code/audit_phase1_ccoi_pa_v2.py`
- `code/cvsrffi/ccoi_causal_audit.py`
- `code/train_phase1_ccoi_pa.py`
- `code/scripts/launch_phase1_ccoi_pa_v2_causal_audit_20260825.sh`

测试文件：

- `code/tests/test_ccoi_causal_audit.py`
- `code/tests/test_phase1_ccoi_pa_causal_audit_runner.py`
- 现有相关CCOI回归测试。

本地证据：

- Conda环境`ssr-gpu`；
- 真实辅助probe state回载测试先因缺少同构回载函数失败，修复后通过；
- 完整相关CCOI套件：`55 passed`；
- 三个生产Python模块：`py_compile PASS`；
- synthetic smoke：7个JSON artifact，全部可解析且UTF-8无替换字符；
- synthetic smoke路径：`E:\type10-7\local_artifacts\ccoi_causal_audit_synthetic_smoke_20260825C`。

## 5.输入、路径与精确命令

输入：

- Core90：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- C4 sidecar：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth`
- WiSig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

输出：

- Release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_83d4453b`
- Run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_causal_audit_20260825`
- Run output：上述root下`PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825B`
- Smoke output：上述run output追加`_REAL_CKPT_NO_QUERY_SMOKE`
- Log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_causal_audit_20260825`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v2_causal_audit_20260825_83d4453b
ROOT="$PWD" GPU=1 RUN_ID=PHASE1_CCOI_PA_V2_CAUSAL_AUDIT_S20260824_20260825B \
CHECKPOINT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl \
SIDECAR=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_20260825/PHASE1_CCOI_PA_V2_S20260824_20260825A/C4/sidecar.pth \
RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v2_causal_audit_20260825 \
LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v2_causal_audit_20260825 \
bash code/scripts/launch_phase1_ccoi_pa_v2_causal_audit_20260825.sh
```

launcher先运行真实checkpoint无query smoke，通过后立即继续正式审计。预期artifact为：
`protocol_and_smoke.json`、`feature_audit.json`、`probe_audit.json`、`pair_geometry.json`、`holdout_factorization.json`、`complementarity.json`、`audit_manifest.json`。

## 6.N607预检

2026-08-25完成B专属只读检查：

- B release目录和归档路径不存在；
- B run、smoke和主日志路径不存在；
- B同名进程不存在；
- GPU1显存约1MiB、利用率0；
- A已退出且不会重启；
- 不干预GPU0既有训练进程。

技术停止仅限协议越界、错误输入/结构、输出碰撞、真实smoke失败、确定性异常、无日志增长或七个artifact不能闭合。低性能只影响科学判定。

## 7.实验结果

尚未发布和启动。后续只追加真实release、smoke、健康状态、七个artifact结果、停止规则逐项判定、暴露问题和下一路线。

