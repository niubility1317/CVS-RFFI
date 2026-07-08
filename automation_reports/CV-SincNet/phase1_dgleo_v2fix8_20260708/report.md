# phase1_dgleo_v2fix8_20260708

## 基本信息

| 字段 | 内容 |
|---|---|
| run_id | `phase1_dgleo_v2fix8_20260708` |
| 日期 | 2026-07-08 |
| operator | Codex Phase1地面训练分析/修复agent |
| 阶段 | Phase1 source-only地面DG训练 |
| 协议边界 | 只使用`ManySig.pkl`源域；不使用真实unknown、`ManyTx.pkl`、target receiver样本或Stage2阈值拟合 |
| 目标 | 修复`direct_metric_acceptance_loss`启用后非有限梯度导致E28以后`optimizer.step()`全部跳过的问题，并按每卡一个实验验证训练是否重新推进 |

## 根因与假设

`phase1_dgleo_v2full32_main8_20260708`的主要异常不是测试集评估没有运行，而是E28启用`direct_metric_acceptance_loss`后，8个候选从E28到E200持续记录`train_skipped_nonfinite_grad=1.0`。因此E30之后测试结果几乎不再变化，本质是模型参数冻结。

修复假设：

1. 当前batch动态原型同时参与门控边界、虚拟unknown生成、类间角距离ratio和source episode支持中心反传，叠加小温度`softplus/sigmoid`和`acos`边界梯度后容易产生非有限梯度。
2. endpoint拒识边界不能由动态dm软门控替代；训练loss应使用稳定参考几何，最终仍由`endpoint_accept_v1`、tail safety、source episode density gate和prototype export guard裁决。
3. 只把虚拟unknown整体detach会让proxy/bridge类loss主要经动态原型路径反传，风险高且与“直接优化proxy_vaccept/bridge/low-density接收”的目标不一致。本轮改为参考原型detach，但`direct_metric_virtual_detach=false`，让shell/outward等hard virtual通过样本合成路径提供有限梯度。

## 本地修改

| 文件 | 修改 |
|---|---|
| `E:\type10-7\code\cvsrffi\losses.py` | `_safe_angle_from_cos`新增`eps`和`nan_to_num`保护；新增`_bounded_softplus`；`direct_metric_acceptance_loss`使用`angle_eps=1e-4`、`softplus_clip=20`、detached geometry reference；source episode支持中心detach；metrics新增`geometry_stabilized`、`geometry_reference_detached`、`angle_clamp_eps`、`softplus_clip` |
| `E:\type10-7\code\scripts\launch_phase1_dgleo_v2full32_20260707.sh` | 将`--direct_metric_virtual_detach true`改为`false`，保留hard virtual通过样本合成路径产生梯度 |
| `E:\type10-7\code\tests\test_direct_metric_acceptance_loss.py` | 新增稳定几何契约测试，覆盖近邻类、小温度、hard virtual和finite gradient |
| `E:\type10-7\code\tests\test_phase1_dgleo_v2full32_launcher.py` | 新增launcher断言，防止远端仍以`direct_metric_virtual_detach true`启动 |

根目录`E:\type10-7\code`不是Git仓库，已创建非Git快照：

`E:\type10-7\code\snapshots\phase1_dgleo_v2fix8_20260708\`

同步到Git承载面：

`E:\type10-7\github_publish\CVS-RFFI-repo\code\...`

## 本地验证

| 命令 | 结果 |
|---|---|
| `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_direct_metric_acceptance_loss.py -q` | 3 passed；RED阶段曾因缺少`geometry_stabilized`失败，修复后通过 |
| `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_dgleo_v2full32_launcher.py -q` | 4 passed |
| `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_v2_control.py -q` | 17 passed |
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code\cvsrffi\losses.py code\SSDG\train_ssdg.py` | pass |
| `bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh --dry-run --only=<main8>` | 8个候选覆盖GPU0-7；命令包含`--direct_metric_virtual_detach false`；仍为source-only ManySig |

pytest存在`.pytest_cache`写入权限warning，不影响测试结论。

## 实验矩阵

每张卡启动一个主候选，用于验证修复后E28以后是否继续更新、测试结果是否重新随epoch变化、open-set几何代理是否出现实际改善。

| GPU | candidate | 机制定位 | 主要观察 |
|---:|---|---|---|
| 0 | `DGLEO_V2FULL32_FULL_STABLE` | 全量稳定版 | `train_skipped_nonfinite_grad`是否归零；整体DG与open-set代理是否均衡 |
| 1 | `DGLEO_V2FULL32_DM_PROXY_ALIGNED` | direct metric/proxy对齐 | `dm_proxy_vaccept`、旧`proxy_unknown_proxy_vaccept`是否同步下降 |
| 2 | `DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE` | receiver-aware/local安全 | `source_episode_overflow`、receiver floor、local component导出 |
| 3 | `DGLEO_V2FULL32_BRIDGE_LOW_DENSITY` | bridge/low-density压力 | `bridge_accept_rate`、`low_density_accept_rate` |
| 4 | `DGLEO_V2FULL32_U_TRISTATE_FULL` | 无标签三态 | `trusted_core/ambiguous_tail/outside_reject`与U_s direct active |
| 5 | `DGLEO_V2FULL32_SAT_OPEN_PAIR` | 星地pair+open-set | `sat_pair_loss`、satellite mean/floor、strict UDU |
| 6 | `DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE` | open-set budget偏高 | open-set代理改善是否压过闭集/星地性能 |
| 7 | `DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE` | export/promotion guard | `p99_delta`、tail safety、final export阻断逻辑 |

## 远端同步计划

| 本地文件 | 远端路径 |
|---|---|
| `E:\type10-7\code\cvsrffi\losses.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py` |
| `E:\type10-7\code\scripts\launch_phase1_dgleo_v2full32_20260707.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_v2full32_20260707.sh` |
| `E:\type10-7\code\tests\test_direct_metric_acceptance_loss.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_direct_metric_acceptance_loss.py` |
| `E:\type10-7\code\tests\test_phase1_dgleo_v2full32_launcher.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase1_dgleo_v2full32_launcher.py` |

## 远端启动命令

待N607预检和同步后执行：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=phase1_dgleo_v2fix8_20260708 \
MAX_ACTIVE_PER_GPU=2 \
LAUNCH_SETTLE_SECONDS=2 \
bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh \
  --only=DGLEO_V2FULL32_FULL_STABLE,DGLEO_V2FULL32_DM_PROXY_ALIGNED,DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE,DGLEO_V2FULL32_BRIDGE_LOW_DENSITY,DGLEO_V2FULL32_U_TRISTATE_FULL,DGLEO_V2FULL32_SAT_OPEN_PAIR,DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE,DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE
```

预检显示GPU4已有一个独立paper reproduction训练，占用约2.5GB显存。为满足本批“每卡新增一个实验”，实际使用`MAX_ACTIVE_PER_GPU=2`；这不会让本批在同一GPU提交多个候选，但GPU4总训练进程数为2。

## 启动后健康检查

启动后4-5分钟检查：

1. 8个候选日志是否存在并更新。
2. 日志是否出现`[CONFIG-LOSS]`、`[CONFIG-AUG]`、`[CONFIG-SAT]`、`[CONCAT-SAT-TRAIN]`、`[EPOCH-BEGIN]`。
3. 是否有Traceback、argparse错误、OOM、NaN。
4. `metrics_epoch.csv`是否写入`direct_metric_geometry_stabilized`/`train_loss_direct_metric_accept`等字段。
5. 到E28之后重点检查`train_skipped_nonfinite_grad`是否不再持续为1，且E30后测试指标是否继续变化。

## 完成后分析标准

不能声明真实unknown_FAR、FPR95、Stage2 old/new成功或部署成功。本组只验证Phase1 source-only下：

- 闭集DG：overall、strict UDU、receiver floor、satellite mean/floor。
- open-set代理：proxy_vaccept、source_overflow、bridge_accept_rate、low_density_accept_rate、tail/overflow accept、radius_to_inter_ratio、zid p50/p95/p99、zid_tail_cvar。
- 训练健康：非有限梯度跳步、best/final gap、p99_delta、prototype export guard。

主成功标准：

1. E28后不再全候选持续`train_skipped_nonfinite_grad=1.0`。
2. E30后测试结果不再冻结。
3. 至少一个候选同时保持strict UDU/receiver floor/satellite floor不退化，并降低p99、source_episode_overflow或proxy/bridge/low-density接收风险。

失败判据：

1. E28后仍大面积非有限梯度跳步。
2. 训练可推进但open-set代理无改善，说明稳定性修复只是恢复训练，不等于解决几何矛盾。
3. open-set代理改善但strict UDU/receiver floor/satellite floor显著下降，只能作为机制诊断，不可推进Stage2/Phase2候选。

## 远端状态

### 预检与同步

N607直连预检通过。远端时间为2026-07-08 15:49 CST，项目根目录存在，GPU0-7可见。预检时GPU4有既有进程：

| GPU | 既有PID | 类型 | 显存 |
|---:|---:|---|---:|
| 4 | 667569 | `paper_reproduction.receiver_agnostic_twostage_uda.train` | 约2552MiB |

同步前已备份远端目标文件到：

`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/phase1_dgleo_v2fix8_20260708_remote_before/`

远端同步后SHA256：

| 文件 | SHA256 |
|---|---|
| `code/cvsrffi/losses.py` | `dd4ee20964eeaf9400646563bd9bca117c6acb38d3e5ab906071f6ecaf781306` |
| `code/scripts/launch_phase1_dgleo_v2full32_20260707.sh` | `a8923d80b931c928efe097550a2993c2fc4285ad47bc37ac9ddf87fa320eae42` |
| `code/tests/test_direct_metric_acceptance_loss.py` | `bf0778100d266a90c5d708b63c1422bf13ccf8e7bc12a1e924224840eec4906e` |
| `code/tests/test_phase1_dgleo_v2full32_launcher.py` | `d7ad2a1756e9dd929b960779773563595e117ebf3a263b936ede21e3e55bb2e8` |

远端验证：

```bash
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py
RUN_ID=phase1_dgleo_v2fix8_20260708 bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh --dry-run --only=<main8>
```

结果：语法编译通过；dry-run确认8个候选写入`runs/phase1_dgleo_v2fix8_20260708/...`和`logs/phase1_dgleo_v2fix8_20260708/...`，命令包含`--direct_metric_virtual_detach false`。

### 启动记录

启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
mkdir -p logs/phase1_dgleo_v2fix8_20260708
RUN_ID=phase1_dgleo_v2fix8_20260708 \
MAX_ACTIVE_PER_GPU=2 \
LAUNCH_SETTLE_SECONDS=2 \
bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh \
  --only=DGLEO_V2FULL32_FULL_STABLE,DGLEO_V2FULL32_DM_PROXY_ALIGNED,DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE,DGLEO_V2FULL32_BRIDGE_LOW_DENSITY,DGLEO_V2FULL32_U_TRISTATE_FULL,DGLEO_V2FULL32_SAT_OPEN_PAIR,DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE,DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE \
  2>&1 | tee logs/phase1_dgleo_v2fix8_20260708/submit.out
```

| GPU | candidate | PID | log |
|---:|---|---:|---|
| 0 | `DGLEO_V2FULL32_FULL_STABLE` | 690629 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_v2fix8_20260708/DGLEO_V2FULL32_FULL_STABLE.out` |
| 1 | `DGLEO_V2FULL32_DM_PROXY_ALIGNED` | 690712 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_v2fix8_20260708/DGLEO_V2FULL32_DM_PROXY_ALIGNED.out` |
| 2 | `DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE` | 690809 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_v2fix8_20260708/DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE.out` |
| 3 | `DGLEO_V2FULL32_BRIDGE_LOW_DENSITY` | 691464 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_v2fix8_20260708/DGLEO_V2FULL32_BRIDGE_LOW_DENSITY.out` |
| 4 | `DGLEO_V2FULL32_U_TRISTATE_FULL` | 692044 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_v2fix8_20260708/DGLEO_V2FULL32_U_TRISTATE_FULL.out` |
| 5 | `DGLEO_V2FULL32_SAT_OPEN_PAIR` | 692454 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_v2fix8_20260708/DGLEO_V2FULL32_SAT_OPEN_PAIR.out` |
| 6 | `DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE` | 692869 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_v2fix8_20260708/DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE.out` |
| 7 | `DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE` | 693283 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_v2fix8_20260708/DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE.out` |

### 启动健康检查

4-5分钟检查：8个主PID均存活，GPU0-7均有本批训练，日志无Traceback、OOM或argparse错误，`metrics_epoch.csv`开始写入。到E9时8个候选均正常推进。

E28后关键检查：

| GPU | candidate | last_epoch | E28+行数 | E28+ `train_skipped_nonfinite_grad`累计 | direct metric loss |
|---:|---|---:|---:|---:|---|
| 0 | `DGLEO_V2FULL32_FULL_STABLE` | 28 | 1 | 0.0 | 非零，最近`99.77` |
| 1 | `DGLEO_V2FULL32_DM_PROXY_ALIGNED` | 36 | 9 | 0.0 | 非零，最近`157.82/99.67/148.21` |
| 2 | `DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE` | 34 | 7 | 0.0 | 非零，最近`118.66/126.47/160.32` |
| 3 | `DGLEO_V2FULL32_BRIDGE_LOW_DENSITY` | 35 | 8 | 0.0 | 非零，最近`161.14/87.65/156.38` |
| 4 | `DGLEO_V2FULL32_U_TRISTATE_FULL` | 34 | 7 | 0.0 | 非零，最近`155.40/125.60/175.73` |
| 5 | `DGLEO_V2FULL32_SAT_OPEN_PAIR` | 35 | 8 | 0.0 | 非零，最近`85.07/85.92/135.12` |
| 6 | `DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE` | 35 | 8 | 0.0 | 非零，最近`140.38/143.99/152.21` |
| 7 | `DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE` | 35 | 8 | 0.0 | 非零，最近`159.92/147.30/160.37` |

结论：旧`phase1_dgleo_v2full32_main8_20260708`的核心失败，即E28启用direct metric后全候选持续`train_skipped_nonfinite_grad=1.0`并导致训练冻结，当前已被打破。当前只能说明训练健康修复成功；测试曲线是否重新变化至少要等E40/E50以后，open-set代理是否改善需要完整run或后续分段分析。

### 已知审计缺口

`direct_metric_acceptance_loss`已返回`geometry_stabilized`、`geometry_reference_detached`、`angle_clamp_eps`和`softplus_clip`，并由本地单测覆盖。但当前`train_ssdg.py`只白名单写出既有`train/dm_accept_*`指标，未把这些新增稳定字段写入`metrics_epoch.csv`。本批运行已启动，不能通过修改文件影响已运行进程。后续若需要更完整artifact parity，应补丁`train_ssdg.py`日志白名单，并在下一批验证。
