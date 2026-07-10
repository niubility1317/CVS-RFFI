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

## 2026-07-10完成结果与终局分析

### 证据完整性

N607只读检查确认8个训练进程均已退出。每个候选都有200行`metrics_epoch.csv`、200条完整epoch记录和约9194行stdout；完整stdout扫描未发现`Traceback`、`RuntimeError`、OOM、Killed、argparse错误或FATAL。E28以后8个候选的`train_skipped_nonfinite_grad`累计均为0，E30以后测试结果各有29至30个不同取值，证明梯度稳定性修复有效，旧主8的“E28后参数冻结/测试不变”问题已经消失。

stdout中的`nan/-inf`主要来自未到启动epoch的inactive指标、未执行test的epoch和从未建立的best checkpoint字段，不是本轮训练loss非有限。该日志设计会污染自动fatal扫描，后续应把inactive/missing改为带reason code的空值或显式状态。

远端每个run目录最终只有`latest_ssdg.pth`、`metrics_epoch.csv/jsonl`，没有`best_joint_safe_ssdg.pth`、`latest_safe_ssdg.pth`或prototype导出包。8个候选`safe_checkpoint_saved`全程为0，stdout的`BEST-JOINT`始终为`E000/val=-inf/test=nan`，最终均输出`prototype_export_skipped=1`。因此本组是完整训练完成的`NON_PROMOTABLE_DIAGNOSTIC`，不是可部署或可进入真实unknown评估的候选包。

### 泛化与星地压力结果

|candidate|best overall@epoch|final overall|best strict@epoch|final strict|final receiver floor|final sat mean|final sat floor|final sat strict floor|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|`BRIDGE_LOW_DENSITY`|89.99@30|89.42|85.95@30|84.73|70.69|78.76|77.80|71.91|
|`BUDGET_OS_HIGH_SAFE`|90.08@10|89.37|85.75@10|84.93|72.53|78.67|77.71|71.92|
|`DM_PROXY_ALIGNED`|89.93@188|89.18|85.79@186|84.27|71.86|78.23|77.26|71.08|
|`EXPORT_PROMOTE_SAFE`|89.74@30|89.25|85.45@80|84.72|71.02|78.41|77.44|71.54|
|`FULL_STABLE`|89.95@184|89.57|85.95@184|85.13|73.13|78.79|77.81|71.82|
|`RECEIVER_LOCAL_SAFE`|89.79@30|89.25|85.80@50|84.90|73.09|78.50|77.54|71.46|
|`SAT_OPEN_PAIR`|90.17@20|89.35|85.88@10|84.68|70.18|78.21|77.27|71.26|
|`U_TRISTATE_FULL`|89.75@130|89.19|85.60@130|84.78|71.47|78.78|77.86|71.98|

V2FIX8的final中位数为overall 89.30、strict UDU 84.76、receiver floor 71.67、sat floor 77.63、sat strict floor 71.68。相对OSFIX16中位数，overall下降0.17pp、strict UDU下降0.16pp，但sat floor提高2.34pp。星地压力鲁棒性是本组最清楚的正向结果；跨日期/跨接收机strict UDU和最弱receiver没有同步提高。8个候选的best-final strict gap约0.82至1.52pp，训练后期仍存在泛化回落。

从组中位曲线看，sat floor从E10的75.29持续升至E200的77.63；strict UDU则从E20的85.46降至E200的84.76，receiver floor从75.18降至71.67。该方向与后期teacher/satellite目标主导一致：模型更适配训练分布内的LEO增强族，但没有学到更强的跨receiver/day不变RFF表征。

### Open-set代理结果

下表是E200同row结果。训练代理仍不等于真实unknown拒识，`old proxy`和`old bridge`指原有`proxy_unknown_*`端点评估代理。

|candidate|p95|p99|tail cvar|source overflow|source episode overflow|dm proxy|old proxy|dm bridge|old bridge|low-density|tail/overflow accept|radius/inter|p99 delta|export|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|`BRIDGE_LOW_DENSITY`|65.70|79.64|74.15|0.792|0.960|0.149|0.575|0.001|1.000|0.024|0.136/0.002|1.011|28.43|blocked|
|`BUDGET_OS_HIGH_SAFE`|70.47|84.00|78.70|0.798|1.000|0.095|0.562|0.001|1.000|0.028|0.168/0.073|1.110|31.87|blocked|
|`DM_PROXY_ALIGNED`|68.45|83.20|78.42|0.761|0.990|0.117|0.588|0.001|1.000|0.027|0.165/0.129|1.135|35.15|blocked|
|`EXPORT_PROMOTE_SAFE`|79.01|84.61|82.30|0.792|0.995|0.155|0.500|0.011|1.000|0.028|0.151/0.103|1.185|35.27|blocked|
|`FULL_STABLE`|53.54|80.10|72.14|0.762|0.946|0.112|0.663|0.000|1.000|0.029|0.150/0.026|0.999|20.49|blocked|
|`RECEIVER_LOCAL_SAFE`|78.20|83.99|81.90|0.758|0.976|0.123|0.588|0.002|1.000|0.025|0.125/0.128|1.102|39.00|blocked|
|`SAT_OPEN_PAIR`|74.52|83.92|79.41|0.777|0.983|0.160|0.488|0.006|1.000|0.027|0.139/0.104|1.127|38.37|blocked|
|`U_TRISTATE_FULL`|68.07|80.23|77.24|0.814|0.987|0.141|0.600|0.002|1.000|0.022|0.144/0.004|1.078|32.47|blocked|

E181-E200的跨候选稳健中位数为p95 72.55、p99 83.09、tail cvar 79.22、dm proxy 0.133、old proxy 0.613、source overflow 0.782、source episode overflow 0.981、dm bridge 0.001、old bridge 1.000、tail accept 0.152、overflow accept 0.083、radius/inter 1.077。相对OSFIX16，p95、p99、tail cvar、dm proxy、tail/overflow accept和radius/inter有小幅改善；source overflow和source episode overflow反而恶化，legacy bridge完全没有改善，legacy proxy只从约0.628降到约0.613。结论是“部分动态DM代理变好”，不是最终拒识边界变好。

`p99 delta`为20.49至39.00，远超2.0/3.5阻断阈值。但当前state machine用每个随机训练batch的历史最小p99作为`best_p99`，再与当前随机batch比较；单batch偶然低值会永久制造巨大delta。该门能fail-closed阻断导出，却不能可靠度量“同一固定评估集上best checkpoint到final的tail扩张”。

### 无标签、local component与损失预算

8个候选E200均为`u_direct_active=0`、`u_quarantine_active=0`、`u_direct weighted loss=0`、`u_quarantine weighted loss=0`。虽然每batch约110个高置信样本被selected，三态全部退化为`trusted_core=110, ambiguous_tail=0, outside_reject=0`，guard持续报告`US_DIRECT_LOSS_IDLE`和`US_TRI_STATE_SOURCE_MISSING`。

代码原因明确：无标签loader使用`shuffle=False`，而U_s direct metric至少需要两个有效类；有序U_s batch很容易成为单类/类数不足batch。quarantine又直接使用`~pseudo_confidence_mask`，当前110/112样本通过高置信门，剩余约2个低于`quarantine_min_count=4`，所以几何三态从未建立。当前实现把“伪标签高置信”近似成“几何core”，无法识别高置信但处在tail/bridge/low-density的样本。

所谓receiver-aware local component也只在`source_episode_three_sigma_loss`中计数leave-one-domain episode。E200虽记录60至66个component和density gate active，但没有启用prototype memory、TX/RX local prototype head、balanced TX-RX sampler或可导出的local component接收球；`source_episode_overflow`仍约0.98。因此它是诊断统计，不是结构性多原型建模。

E181-E200加权loss中位数显示teacher clean KL约4.27、teacher sat KL约9.52、sat CE约2.48；direct metric约1.22、proxy约0.075、source episode约0.0087，U_s direct/quarantine均为0。`B_os_eff`稳健中位约0.07，远低于配置下限0.15。当前`B_os_eff`只是checkpoint/export gate，不会动态重加权、做PCGrad/CAGrad或插入OS-only优化step，所以它能识别冲突，不能纠正梯度主方向。

星地open-set闭环也未完成。训练和评估使用同一组`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`增强族，日志只有clean-sat pair angle，没有clean/sat分视图的p95/p99、proxy_vaccept、bridge、low-density、source overflow和endpoint acceptance。E181-E200的sat pair p95仍约81至83度，而目标是9度；星地分类floor提高并不等于星地视图的open-set几何变紧。

### 候选决策

|candidate|泛化结论|拒识潜力|主要风险|可否推进真实unknown评估|动作|
|---|---|---|---|---|---|
|`FULL_STABLE`|本组final strict和receiver floor相对较好，sat floor高|p95/p99单点较低，但old proxy/bridge和source episode失败|后期回落、全部hard gate阻断|否|保留为修复后健康基线|
|`BRIDGE_LOW_DENSITY`|E30最好，final回落|dm bridge/low-density低|old bridge=1，receiver floor低|否|只作动态门控负例|
|`BUDGET_OS_HIGH_SAFE`|sat较强，strict无增益|dm proxy单点低|实际`B_os_eff`仍不足，source episode=1|否|验证“高权重不等于有效梯度预算”|
|`DM_PROXY_ALIGNED`|后期可达best但final回落最大之一|old proxy无稳定特异改善|动态/legacy不对齐|否|不promotion|
|`EXPORT_PROMOTE_SAFE`|无泛化优势|old proxy单点低但窗口不稳|没有best/safe checkpoint和export artifact|否|保留fail-closed控制负例|
|`RECEIVER_LOCAL_SAFE`|receiver floor接近本组最好|source overflow略低|local component只是统计，source episode仍0.976|否|结构机制未被真正验证|
|`SAT_OPEN_PAIR`|早期overall最好，final strict/floor偏弱|无星地分视图拒识证据|pair p95约82度，sat分类与几何脱节|否|仅作为sat压力诊断|
|`U_TRISTATE_FULL`|sat floor最高之一，strict一般|U_s open-set损失为0|三态完全未落地|否|该候选不能算有效三态实验|

### 最终判断

本组发布是有效的“训练稳定性修复与机制审计实验”：它证明E28非有限梯度已经修复，测试曲线恢复变化，concat_sa+teacher/sat训练可明显提高已见LEO增强族的satellite floor，部分动态DM代理也能下降。

本组不是有效的“可拒识跨域泛化表征”成功实验。没有候选同时提高strict UDU/receiver floor并降低source episode overflow、legacy proxy/bridge和固定endpoint风险；U_s三态、receiver-aware local component、有效open-set梯度预算和星地分视图拒识均未真正生效；8/8无best-safe checkpoint、无prototype artifact、无promotion资格。

当前不能声明真实`unknown_FAR`、`FPR95`、Stage2/Phase2成功、最终拒识边界改善或deployment success。最值得保留的是`FULL_STABLE`作为下一轮机制修复基线，而不是作为Stage2候选。
