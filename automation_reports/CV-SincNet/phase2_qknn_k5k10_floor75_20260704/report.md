# Phase2 qKNN K5/K10十新类稳定性优化报告

## 基本信息

- 实验ID：`phase2_qknn_k5k10_floor75_20260704`
- 时间：2026-07-05 00:15 Asia/Hong_Kong
- 操作方：Codex
- 目标：继续优化qKNN方法，使新类别数量增加时性能不坍塌；限定`K=5,K=10`，不扩大K数量；十个新类内最低类准确率目标为`>=75%`，并记录旧类准确率和逐类证据。
- 数据/特征：`automation_reports/CV-SincNet/phase2_adv3b02_manynew10_supcon_repr_20260704/remote_features/MANYNEW10_SUPCON_HEAD/features_leo_repaired.npz`
- 严格目标域：`old_role=target_old`，`new_role=target_unknown`，target receiver为`7-14`；每类目标域总样本80条。
- 类别设置：
  - old TX：`14-10,14-7,20-15,20-19,6-15,8-20`
  - new TX：`10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3`

## 本地版本状态

- Git工作区：`E:/type10-7/github_publish/CVS-RFFI-repo`
- 新增脚本：`code/scripts/phase2_graph_smooth_qknn_probe.py`
- 未触碰：`local_artifacts/`
- 本轮没有N607远程启动、没有SCP同步、没有服务器状态改变。

## 方法路线

1. 保留既有`support-code qKNN`压缩路线：部署态保存量化support embedding/code与类原型统计，不保存原始support IQ样本。
2. 在K=10上围绕历史最好行做小网格：`topm/proto_mix/radius_norm/neg_lambda/neg_threshold`，检查是否只是超参问题。
3. 固定更优超参后，扫120个支持集种子，检查是否能靠支持集选择达到最低新类75%。
4. 对K=5重复固定配置种子扫，确认低shot边界。
5. 新增无标签query图平滑qKNN变体：先用support-code qKNN得到query分数，再在query批次内构建kNN图进行分数平滑，最后闭集均衡分配。该变体不使用query标签拟合，也不保存原始support样本。

## 运行命令摘要

```powershell
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_graph_smooth_qknn_probe.py
conda run -n ssr-gpu python code\scripts\phase2_confusion_aware_qknn_probe.py ... --k_old 10 --k_new 10 --query_per_old 70 --query_per_new 70 ...
conda run -n ssr-gpu python code\scripts\phase2_confusion_aware_qknn_probe.py ... --k_old 5 --k_new 5 --query_per_old 75 --query_per_new 75 ...
conda run -n ssr-gpu python code\scripts\phase2_graph_smooth_qknn_probe.py ... --k_old 10 --k_new 10 --query_per_old 70 --query_per_new 70 ...
```

## 结果汇总

| 设置 | 最好证据文件 | seed | 方法/配置 | old_acc | min_old | new_acc | min_new | 是否达到十新类最低75% |
|---|---:|---:|---|---:|---:|---:|---:|---|
| K=10 | `strict_n10_k10_seed421046_small.json` | 421046 | qKNN，`topm=4,proto_mix=0.25,radius_norm=0,old_bias=0.001,neg_lambda=0.7,threshold=0.75,balanced_assignment=true` | 83.57% | 65.71% | 81.29% | 71.43% | 否 |
| K=10 | `strict_n10_k10_topm3_pm04_seed120.json` | 421046 | qKNN，`topm=3,proto_mix=0.4,radius_norm=0.1,neg_lambda=0` | 82.86% | 67.14% | 80.57% | 71.43% | 否 |
| K=10 | `strict_n10_k10_scenariobal_fixed_seed120.json` | 421009 | 场景分块均衡分配诊断 | 62.86% | 20.00% | 43.43% | 41.43% | 否，明显失效 |
| K=10 | `strict_n10_k10_graphsmooth_seed20.json` | 421019 | 图平滑qKNN，最好行为`graph_alpha=0`，即退回原qKNN | 84.29% | 67.14% | 79.86% | 64.29% | 否 |
| K=10 | `strict_n10_k10_transproto_seed421046.json` | 421046 | 转导原型qKNN，`query_mix=0,score_mix=0.05`，最好行为不使用query原型更新 | 83.57% | 65.71% | 81.43% | 71.43% | 否 |
| K=5 | `strict_n10_k5_topm3_pm04_seed120.json` | 421011 | qKNN，`topm=3,proto_mix=0.4,radius_norm=0.1,neg_lambda=0` | 81.78% | 72.00% | 76.93% | 61.33% | 否 |
| K=5 | `strict_n10_k5_transproto_seed421011.json` | 421011 | 转导原型qKNN，`query_mix=0.5,score_mix=0.05,iterations=3` | 82.00% | 72.00% | 77.07% | 61.33% | 否 |
| K=5 | 历史`strict_target_domain_n10_k5.json` | 421074 | qKNN历史最好 | 81.56% | 61.33% | 75.60% | 62.67% | 否 |

## K=10逐类详细证据

当前K=10最好行来自`strict_n10_k10_seed421046_small.json`：

| 类别 | role | acc |
|---|---|---:|
| 14-10 | old | 80.00% |
| 14-7 | old | 80.00% |
| 20-15 | old | 94.29% |
| 20-19 | old | 65.71% |
| 6-15 | old | 84.29% |
| 8-20 | old | 97.14% |
| 10-10 | new | 71.43% |
| 11-10 | new | 77.14% |
| 18-5 | new | 84.29% |
| 19-3 | new | 74.29% |
| 2-13 | new | 72.86% |
| 2-5 | new | 84.29% |
| 3-8 | new | 87.14% |
| 4-10 | new | 88.57% |
| 8-18 | new | 88.57% |
| 8-3 | new | 84.29% |

结论：K=10不是整体均值低，而是`10-10`、`2-13`、`19-3`三个新类卡在75%以下；旧类中`20-19`也只有65.71%。在每新类70个query的设置下，最低新类71.43%表示至少还差3个正确样本才能达到75%。

## K=5逐类详细证据

当前K=5最好行来自`strict_n10_k5_topm3_pm04_seed120.json`：

| 类别 | role | acc |
|---|---|---:|
| 14-10 | old | 74.67% |
| 14-7 | old | 80.00% |
| 20-15 | old | 96.00% |
| 20-19 | old | 72.00% |
| 6-15 | old | 72.00% |
| 8-20 | old | 96.00% |
| 10-10 | new | 77.33% |
| 11-10 | new | 62.67% |
| 18-5 | new | 73.33% |
| 19-3 | new | 69.33% |
| 2-13 | new | 61.33% |
| 2-5 | new | 81.33% |
| 3-8 | new | 85.33% |
| 4-10 | new | 89.33% |
| 8-18 | new | 89.33% |
| 8-3 | new | 80.00% |

结论：K=5的坍塌更明显，主要瓶颈为`2-13`、`11-10`、`19-3`、`18-5`；其中`2-13`最低只有61.33%。

## 解释

- 当前qKNN压缩路线在K=10时已经能让多数新类达到80%上下，但最低类无法越过75%。这不是均值问题，而是多新类同时加入后的局部类间混淆。
- `balanced_assignment`能提升整体新类均值，但不能修复`10-10/2-13/19-3`等边界类。
- `scenario_balanced_assignment`把错误按场景切开后反而使分配严重失衡，K=10新类均值降到43.43%，应作为负诊断，不应作为主路线。
- 图平滑qKNN没有收益，最好行是`graph_alpha=0`，即平滑关闭。这说明当前query邻域图没有提供比support-code分数更可靠的类边界信息。
- 转导原型qKNN没有解决最低类瓶颈。K=10最好行仍是`min_new=71.43%`；K=5可微增整体new_acc到77.07%，但最低新类仍是61.33%。这说明无标签query簇心会强化已经容易的类，对`10-10/2-13/19-3`或`2-13/11-10`这类硬边界帮助有限。
- pairwise ridge方向校正的只读诊断显示：它可以把`19-3`提升到75.71%、`2-13`提升到74.29%，但`10-10`仍卡在71.43%；120个支持集种子的固定配置扫描没有出现`min_new>=75%`行。
- 支持集监督的对角缩放adapter出现过拟合，K=10最低新类降到54%左右；既有低秩残差adapter在严格`pool_per=K`下没有额外enrollment验证集，因此不产生严格K=10可用行。
- 在不扩大K的前提下，下一步不应继续盲目扩大qKNN超参网格；应转向特征/度量层面的轻量修复，例如面向`10-10/2-13/19-3/20-19`的pair-aware度量头或支持集原型间margin训练，然后再回到K=5/K=10压缩qKNN头验证。

## 产物

- `strict_n10_k10_seed421046_small.json/csv`
- `strict_n10_k10_topm3_pm04_seed120.json/csv`
- `strict_n10_k5_topm3_pm04_seed120.json/csv`
- `strict_n10_k10_scenariobal_fixed_seed120.json/csv`
- `strict_n10_k10_graphsmooth_seed20.json/csv`
- `strict_n10_k10_transproto_seed421046.json/csv`
- `strict_n10_k5_transproto_seed421011.json/csv`
- 新脚本：`github_publish/CVS-RFFI-repo/code/scripts/phase2_graph_smooth_qknn_probe.py`
- 新脚本：`github_publish/CVS-RFFI-repo/code/scripts/phase2_transductive_proto_qknn_probe.py`

## 2026-07-05补充：现有特征视图上限复核

为避免把`MANYNEW10_SUPCON_HEAD`的单一特征视图误判为qKNN上限，本轮追加复核了已导出的其他五个特征视图。口径保持严格：`K_old=K_new=10`，`pool_per_old=pool_per_new=10`，每类query为70，target receiver仍为`7-14`，new role仍为`target_unknown`，不扩大K数量，不使用未知拒识。

固定qKNN配置为当前K=10最好配置：`topm=4,proto_mix=0.25,radius_norm=0,old_bias=0.001,neg_lambda=0.7,neg_threshold=0.75,neg_margin=0.01,mutual_only=true,scenario_aware=true,balanced_assignment=true`。每个特征视图扫`seed_start=421000,seed_count=60`与`stable_first,scenario_diverse`两种support策略。

| feature view | 证据文件 | best seed | policy | old_acc | min_old | new_acc | min_new | 关键逐新类短板 | verdict |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| `MANYNEW10_SUPCON_HEAD` | `strict_n10_k10_seed421046_small.json` | 421046 | stable_first | 83.57% | 65.71% | 81.29% | 71.43% | `10-10`71.43%,`2-13`72.86%,`19-3`74.29% | 当前最好，仍失败 |
| `MANYNEW10_SUPCON_NORM` | `strict_n10_k10_supcon_norm_seed60.json` | 421056 | stable_first | 83.33% | 67.14% | 78.00% | 61.43% | `2-13`61.43%,`10-10`70.00%,`11-10`71.43% | 失败 |
| `MANYNEW10_HEAD_SEP` | `strict_n10_k10_head_sep_seed60.json` | 421028 | stable_first | 80.24% | 57.14% | 73.71% | 58.57% | `2-13`58.57%,`19-3`60.00%,`10-10`62.86% | 失败 |
| `MANYNEW10_NORM_SEP` | `strict_n10_k10_norm_sep_seed60.json` | 421027 | stable_first | 80.95% | 55.71% | 70.86% | 54.29% | `2-13`54.29%,`19-3`55.71%,`10-10`57.14% | 失败 |
| `MANYNEW10_IDENTITY` from supcon run | `strict_n10_k10_identity_supconrun_seed60.json` | 421034 | stable_first | 76.90% | 50.00% | 73.00% | 51.43% | `2-13`51.43%,`11-10`52.86%,`18-5`64.29% | 失败 |
| `MANYNEW10_IDENTITY` from repair run | `strict_n10_k10_identity_repairrun_seed60.json` | 421034 | stable_first | 76.90% | 50.00% | 73.00% | 51.43% | `2-13`51.43%,`11-10`52.86%,`18-5`64.29% | 失败 |

结论：替代特征视图没有隐藏的K=10达标行，且多数视图的最低新类远低于`SUPCON_HEAD`。当前瓶颈不是“换一个已导出的特征”或继续微调qKNN读出层，而是目标receiver `7-14`下多个新类与旧类/新类邻域仍未被表示层分开。

## 2026-07-05新增远端训练方案

新增N607 launcher：`code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh`。

该方案不改训练主逻辑，复用已有`proxy_unknown_supcon/proto_ce/pair_margin/old_margin`接口，但把训练目标改成更强的old-protected proxy episode约束，并把post-eval改为严格`K=10`和`K=5` qKNN同口径验证，而不是K=20/K=50近似审计。训练仍只使用source receiver上的proxy non-old TX，显式排除固定十个target seen-new TX；target receiver样本只用于导出后的support/query评估。

| variant | trainable part | epochs | lr | 关键变化 |
|---|---|---:|---:|---|
| `MANYNEW10_CONFLICT_HEAD` | `id_feature_head` | 60 | 0.00008 | 提高proxy pair margin、old margin和clean/feature margin，保护旧类同时增强proxy类间间隔 |
| `MANYNEW10_CONFLICT_NORM` | `id_norm_late_feature` | 60 | 0.00006 | 在late feature+norm/gate参数上做同类约束，检查是否更稳 |

post-eval严格设置：

| eval | K | query per class | qKNN配置 | 成功门槛 |
|---|---:|---:|---|---|
| `manynew10_strict_k10_qknn.json` | 10 | 70 | 当前K=10最好配置，`balanced_assignment=true` | `old_acc>=80%`且每个新类`>=75%` |
| `manynew10_strict_k5_qknn.json` | 5 | 75 | 当前K=5最好配置，`balanced_assignment=true` | `old_acc>=80%`且每个新类`>=75%` |

本地验证：

```powershell
bash -n code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh
conda run -n ssr-gpu python -m py_compile code\scripts\phase2_confusion_aware_qknn_probe.py code\scripts\train_apply_phase1_iq_preadapter_20260703.py
bash -lc 'mkdir -p /tmp/type10_conflict_dryrun && env ROOT=/tmp/type10_conflict_dryrun RUNS_ROOT=/tmp/type10_conflict_dryrun/runs LOG_ROOT=/tmp/type10_conflict_dryrun/logs PYTHON=python GPUS_CSV=0,1 bash /mnt/e/type10-7/github_publish/CVS-RFFI-repo/code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh --dry-run'
```

验证结果：全部通过。dry-run确认`target_rx=7-14`、十个新类列表、proxy pool排除target-new、严格成功门槛为`K5 and K10 old_acc>=0.80 and every seen-new class>=0.75`。

N607同步/启动计划：

| local | remote |
|---|---|
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh` |

计划远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup bash code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh > logs/phase2_adv3b02_manynew10_conflict_protected_20260705.driver.out 2>&1 & echo $!
```

预期输出：

| 类型 | 路径 |
|---|---|
| runs | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_manynew10_conflict_protected_20260705/` |
| logs | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_manynew10_conflict_protected_20260705/` |
| driver log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_manynew10_conflict_protected_20260705.driver.out` |
| summary | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_manynew10_conflict_protected_20260705/manynew10_conflict_protected_summary.json` |

## 2026-07-05 N607启动记录

N607 direct preflight已通过：direct SSH config、identity、server time、project root和GPU可见性均正常。启动前远端GPU0-7均为空闲低显存状态，未发现本任务训练进程。

同步文件：

| local | remote | SHA256 |
|---|---|---|
| `E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh` | `B5849654FE33B91A6D4E9A279B9D3B6FD69F965768915E2B11A664CD8EC5D993` |

远端验证：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
sha256sum code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh
bash -n code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh
```

结果：远端SHA256与本地一致，`bash -n`通过。

启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
nohup bash code/scripts/launch_phase2_adv3b02_manynew10_conflict_protected_20260705.sh > logs/phase2_adv3b02_manynew10_conflict_protected_20260705.driver.out 2>&1 & echo $!
```

启动SSH命令在本地等待时超时，不能单独作为失败或成功证据；随后已按N607规则检查并关闭本地残留`ssh.exe`，复查后本地无`ssh.exe`进程、无到`172.31.111.215:22`或`172.31.105.18:22`的ESTABLISHED连接。

后续短命令复核显示run已启动：

| 项目 | 证据 |
|---|---|
| driver log | 已写入`[MANYNEW10-CONFLICT-PROTECTED] run_id=phase2_adv3b02_manynew10_conflict_protected_20260705 dry_run=0 target_rx=7-14 gpus=0,1` |
| launcher PID | `2460901`，父级残留启动shell PID `2460899` |
| variant driver PIDs | `2460903`、`2460904` |
| training PID | `2460908`：`MANYNEW10_CONFLICT_HEAD` |
| training PID | `2460907`：`MANYNEW10_CONFLICT_NORM` |
| GPU状态 | GPU0约643MiB、GPU1约1481MiB，其余GPU低占用；训练处于启动初期 |

当前状态：N607训练已启动但未完成。需要后续读取`logs/phase2_adv3b02_manynew10_conflict_protected_20260705/`和`manynew10_conflict_protected_summary.json`，完成后按K=5/K=10严格逐类指标更新本报告。

## 2026-07-05 conflict-protected完成结果

N607训练、特征导出和补跑严格qKNN评估已完成。首次launcher后处理因远端`phase2_confusion_aware_qknn_probe.py`为旧版、不支持`--balanced_assignment`而中断；随后同步本地新版评估脚本，远端SHA256为`56DB804D661A3304957BA0601630ED276545D688EDE4E1329DB01029AEBA70E4`，`py_compile`通过，并只补跑四个严格评估，不重训。

远端最终状态：无`manynew10_conflict/train_apply/phase2_confusion_aware_qknn_probe`进程，GPU0-7恢复低占用。本地SSH/SCP残留已检查，当前无`ssh.exe`进程，无到N607或lab bridge 22端口的ESTABLISHED连接。

结果文件已拉取到：`E:\type10-7\automation_reports\CV-SincNet\phase2_qknn_k5k10_floor75_20260704\remote_artifacts\conflict_protected_20260705\`。

| variant | K | seed | old_acc | min_old | new_acc | min_new | 是否达标 |
|---|---:|---:|---:|---:|---:|---:|---|
| `MANYNEW10_CONFLICT_NORM` | 10 | 421027 | 84.05% | 71.43% | 80.00% | 72.86% | 否 |
| `MANYNEW10_CONFLICT_HEAD` | 10 | 421045 | 80.24% | 67.14% | 77.71% | 71.43% | 否 |
| `MANYNEW10_CONFLICT_NORM` | 5 | 421113 | 79.78% | 68.00% | 75.60% | 65.33% | 否 |
| `MANYNEW10_CONFLICT_HEAD` | 5 | 421009 | 78.44% | 64.00% | 74.40% | 64.00% | 否 |

### `MANYNEW10_CONFLICT_NORM` K=10逐类

| 类别 | role | acc |
|---|---|---:|
| 14-10 | old | 82.86% |
| 14-7 | old | 77.14% |
| 20-15 | old | 91.43% |
| 20-19 | old | 71.43% |
| 6-15 | old | 84.29% |
| 8-20 | old | 97.14% |
| 10-10 | new | 75.71% |
| 11-10 | new | 72.86% |
| 18-5 | new | 84.29% |
| 19-3 | new | 78.57% |
| 2-13 | new | 72.86% |
| 2-5 | new | 78.57% |
| 3-8 | new | 82.86% |
| 4-10 | new | 90.00% |
| 8-18 | new | 77.14% |
| 8-3 | new | 87.14% |

### `MANYNEW10_CONFLICT_NORM` K=5逐类

| 类别 | role | acc |
|---|---|---:|
| 14-10 | old | 74.67% |
| 14-7 | old | 77.33% |
| 20-15 | old | 93.33% |
| 20-19 | old | 69.33% |
| 6-15 | old | 68.00% |
| 8-20 | old | 96.00% |
| 10-10 | new | 70.67% |
| 11-10 | new | 66.67% |
| 18-5 | new | 85.33% |
| 19-3 | new | 65.33% |
| 2-13 | new | 65.33% |
| 2-5 | new | 78.67% |
| 3-8 | new | 74.67% |
| 4-10 | new | 92.00% |
| 8-18 | new | 68.00% |
| 8-3 | new | 89.33% |

artifact SHA256：

| file | SHA256 |
|---|---|
| `manynew10_conflict_protected_summary.json` | `1211A817DD9DC3D149A610387DE1AD37C1A6CC72B82439F3F8770839CA0B885B` |
| `MANYNEW10_CONFLICT_NORM_k10.json` | `7CCA68D58C8A8133B3B80980AE43ADB1C4355ACA9D3978A7839C60A8B963A1E9` |
| `MANYNEW10_CONFLICT_NORM_k5.json` | `CA3FBF27C3174AFB9AF0AE2997B33420F9B54329B21A62EEC508F1E823485F5E` |
| `MANYNEW10_CONFLICT_HEAD_k10.json` | `74F39C76AEB5C094DD3ADC57DF86B3D4344F5E6945AA3634C83549BF11AF01D5` |
| `MANYNEW10_CONFLICT_HEAD_k5.json` | `624AC38CA955CF5B9949DD8230D93C55B3D7B497BB58E80831643872747F348B` |

解释：conflict-protected表示训练有正向收益，K=10最好最低新类从原`71.43%`提高到`72.86%`，并保持old_acc超过80%。但是它没有达到十新类最低`75%`，且K=5仍明显坍塌，最低新类只有`65.33%`。当前目标仍未完成。下一步应把优化重点从proxy-only source训练转向Stage2-C允许的support-only轻量度量/原型校准，但必须设计不依赖额外target query标签的验证机制；单纯加强proxy episode约束已经不足以让K=5稳定。

## 2026-07-05 support-metric压缩qKNN本地优化结果

新增本地脚本：`code/scripts/phase2_support_metric_qknn_probe.py`。该脚本保持qKNN路线，但不保存原始support样本；部署状态由量化support code、每类prototype、少量support-only度量变换标量组成。度量变换只用target support拟合，query label仅用于事后审计。脚本SHA256：`076FF9C9C227E0DAB89ADB4CBB8D7B0346807E1C2CD528A2BEB38281E62FFE39`。

本轮只评估`K=5`和`K=10`，未扩大K数量；`pool_per_old=K`、`pool_per_new=K`，严格保持K-only support。support和query均来自`R_t=7-14`目标接收机域，并使用已导出的`MANYNEW10_CONFLICT_NORM_features_leo_repaired.npz`星地信道特征。目标仍未达成。

| 方法 | K | seed | 关键配置 | old_acc | min_old | new_acc | min_new | 是否达标 |
|---|---:|---:|---|---:|---:|---:|---:|---|
| support-metric qKNN | 10 | 421029 | `diag_fisher,strength=0.5,topm=4,proto_mix=0.25,radius_norm=0,old_bias=0.001,neg_lambda=0.7` | 83.10% | 67.14% | 83.43% | 72.86% | 否 |
| support-metric qKNN | 5 | 421037 | `diag_whiten_fisher,strength=0.1,topm=5,proto_mix=0.6,radius_norm=0.1,old_bias=0,neg_lambda=0` | 82.67% | 65.33% | 79.47% | 69.33% | 否 |
| transductive proto qKNN | 10 | 421029 | query-unlabeled prototype refine,`query_mix=0.2,score_mix=0.2,iters=2` | 82.86% | 68.57% | 81.86% | 71.43% | 否 |
| transductive proto qKNN | 5 | 421037 | query-unlabeled prototype refine,最佳为无更新等价配置 | 82.67% | 69.33% | 78.93% | 66.67% | 否 |
| support-LOO class-bias qKNN | 10 | 421118 | 每类bias由support LOO拟合 | 71.67% | 38.57% | 80.14% | 62.86% | 否 |
| support-LOO class-bias qKNN | 5 | 421009 | 每类bias由support LOO拟合 | 43.33% | 0.00% | 74.00% | 58.67% | 否 |

### support-metric qKNN K=10逐类

| 类别 | role | acc |
|---|---|---:|
| 14-10 | old | 82.86% |
| 14-7 | old | 78.57% |
| 20-15 | old | 91.43% |
| 20-19 | old | 67.14% |
| 6-15 | old | 82.86% |
| 8-20 | old | 95.71% |
| 10-10 | new | 72.86% |
| 11-10 | new | 75.71% |
| 18-5 | new | 92.86% |
| 19-3 | new | 80.00% |
| 2-13 | new | 72.86% |
| 2-5 | new | 85.71% |
| 3-8 | new | 88.57% |
| 4-10 | new | 91.43% |
| 8-18 | new | 82.86% |
| 8-3 | new | 91.43% |

### support-metric qKNN K=5逐类

| 类别 | role | acc |
|---|---|---:|
| 14-10 | old | 82.67% |
| 14-7 | old | 74.67% |
| 20-15 | old | 92.00% |
| 20-19 | old | 65.33% |
| 6-15 | old | 85.33% |
| 8-20 | old | 96.00% |
| 10-10 | new | 76.00% |
| 11-10 | new | 72.00% |
| 18-5 | new | 85.33% |
| 19-3 | new | 74.67% |
| 2-13 | new | 69.33% |
| 2-5 | new | 78.67% |
| 3-8 | new | 86.67% |
| 4-10 | new | 88.00% |
| 8-18 | new | 78.67% |
| 8-3 | new | 85.33% |

### artifact SHA256

| file | SHA256 |
|---|---|
| `strict_n10_k10_conflict_norm_metric_qknn_seed120.json` | `6A283A435E3123344682CF0DB7F49F347313D960396B377F699AF855E0477400` |
| `strict_n10_k5_conflict_norm_metric_qknn_topm5_seed120.json` | `C8AE32A557538CB6C6B94F0FAB6D61F80BDD2C2253090469987B977FA90593DF` |
| `strict_n10_k10_conflict_norm_transproto_seed421029_focus.json` | `AA17FEC159765D62AC21D160E8AFF024756281E9617986CCD03CB83A2992AA38` |
| `strict_n10_k5_conflict_norm_transproto_seed421037_focus.json` | `32BF3E8BB90261CB7FBF770C563470E6173CB482EACCC7365DB2102EE77DA0DF` |
| `strict_n10_k10_conflict_norm_classbias_seed120.json` | `DA7686E3427B240628671FABDBB421AF773DDA9DA6DA16878F1977B693EA50C5` |
| `strict_n10_k5_conflict_norm_classbias_seed120.json` | `8E4CA684853A8FD9F85A1EE050CD3D947400829FDB92FB36CB05472CDDA7CC97` |

结论：当前最好仍是support-metric qKNN。它相对原conflict-protected qKNN在K=10上把new_acc从`80.00%`提高到`83.43%`，但最低新类仍卡在`10-10/2-13=72.86%`；K=5最低新类从`65.33%`提高到`69.33%`，仍远低于`75%`。class-bias的support LOO信号与query表现不一致，会严重牺牲旧类；query-unlabeled原型更新也没有改善最低类。因此下一步不应继续调bias或transductive prototype，而应针对低类`10-10/2-13/11-10/19-3`做support选择和类间冲突诊断，重点寻找更稳定的support压缩码本或源表征，而不是扩大K。

## 2026-07-05 support-metric qKNN冲突诊断与后处理负证据

本轮继续沿K=5/K=10、不扩大K数量的约束推进。`phase2_support_metric_qknn_probe.py`新增两个不使用query label的压缩头选项：support-only repulsive prototype scoring和support-similarity pairwise quota refine；另新增`phase2_support_metric_confusion_diagnose.py`用于单配置混淆审计。代码验证：`conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py code\scripts\phase2_support_metric_confusion_diagnose.py`通过。

脚本SHA256：

| file | SHA256 |
|---|---|
| `code/scripts/phase2_support_metric_qknn_probe.py` | `219ED05EE42A8545F598468DEE0A038E6F30C550114DCD04AA4D35A7C8C0F207` |
| `code/scripts/phase2_support_metric_confusion_diagnose.py` | `FB1B960E8974CE7B31CFCCD4F46612F4A61F3D7918F88A8FDCC57C4742A96791` |

### 当前最佳配置的混淆瓶颈

| K | seed | 低类 | acc | 主要错分 | support prototype近邻证据 |
|---:|---:|---|---:|---|---|
| 10 | 421029 | `10-10` | 72.86% | `20-19`:11/70 | `20-19`/`10-10` sim=0.951 |
| 10 | 421029 | `2-13` | 72.86% | `11-10`:8/70,`20-15`:4/70 | `11-10`/`2-13` sim=0.966 |
| 5 | 421037 | `2-13` | 69.33% | `11-10`:8/75,`3-8`:4/75 | `11-10`/`2-13` sim=0.847 |
| 5 | 421037 | `11-10` | 72.00% | `2-13`:9/75,`6-15`:9/75 | `6-15`/`11-10` sim=0.918 |
| 5 | 421037 | `19-3` | 74.67% | `14-10`:11/75,`20-19`:5/75 | `14-10`/`19-3` sim=0.932 |

解释：瓶颈不是query数量不足，也不是K值不够大后的统计均值问题，而是K-shot support在星地信道特征中形成了高度相似的类对；balanced assignment只能保证每类预测配额，不能修复这些相似类对内部的成对互换。

### 新后处理机制结果

| 方法 | K | seed | old_acc | min_old | new_acc | min_new | 变化 |
|---|---:|---:|---:|---:|---:|---:|---|
| support-metric qKNN基线 | 10 | 421029 | 83.10% | 67.14% | 83.43% | 72.86% | 当前最佳 |
| repulsive prototype scoring | 10 | 421029 | 83.10% | 67.14% | 83.43% | 72.86% | 未提升 |
| repulsive prototype+proto_mix窄网格 | 10 | 421029 | 83.10% | 67.14% | 83.43% | 72.86% | 未提升 |
| pairwise quota refine | 10 | 421029 | 83.10% | 67.14% | 83.43% | 72.86% | `changed_predictions=0` |
| support-metric qKNN基线 | 5 | 421037 | 82.67% | 65.33% | 79.47% | 69.33% | 当前最佳 |
| pairwise quota refine | 5 | 421037 | 82.67% | 65.33% | 79.47% | 69.33% | `changed_predictions=0` |

artifact SHA256：

| file | SHA256 |
|---|---|
| `strict_n10_k10_support_metric_confusion_seed421029.json` | `833496BC196D15550B07AAB433051020575C137519234E01F5525A87D02069BC` |
| `strict_n10_k5_support_metric_confusion_seed421037.json` | `E9229CB396FEDE687E6109DDBFCF969B043A27061C2EBE06481FA8C383DE85CE` |
| `strict_n10_k10_metric_qknn_pairrefine_seed421029_focus.json` | `B1AE83D0B3219164218460624CC6BA645B39FDE21B9AA5615533893ECFCF1801` |
| `strict_n10_k5_metric_qknn_pairrefine_seed421037_focus.json` | `EDE127F5090E0247F2C5A9E00774F583EA48B0FA4785A76A1258BA12F2D22921` |
| `strict_n10_k10_metric_qknn_repel_seed421029_focus.json` | `FF868FE5C96D04DFB80B4BFC4EF2CF61129A6AF00547FCDE9F434CB24DB0A8CE` |
| `strict_n10_k10_metric_qknn_repel_protomix_seed421029_narrow.json` | `08C40CEAE366FC4A894B2AD959360BD824EB4C72CAB1B7C87A2D12E51690A6C4` |

结论：目标仍未完成。新的repulsive prototype和pairwise quota refine是合规的压缩qKNN变体，但在当前最佳support seed上不能改善最低类；pairwise refine不改变预测，说明balanced assignment在现有score矩阵下已经达到相似类对内的局部配额最优。后续应停止继续扩大后处理网格，转向训练侧或特征侧：围绕`20-19/10-10`、`11-10/2-13`、`6-15/11-10`、`14-10/19-3`加入source-side pair-separation/episode hard-pair loss，或重新导出更能分开这些pair的`MANYNEW10_CONFLICT_NORM`特征，再回到同一K=5/K=10压缩qKNN评估。

## 2026-07-05 proxy hard-pair训练侧优化计划

目标仍为：K只取`5`和`10`，不扩大K数量；十个目标新类别内最低新类准确率不低于`75%`，同时记录旧类准确率和逐类性能。当前最好仍未达标：K=10最低新类`72.86%`，K=5最低新类`69.33%`。

本轮停止继续扩大qKNN后处理网格，转到训练/特征侧。新增训练脚本参数支持`proxy_unknown_hard_pair_ids`，用可读TX标签指定代理未知类硬对，训练时解析为ManyTx原始`tx_i`。该机制只使用source old和proxy_unknown训练池；`PROXY_UNKNOWN_TX_IDS`继续排除十个目标新类`10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3`，不把目标新类标签泄漏进地面训练。

### proxy hard-pair挖掘依据

挖掘输入为已完成的`MANYNEW10_CONFLICT_NORM_features_leo_repaired.npz`，只读取`source`和`proxy_unknown`角色，未使用目标query标签。该文件SHA256为`ABCD59B5A2766CC108DBC977B060A6D1FC62A7D80C3833910F2C1A12096F785C`。选择的proxy hard-pair如下：

`15-1:20-12,20-12:15-1,4-1:7-11,7-11:4-1,1-16:8-13,8-13:1-16,6-6:8-1,8-1:6-6,4-1:7-10,7-10:4-1,1-10:8-13,8-13:1-10,15-19:9-1,9-1:15-19,1-14:6-6,6-6:1-14`

| proxy pair | cosine sim |
|---|---:|
| `15-1:20-12` | 0.9994 |
| `4-1:7-11` | 0.9988 |
| `1-16:8-13` | 0.9988 |
| `6-6:8-1` | 0.9987 |
| `4-1:7-10` | 0.9987 |
| `1-10:8-13` | 0.9983 |
| `15-19:9-1` | 0.9981 |
| `1-14:6-6` | 0.9985 |

### 本地变更与验证

| file | purpose | SHA256 |
|---|---|---|
| `code/scripts/train_apply_phase1_iq_preadapter_20260703.py` | 增加proxy hard-pair loss和TX标签解析 | `C08D913C42D380BEF4C441BB07CBB17EC0B9DFEC5644AD674DE24E76ABB12F15` |
| `code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh` | N607训练+K=5/K=10支持集metric qKNN评估launcher | `FA4C4053988FED5A7F9F528850EE3D132CFABDCFBE6006C57C7F9ADC90B725D8` |

本地验证：

| command | result |
|---|---|
| `conda run -n ssr-gpu python -m py_compile code\scripts\train_apply_phase1_iq_preadapter_20260703.py code\scripts\phase2_support_metric_qknn_probe.py` | PASS |
| `bash -n code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh` | PASS |
| `ROOT=/tmp/cvs-rffi-dryrun RUNS_ROOT=/tmp/cvs-rffi-dryrun/runs/proxy_hp LOG_ROOT=/tmp/cvs-rffi-dryrun/logs/proxy_hp bash code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh --dry-run` | PASS |

### 待发射N607实验

| field | value |
|---|---|
| run_id | `phase2_adv3b02_manynew10_proxy_hardpair_20260705` |
| target receiver domain | `7-14` |
| old labels | `14-10,14-7,20-15,20-19,6-15,8-20` |
| new labels | `10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3` |
| strict support | `pool_per_old=K,pool_per_new=K` |
| strict query | K=10时`70/query/class`，K=5时`75/query/class` |
| variants | `MANYNEW10_PROXY_HP_NORM_SAFE`、`MANYNEW10_PROXY_HP_NORM_STRONG` |
| qKNN eval | 固定当前最强support-metric配置；K=10用`diag_fisher,strength=0.5`，K=5用`diag_whiten_fisher,strength=0.1` |
| success criterion | K=5和K=10均满足`old_acc>=80%`且每个new类`>=75%` |

待执行远端命令：`bash code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh`。发射前必须完成N607 SSH preflight、同步上述两个脚本、远端`py_compile`和`bash -n`验证，并记录PID/GPU/log路径。

### N607发射记录

2026-07-05 03:00 CST已发射正式任务。直连preflight通过：N607主机`dell-DSS8440`，项目根目录`/home/szu2070436088/2510044040/CV-SincNet`可见，GPU 0-7可见。发射前进程/GPU检查显示无训练进程，GPU 0-7空闲。

同步文件：

| local file | remote file |
|---|---|
| `code/scripts/train_apply_phase1_iq_preadapter_20260703.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/train_apply_phase1_iq_preadapter_20260703.py` |
| `code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh` |
| `code/scripts/phase2_support_metric_qknn_probe.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_metric_qknn_probe.py` |

远端验证：

| command | result |
|---|---|
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/train_apply_phase1_iq_preadapter_20260703.py code/scripts/phase2_support_metric_qknn_probe.py` | PASS |
| `bash -n code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh` | PASS |
| `bash code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh --dry-run` | PASS |

正式提交命令：

`nohup env RUN_ID=phase2_adv3b02_manynew10_proxy_hardpair_20260705 GPUS_CSV=0,1 bash code/scripts/launch_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh > logs/phase2_adv3b02_manynew10_proxy_hardpair_20260705/launcher.out 2>&1 &`

| field | value |
|---|---|
| launcher_pid | `2513737` |
| launcher_log | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_manynew10_proxy_hardpair_20260705/launcher.out` |
| SAFE process | PID`2513745`,GPU`0`,`MANYNEW10_PROXY_HP_NORM_SAFE` |
| STRONG process | PID`2513744`,GPU`1`,`MANYNEW10_PROXY_HP_NORM_STRONG` |
| train logs | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_manynew10_proxy_hardpair_20260705/*_train_export.out` |
| expected outputs | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_manynew10_proxy_hardpair_20260705/PHASE2_MANYNEW10_RX7_14/<variant>/` |

提交后短监控：GPU 0/1各有约`1481 MiB`显存占用并运行对应训练进程；GPU 2-7空闲。短SSH命令结束后，本地无残留`ssh.exe`，无到`172.31.111.215:22`的ESTABLISHED连接。

2026-07-05 03:01 CST启动健康检查：两个变体均已进入训练日志并运行到epoch 25；日志中出现`proxy_unknown_hard_pair`和`proxy_unknown_hard_old`字段，说明hard-pair标签解析和loss路径已实际生效。GPU 0/1仍分别运行SAFE/STRONG训练，GPU 2-7空闲。本次短监控后本地再次确认无残留`ssh.exe`，无到N607:22的ESTABLISHED连接。

### N607完成结果与评估修复

2026-07-05 03:05 CST复查：两个训练进程已结束，`features_leo_repaired.npz`和`features_clean_repaired.npz`均已导出；GPU 0/1恢复空闲。launcher原始评估阶段失败，原因是远端缺少`phase2_metric_adapter_probe.py`依赖，首个K=10 eval日志报错`ModuleNotFoundError: No module named 'phase2_metric_adapter_probe'`。处理方式：只同步support-metric评估依赖脚本，不重训；新增并提交eval-only launcher`code/scripts/eval_phase2_adv3b02_manynew10_proxy_hardpair_20260705.sh`，远端`bash -n`通过后补跑四个评估JSON。

| variant | K | seed | old_acc | min_old | new_acc | min_new | 达到最低新类75% |
|---|---:|---:|---:|---:|---:|---:|---|
| `MANYNEW10_PROXY_HP_NORM_SAFE` | 10 | 421029 | 83.81% | 68.57% | 82.43% | 70.00% | 否 |
| `MANYNEW10_PROXY_HP_NORM_STRONG` | 10 | 421029 | 82.62% | 67.14% | 81.86% | 71.43% | 否 |
| `MANYNEW10_PROXY_HP_NORM_SAFE` | 5 | 421037 | 82.22% | 65.33% | 78.93% | 68.00% | 否 |
| `MANYNEW10_PROXY_HP_NORM_STRONG` | 5 | 421037 | 82.44% | 66.67% | 78.13% | 66.67% | 否 |

#### SAFE K=10逐类

| 类别 | role | acc |
|---|---|---:|
| `14-10` | old | 85.71% |
| `14-7` | old | 74.29% |
| `20-15` | old | 88.57% |
| `20-19` | old | 68.57% |
| `6-15` | old | 88.57% |
| `8-20` | old | 97.14% |
| `10-10` | new | 71.43% |
| `11-10` | new | 75.71% |
| `18-5` | new | 91.43% |
| `19-3` | new | 82.86% |
| `2-13` | new | 70.00% |
| `2-5` | new | 82.86% |
| `3-8` | new | 88.57% |
| `4-10` | new | 90.00% |
| `8-18` | new | 81.43% |
| `8-3` | new | 90.00% |

#### SAFE K=5逐类

| 类别 | role | acc |
|---|---|---:|
| `14-10` | old | 84.00% |
| `14-7` | old | 77.33% |
| `20-15` | old | 89.33% |
| `20-19` | old | 65.33% |
| `6-15` | old | 81.33% |
| `8-20` | old | 96.00% |
| `10-10` | new | 74.67% |
| `11-10` | new | 70.67% |
| `18-5` | new | 85.33% |
| `19-3` | new | 74.67% |
| `2-13` | new | 68.00% |
| `2-5` | new | 78.67% |
| `3-8` | new | 86.67% |
| `4-10` | new | 86.67% |
| `8-18` | new | 78.67% |
| `8-3` | new | 85.33% |

#### STRONG K=10逐类

| 类别 | role | acc |
|---|---|---:|
| `14-10` | old | 84.29% |
| `14-7` | old | 72.86% |
| `20-15` | old | 88.57% |
| `20-19` | old | 67.14% |
| `6-15` | old | 85.71% |
| `8-20` | old | 97.14% |
| `10-10` | new | 71.43% |
| `11-10` | new | 75.71% |
| `18-5` | new | 88.57% |
| `19-3` | new | 80.00% |
| `2-13` | new | 71.43% |
| `2-5` | new | 82.86% |
| `3-8` | new | 88.57% |
| `4-10` | new | 90.00% |
| `8-18` | new | 81.43% |
| `8-3` | new | 88.57% |

#### STRONG K=5逐类

| 类别 | role | acc |
|---|---|---:|
| `14-10` | old | 82.67% |
| `14-7` | old | 80.00% |
| `20-15` | old | 89.33% |
| `20-19` | old | 66.67% |
| `6-15` | old | 78.67% |
| `8-20` | old | 97.33% |
| `10-10` | new | 74.67% |
| `11-10` | new | 69.33% |
| `18-5` | new | 80.00% |
| `19-3` | new | 72.00% |
| `2-13` | new | 66.67% |
| `2-5` | new | 80.00% |
| `3-8` | new | 85.33% |
| `4-10` | new | 88.00% |
| `8-18` | new | 80.00% |
| `8-3` | new | 85.33% |

artifact SHA256：

| file | SHA256 |
|---|---|
| `SAFE_k10.json` | `469D06DBD1D08F3CE4C1A994D6A00D9C5B2E9CE00E7F7FF3656EA40646EFF8FC` |
| `SAFE_k5.json` | `4647BFC0A8917F3CB380F6367B8AEB226CE644EA82CC1CFA63599141334B3D40` |
| `STRONG_k10.json` | `C5F19E5B25EC152B8DE2B7E9AC7B87710E11289F7F85DF0913EC4C207AD7E0EC` |
| `STRONG_k5.json` | `63DAE451CD97FD1226BFB105E5857FF284F534B86DEBF8D5F0A27D054E5111B7` |
| `manynew10_proxy_hardpair_summary.json` | `2E375E2F12156639813B5FB59BD66980B151B71313066ECC31FF87CBFAF8E483` |

结论：proxy hard-pair训练侧优化没有达成目标，也没有超过原support-metric qKNN最佳最低类。原最佳仍是`MANYNEW10_CONFLICT_NORM`：K=10的min_new为`72.86%`，K=5的min_new为`69.33%`。proxy hard-pair在K=10提高了SAFE旧类均值到`83.81%`，但牺牲新类最低类；K=5整体也回落。

### 2026-07-05 support-only pair-axis压缩qKNN负证据

新增`phase2_support_metric_qknn_probe.py`中的support-only pair-axis rerank：只用support prototype之间高相似类对构造一维判别轴，部署侧额外保存少量pair axis参数，不保存原始support样本。脚本SHA256：`906963B1D110BFAF534040C358F2323A6104DC0AD7B53AEB1E8107A0CD5D7BBC`。本地验证：`conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py`通过；`SAFE_k10_pairaxis_smoke`通过。

在历史最佳`MANYNEW10_CONFLICT_NORM`特征上快速验证：

| 特征 | K | pair-axis最佳配置 | old_acc | min_old | new_acc | min_new | 结论 |
|---|---:|---|---:|---:|---:|---:|---|
| `MANYNEW10_CONFLICT_NORM` | 10 | `similarity=0.90,weight=0.005` | 83.33% | 67.14% | 83.57% | 72.86% | 均值略升，最低类未升 |
| `MANYNEW10_CONFLICT_NORM` | 5 | `similarity=0.95,weight=0.02` | 83.11% | 65.33% | 79.87% | 69.33% | 均值略升，最低类未升 |

artifact SHA256：

| file | SHA256 |
|---|---|
| `conflict_norm_k10_pairaxis_smoke.json` | `234DB1737A9FCC6C5E4E91F7FF7C2DF7B448EEE63EEA4C7B71450B3C67D32CE5` |
| `conflict_norm_k5_pairaxis_smoke.json` | `67D93048E51FBC62B7BE412CA4B533B86118C87E5E0AAF3AAB93B2CB7DCDE130` |

结论：pair-axis是合规的压缩qKNN变体，但它只改善均值，不改善目标所需的最低类；因此当前瓶颈仍是`10-10/2-13/11-10/19-3`这组目标新类在星地特征空间中的support代表性不足和类间纠缠。下一步应优先做support selection本身的优化，而不是继续加score后处理：例如在K固定时选择低类更稳定的scenario-balanced/anti-nearest-old support，或用source/proxy训练更强的目标类相似度解耦表征。

### 2026-07-05 bootstrap/pair-gaussian/ridge压缩qKNN负证据

本轮继续限定K=5和K=10，不扩大K数量，不使用query标签拟合。`phase2_support_metric_qknn_probe.py`新增三个support-only压缩头：

- `bootstrap_proto`：由K-shot support生成留一子原型，部署侧保存派生原型，不保存原始support。
- `pair_gaussian`：对高相似support prototype类对保存一维pair轴及两侧support投影均值/方差，用高斯似然差修正分数。
- `ridge_head`：用K-shot support闭式训练L2岭线性残差头，与qKNN分数小权重融合，部署侧保存权重矩阵。

脚本SHA256：`6B2FE9A94711D6D98D30C34F0289222BFD4311D8F58F5D442068714719AB94C7`。本地验证：`conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py`通过。

固定历史最佳support seed的小网格结果：

| 变体 | K | 最佳配置 | old_acc | min_old | new_acc | min_new | 逐类瓶颈 |
|---|---:|---|---:|---:|---:|---:|---|
| bootstrap_proto | 10 | `mix=0`仍最佳 | 83.10% | 67.14% | 83.43% | 72.86% | `10-10`72.86%,`2-13`72.86% |
| bootstrap_proto | 5 | `mix=0`仍最佳 | 82.67% | 65.33% | 79.47% | 69.33% | `2-13`69.33%,`11-10`72.00% |
| pair_gaussian | 10 | `sim=0.95,weight=0.005,clip=2` | 83.57% | 67.14% | 84.00% | 72.86% | `10-10`72.86%,`2-13`72.86% |
| pair_gaussian | 5 | `sim=0.85,weight=0.02,clip=2` | 83.33% | 65.33% | 80.13% | 70.67% | `2-13`70.67%,`11-10`73.33% |
| ridge_head | 10 | `weight=0.002,alpha=0.1` | 83.57% | 68.57% | 83.57% | 72.86% | `10-10`72.86%,`2-13`72.86% |
| ridge_head | 5 | `weight=0.01,alpha=0.1` | 82.89% | 64.00% | 79.73% | 69.33% | `2-13`69.33% |

120个support seed复核：

| 变体 | K | 固定配置 | best seed | old_acc | min_old | new_acc | min_new | 结论 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| pair_gaussian | 10 | `sim=0.95,weight=0.005,clip=2` | 421029 | 83.57% | 67.14% | 84.00% | 72.86% | 未过75%最低类 |
| pair_gaussian | 5 | `sim=0.85,weight=0.02,clip=2` | 421037 | 83.33% | 65.33% | 80.13% | 70.67% | 未过75%最低类 |

artifact SHA256：

| file | SHA256 |
|---|---|
| `strict_n10_k10_bootproto_seed421029_focus.json` | `7D7DC85A7C9A44E6A0EB9BF7313DA11C55DBEE8112D72AABDAEF36227638CA8A` |
| `strict_n10_k5_bootproto_seed421037_focus.json` | `B5FA4FE8DABC982AC47CDB1F8DEA37405C56CC7FD1242B4CF795E710B9C99643` |
| `strict_n10_k10_pairgauss_seed120.json` | `B9C3DB6DB065A0776CB209B51FB2740EE257605A853C89FDE7C0A077B101259D` |
| `strict_n10_k5_pairgauss_seed120.json` | `0DF611041D988269626601235BD06772157FA09046803FDD6FF7ED99304B1860` |
| `strict_n10_k10_ridgehead_seed421029_focus.json` | `DE528EF9507CA6A3EE14816523BDEC2EB5D09291FA577D4D0D6421BA6506055A` |
| `strict_n10_k5_ridgehead_seed421037_focus.json` | `C9C60892A422DC36B2827791B57996D05329AD159D31332AE16B00CF87E22F3A` |

结论：新增三个压缩qKNN头均合规且部署侧不保存原始support，但仍无法把十新类最低类抬到75%。pair_gaussian是本轮最有效的后处理：K=5最低类从69.33%升至70.67%，K=10新类均值从83.43%升至84.00%，但`10-10/2-13`硬对仍卡在72.86%。下一步不应继续扩大后处理网格；在严格K-shot且`pool_per_class=K`时，support selection本身没有候选空间，必须转向表征侧或训练侧生成更可分的星地特征，再用同一K=5/K=10压缩qKNN头复核。

### 2026-07-05 Mahalanobis与dual-view压缩qKNN继续优化

本轮在不扩大K数量、不使用query标签拟合的约束下，继续加入两个压缩路线：

- `mahal_proto`：由K-shot support估计收缩协方差逆矩阵，使用Mahalanobis prototype分数作为qKNN残差；部署侧保存类原型和协方差逆矩阵，不保存原始support。
- `dual-view qKNN`：对同一support在`MANYNEW10_CONFLICT_NORM`和`MANYNEW10_CONFLICT_HEAD`两个已导出表征视图中分别建压缩qKNN分数，再按权重融合；部署侧保存两个视图的压缩原型/变换参数，不保存support原样本。

本地验证：`conda run -n ssr-gpu python -m py_compile code\scripts\phase2_support_metric_qknn_probe.py`通过。

新增结果：

| 路线 | K | 配置 | old_acc | min_old | new_acc | min_new | 关键逐类 |
|---|---:|---|---:|---:|---:|---:|---|
| Mahalanobis residual | 10 | `weight=0.05,alpha=0.1,diag_mix=0` | 84.05% | 68.57% | 84.14% | 72.86% | `10-10`72.86%,`2-13`72.86% |
| Mahalanobis residual | 5 | `weight=0.05,alpha=1,diag_mix=0` | 82.22% | 62.67% | 79.60% | 70.67% | `2-13`70.67%,`11-10`72.00% |
| dual-view grid | 10 | `NORM primary + HEAD aux; aux_w=0` | 83.57% | 67.14% | 84.00% | 72.86% | `10-10`72.86%,`2-13`72.86% |
| dual-view grid | 5 | `NORM primary + HEAD aux_w=0.8,topm=4,proto_mix=0.4` | 82.44% | 62.67% | 81.20% | 74.67% | `10-10`74.67%,`2-13`74.67% |

dual-view K=5是当前最接近目标的qKNN头：逐新类为`10-10`74.67%,`11-10`76.00%,`18-5`84.00%,`19-3`82.67%,`2-13`74.67%,`2-5`80.00%,`3-8`86.67%,`4-10`88.00%,`8-18`81.33%,`8-3`84.00%。它距离75%最低类只差每个瓶颈类1个正确query样本，但仍不能报告达标。

120个support seed复核：

| 路线 | K | 固定配置 | best seed | old_acc | min_old | new_acc | min_new | 结论 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| dual-view | 10 | `aux_w in {0,0.2,0.4,0.6,0.8}` | 421029 | 83.57% | 67.14% | 84.00% | 72.86% | 未过75% |
| dual-view | 5 | `aux_w in {0.6,0.7,0.8,0.9}` | 421037 | 82.44% | 62.67% | 81.20% | 74.67% | 未过75%，差1个query |

已有10个本地表征重扫显示，当前最强候选仍来自conflict-protected表征族：K=10最佳是`MANYNEW10_CONFLICT_NORM`，K=5最佳是`MANYNEW10_CONFLICT_HEAD/NORM dual-view`。`SUPCON/SEP/IDENTITY/proxy_hardpair`表征在最低新类上均未超过上述结果。

artifact SHA256：

| file | SHA256 |
|---|---|
| `strict_n10_k10_mahalproto_seed421029_focus.json` | `A7A1231439ACFD3B9488E72FC616EDE65E3C13F50C63AB86E3115CB13323238B` |
| `strict_n10_k5_mahalproto_seed421037_focus.json` | `B378736156CA96CB5326CEF7F49BC7358041C74324CE3103F611477BF1FF53BC` |
| `norm_head_k10_dualview_grid.json` | `38941BE38B8798C34C7E2FBF6CEAEB1896C600C21C1DD7FFB1B7A2FC710ED57D` |
| `norm_head_k5_dualview_grid.json` | `2CA8C7F60DE7D3298709F51BA21C805C89EF9980E97E58FFD03A261648C9C97E` |
| `norm_head_k10_dualview_seed120.json` | `98E74996951D2EDD81942F50D280AA4106FB6AD6B8463286188919C4DB8BCBA3` |
| `norm_head_k5_dualview_seed120.json` | `B61DEAD5382497BCC16C982D91E5EE245C47224BD336C2834AD44CA1019C248C` |

结论：qKNN压缩头侧已经把K=5最低类推进到74.67%，但K=10仍为72.86%。这说明当前主要瓶颈不是support存储形式，而是`10-10/2-13`在现有星地表征空间中的类间可分性不足。下一步应围绕`CONFLICT_NORM/CONFLICT_HEAD`的双视图思路做训练侧表征改造，而不是继续扩大K或继续加纯后处理网格。
