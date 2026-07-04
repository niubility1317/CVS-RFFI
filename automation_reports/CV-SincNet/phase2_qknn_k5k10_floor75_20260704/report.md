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
