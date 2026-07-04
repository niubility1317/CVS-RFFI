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

当前状态：本地脚本和报告已准备；尚未启动N607，需先执行N607 direct preflight并检查GPU/进程占用。
