# phase2_adv3b02_stage2c_qknnv42_registered_only_20260707

## 基本信息

| 字段 | 内容 |
|---|---|
| 实验ID | `phase2_adv3b02_stage2c_qknnv42_registered_only_20260707` |
| 时间 | 2026-07-07 |
| 操作者 | Codex |
| 目标 | 在qKNNV42当前基础上按Phase2主线优化旧类目标域适应、seen-new注册识别、最低类性能；unknown拒识不作为Phase2成功门槛 |
| 协议依据 | `AGENTS.md`、`项目.md` |
| 数据协议 | `R_t=7-14`，`Y_old={14-10,14-7,20-15,20-19,6-15,8-20}`，`Y_new`来自ManyTx非旧类，support/query均为同一目标接收机LEO视图 |
| 对照 | `phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_retry1_20260707`最佳qKNNV42实际行为`old_acc=0.6119,min_old=0,seen_new_acc=0` |

## 假设

旧类上限诊断已显示同一feature包中target-old信号可达`old_acc=0.9286,min_old=0.8143`，而qKNNV42实际行仅`old_acc=0.6119,min_old=0,seen_new_acc=0`。主要瓶颈不是feature不可分，而是Phase2主线中仍使用unknown/open-set风险仲裁，导致已注册old/seen-new样本被`unknown_reject`或`defer`压掉。本实验新增`fusion_policy=phase2_registered_only`，只在已注册`Y_old∪Y_new`内做支持证据仲裁，unknown风险保留为Phase3备用诊断列，不参与Phase2主排序。

## 本地改动

| 文件 | 目的 |
|---|---|
| `code/evaluation/collaborative_open_set_qknn_eval.py` | 新增`phase2_registered_only`融合策略；对old/seen-new注册标签按candidate receiver、top1、p-value、receiver reliability、score、margin和score gap接受；unknown风险不触发Phase2拒识；新增`phase2_registered_only_accept_count/rate/by_role`聚合字段 |
| `code/scripts/phase2_collaborative_open_set_qknn_eval.py` | CLI开放`--fusion_policy phase2_registered_only` |
| `code/tests/test_collaborative_open_set_qknn_eval.py` | 增加高unknown风险下registered old/seen-new仍被接受的回归测试 |
| `code/tests/test_phase2_collaborative_open_set_qknn_eval.py` | 增加CLI解析测试 |
| `code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_registered_only_20260707.sh` | 新增16行小矩阵：2个feature variant×4个profile×K=5/10；Phase2排序只使用old/seen-new和`H_old_new` |

## 本地验证

| 命令 | 结果 |
|---|---|
| `conda run -n ssr-gpu python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py -k phase2_registered_only -q` | PASS，新增核心测试1 passed |
| `conda run -n ssr-gpu python -m pytest code/tests/test_phase2_collaborative_open_set_qknn_eval.py -k phase2_registered_only -q` | PASS，新增CLI测试1 passed |
| `conda run -n ssr-gpu python -m pytest code/tests/test_collaborative_open_set_qknn_eval.py code/tests/test_phase2_collaborative_open_set_qknn_eval.py -q` | PASS，根目录130 passed；Git承载面130 passed |
| `conda run -n ssr-gpu python -m py_compile code/evaluation/collaborative_open_set_qknn_eval.py code/scripts/phase2_collaborative_open_set_qknn_eval.py` | PASS |
| `bash -n code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_registered_only_20260707.sh` | PASS |
| `bash -lc "ROOT=/tmp/CV-SincNet-qknnv42-regtest DRY_RUN=1 bash code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_registered_only_20260707.sh --dry-run"` | PASS，展开16个candidate |

## 版本状态

| 项 | 内容 |
|---|---|
| 根目录Git状态 | `E:\type10-7\code`不是Git仓库 |
| 本地快照 | `E:\type10-7\code\snapshots\phase2_adv3b02_stage2c_qknnv42_registered_only_20260707\` |
| Git承载面 | `E:\type10-7\github_publish\CVS-RFFI-repo` |
| Git状态 | 分支`codex/cvs-rffi-release-20260626`，本次5个代码/测试文件修改、1个launcher新增；既有未跟踪`local_artifacts/phase2_adv3b02_proxy_mined_20260704/`和`local_artifacts/phase2_adv3b02_smec_ci_20260704/`非本次改动 |

## N607计划

| 项 | 内容 |
|---|---|
| 远端ROOT | `/home/szu2070436088/2510044040/CV-SincNet` |
| 远端脚本 | `code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_registered_only_20260707.sh` |
| 远端run root | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_qknnv42_registered_only_20260707` |
| 远端log root | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_qknnv42_registered_only_20260707` |
| 源feature run | `phase2_adv3b02_stage2c_normsep_protocol_20260707` |
| 预期summary | `logs/phase2_adv3b02_stage2c_qknnv42_registered_only_20260707/stage2c_qknnv42_registered_only_summary.json`和`.csv` |
| 成功观察 | 首先看`old_acc>=0.80`且`min_old_class_acc>0`；若达成，再看`seen_new_acc`、`min_seen_new_class_acc`和`H_old_new`；unknown列只作Phase3备用诊断 |

## 风险与检查点

| 风险 | 检查方式 |
|---|---|
| 无unknown拒识后unknown false accept上升 | 只作为Phase3备用诊断记录，不阻塞Phase2主线；报告中分表解释 |
| single target receiver导致collab_count=1，最低类仍可能为0 | 查看`per_old_class_acc`、`per_seen_new_class_acc`和`phase2_registered_only_accept_by_role` |
| center/contrast profile可能提升seen-new但伤害old | 按同row`old_acc/min_old/seen_new/min_seen/H_old_new`排序，不用单项最大值 |
| 远端已有活跃训练 | 按N607规则先preflight和只读进程/GPU检查；若不适合启动则停在已验证待同步状态 |

## N607执行记录

| 项 | 内容 |
|---|---|
| preflight | 2026-07-07 19:16 CST，直连`N607`通过；项目根目录可见；GPU2-GPU7空闲，GPU0/GPU1后续发现各有一个独立RIEI/DRIFT训练进程 |
| 活跃进程边界 | PID`131101`为`baselines.riei_fd.train`，PID`131105`为`baselines.drift.train`；本次qKNNV42为no-training评估矩阵，不介入这些进程 |
| 同步文件 | `collaborative_open_set_qknn_eval.py`、`phase2_collaborative_open_set_qknn_eval.py`、launcher、两个测试文件 |
| 远端hash | eval=`c31f65b42cf27bc9372bc0e742ca8eecd3507cde0dc15132e090a855e608dc35`；CLI=`08f31773456ea7733ff661220d7b43b2c52ecca4ac74d9b9a4ca226a28d83f02`；launcher=`337c4cbb622cda4c1dd8c46a05401500d3c91aa6b1f8a16d0a8353a78b3a0a97` |
| 远端验证 | `python3 -m py_compile ...` PASS；`bash -n` PASS；`DRY_RUN=1` PASS，展开16个candidate |
| 失败启动处理 | 早期失败日志`STAGE2C_NORM_SEP_REGISTERED_BASE_k5.out`已备份到`logs/phase2_adv3b02_stage2c_qknnv42_registered_only_20260707/failed_startup_backup_20260707_1918/` |
| 最终启动 | `LAUNCHER_PID=134690`，命令为`nohup bash code/scripts/launch_phase2_adv3b02_stage2c_qknnv42_registered_only_20260707.sh > logs/phase2_adv3b02_stage2c_qknnv42_registered_only_20260707.launch.out 2>&1 < /dev/null &` |
| 最终状态 | 2026-07-07 19:20 CST完成，`[STAGE2C-QKNNV42-REGISTERED-ONLY-DONE]`；无`Traceback`、`TypeError`或`unrecognized arguments` |
| 本地拉取 | `stage2c_qknnv42_registered_only_summary.json`和`.csv`已拉到本报告目录 |

## 最终结果表

排序规则：`phase2_old80_ready`优先，然后按`phase2_joint_score`、`old_acc`、`min_old_class_acc`、`H_old_new`、`seen_new_acc`、`min_seen_new_class_acc`降序。unknown拒识不是Phase2主线门槛，`unknown_FAR`只作为Phase3诊断列。

| Rank | Variant | Profile | K | old80 | joint | old_acc | seen_new_acc | H_old_new | min_old | min_seen | coverage | defer | unknown_FAR | p95_ms |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `STAGE2C_HEAD_SEP` | `REGISTERED_CENTER_CONTRAST` | 5 | true | 3.236613 | 0.826190 | 0.321429 | 0.462804 | 0.657143 | 0.142857 | 1.000000 | 0.000000 | 1.000000 | 1.417 |
| 2 | `STAGE2C_HEAD_SEP` | `REGISTERED_CENTER` | 5 | true | 3.231477 | 0.823810 | 0.321429 | 0.462429 | 0.657143 | 0.142857 | 1.000000 | 0.000000 | 1.000000 | 1.308 |
| 3 | `STAGE2C_NORM_SEP` | `REGISTERED_BASE` | 10 | true | 3.225403 | 0.807143 | 0.387500 | 0.523617 | 0.514286 | 0.185714 | 1.000000 | 0.000000 | 1.000000 | 1.588 |
| 4 | `STAGE2C_NORM_SEP` | `REGISTERED_CONTRAST` | 10 | true | 3.225403 | 0.807143 | 0.387500 | 0.523617 | 0.514286 | 0.185714 | 1.000000 | 0.000000 | 1.000000 | 1.955 |
| 5 | `STAGE2C_NORM_SEP` | `REGISTERED_CENTER` | 10 | true | 3.211494 | 0.816667 | 0.376786 | 0.515661 | 0.528571 | 0.157143 | 1.000000 | 0.000000 | 1.000000 | 1.639 |
| 6 | `STAGE2C_NORM_SEP` | `REGISTERED_CENTER_CONTRAST` | 10 | true | 3.211494 | 0.816667 | 0.376786 | 0.515661 | 0.528571 | 0.157143 | 1.000000 | 0.000000 | 1.000000 | 1.961 |
| 7 | `STAGE2C_NORM_SEP` | `REGISTERED_CENTER` | 5 | true | 3.093872 | 0.830952 | 0.282143 | 0.421253 | 0.657143 | 0.071429 | 1.000000 | 0.000000 | 1.000000 | 1.283 |
| 8 | `STAGE2C_NORM_SEP` | `REGISTERED_CENTER_CONTRAST` | 5 | true | 3.093872 | 0.830952 | 0.282143 | 0.421253 | 0.657143 | 0.071429 | 1.000000 | 0.000000 | 1.000000 | 1.481 |
| 9 | `STAGE2C_HEAD_SEP` | `REGISTERED_CONTRAST` | 5 | true | 3.087450 | 0.823810 | 0.300000 | 0.439831 | 0.628571 | 0.071429 | 1.000000 | 0.000000 | 1.000000 | 1.563 |
| 10 | `STAGE2C_HEAD_SEP` | `REGISTERED_BASE` | 5 | true | 3.087450 | 0.823810 | 0.300000 | 0.439831 | 0.628571 | 0.071429 | 1.000000 | 0.000000 | 1.000000 | 1.229 |
| 11 | `STAGE2C_NORM_SEP` | `REGISTERED_CONTRAST` | 5 | true | 3.057613 | 0.823810 | 0.285714 | 0.424280 | 0.642857 | 0.057143 | 1.000000 | 0.000000 | 1.000000 | 1.618 |
| 12 | `STAGE2C_NORM_SEP` | `REGISTERED_BASE` | 5 | true | 3.012679 | 0.821429 | 0.280357 | 0.418037 | 0.628571 | 0.042857 | 1.000000 | 0.000000 | 1.000000 | 1.274 |
| 13 | `STAGE2C_HEAD_SEP` | `REGISTERED_BASE` | 10 | false | 3.242623 | 0.792857 | 0.353571 | 0.489052 | 0.600000 | 0.214286 | 1.000000 | 0.000000 | 1.000000 | 1.577 |
| 14 | `STAGE2C_HEAD_SEP` | `REGISTERED_CONTRAST` | 10 | false | 3.242623 | 0.792857 | 0.353571 | 0.489052 | 0.600000 | 0.214286 | 1.000000 | 0.000000 | 1.000000 | 1.954 |
| 15 | `STAGE2C_HEAD_SEP` | `REGISTERED_CENTER_CONTRAST` | 10 | false | 3.230052 | 0.795238 | 0.351786 | 0.487790 | 0.571429 | 0.228571 | 1.000000 | 0.000000 | 1.000000 | 2.051 |
| 16 | `STAGE2C_HEAD_SEP` | `REGISTERED_CENTER` | 10 | false | 3.224841 | 0.792857 | 0.351786 | 0.487341 | 0.571429 | 0.228571 | 1.000000 | 0.000000 | 1.000000 | 1.517 |

## 更正结论

本实验不能登记为qKNNV42当前最佳版本。

原因：本实验仍然使用带独立`target_unknown`字段的冻结诊断包作为输入，并在报告中按“完整Stage2-C互斥unknown包”描述结果。该口径与当前Phase2主线要求不一致。当前Phase2主线只评估K=5/K=10目标域LEO旧类适应和seen-new注册识别；unknown互斥/拒识不得作为当前最佳版本的主评价包或主排序依据。

本实验只能保留为错误口径诊断：它说明在该冻结诊断包上去掉unknown拒识主导后，old/seen-new仍没有恢复到no-unknown主线水平，最佳行只有`old_acc=0.826190`、`min_old=0.657143`、`seen_new_acc=0.321429`、`min_seen=0.142857`。它不得覆盖`phase2_qknn_hardpair_n20_20260706`中no-unknown主线的当前最佳结论。

当前应引用的no-unknown主线最佳为`phase2_qknn_hardpair_n20_20260706`报告中的K5严格高floor候选：`seed=421070`，参数`aux_score_weight=0.34,labelprop_weight=0.025,labelprop_alpha=0.76,scenario_residual_weight=0.5,old_bias=0.001`，指标`old=94.52%,min_old=85.71%,seen_new=90.14%,min_new=81.43%`。该候选使用N20 HP08L5注册新类包，目标域support/query均为LEO叠加视图；不以unknown互斥或unknown FAR为主任务。
