# phase1_accept_domain_verify_20260701_130328

## 基本信息

| 字段 | 内容 |
| --- | --- |
| experiment ID | `phase1_accept_domain_verify_20260701_130328` |
| timestamp | `2026-07-01 13:03:28 Asia/Hong_Kong` |
| operator/agent | Codex |
| objective | 在不使用GPU0的前提下，为GPU1-GPU7各提交2个Phase1验证实验，验证accept-domain局部组件原型导出修复和已接入训练循环的FSP/vacuum/source-episode/fusion参数族。 |
| protocol | `Safe-SSDG-CVS-R01`，`split_mode=tx_rx_day_1_7_2`，`labeled_ratio=0.10`，`unlabeled_ratio=0.70`，`source_val_ratio=0.20`。 |
| target receiver visibility | 训练期保持`source_only_ground_training_no_target_receiver`；不使用target receiver query/test作为支持或调参来源。 |
| concurrency policy | `STAGE2_MAX_ACTIVE_PER_GPU=2`；GPU0排除；GPU1-GPU7各2个候选，启动脚本会在单卡活跃计算进程数达到2时等待。 |

## 假设和对照

对照为`phase1_fsp_vacuum_20260701`中的强候选和技术报告结论：`FSP_VAC_R17_Q2_HARDK3_E280`主性能最强，`FSP_VAC_R20_Q2_SAT70_E280`闭集与receiver floor更稳，`FSP_VAC_R28_Q2_SAT72_E300`proxy vacuum更低但主性能下降，`FSP_VAC_T13_LATE60_SAT68_E260`可作为保守尾部对照。本轮不使用新CLI占位项`--neg_shell_ratio`、`--tail_quarantine`、`--unlabeled_risk_buffer`作为训练机制证据，因为当前训练环节尚未消费这些参数。本轮只验证已经接入训练循环的`lambda_open_world_feat`、`lambda_zid_compact`、`lambda_proxy_unknown`、`lambda_source_episode`和`phase2_fuse_prototypes`相关路径。

核心假设：

| 假设 | 验证方式 | 成功信号 |
| --- | --- | --- |
| R17/R20附近的E260缩短训练能保留主性能并降低E300过拟合风险 | GPU1/GPU2的R17/R20变体 | `best_joint_test_tx`接近或超过旧R17/R20，`final_strict_udu`不退化，`receiver_floor`不低于旧强候选。 |
| proxy/vacuum强度需要局部提高但不能牺牲闭集 | GPU1、GPU2、GPU7压力组 | `proxy_vac_rate`下降，同时`final_strict_udu`和`sat_strict_floor`没有大幅下降。 |
| fusion组件约束需要验证导出修复后是否稳定 | GPU3、GPU6 fusion组 | `phase2_zid_prototypes.pt`可导出，含局部组件融合元数据；`p95`、`min_inter`与分类指标同排可解释。 |
| source-episode溢出需要被降档或限半径 | GPU5 source组 | `source_episode_overflow`低于旧R组，同时joint/strict不出现明显回退。 |
| T13保守路线可作为低尾部对照 | GPU4 T13组 | tail/source指标低于R组，并提供闭集-拒识权衡下界。 |

## 候选矩阵

| GPU | 候选 | seed | epochs | 设计角色 | 关键差异 |
| --- | --- | ---: | ---: | --- | --- |
| 1 | `ADV2_R17_CORESTRICT_E260` | 364601 | 260 | R17主线缩短训练 | 保留R17 hard-k3、E260、局部组件融合。 |
| 1 | `ADV2_R17_PROXYHI_E260` | 364602 | 260 | R17拒识压力 | 提高`lambda_proxy_unknown`和proxy vacuum宽度。 |
| 2 | `ADV2_R20_SAT70_E260` | 364611 | 260 | R20稳定主线 | 继承SAT70、较低vacuum强度。 |
| 2 | `ADV2_R20_VACMID_E260` | 364612 | 260 | R20中等vacuum | 提高OW/proxy vacuum到中档。 |
| 3 | `ADV2_R28_PROXYLOW_E260` | 364621 | 260 | R28 proxy低风险对照 | 保留R28低proxy-vac风格但缩短到E260。 |
| 3 | `ADV2_R28_FUSE6_E260` | 364622 | 260 | R28严格融合 | `phase2_fuse_max_components=6`、`merge=2.0`、`radius_cap=15`。 |
| 4 | `ADV2_T13_CONSERVE_E260` | 364631 | 260 | T13保守对照 | 低OW、低source-episode、SAT68。 |
| 4 | `ADV2_T13_TAILGUARD_E260` | 364632 | 260 | T13尾部保护 | 轻度提高tail/vacuum并收紧source radius。 |
| 5 | `ADV2_SRCLOW_R17_E260` | 364641 | 260 | source overflow修复 | 降低`lambda_source_episode`，扩大source半径容忍。 |
| 5 | `ADV2_SOURCECAP32_R20_E260` | 364642 | 260 | source半径上限压力 | `source_episode_radius_cap_deg=32`，验证溢出-闭集折中。 |
| 6 | `ADV2_FUSE6_R17_E260` | 364651 | 260 | R17融合压力 | fusion 6组件、merge 2.0、radius cap 15。 |
| 6 | `ADV2_FUSE5_R20_E260` | 364652 | 260 | R20融合中档 | fusion 5组件、merge 2.5、radius cap 16。 |
| 7 | `ADV2_TAILCV_R17_E260` | 364661 | 260 | R17尾部压力 | 高tail、高vacuum，用于观察拒识增益代价。 |
| 7 | `ADV2_TAILCV_R20_E260` | 364662 | 260 | R20尾部压力 | 中高tail/vacuum，保留SAT70稳定项。 |

## 本地文件变更

| 文件 | 目的 |
| --- | --- |
| `code/scripts/launch_phase1_accept_domain_verify_20260701.sh` | 新增GPU1-GPU7双实验验证启动器，默认排除GPU0并限制单卡最多2个活跃训练进程。 |
| `automation_reports/CV-SincNet/phase1_accept_domain_verify_20260701_130328/report.md` | 记录实验设计、协议边界、候选矩阵、验证命令、同步与启动证据。 |

## 待执行验证命令

本地：

```bash
bash -n code/scripts/launch_phase1_accept_domain_verify_20260701.sh
bash code/scripts/launch_phase1_accept_domain_verify_20260701.sh --dry-run
conda run --no-capture-output -n ssr-gpu python -m py_compile code/SSDG/train_ssdg.py code/cvsrffi/phase2_prototypes.py
```

N607预检：

```powershell
powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1
conda run --no-capture-output -n ssr-gpu python tools/n607_training_inventory.py --direct-only --pretty
```

N607目标启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
mkdir -p logs/phase1_accept_domain_verify_20260701_130328
nohup bash code/scripts/launch_phase1_accept_domain_verify_20260701.sh > logs/phase1_accept_domain_verify_20260701_130328/scheduler.out 2>&1 & echo $!
```

## 预期输出

| 路径 | 内容 |
| --- | --- |
| `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_accept_domain_verify_20260701_130328/<candidate>/` | 每个候选的checkpoint、metrics和`phase2_zid_prototypes.pt`。 |
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_accept_domain_verify_20260701_130328/<candidate>.out` | 单候选训练日志。 |
| `/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_accept_domain_verify_20260701_130328/scheduler.out` | 调度器提交日志和各候选PID。 |

## 风险和检查点

| 风险 | 处理 |
| --- | --- |
| GPU1-GPU7已有2个训练进程 | 启动脚本等待；如果预检发现所有目标卡已满，本轮记录为容量延后，不强行超额。 |
| 当前新增accept-domain CLI占位项未进入训练循环 | 本轮不把这些占位项作为训练实验变量，只用已接入损失和融合导出路径。 |
| 远端代码未同步到本地修复版本 | 本地验证通过后再`scp`同步所需脚本和核心模块，远端只做短命令验证。 |
| 训练完成前无法给出最终性能 | 本报告先记录启动与预期检查项；完成后再追加同候选同排指标表。 |

## 执行状态

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| 本地脚本创建 | DONE | `code/scripts/launch_phase1_accept_domain_verify_20260701.sh` |
| 本地验证 | DONE | `bash -n`通过；`--dry-run`展开14个候选；GPU0候选0条；GPU1-GPU7各2条；`py_compile`通过。 |
| Git/快照 | DONE | 快照`E:\type10-7\code\snapshots\phase1_accept_domain_verify_20260701_130328\`；Git提交`fb3de83`。 |
| N607预检 | DONE | 直连预检通过；训练清单为空；GPU1-GPU7可提交。 |
| N607同步 | DONE | 4个文件已同步，远端`bash -n`、`py_compile`、干跑计数和SHA256验证通过。 |
| N607启动 | SUBMITTED | scheduler PID`2548586`；`submit_complete=1`；14条候选全部提交；GPU1-GPU7各2条；GPU计算PID数14。 |

## 本地验证记录

| 命令 | 结果 |
| --- | --- |
| `bash -n code/scripts/launch_phase1_accept_domain_verify_20260701.sh` | PASS |
| `bash code/scripts/launch_phase1_accept_domain_verify_20260701.sh --dry-run` | PASS，展开14个候选和完整训练命令。 |
| 结构化干跑计数 | `candidate_lines=14`，`gpu0_candidate_lines=0`，`gpu1=2`，`gpu2=2`，`gpu3=2`，`gpu4=2`，`gpu5=2`，`gpu6=2`，`gpu7=2`。 |
| `conda run --no-capture-output -n ssr-gpu python -m py_compile code/SSDG/train_ssdg.py code/cvsrffi/phase2_prototypes.py code/cvsrffi/losses.py` | PASS |

## 本地快照和哈希

| 文件 | SHA256 |
| --- | --- |
| `code/scripts/launch_phase1_accept_domain_verify_20260701.sh` | `6B821CF9EE5D0045D701409A8C01387C2F6B2518D225A340D6873E2B4979C429` |
| `code/SSDG/train_ssdg.py` | `64E50640D32913DFAECBC1E2E8B2E1ECF875E6FB27070FC41E2B5AF5ABFA545A` |
| `code/cvsrffi/phase2_prototypes.py` | `D0A5C59C8D615FFC9935808367A9A812BC830D8D11E83CF953A13E871A8CC85E` |
| `code/cvsrffi/losses.py` | `C74BEB4CF156320E21865AD53BA0A22319745A88AD7A09F5CE455448B86D5C8F` |

快照路径：`E:\type10-7\code\snapshots\phase1_accept_domain_verify_20260701_130328\`。

Git发布树提交：`fb3de83 Add accept-domain validation launch matrix`。

## N607预检和占用

| 检查 | 结果 |
| --- | --- |
| `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1` | PASS，直连`N607`；服务器`dell-DSS8440`；项目根目录`/home/szu2070436088/2510044040/CV-SincNet`可见；GPU0-GPU7均为RTX3090。 |
| 预检后本地SSH清理 | `ssh_process_count=0`，`n607_or_bridge_established_count=0`。 |
| `conda run --no-capture-output -n ssr-gpu python tools/n607_training_inventory.py --direct-only --pretty` | PASS，`gpu_compute=[]`，`active_training_processes=[]`，`route_used=direct`。 |
| 清单后本地SSH清理 | `ssh_process_count=0`，`n607_or_bridge_established_count=0`。 |

## N607同步和远端验证

| 本地文件 | N607目标 |
| --- | --- |
| `code/scripts/launch_phase1_accept_domain_verify_20260701.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_accept_domain_verify_20260701.sh` |
| `code/SSDG/train_ssdg.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py` |
| `code/cvsrffi/phase2_prototypes.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/phase2_prototypes.py` |
| `code/cvsrffi/losses.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py` |

远端验证：

| 检查 | 结果 |
| --- | --- |
| `bash -n code/scripts/launch_phase1_accept_domain_verify_20260701.sh` | PASS |
| `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/SSDG/train_ssdg.py code/cvsrffi/phase2_prototypes.py code/cvsrffi/losses.py` | PASS |
| 远端干跑计数 | `remote_candidate_lines=14`，`remote_gpu0_candidate_lines=0`，`remote_gpu1=2`，`remote_gpu2=2`，`remote_gpu3=2`，`remote_gpu4=2`，`remote_gpu5=2`，`remote_gpu6=2`，`remote_gpu7=2`。 |
| 远端SHA256 | 与本地快照一致：`6b821cf9...`、`64e50640...`、`d0a5c59c...`、`c74beb4c...`。 |
| 同步/验证后本地SSH清理 | 每次检查均为`ssh_process_count=0`，`n607_or_bridge_established_count=0`。 |

## N607启动记录

启动时间：`2026-07-01 13:11:36 CST`。

工作目录：`/home/szu2070436088/2510044040/CV-SincNet`。

服务端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
RUN_ID=phase1_accept_domain_verify_20260701_130328
mkdir -p logs/$RUN_ID
nohup bash code/scripts/launch_phase1_accept_domain_verify_20260701.sh > logs/$RUN_ID/scheduler.out 2>&1 & echo $!
```

调度器PID：`2548586`。

调度器日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_accept_domain_verify_20260701_130328/scheduler.out`。

run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_accept_domain_verify_20260701_130328`。

提交核对：

| 证据 | 值 |
| --- | --- |
| `submit_complete` | `1` |
| `launched_total` | `14` |
| `launched_gpu1` | `2` |
| `launched_gpu2` | `2` |
| `launched_gpu3` | `2` |
| `launched_gpu4` | `2` |
| `launched_gpu5` | `2` |
| `launched_gpu6` | `2` |
| `launched_gpu7` | `2` |
| `run_dir_count` | `14` |
| `log_count` | `14` |
| `gpu_compute_count` | `14` |
| `gpu_compute_pids` | `2548665,2552044,2548646,2552028,2548641,2552020,2548662,2552040,2548651,2552038,2548653,2552024,2548661,2552033` |

候选PID：

| GPU | 候选 | PID | log |
| --- | --- | ---: | --- |
| 1 | `ADV2_R17_CORESTRICT_E260` | 2548665 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_R17_CORESTRICT_E260.out` |
| 1 | `ADV2_R17_PROXYHI_E260` | 2552044 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_R17_PROXYHI_E260.out` |
| 2 | `ADV2_R20_SAT70_E260` | 2548646 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_R20_SAT70_E260.out` |
| 2 | `ADV2_R20_VACMID_E260` | 2552028 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_R20_VACMID_E260.out` |
| 3 | `ADV2_R28_PROXYLOW_E260` | 2548641 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_R28_PROXYLOW_E260.out` |
| 3 | `ADV2_R28_FUSE6_E260` | 2552020 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_R28_FUSE6_E260.out` |
| 4 | `ADV2_T13_CONSERVE_E260` | 2548662 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_T13_CONSERVE_E260.out` |
| 4 | `ADV2_T13_TAILGUARD_E260` | 2552040 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_T13_TAILGUARD_E260.out` |
| 5 | `ADV2_SRCLOW_R17_E260` | 2548651 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_SRCLOW_R17_E260.out` |
| 5 | `ADV2_SOURCECAP32_R20_E260` | 2552038 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_SOURCECAP32_R20_E260.out` |
| 6 | `ADV2_FUSE6_R17_E260` | 2548653 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_FUSE6_R17_E260.out` |
| 6 | `ADV2_FUSE5_R20_E260` | 2552024 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_FUSE5_R20_E260.out` |
| 7 | `ADV2_TAILCV_R17_E260` | 2548661 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_TAILCV_R17_E260.out` |
| 7 | `ADV2_TAILCV_R20_E260` | 2552033 | `logs/phase1_accept_domain_verify_20260701_130328/ADV2_TAILCV_R20_E260.out` |

状态边界：本轮已经完成本地设计、验证、同步和N607提交；这不是训练完成、指标达标或部署成功声明。后续完成后需追加同候选同排指标表，至少包含`best_joint_test_tx`、`final_strict_udu`、`receiver_floor`、`sat_strict_floor`、`p95/p99/min_inter`、`proxy_vac_rate`、`source_episode_overflow`、导出原型文件状态和最终判定。

## 完成后分析更新

更新时间：`2026-07-01 17:18 Asia/Hong_Kong`。

最终状态：14/14候选完成260个epoch，14/14导出`metrics_epoch.csv/jsonl`、`phase2_zid_prototypes.json/pt`，fatal总数0。日志中存在`NAN_SKIPPED_TEST_PLACEHOLDER`和`NAN_AUX_GRAD_TELEMETRY`，但未发现影响最终指标的`NAN_REAL_LOSS`、`NAN_REAL_METRIC`或`NAN_FATAL`。

完整分析报告：`E:\type10-7\automation_reports\CV-SincNet\phase1_accept_domain_verify_20260701_130328\adv2_full_analysis_report.md`。

结构化表格目录：`E:\type10-7\automation_reports\CV-SincNet\phase1_accept_domain_verify_20260701_130328\adv2_analysis_outputs\`。

复核脚本：`E:\type10-7\automation_reports\CV-SincNet\phase1_accept_domain_verify_20260701_130328\adv2_analysis_outputs\generate_adv2_analysis.py`。

核心结论：

| 项目 | 结论 |
| --- | --- |
| 协议边界 | 本轮仍是Phase1 source-only地面训练，不含真实`Y_unknown` query，不能声明`unknown_FAR`、`FPR95`、真实unknown AUROC或Stage2-C成功。 |
| 闭集均值 | ADV2`final_overall_tx`均值86.92%，相对lateopt变化+0.48pp，相对vacuum32变化-0.31pp；`final_strict_udu`均值80.09%，相对lateopt变化-0.15pp，相对vacuum32变化-0.62pp。 |
| proxy失败面 | `final_proxy_vaccept`均值0.9995，virtual unknown几乎仍被接收，不能作为unknown拒识成功证据。 |
| 几何风险 | `final_p95`均值52.20deg略低于vacuum32，但`final_p99`均值75.78deg、source overflow均值0.3442，说明极端尾部和跨域越界仍是主要风险。 |
| fusion审计 | 14/14原型JSON包含`fusion_components`、`fused_tx_prototypes`和`fusion_config`，`global_ball_accept=False`、`tail_auto_accept=False`；这证明fusion字段已导出，但不证明local component gate已部署成功。 |
| 主推进候选 | `ADV2_SRCLOW_R17_E260`、`ADV2_FUSE5_R20_E260`、`ADV2_R28_FUSE6_E260`。 |
| 下一步 | 优先做不重训local component hard gate dry-run、shell/inter/bridge negative评估和真实Stage2-A/C unknown评估，再决定是否重训negative-space filling。 |

Git发布树记录：`E:\type10-7\github_publish\CVS-RFFI-repo\automation_reports\CV-SincNet\phase1_accept_domain_verify_20260701_130328\adv2_full_analysis_report.md`，提交`0669fab Add ADV2 analysis report`。
