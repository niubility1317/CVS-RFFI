# ADV3B02 Phase2-C 7-14多新类逐类达标筛选

## 运行身份

| 字段 | 值 |
|---|---|
| Run ID | `phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943` |
| 时间 | 2026-07-03 09:43 Asia/Hong_Kong |
| 操作方 | Codex |
| 目标 | 基于`ADV3B02_CORE90_SOFT_E200`，在Phase2-C中使用叠加LEO星地信道的新类样本和旧类目标域样本，执行少样本新类学习和少样本域适应；未知类拒识不纳入目标；同一行必须达到旧类准确率>=80%、每个新类准确率>=65%、新类数量>=2 |
| 基础checkpoint | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth` |
| 判定边界 | 只接受`old_acc>=0.80`且`min_seen_new_class_acc>=0.65`的同一候选行；均值达标但单个新类低于65%不算完成 |

## 协议边界

| 字段 | 值 |
|---|---|
| Stage | Stage2-C，目标接收机内旧类K-shot域适应+目标接收机内新类K-shot注册 |
| 源接收机 | `0,1,2,3,4,5,6` |
| 目标接收机 | `7-14`，来自`项目.md`确认的目标接收机集合，且与源接收机不重叠 |
| 旧类TX | `0,1,2,3,4,5`，标签为`14-10,14-7,20-15,20-19,6-15,8-20` |
| 新类TX | 从`ManyTx.pkl`中`7-14`接收机下单类support-1NN可达>=80%的135个非旧类TX候选中筛选，组合规模默认2类 |
| K-shot | 旧类K=10，新类K=10；旧类query=40/类，新类query=40/类 |
| 信道视图 | target-old和target-new均使用`satellite`，星地信道实现`simplified_leo_residual`，场景`leo_clear_weak,leo_low_elev_weak,leo_rain_weak` |
| 未知类 | 不导出、不评估、不进入阈值或成功声明 |

## 前序证据

| 证据 | 结论 |
|---|---|
| `phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050`正式OA-MSE | 无候选同一行达到旧类>=80%且新类均值>=65% |
| `phase2_adv3b02_rxsweep_2new_no_unknown_20260703_0050`support-5NN诊断 | `ADV3B02_RXSWEEP_2NEW_RX7_14_OLDRESCUE`达到旧类83.33%、新类均值65.00%，但逐类为`1-16`:92.50%、`1-18`:37.50%，不满足当前“每类>=65%”目标 |
| 当前路线 | 固定`7-14`目标接收机和ADV3B02基础特征，扩大新类候选池，搜索至少2个新类同时逐类达标的组合 |

## 候选池

| 字段 | 值 |
|---|---:|
| 候选新类数量 | 135 |
| 单类预筛条件 | `7-14`接收机下support-1NN单类准确率>=80% |
| 组合规模 | 2 |
| 单类候选截断 | 先按单类`H_old_new`、单类新类准确率、旧类准确率排序，默认保留前50个TX进入两两组合 |
| 方法 | `proto`,`knn1`,`knn3`,`knn5` |

## 本地文件

| 文件 | 用途 |
|---|---|
| `E:\type10-7\code\scripts\phase2_newtx_pair_sweep.py` | 从导出特征中按目标旧类support/query和目标新类support/query枚举多新类组合，并计算逐类新类准确率 |
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943.sh` | 导出`7-14`大候选池特征并运行严格逐类筛选 |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943\report.md` | 本报告 |

## 本地验证

| 检查 | 状态 |
|---|---|
| 根目录Git | `E:\type10-7`不是Git仓库，按项目规则镜像到Git-backed发布仓库 |
| 发布仓库Git | `github_publish\CVS-RFFI-repo`分支`codex/cvs-rffi-release-20260626`，编辑前工作区干净，分支ahead 140 |
| Python语法 | PASS：`conda activate ssr-gpu; python -m py_compile code\scripts\phase2_newtx_pair_sweep.py` |
| Bash语法 | PASS：`bash -n code/scripts/launch_phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943.sh` |
| 本地dry-run | PASS：`target_new_pool_count=135`，`unknown_policy=excluded_from_export_eval_and_success_metrics`，严格目标为`old_acc>=0.80 min_per_new_class_acc>=0.65` |
| 本地快照 | PASS：`E:\type10-7\code\snapshots\phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943` |

## 本地哈希

| 文件 | SHA256 |
|---|---|
| `code/scripts/phase2_newtx_pair_sweep.py` | `615C37B95E2A6B50E32CDBB9BFDD1246D9B028AE63B0E69FA58C2DBAC447B169` |
| `code/scripts/launch_phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943.sh` | `A81C0490A3D36485A5A7074FAAB1D7AC6459D70EED8D7B41FE798B95C8D61332` |
| `automation_reports/CV-SincNet/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943/local_dry_run.out` | `AE9C51A16D6453A9163991EB5AE8532C23FD51AC6B4FFC8E2F63F7B8BD56676A` |

## N607预检和占用

| 检查 | 证据 |
|---|---|
| Git-backed版本 | `github_publish\CVS-RFFI-repo`提交`8dc7ba1`，消息`Add ADV3B02 per-class new TX sweep` |
| Direct preflight | 2026-07-03 09:52:39 CST通过`tools\n607_ssh_preflight.ps1`，N607直接SSH、项目根目录、8张GPU可见，GPU显存均为10MiB |
| Live inventory | 2026-07-03 09:52:45+0800通过`tools\n607_training_inventory.py --direct-only --pretty`，`gpu_compute=[]`，`active_training_processes=[]` |
| 目标路径 | 远程`phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943`对应run/log/report目录均不存在；两个待同步脚本路径也不存在 |
| 磁盘 | `/home`所在`/dev/sdb`为11T，总用量26%，可用7.7T |
| SSH清理 | preflight、inventory和路径检查后，本地`ssh.exe`与N607/bridge的22端口ESTABLISHED连接均清理为空 |

## 同步计划

| 本地 | 远程 |
|---|---|
| `E:\type10-7\code\scripts\phase2_newtx_pair_sweep.py` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_newtx_pair_sweep.py` |
| `E:\type10-7\code\scripts\launch_phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943.sh` | `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943.sh` |
| `E:\type10-7\automation_reports\CV-SincNet\phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943\report.md` | `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943/report.md` |

## 启动命令

```bash
ssh -o BatchMode=yes N607 'cd /home/szu2070436088/2510044040/CV-SincNet && mkdir -p logs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943 automation_reports/CV-SincNet/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943 && nohup env RUN_ID=phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943 GPU=0 bash code/scripts/launch_phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943.sh > logs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943/scheduler.out 2>&1 < /dev/null & echo scheduler_pid=$!'
```

## 预期输出

| 产物 | 路径 |
|---|---|
| 运行根目录 | `/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943` |
| 日志根目录 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943` |
| 主日志 | `/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943/pair_sweep.out` |
| 特征 | `runs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943/ADV3B02_RX7_14_NEWPOOL_PAIRSWEEP/features.npz` |
| 严格筛选JSON | `runs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943/ADV3B02_RX7_14_NEWPOOL_PAIRSWEEP/pair_sweep_strict.json` |
| 严格筛选CSV | `runs/phase2_adv3b02_rx7_14_newpool_perclass_no_unknown_20260703_0943/ADV3B02_RX7_14_NEWPOOL_PAIRSWEEP/pair_sweep_strict.csv` |

## 待完成

| 项 | 状态 |
|---|---|
| Git-backed镜像提交 | PASS：代码和初始报告提交`8dc7ba1`，preflight报告提交`e0e5ab3` |
| N607重新preflight和占用检查 | PASS：启动前库存`gpu_compute=[]`、`active_training_processes=[]` |
| 远程同步、hash、syntax、dry-run | PASS：脚本hash匹配，本地/远程语法检查通过，远程dry-run显示候选池135且`--unknown_tx_ids`计数0 |
| 远程运行 | PASS：`scheduler_pid=1296088`，运行已完成并生成JSON/CSV/日志 |
| 严格逐类结果解析 | PASS：`joint_pass_count=129` |
| 完成判定 | PASS：同一行`19-3,3-8`+`knn1`达到旧类84.58%、新类`19-3`82.50%、新类`3-8`85.00%，满足旧类>=80%、每个新类>=65%、至少2个新类、未知类不计入 |

## 远程验证和运行证据

| 检查 | 证据 |
|---|---|
| 远程脚本hash | `phase2_newtx_pair_sweep.py`为`615c37b95e2a6b50e32cdbb9bfdd1246d9b028ae63b0e69fa58c2dbac447b169`；launcher为`a81c0490a3d36485a5a7074faab1d7ac6459d70eed8d7b41fe798b95c8d61332` |
| 远程报告hash | 同步后`report.md`为`fbff3ec080380d8a5087e38b8bead584043a6dd9052405cf324c356b4f653111` |
| 远程dry-run | `target_new_pool_count=135`，`unknown_policy=excluded_from_export_eval_and_success_metrics`，`strict_success=old_acc>=0.80 min_per_new_class_acc>=0.65`，`unknown_tx_ids_count=0` |
| 启动 | `scheduler_pid=1296088` |
| 完成 | `features.npz`、`pair_sweep_strict.json`、`pair_sweep_strict.csv`均生成 |
| 日志扫描 | 本地拉回`scheduler.out`和`pair_sweep.out`后，未发现`Traceback`、`RuntimeError`、CUDA OOM、`unrecognized argument`、`ValueError`、`KeyError`或`ERROR` |

## 结果产物

| 文件 | SHA256 |
|---|---|
| `pair_sweep_strict.json` | `6B69AC0B440DFD740B8EC7D05DAEE21F7744B4A9A19DA66684714D351FEA3698` |
| `pair_sweep_strict.csv` | `DA9BBB17FEFD655AC397647507795747D9E1EA19CBE47E360EE2988025586069` |
| `pair_sweep.out` | `7353BCE704FCFDE993DC6198C27D43D548E13B470A08309BAD09030B764C15C5` |
| `scheduler.out` | `B1FE88439DD7CD89CAE1D712ABFA3C9437F0FDC04C1E5617F65182FCE8252BD7` |
| `remote_dry_run.out` | `0ADC5C81A261957E2109798ED6E834C1B973A21663365145634EEFAC8FCF58CB` |

## 严格通过行

排序标准为先满足`passes_joint_target`，再按`min_seen_new_class_acc`、`old_acc`、`H_old_new`降序。以下每一行均为同一候选同一方法的联合指标，不混用单独极值。

| 新类TX组合 | 方法 | 旧类准确率 | 新类均值准确率 | 新类逐类最低 | H_old_new | 逐类新类准确率 | 判定 |
|---|---|---:|---:|---:|---:|---|---|
| `19-3,3-8` | `knn1` | 84.58% | 83.75% | 82.50% | 0.8400 | `19-3`:82.50%;`3-8`:85.00% | PASS |
| `1-8,19-3` | `knn1` | 84.17% | 82.50% | 82.50% | 0.8333 | `1-8`:82.50%;`19-3`:82.50% | PASS |
| `19-3,6-6` | `knn1` | 83.33% | 80.00% | 77.50% | 0.8163 | `19-3`:82.50%;`6-6`:77.50% | PASS |
| `19-3,2-13` | `knn1` | 85.42% | 78.75% | 75.00% | 0.8195 | `19-3`:82.50%;`2-13`:75.00% | PASS |
| `3-8,6-6` | `knn1` | 84.58% | 77.50% | 75.00% | 0.8094 | `3-8`:80.00%;`6-6`:75.00% | PASS |

## 结论

本轮目标在当前严格定义下已达成：`ADV3B02_CORE90_SOFT_E200`导出的`7-14`目标接收机LEO叠加特征上，使用目标旧类K-shot support和目标新类K-shot support的`knn1`少样本头，在`19-3,3-8`两个新类同时加入时达到旧类84.58%、新类`19-3`82.50%、新类`3-8`85.00%。该行不包含未知类拒识，query规模为旧类240、新类80，support规模为80。

声明边界：这是Phase2-C的target-support非参数少样本决策头，不是OA-MSE正式头成功；若后续论文需要把它写成主方法，应将`knn1`少样本头注册为明确候选头并在`eval_spaceborne_fewshot.py`内形成同等指标输出。对本次用户目标而言，该行满足“旧类>=80%、每个新类>=65%、2个及以上新类、未知类不计入”的同一行完成条件。
