# D26 compact-diag高维拼接support-only实验

## 启动前记录

- experiment ID：`d26_compact_diag_20260718/support_screen_v1`。
- 日期：2026-07-18；operator：Codex；状态：`DEVELOPMENT_SUPPORT_ONLY_COMPLETE_NEGATIVE`。
- 目标：压缩D25 v4中性能最强但60 optimizer steps超限的B3诊断结构，在单IQ 288D拼接下用≤30步完成Stage2-B适配和Stage2-C新类注册，并显式保护旧类floor。
- 假设：B3相对C3的主要收益来自逐类可学习cosine head；把mini-batch 40+20步压成全批次15+(0/10/15)步，并增加support-only new-group bias，可保留B3大部分old/new收益，同时把峰值活动参数降到2,016、总步数最多30。
- 比较目标：Z0、B3、C0、D26-A `15+0`、D26-B `15+10`、D26-C `15+15`。
- 矩阵：6候选×3个LEO_weak场景×5个held-rank fold=90行；receiver `20-1`、开发seed `713101`、K=10，每fold每类fit8/held2，旧6类+seen-new5类。

## 本轮不再做数据准备

- 直接复用D25/C3已验证的sealed enrollment-only LEO_weak support，不新增、不重新叠加、不派生任何星地信道样本。
- 每个物理IQ仍只有一个LEO_weak观测；`z160/FFT96/RF32`只是同一接收IQ的一条288D拼接行，`support_view_count=1`、`support_row_multiplicity=1`。
- query/test不打开，不参与适配、bias选择、回滚或ranking。

## 方法与选择锁

- Stage2-B：shared 288维对角+6个逐类288维weight，全批次15步；不更新backbone。
- Stage2-C：只更新5个new suffix weight，分别0/10/15步；旧weight、shared diagonal和old raw score列冻结。
- new-group bias只从预锁定`[-2,-1,-0.5,0,0.5]`由新类support LOO和旧support安全门选择；它作用于注册类身份，不读取query角色。K=1强制0且不伪造LOO。
- fold与full-K10都比较注册前后旧support逐类准确率和floor；任何退化不得晋级。
- 每场景pooled old/new floor相对C0均至少提升10pp、任一类下降不超过10pp、H不低于C0、forgetting不高于C0；B3仅为性能参考。

## 本地版本与验证

- Git仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`；根目录不是Git仓库，本报告在根目录与Git承载面保持镜像。
- D26核心提交：`0a9fbb20`；runner提交：`e4681cee`。
- runner SHA256：`4b664deff293571e44a86c3157f918146adda42e35167adb4d9836f2cbffcb52`。
- D26核心SHA256：`c03d2990a88b1526d40728fa616e4e4e6a43bc42c3e67e24388687aee35d6899`。
- launcher SHA256：`d49e7626a90b0c2b068f83651d5760033365c8b2cd60401769822f3b6434a2e3`。
- 55项D26+D25/C3相邻回归PASS；`py_compile`、`bash -n`、`git diff --check`PASS。
- 独立review未发现协议或算法高严重度阻断；已修复D26实现Git归因和FFT96/RF32 operator闭包遗漏。

## N607计划

- 已有2026-07-18 01:16 CST直连preflight PASS；正式启动前重新读取live process/GPU inventory。
- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 计划GPU：优先空闲GPU0，但不超过每GPU两个训练任务。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d26_compact_diag_20260718/support_screen_v1.log`。
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d26_compact_diag_20260718/output/support_screen_v1`。
- 精确命令：`cd /home/szu2070436088/2510044040/CV-SincNet && D26_GPU=0 bash code/scripts/launch_d26_compact_diag_support_20260718.sh`。
- 只同步runner、D26核心和launcher；不会覆盖远端现有`stage2_diag_cosine_exploration.py`。launcher与runtime candidate lock会校验并记录远端实际operator SHA=`14ec919395f9bf9f13214c677b1a3d640764214668d1d00e9109f5b149ec41ca`。
- 03:17 CST直连preflight再次PASS；8张RTX3090均0%利用、约10MiB显存。live inventory：`gpu_compute=[]`、`active_training_processes=[]`、`unknown_training_active=false`，允许使用GPU0。
- 已同步映射：runner→`code/scripts/run_d25_support_only_concat.py`；D26核心→`code/cvsrffi/stage2_multimodal_compact_diag.py`；launcher→`code/scripts/launch_d26_compact_diag_support_20260718.sh`。SCP结束后本地SSH/TCP22连接为0。
- 远端SHA、`py_compile`、launcher `bash -n`和output不存在门均PASS。03:18 CST启动PID=`3616036`，GPU0；启动后本地SSH/TCP22连接为0。

## 预期产物与判定

- 六件固定artifact：`training_log.jsonl`、`support_audit.json`、`selection.json`、`resource_audit.json`、`geometry_audit.json`、`RECEIPT.json`。
- 完成后报告候选联合行、逐场景、逐类old/new floor、bias选择分布、注册前后旧support、完整loss、参数/step/状态/MAC/时延/显存和相对qKNN Pareto。
- 本轮仍是开发support-only筛选，不是正式query性能；只有正向路线才进入joint bundle method lock和正式5receiver×确认seed×3scene矩阵。

## v1完成状态

- PID`3616036`正常退出，耗时16.507秒；`query_opened=false`，90/90行齐全，stdout与结构化日志无异常或非有限数值。
- 最终与pre-full-K10选择均为C0，`selected_positive_route=false`；不是full-K10异常回退，而是三条D26在fold初筛已失败。
- artifact哈希：training=`1e261ecd1acd1af48ee48c9c4bcf38eb99aea3313aae62469b4bbf9a0da6a8f7`、selection=`195b2b4584d4f5a3a797308ba9a686831dfc2848fc791affe972d4fb452e4212`、support=`d369a3ec633733f79f9ff09f1ab75046d34cdad25e7624b60020389dabe3beec`、resource=`115dd4541313bb5b0d5994e395a19012535721b8644d89e0905b552e7119411b`、geometry=`3820ae423839f6e3608dad858cf06ae5b16274498fb144cd0b32048d604527a2`、receipt=`16380d63fe453efd2d72e5985a25449dbca8884193d1eb58214dfdc124c6a346`。
- 本地完整产物：`E:\type10-7\automation_reports\CV-SincNet\d26_compact_diag_support_20260718\remote_output_v1`。

## 候选联合结果

| 候选 | 注册前old | 注册后old | seen-new | H | forgetting | fit-old非退化 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 71.67% | 50.56% | 54.00% | 50.35% | 21.11pp | N/A |
| B3诊断 | 86.67% | 73.33% | 73.33% | 72.65% | 13.33pp | N/A |
| D26-A 15+0 | 80.00% | 9.44% | 69.33% | 15.00% | 70.56pp | 0/15 |
| D26-B 15+10 | 80.00% | 23.89% | 70.67% | 33.16% | 56.11pp | 0/15 |
| D26-C 15+15 | 80.00% | 20.56% | 74.00% | 30.32% | 59.44pp | 0/15 |

核心判断：class-specific head本身有效，注册前old比C0提高8.33pp；失败集中发生在新类注册后的全类竞争。三条路线的bias在15/15fold都选择0，fit-old注册前为100%，注册后仅16.81%/41.53%/35.69%。当前bias门只保证“不劣于注册后bias=0”，没有保护Stage2-B old-only基线。

## 逐场景与逐类摘要

| 候选 | 场景 | 注册前old | 注册后old | new | H | pooled old/new floor |
|---|---|---:|---:|---:|---:|---:|
| D26-B | clear | 81.67% | 31.67% | 68.00% | 42.79% | 10%/30% |
| D26-B | low-elev | 75.00% | 10.00% | 76.00% | 15.44% | 0%/30% |
| D26-B | rain | 83.33% | 30.00% | 68.00% | 41.26% | 0%/40% |
| D26-C | clear | 81.67% | 26.67% | 72.00% | 38.11% | 10%/40% |
| D26-C | low-elev | 75.00% | 11.67% | 78.00% | 18.94% | 0%/20% |
| D26-C | rain | 83.33% | 23.33% | 72.00% | 33.92% | 0%/50% |

D26-B注册前逐类old为`14-10=70.00%`、`14-7=80.00%`、`20-15=96.67%`、`20-19=56.67%`、`6-15=86.67%`、`8-20=90.00%`；注册后分别降到6.67%、20.00%、46.67%、13.33%、6.67%、50.00%。seen-new逐类为`cls_09f8=36.67%`、`cls_1c2a=96.67%`、`cls_b8fb=86.67%`、`cls_d3af=86.67%`、`cls_f608=46.67%`，已经明显好于C3，但以旧类崩塌为代价。

## loss、bias与资源

- Stage2-B平均loss由0.5548降到0.0664，fit support准确率达到100%；Stage2-C 10/15步把loss降到0.2570/0.2229，新support准确率达到96.0%/96.67%。这进一步证明问题是注册score尺度，而非训练不收敛。
- 离线读取预锁定候选证据：D26-B在bias=-2时fit-old总体/平均floor可从bias0的41.53%/9.17%恢复到77.40%/44.17%，new LOO总体仍有78.8%；负bias方向有效，但当前网格与安全基线不足。

| 候选 | 峰值参数 | epoch/step | 状态 | MAC/query | 适配+注册 | CPU FP32 head |
|---|---:|---:|---:|---:|---:|---:|
| D26-A | 2,016 | 15/15 | 23,956B | 3,456 | 74.82ms | 0.0879ms |
| D26-B | 2,016 | 25/25 | 23,988B | 3,456 | 96.87ms | 0.0873ms |
| D26-C | 2,016 | 30/30 | 23,982B | 3,456 | 107.80ms | 0.0865ms |
| C0 | 0 | 0/0 | 17,616B | 3,456 | 约16ms | 0.1240ms |
| B3 | 3,456 | 20epoch/60step | 14,618B | 3,456 | 诊断 | N/A |

D26比B3减少41.67%峰值活动参数和50%–75% optimizer steps，head时延也低于C0；资源目标已达成，但性能门失败，不晋级。

## v2直接修复决定

不增加数据、不改288D拼接、不扩大模型。D26-v2只修改1标量bias选择：网格扩为更负区间，安全基线从“注册后bias0”改为“Stage2-B注册前old-only”，要求逐旧类准确率不降且保留所有原先正确old support行；K=1不伪造LOO，在旧保护可行bias中选最接近0者。随后复用同一90行support矩阵快速重跑。

## v2启动前记录

- experiment ID：`d26_compact_diag_20260718/support_screen_v2`；状态：`READY_FOR_N607_SUPPORT_SCREEN`。
- 研发变化仅限注册bias保护：候选网格锁为`[-12,-8,-6,-4,-3,-2,-1,0]`，逐旧类准确率与所有Stage2-B old-only正确support行都不得退化；K>1在安全bias中按新类support LOO选择，K=1不伪造LOO并选最接近0的安全bias；无安全bias即fail closed。
- 数据、288D拼接、旧类head、Stage2-B/C步数、query不可达边界均不变；直接复用v1同一sealed LEO_weak support和同一90行矩阵。
- D26-v2核心提交：`55d69d0efa3d5ef4d43e9702058d15c20e7f95e5`；runner提交：`bbaf5958`。
- runner SHA256：`5adfd865bde4f896edec848adda36d6a98f5de8ad4c70fc35c6780dc51031065`；核心SHA256：`9d7d1ef87fed216aa35c5dee056a067da884f95af3b233d96d05eab9d6da34d6`；launcher SHA256：`0ac594f95b3441ce840cf8092806e22a725ebe2535d54c4bcd6782f428a36ce6`。
- 本地验证：58项D26/D25/C3相邻回归PASS，`py_compile`、`bash -n`、`git diff --check`PASS。
- 远端output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d26_compact_diag_20260718/output/support_screen_v2`；log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d26_compact_diag_20260718/support_screen_v2.log`。
- 精确命令：`cd /home/szu2070436088/2510044040/CV-SincNet && D26_GPU=0 bash code/scripts/launch_d26_v2_strict_bias_support_20260718.sh`。
- 03:32 CST直连preflight再次PASS；8张RTX3090均0%利用、约10MiB显存。live inventory为`gpu_compute=[]`、`active_training_processes=[]`、`unknown_training_active=false`，允许使用GPU0。
- 已同步runner、D26-v2核心和v2 launcher；远端SHA、`py_compile`、`bash -n`、实际FFT96/RF32 operator SHA和output不存在门均PASS。同步及验证后本地`ssh.exe`与N607 TCP22连接均为0。
