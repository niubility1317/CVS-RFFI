# RIEI Table III论文一致性修复与优化报告

- 实验ID：`paper_repro_riei_parity_repair_20260714_145800`
- 目标：修复RIEI期刊Table III的预处理、优化器和评估窗口偏差，先做Table III第1行8候选受控消融，再以固定配置确认完整12行。
- 论文目标：12行均值`73.30%`；第1行`77.88±2.23%`。

## 已确认问题

| 问题 | 修复前 | 当前修复 |
|---|---|---|
| 信号预处理 | `riei_original`硬编码逐包RMS归一化 | 严格候选仅信道均衡并关闭RMS；保留RMS control |
| 优化器 | 固定Adam，FED同一批次连续两个Adam step | 增加无momentum SGD及Adam消融，保持Eq.20–21交替顺序 |
| 评价窗口 | Table III使用last10 | 2025期刊版默认last5；旧会议版last10只作历史口径 |

当前fixopt前8行均值`60.34%`，论文前8行均值`73.37%`，平均差`-13.03pp`；source validation接近`100%`而target receiver偏低，主因是跨接收机泛化，不是NaN/OOM。

## 本地验证

- 修改：`baselines/common/cvs_data.py`、`baselines/riei_fd/train_cvs.py`、`run_wisig_paper_scope_queue.sh`、`code/scripts/launch_riei_parity_repair_matrix_20260714.sh`、`tests/test_riei_parity_repair.py`。
- 根目录聚焦测试`15 passed`；Git镜像聚焦测试`3 passed`。
- `py_compile`、`bash -n`和8-job dry-run通过。
- 发现矩阵固定第1行、seed1337、200epoch、last5；目标域间隔曲线只作诊断，禁止target-oracle选epoch。
- 当前N607 fixopt仍有4个RIEI训练；未同步、未启动、未影响Phase1。后续先完成DRIFT v2，再在容量门通过后启动本RIEI矩阵。

## 2026-07-14 17:20启动前门控

- DRIFT五seed确认已5/5完成并退出；正式final均值`72.75±5.93%`，与论文73.54%差-0.79pp。Git提交`be02078`已将mean/no-cap配置固定为唯一支持的DRIFT论文复现入口。
- 本地重新验证：`ssr-gpu`下`py_compile`通过；RIEI launcher与canonical DRIFT launcher的`bash -n`通过；RIEI 8-job dry-run完整展开P01–P08；聚焦测试根目录15 passed、Git镜像3 passed。
- 待同步文件SHA256：`cvs_data.py=a2093e0a...`、`riei_fd/train_cvs.py=950b6008...`、paper queue=`2ba90874...`、RIEI launcher=`e2a87932...`。
- 直接N607预检通过；实时GPU compute仅GPU3有1个既有Phase1进程，GPU0–2、4–7为空。RIEI矩阵每GPU新增1个，峰值GPU3=2、其余=1，满足每GPU不超过2。
- 计划正式命令：`bash code/scripts/launch_riei_parity_repair_matrix_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。独立run/log根为`paper_repro_riei_parity_repair_20260714_145800`，不会覆盖既有产物。

## 2026-07-14 17:37正式启动与5分钟健康检查

- 远程工作目录：`/home/szu2070436088/2510044040/CV-SincNet`；环境：`CVS-RFFI`。
- 正式命令：`bash code/scripts/launch_riei_parity_repair_matrix_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。启动前实时容量为每GPU 0个compute，计划每GPU 1个，峰值1≤2。
- 同步与校验：`cvs_data.py=a2093e0a...`、`riei_fd/train_cvs.py=950b6008...`、paper queue=`2ba90874...`、RIEI launcher=`e2a87932...`；本地/远程SHA256一致，远程`bash -n`和8-job dry-run通过。
- launcher PID：P01=`481632`、P02=`481635`、P03=`481640`、P04=`481646`、P05=`481654`、P06=`481663`、P07=`481676`、P08=`481692`。训练PID：`481774,481799,481813,481829,481840,481858,481862,481863`，GPU0–7各1个。
- run根：`paper_reproduction/runs/paper_repro_riei_parity_repair_20260714_145800`；log根：`paper_reproduction/logs/paper_repro_riei_parity_repair_20260714_145800`。预期每候选产生200epoch训练日志、`metrics_epoch.csv`、`riei_last5`汇总和checkpoint。
- 17:42健康检查：8/8 launcher与8/8 trainer均运行；P01–P08分别进入epoch `14,15,14,14,14,14,13,13/200`；8块GPU各1个本任务compute；硬错误0，未见Traceback、RuntimeError、OOM、Killed、NaN或Inf；本地SSH连接已完全退出。
- 当前判定：`RUNNING_HEALTHY`。中间target曲线仅作诊断；正式选型仍固定为epoch196–200的last5，禁止按target peak选epoch。完成后将对8份完整200epoch日志做同row联合比较，再固定胜出配置执行Table III完整12行确认。

## 2026-07-14 17:51心跳监控

- 直连N607只读预检通过：身份、项目根目录、服务器时间与8块GPU可见。
- 8/8 launcher与8/8 trainer仍在运行；训练PID与启动记录一致。P01–P08最新完整日志进度为epoch `39,39,40,40,39,39,39,37/200`，对应日志行数`171,247,150,158,155,159,175,141`。
- GPU0–7各有且仅有1个本任务compute，显存约`470–500MiB`，GPU利用率`7%–24%`，每GPU训练数未超过2。
- 完整扫描当前已写入日志，未见Traceback、RuntimeError、OOM、Killed、NaN、Inf或未识别参数；硬错误计数0。`metrics_epoch.csv=0/8`、`PAPER-EVAL-SUMMARY=0/8`，符合产物在训练完成时落盘的当前阶段。
- 证据边界：本次只能判定为`RUNNING_HEALTHY_THROUGH_LATEST_PARSED_EPOCH`，不是完整训练分析或结果达标。全部8个候选完成后再读取8×200epoch完整日志和正式last5产物。
- 本次SSH命令完成后，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`；未干预、重启、覆盖或删除任何远程状态。

## 2026-07-14 18:21心跳监控

- 直连N607只读预检再次通过；8/8 launcher和8/8 trainer仍与启动PID一致。
- P01–P08最新完整日志进度为epoch `124,122,127,127,122,126,121,114/200`；日志行数分别为`462,628,451,455,440,456,453,404`，文件大小为`30.3–41.2KiB`。P08稍慢但持续前进，不构成stale log。
- GPU0–7各仅1个本任务compute，显存`470–500MiB`，GPU利用率`10%–21%`；容量门仍满足。
- 完整扫描截至18:21已写入的所有训练与launcher日志，硬错误计数0；未见Traceback、RuntimeError、OOM、Killed、NaN、Inf或参数错误。`metrics_epoch.csv=0/8`、`PAPER-EVAL-SUMMARY=0/8`，仍属训练中正常状态。
- 当前证据只覆盖最新epoch114–127，判定为`RUNNING_HEALTHY_THROUGH_LATEST_PARSED_EPOCH`，不构成last5结果或论文复现结论。
- 短连接已退出：本地`ssh.exe=0`、N607 TCP22已建立连接`=0`；未修改任何N607文件、进程或产物。

## 2026-07-14 18:57发现矩阵完成与全日志分析

- 完成性：8/8 trainer和8/8 launcher自然退出，8/8`PAPER-EVAL-SUMMARY`和8/8`FINAL-TEST`完整，硬错误0。实际结构化产物名为`metrics.json`，非运行中监控预期的`metrics_epoch.csv`；8份`metrics.json`均包含严格epoch1–200。
- 证据集：本地`analysis_tmp/paper_repro_riei_parity_repair_20260714_145800/final_1857`中仅拉取了8份metrics、8份manifest、2份调度TSV和24份日志；未拉取checkpoint。完整解析8×200=1600个epoch、24份日志共6100行/476599字节，未见Traceback、RuntimeError、OOM、Killed、NaN、Inf或argparse错误。

|ID|优化器|reduction|RMS|feature norm|last5|final|best-val target诊断|last5 CE/MI/IE/FN|较论文77.88%|
|---|---|---|---:|---:|---:|---:|---:|---|---:|
|P01|SGD|sum|0|0|74.52±0.59%|74.62%|80.42%@29|0.0048/0.0056/159.012/5.804|-3.36pp|
|P02|SGD|mean|0|0|**80.12±0.58%**|79.02%|79.88%@193|0.0125/0.0117/2.480/5.205|**+2.24pp**|
|P03|Adam|sum|0|0|65.84±0.50%|65.44%|63.71%@58|0.1189/0.0060/158.045/0.691|-12.04pp|
|P04|Adam|mean|0|0|61.79±1.40%|59.15%|65.38%@22|0.0029/0.0001/2.457/0.672|-16.09pp|
|P05|Adam|sum|0|`1e-4`|63.76±0.72%|64.15%|65.02%@128|0.1399/0.0093/157.777/0.801|-14.12pp|
|P06|SGD|sum|1|0|74.43±1.00%|73.46%|77.62%@21|0.0036/0.0050/159.013/5.418|-3.45pp|
|P07|SGD|sum|0|`1e-4`|72.79±1.48%|73.81%|79.54%@33|0.0054/0.0059/159.011/5.751|-5.09pp|
|P08|Adam|sum|1|`1e-4`|63.75±2.02%|64.00%|57.52%@17|0.1105/0.0458/157.313/0.710|-14.13pp|

### 选型结论

- 胜出配置为P02：无momentum SGD、CE/MI/IE全部mean reduction、关闭逐包RMS、关闭feature-norm guard。正式last5=`80.12±0.58%`，较论文`77.88±2.23%`高`2.24pp`，约为论文`+1.00SD`，进入论文均值`±2SD`区间；last5自身波动仅`0.58pp`。
- 动力学证据：P02的source validation从epoch1的`32.71%`逐步升至epoch42首次达`99%`，target诊断在epoch20首次超过`77.88%`；训练后段仍依照预先固定的epoch196–200计分，未使用target-oracle选epoch。
- 机制归因：同为SGD时，mean比sum提高`5.60pp`；同为sum/no-RMS时，SGD比Adam提高`8.68pp`。RMS对SGD-sum仅影响`-0.09pp`，feature-norm对SGD-sum影响`-1.73pp`。因此主要修复是优化器与loss尺度的联合语义，不是RMS或feature-norm。
- 声明边界：论文未公开优化器名称和所有reduction细节；P02是受Eq.20–21显式梯度下降式与同row消融支持的最接近实现，不将未公开细节写成已证实事实。
- 下一步：本地实现`code/scripts/launch_riei_table3_confirm_sgd_mean_20260714.sh`，固定P02配置并运行Table III完整12行。预定run ID为`paper_repro_riei_table3_confirm_sgd_mean_seed1337_20260714_190100`；只有12行逐行与整体达到MAE≤3pp且至少10/12进入论文±2SD，才能判定Table III初步复现。

## 2026-07-14 19:01完整12行launcher本地落地

- 新增：`code/scripts/launch_riei_table3_confirm_sgd_mean_20260714.sh`、`tests/test_riei_table3_confirmation.py`，并更新`baselines/README.md`、本报告和`traceability.md`；根目录与Git镜像同步。
- launcher SHA256：`22e844e0eb0eb401fb018752472e9cefd38b28711ec05cdfa227fd32347d3d70`，根目录与Git镜像完全一致。
- `bash -n`通过；12-job dry-run完整覆盖Table III行1–12、8个capacity gate、GPU0–3各2个顺序job、GPU4–7各1个job，所有12条命令均固定SGD+mean+no-RMS+no-FN、200epoch和last5。
- 测试：首次`conda run -n ssr-gpu`再次触发本机已知GBK Unicode包装器错误；按`AGENTS.md`用同一`ssr-gpu`解释器串行重跑，根目录聚焦测试`5 passed`，Git镜像`5 passed`。根目录唯一warning为`.pytest_cache`无写权，不影响测试结果。
- 当前边界：本地已实现并验证，但尚未同步或启动N607；必须先提交Git任务变更，再重新执行实时容量门。

## 2026-07-14 19:29完整Table III确认矩阵启动

- Git版本：`0c47f42 confirm RIEI Table III with SGD mean scaling`。N607同步文件仅为`code/scripts/launch_riei_table3_confirm_sgd_mean_20260714.sh`；本地与远端SHA256均为`22e844e0eb0eb401fb018752472e9cefd38b28711ec05cdfa227fd32347d3d70`。
- 远端依赖复核：`cvs_data.py=a2093e0a...`、`riei_fd/train_cvs.py=950b6008...`、paper queue=`2ba90874...`；远端`bash -n`通过，dry-run计数为12个job、8个capacity gate、12条SGD+mean固定配置。
- 启动前直接预检通过；实时既有compute为GPU1–5、7各1个Phase1，GPU0、6为空。确认矩阵计划每GPU新增1个训练，启动器复核峰值为GPU0、6各1个，GPU1–5、7各2个，全部满足每GPU不超过2。
- 正式命令：`bash code/scripts/launch_riei_table3_confirm_sgd_mean_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。run/log ID为`paper_repro_riei_table3_confirm_sgd_mean_seed1337_20260714_190100`，不存在旧目录且未覆盖任何产物。
- 8个顺序queue PID：GPU0–7依次为`569608,569610,569613,569618,569624,569627,569635,569642`；GPU0–3各排2行，GPU4–7各排1行。首批Table III第1–8行训练PID为`569767,569783,569811,569815,569827,569828,569830,569831`。
- 约4分钟健康检查：8/8 queue与8/8 trainer运行；第1–8行分别进入epoch`11,11,10,11,11,11,10,11/200`，全部日志持续增长；硬错误0，`PAPER-EVAL-SUMMARY=0`、完成job=0，符合训练早期状态。8份`metrics.json`已创建但尚未完成，不能据此给出结果。
- GPU occupancy：GPU0、6各1个本任务compute；GPU1–5、7各1个本任务加1个Phase1，总数均未超过2。既有Phase1 PID`549385,549925,552673,551328,550859,551794`均继续运行，未被干预。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_10_11`。正式结论仍固定epoch196–200 last5；12行全部完成后计算逐行差值、MAE和论文`±2SD`命中数。短连接退出后本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-14 19:51心跳监控

- 直连N607只读预检通过，服务器时间、项目根和8块GPU均可见。8/8顺序queue仍为原PID并已运行约22分55秒；首批Table III第1–8行的8/8 trainer仍为原PID，未发生重启或替换。
- 对当前run全部8份训练日志读取至最新完整记录：第1–8行分别到epoch`65,66,65,63,65,65,66,65/200`；日志行数`397,352,385,391,397,389,372,397`，文件大小`24.3–26.4KiB`，所有日志均持续推进。
- 完整扫描当前run已写入的训练及queue日志，硬错误0；未见Traceback、RuntimeError、OOM、Killed、NaN、Inf或参数错误。当前`metrics.json=8`、`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、完成job=0，属于首批训练中状态，不能作为正式结果。
- 19:51预检GPU证据：GPU0、6约`485MiB`且利用率`15%–16%`，对应各1个本任务训练；GPU1–5、7显存约`2604–4122MiB`且利用率`75%–82%`，与各1个本任务加1个Phase1的容量布局一致，未超过每GPU2个训练。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_63_66`。这是截至最新完整epoch的在途证据，不是12行完成、last5结果或论文复现结论。
- SSH收尾审计发现另一个既有任务正在执行`scp ... ManyTx.pkl`：父PID=`46384`、SFTP子PID=`21604`，本地文件已写入约`1.44GB`。该连接不属于本RIEI监控，未被终止或干预；本次监控SSH已退出，并因保护并行任务未继续发起额外N607连接。
