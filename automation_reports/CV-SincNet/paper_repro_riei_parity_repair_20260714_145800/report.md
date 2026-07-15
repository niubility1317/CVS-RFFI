# RIEI Table III论文一致性修复与优化报告

## 基本信息

- 实验ID：`paper_repro_riei_parity_repair_20260714_145800`
- 时间：2026-07-14
- 操作者：Codex
- 目标：修复RIEI期刊Table III的预处理、优化器和评估窗口偏差，先完成单row受控消融，再以固定配置重跑完整12行。
- 对照：RIEI Table III论文12行均值`73.30%`；逐行论文均值/标准差以当前fixopt报告中的表格为准。

## 当前诊断

1. 当前`riei_original`强制逐包RMS归一化；论文数据集专段只明确无信号段去除和信道均衡，因此该额外预处理可能削弱跨接收机幅相指纹。
2. 当前训练器固定Adam。论文Eq.20–21给出交替梯度下降，并强调中间FED更新；Adam在一个mini-batch内对FED连续推进两次内部时间步，属于未披露实现假设。
3. 当前Table III启动器使用last10。2025期刊扩展版明确报告最终5个epoch；旧会议版才写last10。
4. 当前fixopt前8行均值`60.34%`，论文前8行均值`73.37%`，平均差`-13.03pp`；仅第4行进入论文`±2SD`范围。feature-norm guard改善不均匀，未解决主因。
5. 现行日志的source validation接近`100%`，而target receiver显著偏低，表现为跨接收机泛化不足，不是NaN/OOM或训练未收敛。

## 计划矩阵

发现矩阵固定Table III第1行`Rx(1-1),Rx(7-7)→Rx(1-19)`、seed1337、batch64、lr1e-4、`lambda_mi=lambda_ie=1.2`，只改变以下变量：SGD/Adam、sum/mean、RMS开关、feature-norm guard。所有正式候选使用预先固定的200epoch和last5；任何目标域中间峰值只作诊断。

## N607安全边界

- 当前`paper_repro_fixopt_riei_drift_seed1337_20260714_105000`仍有4个RIEI训练，GPU0–3各有1个本任务训练和1个Phase1训练；GPU4–7仅有Phase1。
- 当前queue未完全退出前不修改远端共享入口、不启动DRIFT v2或RIEI新矩阵。
- 后续启动前重新执行直接预检，并验证`existing_compute+planned_peak<=2`。

## 预期产物

- 发现矩阵：8个同row候选完整200epoch日志、`metrics.json`及loss/feature norm轨迹。
- 确认矩阵：胜出配置的Table III完整12行；若达到逐行/整体阈值，再做多seed稳定性确认。
- 所有结论回写本报告和Git镜像报告，并保持DRIFT v2独立论文口径。

## 本地实现与验证

- 根目录`E:\type10-7`不是有效Git仓库；版本化承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`。
- 修改：`baselines/common/cvs_data.py`、`baselines/riei_fd/train_cvs.py`、`run_wisig_paper_scope_queue.sh`、`code/scripts/launch_riei_parity_repair_matrix_20260714.sh`、相关测试。
- `ssr-gpu`环境`py_compile`通过。
- 根目录聚焦测试：`15 passed`；唯一警告为根目录`.pytest_cache`访问权限，不影响项目验证。
- Git镜像聚焦测试：`3 passed`。
- `bash -n`通过；8-job dry-run输出8个候选、8个容量门、last5、SGD/Adam、RMS开关和fixopt control均正确。
- 当前未同步或启动N607新任务；共享入口保持远端现行版本，等待fixopt及随后DRIFT v2按队列顺序完成。

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

## 2026-07-14 20:22心跳监控

- 直连N607只读预检通过；上轮并行`ManyTx.pkl`传输已自然退出，监控开始前本地无残留SSH/SCP连接。8/8顺序queue仍为原PID，运行约53分13秒；首批第1–8行8/8 trainer仍为原PID。
- 完整读取当前8份训练日志至最新记录：第1–8行分别到epoch`153,156,155,150,154,156,153,154/200`；日志行数`745,690,723,764,744,710,697,744`，文件大小`47.0–49.9KiB`，全部持续推进且无stale log。
- 8份`metrics.json`均可完整解析，已写入epoch数分别为`152,155,154,149,153,155,152,153`，各自最后epoch与日志当前epoch仅差1，且均尚无`final`字段；`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、完成job=0，符合首批训练未到epoch200的状态。
- 对当前run全部已写入训练、queue日志做完整硬错误扫描，计数0；未见Traceback、RuntimeError、OOM、Killed、NaN、Inf或参数错误。
- GPU进程证据：GPU0、6各1个本任务trainer；GPU1–5、7各1个本任务trainer加1个既有Phase1 trainer，总训练数分别为`1,2,2,2,2,2,1,2`，未超过容量上限2。Phase1 PID`549385,549925,552673,551328,550859,551794`仍运行，未被干预。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_150_156`。这是截至最新完整epoch的在途分析，不是last5或Table III复现结论。监控SSH完成后本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-14 20:51心跳监控

- 直连N607只读预检通过。Table III第1–8行均自然完成epoch200，8份日志各有1个`PAPER-EVAL-SUMMARY`和1个`FINAL-TEST`；queue日志中的8个`QUEUE-JOB-END status=0`与之对应，没有异常退出。
- GPU4–7的单job queue已自然退出；GPU0–3的queue PID`569608,569610,569613,569618`继续运行第二批。当前4个trainer分别为row9 PID`600349`、row10 PID`599644`、row11 PID`600201`、row12 PID`601314`，进度为epoch`36,39,37,33/200`。
- 12份`metrics.json`均可解析：row1–8各完整200epoch且有`final`字段；row9–12分别完整写入epoch`35,38,36,32`且尚无`final`。当前汇总为`PAPER-EVAL-SUMMARY=8/12`、`FINAL-TEST=8/12`、成功完成job=`8/12`。
- 完整扫描当前run全部已写入训练及queue日志，硬错误0；未见Traceback、RuntimeError、OOM、Killed、NaN、Inf或参数错误。row1–8的正式数值暂不脱离完整12行单独下结论，待全矩阵完成后按同row口径统一计算MAE和论文`±2SD`命中数。
- GPU进程证据：仅GPU0–3各有1个本任务trainer，GPU4–7无compute；此前Phase1训练均已自然退出。当前每GPU训练数为`1,1,1,1,0,0,0,0`，容量合规且未发生干预。
- 当前判定：`RUNNING_HEALTHY_8_OF_12_COMPLETE`。SSH短连接均已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-14 21:21心跳监控

- 直连N607只读预检通过。row1–8保持完整200epoch、`final`字段和8个成功`QUEUE-JOB-END status=0`；GPU0–3的4个queue及row9–12原trainer PID继续运行，无重启或进程替换。
- 完整读取全部12份训练日志至最新记录：row9–12分别到epoch`121,127,123,117/200`；对应日志行数`545,643,587,641`、文件大小`37.3–42.3KiB`，均持续推进。row1–8完整日志保持不变。
- 12份`metrics.json`均可解析：row1–8各200epoch且有`final`；row9–12分别完整写入epoch`120,126,122,116`且无`final`，与日志当前epoch只差1。当前`PAPER-EVAL-SUMMARY=8/12`、`FINAL-TEST=8/12`、成功完成job=`8/12`。
- 对当前run全部已写入训练及queue日志做完整硬错误扫描，计数0；未见Traceback、RuntimeError、OOM、Killed、NaN、Inf或参数错误。
- GPU0–3各仅1个本任务trainer，GPU4–7无compute；当前每GPU训练数为`1,1,1,1,0,0,0,0`，容量合规。
- 当前判定：`RUNNING_HEALTHY_8_OF_12_COMPLETE_THROUGH_EPOCH_117_127`。这是在途分析，不构成剩余4行last5或完整Table III复现结论。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-14 21:51完整Table III确认结果

- 完成性：12/12训练均自然完成epoch200，12个`QUEUE-JOB-END status=0`、12个`PAPER-EVAL-SUMMARY`和12个`FINAL-TEST`齐全；8个queue和全部trainer均已退出，8块GPU空闲，硬错误0。
- 小型证据包：仅打包日志、`metrics.json`、manifest和TSV，不含dataset/checkpoint。远端与本地SHA256均为`92b6abc91a451b748de1947f3fe00432f23aa6da82e73b5b9fc96de3f5909519`；本地路径为`analysis_tmp/paper_repro_riei_table3_confirm_sgd_mean_seed1337_20260714_190100/final_2151`。
- 完整分析：12份`metrics.json`均严格包含epoch1–200，共2400个epoch；扫描32份日志共11020行、802910字节，未见Traceback、RuntimeError、OOM、Killed、NaN、Inf或参数错误。正式分数仍按本轮预注册epoch196–200的last5计算；目标域中间曲线只作诊断。

|行|训练接收机→测试接收机|论文均值±SD|本轮last5均值±SD|差值|较fixopt提升|final|source val last5|last5 CE/MI/IE/FN|论文±2SD|
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
|1|`1-1,7-7`→`1-19`|77.88±2.23%|79.85±0.51%|+1.97pp|+10.84pp|79.02%|99.90±0.01%|0.0125/0.0116/2.481/5.207|命中|
|2|`1-1,8-8`→`1-19`|79.43±1.66%|81.17±1.86%|+1.74pp|+12.64pp|79.77%|99.84±0.01%|0.0114/0.0108/2.481/5.203|命中|
|3|`1-1,14-7`→`1-19`|66.09±0.67%|75.73±2.08%|+9.64pp|+23.66pp|78.00%|99.89±0.01%|0.0127/0.0138/2.481/5.002|未命中|
|4|`7-7,8-8`→`1-19`|70.51±3.53%|79.02±2.38%|+8.51pp|+6.60pp|81.52%|99.85±0.02%|0.0114/0.0104/2.481/5.178|未命中|
|5|`7-7,14-7`→`1-19`|77.35±1.53%|73.56±2.98%|-3.79pp|+7.40pp|75.10%|99.91±0.01%|0.0120/0.0143/2.481/5.007|未命中|
|6|`8-8,14-7`→`1-19`|75.48±1.21%|65.88±1.59%|-9.60pp|+7.63pp|63.60%|99.90±0.00%|0.0108/0.0117/2.481/5.045|未命中|
|7|`1-1,1-19`→`14-7`|71.91±2.08%|71.37±1.73%|-0.54pp|+17.72pp|72.48%|99.81±0.02%|0.0123/0.0098/2.481/5.124|命中|
|8|`1-1,7-7`→`14-7`|68.33±2.37%|69.20±1.92%|+0.87pp|+23.43pp|70.50%|99.90±0.01%|0.0125/0.0116/2.481/5.207|命中|
|9|`1-1,8-8`→`14-7`|73.54±1.27%|70.58±1.45%|-2.96pp|+20.11pp|68.50%|99.86±0.01%|0.0114/0.0107/2.481/5.203|未命中|
|10|`1-19,7-7`→`14-7`|73.52±3.15%|62.19±0.61%|-11.33pp|+24.95pp|62.10%|99.90±0.01%|0.0114/0.0107/2.481/5.274|未命中|
|11|`1-19,8-8`→`14-7`|72.05±2.71%|70.75±1.35%|-1.30pp|+20.35pp|68.58%|99.82±0.01%|0.0105/0.0089/2.481/5.239|命中|
|12|`7-7,8-8`→`14-7`|73.46±2.00%|67.88±1.01%|-5.58pp|+18.98pp|69.10%|99.84±0.01%|0.0112/0.0104/2.481/5.178|未命中|

### 聚合判定与动力学

- 本轮12行均值`72.26%`，论文12行均值`73.30%`，整体偏差仅`-1.03pp`；但逐行MAE=`4.82pp`、RMSE=`6.12pp`、论文`±2SD`命中仅`5/12`，未达到预注册阈值`MAE≤3pp且命中≥10/12`，正式结论为`NOT_REPRODUCED`。
- 相比fixopt，12行均值从`56.07%`升至`72.26%`，提升`16.19pp`，且12行全部提升；说明SGD+mean联合修复有效，但不能据聚合均值接近就宣称逐行复现。
- 所有行source validation后段均为`99.81%–99.91%`，CE/MI/IE和feature norm稳定，无数值崩溃。剩余差异是receiver组合相关的跨域泛化差异，不是未收敛。row10的target诊断峰值`77.17%@43`而正式last5仅`62.19%`，显示source继续拟合可损害target，但禁止用该target峰值选epoch。

## 2026-07-14 22:02剩余协议问题与最小修复

- 原论文实际是IEEE SPAWC 2023会议论文（DOI`10.1109/SPAWC53906.2023.10304544`），正文明确Table III统计“last 10 epochs”；现有报告把它误称为期刊版并固定last5，属于证据口径错误。下一轮改为论文原始last10，但本轮last5结果仍完整保留，不回写或覆盖。
- 更关键的数据协议问题：旧`make_wisig_riei_receiver_holdout_split`在每个Table III组合内部共用一个顺序RNG。某接收机的2400训练/800保留样本会随“另一个source receiver是谁”和循环顺序而变化；target的800样本又来自独立`seed+7919`流。因此12行没有复用论文所述的一次随机train/test partition。
- 本地修复：`code/dataset_wisig.py`对每个`(tx_i,rx_i,eq_i)`使用由`seed+group identity`构造的稳定排列；source取前2400，source validation和该接收机作为target时均复用后续800。这样同一接收机在所有Table III行保持完全相同的全局partition，且不接触target标签进行选型。
- 新增`code/scripts/launch_riei_table3_partition_repair_20260714.sh`，run ID为`paper_repro_riei_table3_partition_repair_seed1337_20260714_220200`；保持P02的SGD、mean、no-RMS、no-feature-norm、200epoch不变，只修复全局partition并改为论文last10。
- 本地验证：`ssr-gpu`解释器下`py_compile`通过；聚焦测试`6 passed`，新增测试验证同一receiver跨source组合的train/validation集合一致，并验证该receiver作为target时test集合等于全局保留集合；`bash -n`通过，dry-run完整展开12个job、8个capacity gate和12条last10命令。
- 下一步：镜像并提交上述最小变更；重新执行N607直接预检，确认每GPU`existing_compute+planned_peak≤2`后，仅同步`code/dataset_wisig.py`和新launcher，核对hash、远端`bash -n`与12-job dry-run，再启动。若逐行仍失败，将优先检查未公开的ResNet1D-18具体结构/随机种子不确定性，不使用target-oracle调参。

## 2026-07-14 22:07分区修复矩阵启动

- Git版本：`ee54a31 Repair RIEI Table III global partition`。根目录不是Git仓库，变更已镜像到`github_publish/CVS-RFFI-repo`并只提交本任务7个文件；本地快照位于`code/snapshots/paper_repro_riei_table3_partition_repair_seed1337_20260714_220200/`。
- 同步映射：`code/dataset_wisig.py`→N607同路径，SHA256=`bb2ccb83a57505066c2d156e8923a77d0c2dff7f40013b45e9ed952c25aa62ff`；`code/scripts/launch_riei_table3_partition_repair_20260714.sh`→N607同路径，SHA256=`dd2896436ff00d24237b022961647a307a1d35954e639f12a412a7cea5414511`。同步前远端`dataset_wisig.py`哈希为`8bf22bd8...`，与本地修复前文件完全一致，未覆盖未归属变更。
- N607直接预检通过；启动前8块GPU均无compute或trainer，目标run/log目录均不存在。远端hash复核、`bash -n`及dry-run均通过：12个job、8个capacity gate、12条last10命令。
- 正式命令：`bash code/scripts/launch_riei_table3_partition_repair_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。8个queue PID为`640456,640458,640461,640464,640469,640476,640481,640487`；GPU0–3各排2个job，GPU4–7各排1个job，计划峰值每GPU仅1个本任务训练。
- 约5分钟健康检查：8/8 queue、8/8 trainer运行；row1–8分别到epoch`12,13,13,12,12,13,12,13/200`，8份`metrics.json`均已创建，`PAPER-EVAL-SUMMARY=0`符合训练早期。硬错误0。
- GPU occupancy：GPU0–7均只有1个本任务compute，显存各约`485MiB`，利用率`16%–22%`；每GPU总训练数1，低于上限2。日志中的`split_info.partition_strategy`均为`stable_group_seed_shared_train_test_holdout`，命令均为`paper_eval_last_n=10`。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_12_13`。不得干预、重启、覆盖或删除产物；完成后必须按12×200epoch完整日志及论文last10逐行重新计算MAE和`±2SD`命中数。

## 2026-07-14 22:17状态复核与声明边界

- N607直接只读预检通过。8个queue与8个首批trainer持续运行，row1–8分别到epoch`28,28,28,27,28,27,27,28/200`；8份`metrics.json`持续写入，`PAPER-EVAL-SUMMARY=0`符合尚未进入最终last10窗口。
- 完整扫描当前已写日志的硬错误计数0；GPU0–7各仅1个本任务compute，显存约`485MiB`，利用率`8%–28%`且epoch持续推进。监控SSH退出后本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。
- 术语边界：RIEI的`SGD+mean`表示无momentum SGD优化器，以及CE/MI/IE在mini-batch内按均值约简；DRIFT并非`SGD+mean`，其固定复现配置是Adam加negative-MSE的mean reduction/no-cap。
- 当前声明：DRIFT五seed final均值`72.75±5.93%`与论文`73.54%`差`-0.79pp`，可写作“聚合均值复现”，但大seed波动必须披露，不能写成稳定逐seed复现。RIEI旧12行虽均值`72.26%`接近论文`73.30%`，但MAE=`4.82pp`且论文`±2SD`仅命中`5/12`，仍为`NOT_REPRODUCED`；新分区修复run尚未完成，不能提前改判。

## 2026-07-14 22:20心跳监控

- N607直接预检通过；8/8原queue与8/8首批trainer持续运行，无PID重启迹象。row1–8最新完整epoch为`39,39,39,38,38,38,38,40/200`，对应日志均持续增长至`219–267`行、`15.7–18.0KiB`。
- 当前`metrics.json=8`、`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、成功完成job=`0/12`，符合首批训练阶段；8份训练日志均确认`partition_strategy=stable_group_seed_shared_train_test_holdout`。
- 完整扫描当前run全部已写日志，硬错误0；未见Traceback、RuntimeError、CUDA OOM、Killed、NaN或参数错误。GPU0–7各仅1个本任务compute，显存约`485MiB`，利用率`10%–26%`且epoch持续推进，容量合规。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_38_40`，不是last10或复现成功证据。SSH短连接均已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-14 22:50心跳监控

- N607直接预检通过；8/8原queue与8/8首批trainer持续运行。row1–8最新完整epoch为`125,125,125,124,122,124,122,127/200`，日志均持续增长至`584–666`行、`40.1–44.4KiB`。
- 当前`metrics.json=8`、`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、成功完成job=`0/12`；8份训练日志继续确认稳定全局partition marker，无协议回退。
- 完整扫描当前run全部已写日志，硬错误0；GPU0–7各仅1个本任务compute，显存约`485MiB`，利用率`16%–30%`且训练持续推进，容量合规。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_122_127`，仍不是论文last10或复现成功证据。SSH短连接均已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-14 23:20心跳监控

- N607直接预检通过。row1–8均已自然完成epoch200，8个`QUEUE-JOB-END status=0`、8个`PAPER-EVAL-SUMMARY`和8个`FINAL-TEST`齐全；GPU4–7的单job queue已退出。
- GPU0–3的4个queue继续第二批row9–12，最新完整epoch分别为`10,13,13,11/200`；当前`metrics.json=12`，12份训练日志均确认稳定全局partition marker。
- 完整扫描当前run全部已写日志，硬错误0；仅GPU0–3各1个本任务compute，GPU4–7空闲，任何GPU总训练均未超过1。
- 当前判定：`RUNNING_HEALTHY_8_OF_12_COMPLETE_THROUGH_EPOCH_10_13`。row1–8的last10暂不脱离完整12行单独宣称，待row9–12完成后统一逐行计算。SSH短连接均已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-14 23:50心跳监控

- N607直接预检通过；row1–8保持完整完成，GPU0–3的4个queue与row9–12 trainer持续运行。row9–12最新完整epoch为`103,103,104,98/200`，日志持续增长至`519–560`行、`35.5–37.0KiB`。
- 当前`metrics.json=12`、`PAPER-EVAL-SUMMARY=8`、`FINAL-TEST=8`、成功完成job=`8/12`；12份训练日志均保留稳定全局partition marker，硬错误0。
- GPU0–3各有1个本任务trainer及1个不属于本任务的并行trainer，总compute为2；GPU4–7各1个其他任务compute。所有GPU均未超过上限2，本任务未干预其他运行。
- 当前判定：`RUNNING_HEALTHY_8_OF_12_COMPLETE_THROUGH_EPOCH_98_104`，仍不是完整Table III复现结论。SSH短连接均已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-15 00:23全局partition修复结果

- 完成性：Table III 12/12行均自然完成epoch200，12个`QUEUE-JOB-END status=0`、12个`PAPER-EVAL-SUMMARY`和12份含`FINAL-TEST`的训练日志齐全；全部queue/trainer退出，N607八张GPU无compute process，未干预其他任务。
- 小型证据包：仅包含32份日志、12份`metrics.json`、12份manifest和调度TSV，不含dataset/checkpoint。远端与本地SHA256均为`edc3886f1208be2afd9ceef6404e498ff8c8156fc7063784ed90a5477fe4dd1d`；本地路径为`analysis_tmp/paper_repro_riei_table3_partition_repair_seed1337_20260714_220200/final_0023`。
- 完整性审计：12份metrics均严格包含epoch1–200，共2400个epoch；完整读取32份日志共11116行/808899字节，12份训练日志均包含`stable_group_seed_shared_train_test_holdout`，硬错误0。正式数值固定论文epoch191–200的last10；target中间曲线只作诊断。

|行|训练接收机→测试接收机|论文均值±SD|新run last10均值±SD|差值|较旧run变化|final|source val last10|last10 CE/MI/IE/FN|论文±2SD|
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
|1|`1-1,7-7`→`1-19`|77.88±2.23%|79.52±0.42%|+1.64pp|-0.33pp|79.77%|99.79±0.03%|0.0119/0.0119/2.481/5.207|命中|
|2|`1-1,8-8`→`1-19`|79.43±1.66%|82.08±3.12%|+2.65pp|+0.91pp|79.08%|99.87±0.01%|0.0112/0.0109/2.481/5.204|命中|
|3|`1-1,14-7`→`1-19`|66.09±0.67%|74.08±2.39%|+7.99pp|-1.65pp|71.85%|99.81±0.01%|0.0119/0.0142/2.481/4.975|未命中|
|4|`7-7,8-8`→`1-19`|70.51±3.53%|78.61±2.04%|+8.10pp|-0.41pp|79.94%|99.86±0.02%|0.0115/0.0105/2.481/5.172|未命中|
|5|`7-7,14-7`→`1-19`|77.35±1.53%|75.26±1.38%|-2.09pp|+1.70pp|74.40%|99.79±0.02%|0.0114/0.0144/2.481/5.003|命中|
|6|`8-8,14-7`→`1-19`|75.48±1.21%|65.81±2.72%|-9.67pp|-0.07pp|66.48%|99.85±0.02%|0.0107/0.0120/2.481/5.038|未命中|
|7|`1-1,1-19`→`14-7`|71.91±2.08%|71.77±2.24%|-0.14pp|+0.40pp|71.25%|99.75±0.01%|0.0120/0.0101/2.481/5.122|命中|
|8|`1-1,7-7`→`14-7`|68.33±2.37%|70.49±2.54%|+2.16pp|+1.29pp|70.08%|99.80±0.03%|0.0120/0.0119/2.481/5.207|命中|
|9|`1-1,8-8`→`14-7`|73.54±1.27%|70.78±1.27%|-2.76pp|+0.20pp|68.71%|99.87±0.01%|0.0112/0.0109/2.481/5.205|未命中|
|10|`1-19,7-7`→`14-7`|73.52±3.15%|63.62±1.15%|-9.90pp|+1.43pp|64.02%|99.77±0.02%|0.0113/0.0111/2.481/5.255|未命中|
|11|`1-19,8-8`→`14-7`|72.05±2.71%|72.05±3.27%|-0.00pp|+1.30pp|76.60%|99.77±0.02%|0.0105/0.0090/2.481/5.239|命中|
|12|`7-7,8-8`→`14-7`|73.46±2.00%|68.54±1.09%|-4.92pp|+0.66pp|68.21%|99.86±0.02%|0.0115/0.0105/2.481/5.173|未命中|

### 判定与剩余问题

- 新run的12行均值为`72.72%`，论文均值为`73.30%`，聚合偏差仅`-0.58pp`；逐行MAE=`4.34pp`、RMSE=`5.56pp`、论文`±2SD`命中`6/12`。相对旧run，MAE只从`4.82pp`降至`4.34pp`，命中数从`5/12`升至`6/12`，仍未达到`MAE≤3pp且命中≥10/12`，正式结论保持`NOT_REPRODUCED`。
- 分区修复没有消除receiver组合相关的方向性偏差：row3/4仍高出论文约`8pp`，row6/10仍低约`10pp`，row12低`4.92pp`。所有行source validation均为`99.75%–99.87%`，CE/MI/IE、feature norm和数值状态稳定，因此剩余差异不是训练未收敛或随机分区漂移。
- 原论文Eq.(5)、Eq.(7)和Eq.(8)把MI与IE明确写成样本/receiver求和，但当前完整12行使用了单row消融胜出的mean reduction。sum不是新机制，而是论文公式的字面尺度；发现矩阵中的SGD+sum/no-RMS/no-FN第1行last5=`74.52±0.59%`，仍进入论文第1行`±2SD`。在猜测未公开ResNet1D-18细节或按target选择seed之前，应先用已修复partition和论文last10完成sum的12行同协议确认。
- 论文没有公开ResNet1D-18的卷积核、stem/max-pool、通道数、feature split维数、优化器名称或随机seed；公开检索也未找到作者提供的实现仓库。若sum完整矩阵仍失败，下一轮只把这些未公开项作为受控架构/seed敏感性诊断，不把target峰值用于选型。

## 2026-07-15 00:30下一轮最小论文字面尺度确认设计

- 预定run ID：`paper_repro_riei_table3_sum_literal_seed1337_20260715_003000`。保持稳定全局partition、SGD momentum0、no-RMS、no-feature-norm、batch64、学习率`1e-4`、`lambda_mi=lambda_ie=1.2`、200epoch和last10不变；唯一变量是CE/MI/IE从mean改为sum。
- 本地launcher在`code/scripts/launch_riei_table3_partition_repair_20260714.sh`新增`RIEI_REDUCTION`显式参数，默认仍为mean；新run固定`RIEI_REDUCTION=sum`。非法值将fail closed，避免静默回退。
- 成功阈值保持`MAE≤3pp且至少10/12进入论文±2SD`。若不达标，继续保留DRIFT唯一固定版本，RIEI不宣称复现；后续诊断优先比较ResNet1D-18常见实现与seed敏感性。
- 本地验证：`ssr-gpu`解释器运行聚焦测试`20 passed,8 subtests passed`；唯一warning是根目录`.pytest_cache`无写权。launcher的`bash -n`通过；显式前缀`RUN_ID=... RIEI_REDUCTION=sum`的dry-run展开12个job、8个capacity gate，12条命令的CE/MI/IE均为sum且mean计数0。PowerShell仅设置`$env:`时，WSL Bash没有继承变量并回退默认mean，因此正式命令必须把两个环境变量写在同一Bash命令前缀中。

## 2026-07-15 00:35 sum矩阵同步前容量门

- 根目录不是Git仓库；脚本、测试、README、报告和traceability已镜像到Git承载面并以`b560cb9 Record RIEI partition result and sum diagnostic`提交。脚本快照位于`code/snapshots/paper_repro_riei_table3_sum_literal_seed1337_20260715_003000/`。
- 直接N607预检通过。实时进程/CWD/cmdline检查没有训练进程，`nvidia-smi`的compute process计数为0；计划每GPU新增1个训练，所以`existing_compute+planned_peak=1≤2`。目标run/log目录均不存在。
- 同步前远端launcher SHA256=`dd2896436ff00d24237b022961647a307a1d35954e639f12a412a7cea5414511`，等于上一轮启动时记录的已知版本；本地新launcher SHA256=`6c90bb827f0682c286fe93e69f46a4d67ea46b916de4442782f2fff5f7801374`。`code/dataset_wisig.py`远端SHA256仍为已验证的`bb2ccb83a57505066c2d156e8923a77d0c2dff7f40013b45e9ed952c25aa62ff`，本轮无需重传。
- 唯一同步映射：本地`code/scripts/launch_riei_table3_partition_repair_20260714.sh`→N607同路径。同步后必须复核hash、远端`bash -n`和sum dry-run，再执行：`RUN_ID=paper_repro_riei_table3_sum_literal_seed1337_20260715_003000 RIEI_REDUCTION=sum bash code/scripts/launch_riei_table3_partition_repair_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。

## 2026-07-15 00:36 sum矩阵启动与健康检查

- 仅同步已验证launcher；远端SHA256复核为`6c90bb827f0682c286fe93e69f46a4d67ea46b916de4442782f2fff5f7801374`。远端`bash -n`通过，显式Bash前缀dry-run确认12个job、8个capacity gate、CE/MI/IE sum各12条、mean 0条，目标run/log目录在启动前不存在。
- 正式命令与00:35记录一致。launcher再次执行容量门，GPU0–7的`current=0`、`planned_peak=1`、`total_peak=1≤2`；8个queue PID依次为`784884,784886,784889,784893,784898,784903,784911,784918`。GPU0–3各排2个顺序job，GPU4–7各1个job。
- 启动约5分钟后，8/8 queue和8/8本任务Python进程持续运行；首批row1–8最新完整epoch为`15,14,15,14,15,15,14,15/200`。8份`metrics.json`已建立，所有训练日志均确认`ce/mi/ie_reduction=sum`及`stable_group_seed_shared_train_test_holdout`。
- 完整扫描当前已写日志，硬错误0；`PAPER-EVAL-SUMMARY=0`、完成job=0符合训练早期。`nvidia-smi pmon`显示GPU0–7各1个本任务compute，SM占用约13%–18%，没有其他compute；每GPU总训练数1，容量合规。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_14_15`。本次所有SSH/SCP短连接均已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`；不得干预、重启或覆盖产物。

## 2026-07-15 00:51心跳监控

- 直接N607只读预检通过。8个原queue PID均存活并运行约869–870秒；8个本任务Python trainer持续运行，命令行逐一确认SGD momentum0、CE/MI/IE sum、no-RMS、no-feature-norm、200epoch和last10，没有参数回退。
- 完整读取当前8份训练日志至最新记录：row1–8分别到epoch`43,43,43,43,44,44,42,42/200`；日志为`164–190`行、`13.5–14.9KiB`，全部持续增长。8份日志均含1个sum配置marker和1个稳定全局partition marker。
- 8份`metrics.json`均可完整解析，epoch序列连续且无重复/缺口；已完整写入epoch`42,42,42,42,43,43,41,41`，尚无`final`字段。当前`PAPER-EVAL-SUMMARY=0`、含`FINAL-TEST`日志=0、成功完成job=0，符合首批训练阶段。
- 对当前run全部已写日志做完整硬错误扫描，计数0；未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN/Inf或参数错误。
- `nvidia-smi pmon`确认GPU0–7各仅1个本任务compute，SM占用约16%–27%，无其他compute；每GPU训练数1，未超过容量上限2。本次短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_42_44`。这是在途证据，不构成sum last10或Table III复现结论。

## 2026-07-15 01:21心跳监控

- 直接N607只读预检通过。8个原queue PID均存活并运行约2663秒；8个本任务trainer PID=`785062,785064,785085,785096,785099,785105,785109,785112`持续运行，完整命令行仍固定sum、稳定partition及其余预注册参数。
- 完整读取首批8份训练日志至最新记录：row1–8分别到epoch`131,130,131,130,129,132,128,130/200`；日志为`464–486`行、`35.3–36.7KiB`，均持续增长并保留sum/partition marker。
- 8份`metrics.json`均可完整解析，epoch序列连续且无重复/缺口；已写入epoch`130,129,130,129,128,131,127,129`，尚无`final`字段。当前`PAPER-EVAL-SUMMARY=0`、含`FINAL-TEST`日志=0、成功完成job=0，符合训练中状态。
- 全量扫描当前run已写日志，硬错误0；未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN/Inf或参数错误。
- GPU0–7各仅1个本任务compute，SM占用约15%–23%，无其他compute；任何GPU训练总数均为1。短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。
- 当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_128_132`。正式结论继续等待完整12行论文last10结果。

## 2026-07-15 01:51心跳监控

- 直接N607只读预检通过。row1–8均自然完成epoch200，8个`QUEUE-JOB-END status=0`、8个`PAPER-EVAL-SUMMARY`和8份含`FINAL-TEST`的训练日志齐全；GPU4–7的单job queue已自然退出。
- GPU0–3的queue PID=`784884,784886,784889,784893`继续第二批；row9–12 trainer PID分别为`829768,830410,830638,829883`，最新完整日志epoch为`17,16,17,17/200`。四条完整命令行继续确认sum、稳定partition及其余预注册参数。
- 12份`metrics.json`均可解析且epoch连续：row1–8各200epoch并有`final`字段；row9–12分别完整写入epoch`16,15,16,16`且无`final`。当前`PAPER-EVAL-SUMMARY=8/12`、含`FINAL-TEST`日志=`8/12`、成功完成job=`8/12`。
- 完整扫描当前run全部已写日志，硬错误0；未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN/Inf或参数错误。已完成row1–8的数值不脱离完整12行单独选型或宣称。
- GPU0–3各仅1个本任务compute，GPU4–7无compute；任何GPU训练数均不超过1。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。
- 当前判定：`RUNNING_HEALTHY_8_OF_12_COMPLETE_THROUGH_EPOCH_16_17`。完整Table III sum结论等待row9–12自然完成。

## 2026-07-15 02:21心跳监控

- 直接N607只读预检通过。row1–8保持完整200epoch、`final`字段及8个成功`QUEUE-JOB-END`；GPU0–3的4个queue与row9–12原trainer PID持续运行，无重启或进程替换。
- 完整读取全部12份训练日志：row9–12最新完整epoch为`105,108,103,104/200`；对应日志为`382–398`行、`29.2–30.3KiB`，持续增长并保留sum/partition marker。
- 12份`metrics.json`均可解析且epoch连续：row1–8各200epoch并有`final`；row9–12分别完整写入epoch`104,107,102,103`且无`final`。当前`PAPER-EVAL-SUMMARY=8/12`、含`FINAL-TEST`日志=`8/12`、成功完成job=`8/12`。
- 完整扫描当前run全部已写日志，硬错误0；未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN/Inf或参数错误。
- GPU0–3各仅1个本任务compute，SM占用约14%–16%；GPU4–7无compute，容量合规。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。
- 当前判定：`RUNNING_HEALTHY_8_OF_12_COMPLETE_THROUGH_EPOCH_103_108`。剩余4行尚未进入论文last10正式窗口。

## 2026-07-15 02:55论文字面sum矩阵完整结果

- 完成性：12/12训练均自然完成epoch200，12个`QUEUE-JOB-END status=0`、12份含`PAPER-EVAL-SUMMARY`的训练日志和12份含`FINAL-TEST`的训练日志齐全；8个queue与全部trainer均已退出，硬错误0。
- 小型证据包仅含32份完整日志、12份`metrics.json`、manifest和scheduler TSV，不含dataset/checkpoint。远端与本地SHA256均为`610158161ec926b5ca998fee2a45acbd6d901e5338163abc5bef9536204e13ff`；本地路径为`analysis_tmp/paper_repro_riei_table3_sum_literal_seed1337_20260715_003000/final_0255`。
- 完整分析覆盖12份`metrics.json`的epoch1–200，共2400个epoch；扫描32份日志共8952行、711816字节。12份日志均确认`stable_group_seed_shared_train_test_holdout`，且配置均为CE/MI/IE=`sum`；未见Traceback、RuntimeError、CUDA OOM、Killed、NaN、Inf或参数错误。

|行|训练接收机→测试接收机|论文均值±SD|sum last10均值±SD|差值|相对mean变化|final|source val last10|last10 CE/MI/IE/FN|论文±2SD|
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
|1|`1-1,7-7`→`1-19`|77.88±2.23%|73.22±1.02%|-4.66pp|-6.30pp|72.71%|99.88±0.02%|0.0044/0.0052/159.012/5.816|未命中|
|2|`1-1,8-8`→`1-19`|79.43±1.66%|69.07±1.07%|-10.36pp|-13.01pp|69.15%|99.89±0.01%|0.0046/0.0047/159.015/5.606|未命中|
|3|`1-1,14-7`→`1-19`|66.09±0.67%|70.38±4.16%|+4.28pp|-3.70pp|70.58%|99.89±0.02%|0.0044/0.0053/159.013/5.545|未命中|
|4|`7-7,8-8`→`1-19`|70.51±3.53%|78.72±2.33%|+8.21pp|+0.11pp|77.35%|99.89±0.01%|0.0053/0.0047/159.013/5.571|未命中|
|5|`7-7,14-7`→`1-19`|77.35±1.53%|72.51±3.46%|-4.84pp|-2.75pp|67.21%|99.91±0.01%|0.0046/0.0063/159.014/5.539|未命中|
|6|`8-8,14-7`→`1-19`|75.48±1.21%|64.79±0.38%|-10.69pp|-1.02pp|64.56%|99.92±0.01%|0.0044/0.0050/159.016/5.518|未命中|
|7|`1-1,1-19`→`14-7`|71.91±2.08%|72.18±2.74%|+0.27pp|+0.41pp|73.52%|99.83±0.01%|0.0045/0.0055/159.014/5.733|命中|
|8|`1-1,7-7`→`14-7`|68.33±2.37%|63.59±2.57%|-4.74pp|-6.90pp|64.85%|99.87±0.01%|0.0042/0.0053/159.013/5.805|未命中|
|9|`1-1,8-8`→`14-7`|73.54±1.27%|61.66±2.06%|-11.88pp|-9.12pp|59.94%|99.89±0.01%|0.0043/0.0046/159.014/5.716|未命中|
|10|`1-19,7-7`→`14-7`|73.52±3.15%|56.91±3.89%|-16.61pp|-6.71pp|60.44%|99.89±0.01%|0.0592/0.0074/159.004/5.851|未命中|
|11|`1-19,8-8`→`14-7`|72.05±2.71%|79.54±3.78%|+7.49pp|+7.49pp|82.67%|99.91±0.01%|0.0044/0.0061/159.017/5.876|未命中|
|12|`7-7,8-8`→`14-7`|73.46±2.00%|62.46±1.89%|-11.00pp|-6.08pp|62.96%|99.89±0.01%|0.0055/0.0047/159.014/5.552|未命中|

### sum判定与动力学诊断

- 论文12行均值为`73.30%`，sum复现均值为`68.75%`，有符号偏差`-4.54pp`；逐行MAE=`7.92pp`、RMSE=`8.99pp`、论文`±2SD`仅命中`1/12`。未达到`MAE≤3pp且命中≥10/12`，正式结论为`NOT_REPRODUCED`。
- 相对稳定partition的mean run，sum仅row4、7、11提高，另外9行降低，平均由`72.72%`降至`68.75%`。因此Eq.(2)–(8)的论文字面求和不能作为当前batch64实现的正确数值尺度；mean仍是后续唯一保留的训练尺度。
- sum后段IE约为`159.01`，使last10总训练loss约为`-190.8`；source validation仍达到`99.83%–99.92%`，但大多数target诊断峰值出现在epoch1–25，随后跨receiver泛化明显退化。这是loss尺度导致的目标域过拟合，不是训练未收敛或数值崩溃；诊断峰值不用于选epoch。
- 三轮完整Table III的正式边界依次为：旧组合内随机split的mean/last5=`MAE4.82pp,5/12`；稳定全局partition的mean/last10=`MAE4.34pp,6/12`；稳定partition的sum/last10=`MAE7.92pp,1/12`。协议与loss尺度已排除，剩余首要不确定性是论文未公开的ResNet1D-18具体stem/下采样结构，其次是优化器细节与随机seed。

## 2026-07-15 03:00下一轮最小受控架构诊断设计

- 论文只给出“ResNet 1D-18”，未公开stem卷积核、首层stride和是否使用max-pool。当前FED是ImageNet式`kernel7/stride2 + maxpool3/stride2`；对长度256的I/Q输入在进入残差stage前即压缩到64点。短序列RFFI常见的另一种合理解释是CIFAR式`kernel3/stride1、无max-pool`，保留256点后再由残差stage下采样。
- 下一轮只改变FED stem，保持已验证较优的SGD、mean、no-RMS、no-feature-norm、稳定全局partition、200epoch和论文last10不变。预注册诊断行固定为partition-mean中绝对误差最大的row3、6、10、12；每行对照`imagenet1d`和候选`short_stem1d`，共8个job，禁止target峰值选epoch。
- 预注册筛选分数为四行last10的MAE，且候选至少在3/4行降低绝对误差，才允许进入完整12行确认；否则拒绝该架构，不通过事后挑行或挑epoch保留。该诊断只能缩小未公开实现空间，不能单独形成Table III复现成功声明。

### 本地实现与验证

- `baselines/riei_fd/architecture.py`新增fail-closed的FED`variant`：`imagenet1d`保持既有结构，`short_stem1d`仅把stem改为`kernel3/stride1`且取消max-pool；四个残差stage、512维输出和EC/RC均不变。`model.py`、`train_cvs.py`与paper queue完成显式参数接线和日志marker。
- 新launcher：`code/scripts/launch_riei_table3_architecture_probe_20260715.sh`，run ID为`paper_repro_riei_archprobe_seed1337_20260715_030500`。8个job恰好覆盖4行×2个variant，每GPU计划峰值1个训练，并保留唯一run/log根保护。
- `ssr-gpu`下`py_compile`通过；根目录聚焦测试`7 passed`，Git镜像同组测试`7 passed`。根目录仅有既知`.pytest_cache`无写权warning，不影响测试。`bash -n`通过，8-job dry-run计数为8个job、8个capacity gate、`imagenet1d=4`、`short_stem1d=4`。
- 根目录不是Git仓库；本地代码快照为`code/snapshots/paper_repro_riei_archprobe_seed1337_20260715_030500/`。关键SHA256：paper queue=`5a1fe1f1...`、architecture=`bf1d8e1f...`、model=`c936c09b...`、root train=`98e974aa...`、launcher=`0760d3cc...`、test=`6cf13d3b...`。
- Git镜像的`train_cvs.py`已有不属于本任务的augmentation-consistency接线；镜像时保留该既有逻辑，只叠加本任务3处`fed_variant`接线，未用根目录旧副本覆盖这些并行变更。当前尚未同步或启动N607；下一步必须先提交仅本任务文件，再重新执行实时容量门。

### 03:09同步与启动前容量门

- Git提交：`a53e259 Probe RIEI ResNet1D stem parity`，仅包含本任务9个文件。N607同步前所有GPU无compute、无训练或launcher进程；目标run/log目录不存在，计划每GPU新增1个训练，满足`existing_compute+planned_peak=1≤2`。
- 同步前远端SHA256：architecture=`899f66df...`、model=`99b1194a...`、train=`950b6008...`、paper queue=`2ba90874...`，均为本任务既有远端版本；新launcher不存在。同步目标依次为N607同路径的`baselines/riei_fd/{architecture.py,model.py,train_cvs.py}`、`run_wisig_paper_scope_queue.sh`和`code/scripts/launch_riei_table3_architecture_probe_20260715.sh`。
- 同步后远端SHA256与根目录本地文件一致：architecture=`bf1d8e1f...`、model=`c936c09b...`、train=`98e974aa...`、paper queue=`5a1fe1f1...`、launcher=`0760d3cc...`。远端`bash -n`、8-job dry-run和两种FED的`[2,2,256]→[2,512]`前向smoke均通过。
- 计划正式命令：`bash code/scripts/launch_riei_table3_architecture_probe_20260715.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，工作目录为`/home/szu2070436088/2510044040/CV-SincNet`；预期产物为每job的完整日志、200epoch`metrics.json`、final last10、manifest及queue PID/状态。

### 03:10正式启动与4.5分钟健康检查

- 正式命令按上述预注册内容执行成功。8个queue PID依次为`885107,885109,885112,885117,885122,885130,885137,885147`；8个trainer PID为`885263,885278,885314,885316,885317,885329,885335,885334`，GPU0–7各1个。完整命令行确认4个`imagenet1d`与4个`short_stem1d`，其余训练配置一致。
- 03:14:57健康检查时，8个queue与8个trainer均存活，8份`metrics.json`已创建；row3/6/10/12的两种variant分别进入epoch12–14/200。所有日志持续增长至39–85行、6.3–7.2KiB，并保留正确`fed_variant`marker。
- 全量扫描当前已写日志，硬错误0；尚无`PAPER-EVAL-SUMMARY`符合训练早期状态。GPU0–7各仅1个本任务compute：ImageNet式stem约485MiB、short stem约639MiB，利用率9%–38%，任何GPU总训练数均为1。
- 当前判定：`RUNNING_STARTUP_HEALTHY_THROUGH_EPOCH_12_14`。不得干预、重启、覆盖或删除产物；正式选择只能使用各job epoch191–200的last10。SSH短连接全部退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-15 03:51架构诊断心跳监控

- 直接N607只读预检通过。8个原queue PID与8个原trainer PID均持续运行约2438秒，无重启或进程替换；完整命令行仍保持4个`imagenet1d`、4个`short_stem1d`及相同的SGD+mean协议。
- 完整读取8份训练日志至最新记录：row3/6/10/12两种variant分别到epoch`113,113,116,113,114,116,114,113/200`；日志为580–641行、39.3–42.1KiB，全部持续增长。
- 8份`metrics.json`均可完整解析，epoch序列分别写入`112,112,115,112,113,115,113,112`且均无`final`字段；与日志最新epoch只差1。当前`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`符合尚未进入epoch191–200正式窗口。
- 对当前run全部已写训练日志做完整硬错误扫描，计数0；未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN或参数错误。
- GPU0–7各仅1个本任务compute，ImageNet式stem约485MiB、short stem约639MiB，利用率16%–33%；任何GPU总训练数均为1。当前判定：`RUNNING_HEALTHY_THROUGH_EPOCH_113_116`，不是正式架构选型结果。SSH短连接全部退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-15 04:23架构诊断完整结果

- 完成性：8/8个job均自然完成epoch200；8个`QUEUE-JOB-END status=0`、8个`PAPER-EVAL-SUMMARY`和8份含`FINAL-TEST`的训练日志齐全，原queue与trainer均已退出。完整分析覆盖8份metrics的1600个epoch以及24份日志的7600行/548720字节，硬错误0。
- 小型证据包仅包含日志、metrics、manifest和调度文件，不含dataset/checkpoint。远端与本地SHA256均为`e9c450044328288835f9f2b79c1cacbc2f9af8a3af120181160608c49aefc498`；本地路径为`analysis_tmp/paper_repro_riei_archprobe_seed1337_20260715_030500/final_0423`。
- 正式比较严格采用预注册的epoch191–200 last10，不使用target中间峰值选型。每个同row对照的partition、seed、SGD、mean、no-RMS、no-feature-norm和200epoch均一致，唯一变量为FED stem。

|行|训练接收机→测试接收机|variant|论文均值±SD|last10均值±SD|差值|final|source val last10|last10 loss|CE/MI/IE/FN|论文±2SD|
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
|3|`1-1,14-7`→`1-19`|`imagenet1d`|66.09±0.67%|74.05±2.40%|+7.96pp|71.83%|99.81±0.01%|-2.9479|0.0119/0.0142/2.4806/4.976|未命中|
|3|`1-1,14-7`→`1-19`|`short_stem1d`|66.09±0.67%|68.95±2.33%|+2.86pp|68.60%|99.82±0.01%|-2.9007|0.0208/0.0307/2.4653/2.952|未命中|
|6|`8-8,14-7`→`1-19`|`imagenet1d`|75.48±1.21%|66.03±2.89%|-9.45pp|65.67%|99.86±0.02%|-2.9522|0.0106/0.0120/2.4810/5.038|未命中|
|6|`8-8,14-7`→`1-19`|`short_stem1d`|75.48±1.21%|72.28±1.12%|-3.20pp|72.67%|99.87±0.01%|-2.8491|0.0234/0.0304/2.4242/3.224|未命中|
|10|`1-19,7-7`→`14-7`|`imagenet1d`|73.52±3.15%|63.59±1.16%|-9.93pp|64.04%|99.76±0.02%|-2.9525|0.0113/0.0111/2.4809/5.254|未命中|
|10|`1-19,7-7`→`14-7`|`short_stem1d`|73.52±3.15%|70.11±1.51%|-3.41pp|68.79%|99.74±0.02%|-2.9180|0.0198/0.0237/2.4719/3.929|命中|
|12|`7-7,8-8`→`14-7`|`imagenet1d`|73.46±2.00%|69.14±1.83%|-4.32pp|66.98%|99.86±0.01%|-2.9529|0.0114/0.0105/2.4807/5.172|未命中|
|12|`7-7,8-8`→`14-7`|`short_stem1d`|73.46±2.00%|66.84±1.14%|-6.62pp|69.04%|99.83±0.02%|-2.9245|0.0196/0.0235/2.4769/3.736|未命中|

### 预注册门槛判定

|variant|四行均值|有符号偏差|MAE|RMSE|论文±2SD命中|
|---|---:|---:|---:|---:|---:|
|`imagenet1d`|68.20%|-3.93pp|7.91pp|8.21pp|0/4|
|`short_stem1d`|69.54%|-2.59pp|4.02pp|4.30pp|1/4|

- `short_stem1d`在row3、6、10分别把绝对误差降低`5.10pp`、`6.25pp`和`6.52pp`，仅row12恶化`2.30pp`。因此它同时满足“四行MAE低于imagenet1d”和“至少3/4行降低绝对误差”两个预注册条件，架构筛选结果为`PASS_TO_FULL_TABLE3_CONFIRMATION`。
- 该结果只支持进入完整12行确认，不能单独宣称RIEI复现成功。row12的反向变化说明short stem不是逐行单调改善；最终仍须用同一short-stem配置完成Table III全部12行，并达到`MAE≤3pp且至少10/12进入论文±2SD`。

## 2026-07-15 04:30short-stem完整12行确认设计

- 新launcher：`code/scripts/launch_riei_table3_shortstem_confirm_20260715.sh`；预定run ID：`paper_repro_riei_table3_shortstem_confirm_seed1337_20260715_043000`。
- 12行receiver组合、论文均值/SD、稳定全局partition、seed1337、SGD momentum0、CE/MI/IE mean、no-RMS、no-feature-norm、200epoch和论文last10全部固定；唯一相对现有完整mean矩阵的变量为`RIEI_FED_VARIANT=short_stem1d`。
- GPU0–3各排2个顺序job，GPU4–7各1个job，planned peak为每GPU 1个训练。launcher保持唯一run/log根保护及每GPU训练总数不超过2的容量门。
- 本地`bash -n`通过；dry-run完整展开12个job、8个capacity gate、12条short-stem命令和12条mean命令，sum命令0条。启动前还必须执行直接N607预检、实时process/CWD/cmdline与GPU容量检查、同步后hash核对、远端`bash -n`和12-job dry-run。

### 04:31同步前版本与容量门

- 根目录仍不是Git仓库；新launcher已镜像到Git承载面，并与本报告及traceability一起以提交`1a649bc Confirm RIEI short-stem Table III`版本化。根目录launcher、本地快照和Git镜像SHA256均为`ce25286443d34648047dcbe6afd2537aac43162fb0f3c6487c1ae0aafe0205bb`；快照位于`code/snapshots/paper_repro_riei_table3_shortstem_confirm_seed1337_20260715_043000/`。
- 直接N607预检通过；服务器时间、项目根和8块GPU可见。实时process/CWD/cmdline检查没有训练或RIEI/DRIFT launcher，GPU0–7的compute process均为0；计划每GPU峰值1个训练，因此`existing_compute+planned_peak=1≤2`。
- 目标run/log目录在同步前均不存在。唯一待同步文件为本地`code/scripts/launch_riei_table3_shortstem_confirm_20260715.sh`→N607同路径；同步后必须复核SHA256、远端`bash -n`和12-job dry-run，再执行正式命令。所有预检SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

### 04:32正式启动与4分钟健康检查

- 远端launcher SHA256复核为`ce25286443d34648047dcbe6afd2537aac43162fb0f3c6487c1ae0aafe0205bb`，与本地、快照及Git镜像一致；远端`bash -n`通过，dry-run确认12个job、8个capacity gate、short stem/mean各12条、sum 0条。
- 正式命令：`bash code/scripts/launch_riei_table3_shortstem_confirm_20260715.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。launcher再次确认GPU0–7的`current=0`、`planned_peak=1`、`total_peak=1≤2`。
- 8个queue PID为`928971,928973,928977,928982,928988,928994,929003,929013`；首批row1–8 trainer PID为`929140,929141,929170,929178,929198,929199,929200,929203`。GPU0–3各排2个顺序job，GPU4–7各1个。
- 04:36:47健康检查：8/8 queue与8/8 trainer均持续运行；row1–8最新完整epoch为`13,13,13,13,12,12,13,14/200`，8份`metrics.json`已建立。完整命令行与日志marker均确认`short_stem1d`、SGD momentum0、CE/MI/IE mean、no-RMS、no-feature-norm、稳定全局partition和paper last10。
- 全量扫描当前已写日志，硬错误0；`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、完成job=0符合训练早期。GPU0–7各1个本任务compute，SM约21%–33%，每GPU训练总数1。当前判定为`RUNNING_STARTUP_HEALTHY_THROUGH_EPOCH_12_14`，不构成RIEI复现结论。所有SSH/SCP短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

### 04:51只读心跳监控

- 直接N607预检通过；8个原queue PID与8个原trainer PID均持续运行约1148秒，无重启或进程替换。完整命令行继续固定`short_stem1d`、稳定全局partition、SGD momentum0、CE/MI/IE mean、no-RMS、no-feature-norm、200epoch及paper last10。
- 8份`metrics.json`均可完整解析，epoch序列连续且无重复或缺口；row1–8分别完整写入epoch`53,52,52,51,50,50,52,53/200`，尚无`final`字段。当前`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、成功完成job=`0/12`，符合首批训练阶段。
- 完整读取当前run的24份日志，共2423行/197300字节；8份训练日志均含稳定partition和short-stem marker。全量硬错误扫描计数0，未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN、Inf或参数错误。
- `nvidia-smi pmon`确认GPU0–7各仅1个本任务compute，SM占用约28%–29%，没有其他compute；任何GPU训练总数均为1。当前判定为`RUNNING_HEALTHY_THROUGH_EPOCH_50_53`，仍不是Table III正式结果。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

### 05:21只读心跳监控

- 直接N607预检通过；8个原queue与8个原trainer PID保持不变并持续运行，无重启或进程替换。当前仍为首批row1–8，第二批row9–12尚未开始。
- 8份`metrics.json`全部可解析，epoch序列连续且无缺口；row1–8分别完整写入epoch`133,133,134,132,133,132,135,136/200`，均无`final`字段。`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、成功完成job=`0/12`，与训练阶段一致。
- 完整读取当前run的24份日志，共5600行/404130字节；8份训练日志均确认稳定partition和`short_stem1d`。全量硬错误扫描计数0，未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN、Inf或参数错误。
- GPU0–7各仅1个本任务compute；SM占用约26%–48%，其余GPU进程仅Xorg。容量门持续合规。当前判定为`RUNNING_HEALTHY_THROUGH_EPOCH_132_136`，正式last10仍须等待epoch191–200及完整12行。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

### 05:51只读心跳监控

- 直接N607预检通过。row1–8均自然完成epoch200，8个`QUEUE-JOB-END status=0`、8个`PAPER-EVAL-SUMMARY`和8份含`FINAL-TEST`的训练日志齐全；GPU4–7的单job queue已自然退出。
- GPU0–3的queue PID=`928971,928973,928977,928982`继续第二批；row9–12 trainer PID依次为`967912,967672,967560,968564`。12份`metrics.json`均可解析且epoch连续：row1–8各200epoch并有`final`字段，row9–12分别完整写入epoch`16,17,17,13/200`且无`final`。
- 当前完整读取32份日志，共8009行/598087字节；12份训练日志均确认稳定partition和`short_stem1d`。`PAPER-EVAL-SUMMARY=8/12`、`FINAL-TEST=8/12`、成功完成job=`8/12`；全量硬错误扫描计数0。
- GPU0–3各仅1个本任务compute，SM占用约28%–36%；GPU4–7无compute，容量合规。当前判定为`RUNNING_HEALTHY_8_OF_12_COMPLETE_THROUGH_EPOCH_13_17`。已完成前8行不脱离完整12行进行目标域选型或复现声明。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

### 06:21只读心跳监控

- 直接N607预检通过；row1–8保持完整完成，GPU0–3的4个原queue与row9–12原trainer PID持续运行，无重启或进程替换。
- 12份`metrics.json`均可解析且epoch连续：row1–8各200epoch并有`final`字段；row9–12分别完整写入epoch`98,97,99,95/200`且无`final`。当前`PAPER-EVAL-SUMMARY=8/12`、`FINAL-TEST=8/12`、成功完成job=`8/12`。
- 完整读取当前32份日志，共9787行/710103字节；12份训练日志均确认稳定partition和`short_stem1d`。全量硬错误扫描计数0，未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN、Inf或参数错误。
- GPU0–3各仅1个本任务compute，SM占用约24%–32%；GPU4–7无compute。当前判定为`RUNNING_HEALTHY_8_OF_12_COMPLETE_THROUGH_EPOCH_95_99`，剩余4行尚未进入论文last10正式窗口。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

### 06:51只读心跳监控

- 直接N607预检通过；row1–8保持完整完成，GPU0–3的4个原queue与row9–12原trainer PID仍持续运行，无重启或进程替换。
- 12份`metrics.json`均可解析且epoch连续：row1–8各200epoch并有`final`字段；row9–12分别完整写入epoch`183,180,184,181/200`且无`final`。当前`PAPER-EVAL-SUMMARY=8/12`、`FINAL-TEST=8/12`、成功完成job=`8/12`。
- 完整读取当前32份日志，共11128行/802876字节；12份训练日志均确认稳定partition和`short_stem1d`。全量硬错误扫描计数0，未见Traceback、RuntimeError、CUDA OOM、Killed、AssertionError、FileNotFound、NaN、Inf或参数错误。
- GPU0–3各仅1个本任务compute，SM占用约29%–38%；GPU4–7无compute。当前判定为`RUNNING_HEALTHY_8_OF_12_COMPLETE_THROUGH_EPOCH_180_184`，剩余4行即将进入论文last10正式窗口，但仍未形成完整结果。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-15 06:59 short-stem完整Table III结果

- 完成性：12/12行均自然完成epoch200；12个`QUEUE-JOB-END status=0`、12个`PAPER-EVAL-SUMMARY`和12份含`FINAL-TEST`的训练日志齐全，全部queue与trainer均已退出。
- 小型证据包仅含32份日志、12份`metrics.json`、manifest及scheduler TSV，不含dataset/checkpoint。远端与本地SHA256均为`b71f3066a71508ad3807ce3b7547d36fe639d80ae69c21b79ba77e793e57eccf`；本地路径为`analysis_tmp/paper_repro_riei_table3_shortstem_confirm_seed1337_20260715_043000/final_0700`。
- 完整分析覆盖12份metrics的epoch1–200，共2400个epoch；读取32份日志共11444行/826541字节。12份训练日志均确认稳定partition和`short_stem1d`，全量硬错误扫描计数0。正式数值固定论文epoch191–200 last10；target中间峰值未用于选型。

|行|训练接收机→测试接收机|论文均值±SD|short last10均值±SD|差值|相对image变化|final|source val last10|last10 loss|CE/MI/IE/FN|论文±2SD|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
|1|`1-1,7-7`→`1-19`|77.88±2.23%|77.51±0.83%|-0.37pp|-2.01pp|78.94%|99.82±0.01%|-2.9215|0.0196/0.0276/2.4785/3.994|命中|
|2|`1-1,8-8`→`1-19`|79.43±1.66%|79.06±1.59%|-0.37pp|-3.02pp|81.77%|99.85±0.01%|-2.9305|0.0178/0.0223/2.4792/4.204|命中|
|3|`1-1,14-7`→`1-19`|66.09±0.67%|68.95±2.33%|+2.86pp|-5.13pp|68.60%|99.82±0.01%|-2.9007|0.0208/0.0307/2.4653/2.951|未命中|
|4|`7-7,8-8`→`1-19`|70.51±3.53%|79.32±1.31%|+8.81pp|+0.71pp|80.17%|99.83±0.01%|-2.9235|0.0195/0.0239/2.4764/3.725|未命中|
|5|`7-7,14-7`→`1-19`|77.35±1.53%|78.25±0.80%|+0.90pp|+2.99pp|78.25%|99.85±0.01%|-2.9037|0.0200/0.0340/2.4705/3.139|命中|
|6|`8-8,14-7`→`1-19`|75.48±1.21%|72.29±1.12%|-3.19pp|+6.48pp|72.60%|99.87±0.01%|-2.8492|0.0234/0.0304/2.4243/3.223|未命中|
|7|`1-1,1-19`→`14-7`|71.91±2.08%|74.64±0.85%|+2.73pp|+2.87pp|75.31%|99.79±0.02%|-2.9319|0.0179/0.0219/2.4801/4.146|命中|
|8|`1-1,7-7`→`14-7`|68.33±2.37%|68.68±1.21%|+0.35pp|-1.81pp|70.04%|99.82±0.01%|-2.9215|0.0196/0.0276/2.4785/3.993|命中|
|9|`1-1,8-8`→`14-7`|73.54±1.27%|70.29±1.58%|-3.25pp|-0.49pp|72.19%|99.86±0.01%|-2.9305|0.0178/0.0223/2.4792/4.204|未命中|
|10|`1-19,7-7`→`14-7`|73.52±3.15%|70.33±1.62%|-3.19pp|+6.71pp|68.85%|99.73±0.02%|-2.9178|0.0199/0.0237/2.4718/3.928|命中|
|11|`1-19,8-8`→`14-7`|72.05±2.71%|67.03±0.90%|-5.02pp|-5.02pp|69.08%|99.87±0.01%|-2.9392|0.0154/0.0193/2.4815/4.265|命中|
|12|`7-7,8-8`→`14-7`|73.46±2.00%|66.83±1.14%|-6.63pp|-1.71pp|69.02%|99.83±0.02%|-2.9245|0.0196/0.0235/2.4769/3.740|未命中|

### 判定与下一问题

- short-stem复现均值为`72.76%`，论文均值为`73.30%`，聚合偏差仅`-0.53pp`；逐行MAE=`3.14pp`、RMSE=`4.03pp`、论文`±2SD`命中`7/12`。相对ImageNet式stem完整mean矩阵，MAE从`4.34pp`降至`3.14pp`，命中从`6/12`升至`7/12`，但仍未达到`MAE≤3pp且命中≥10/12`，正式结论保持`NOT_REPRODUCED`。
- short stem显著修复row3、6、10，却在row4、11、12保留方向相反的大误差；12行平均变化仅`+0.05pp`，说明它主要重分配receiver组合误差，而非简单抬高整体准确率。所有行source validation为`99.73%–99.87%`，训练loss与数值状态稳定，剩余问题不是欠拟合、崩溃或数据partition漂移。
- 论文未公开SGD是否使用momentum。当前实现为momentum0，Eq.(20)–(21)只能证明梯度更新顺序，不能排除常用的momentum SGD。下一轮先在误差最大的row3、4、11、12比较固定`momentum=0.9`与既有momentum0同row结果；其余配置保持short stem、mean及稳定partition不变。预注册门槛为四行MAE降低且至少3/4行绝对误差下降；未通过则拒绝momentum并转向固定partition的模型seed敏感性诊断。

## 2026-07-15 07:15 momentum0.9最小受控诊断设计

- 新launcher：`code/scripts/launch_riei_momentum09_probe_20260715.sh`；预定run ID为`paper_repro_riei_momentum09_probe_seed1337_20260715_071500`。只覆盖row3、4、11、12，各使用一个GPU，共4个job。
- 相对short-stem完整矩阵，唯一训练变量为`RIEI_SGD_MOMENTUM=0.9`；seed1337、稳定partition、short stem、CE/MI/IE mean、no-RMS、no-feature-norm、学习率`1e-4`、200epoch和paper last10均保持不变。既有momentum0同row last10写入scheduler manifest作为预注册control。
- 本地`bash -n`通过；dry-run完整展开4个job、4个capacity gate、4条momentum0.9、4条short-stem和4条mean命令。launcher根目录、Git镜像及非Git快照SHA256均为`ee692628dd8725f923f9aa7ba1edd8ddfa98a86cb700fdeca4e8a32c2f9455d5`；快照位于`code/snapshots/paper_repro_riei_momentum09_probe_seed1337_20260715_071500/`。
- Git提交为`23ca1c2 Probe RIEI SGD momentum parity`。07:07直接N607预检与实时process/CWD/cmdline检查通过：GPU0–7均无compute，目标run/log目录不存在；计划GPU0–3各新增1个训练，满足`existing_compute+planned_peak=1≤2`。
- 仅同步新launcher；远端SHA256与本地一致，远端`bash -n`及4-job dry-run通过。正式命令为`bash code/scripts/launch_riei_momentum09_probe_20260715.sh --launch --gpu-ids 0,1,2,3 --max-train-per-gpu 2`。

### 正式启动与4分钟健康检查

- 正式命令按上述记录执行；launcher再次确认GPU0–3的`current=0`、`planned_peak=1`、`total_peak=1≤2`。4个queue PID为`1012674,1012676,1012678,1012684`，4个trainer PID为`1012778,1012785,1012786,1012792`。
- 健康检查时row3、4、11、12分别完整写入epoch`11,10,10,10/200`，4份`metrics.json`均连续。12份当前日志共359行/43841字节，4份训练日志均确认稳定partition、`short_stem1d`和`sgd_momentum=0.9`。
- 全量硬错误扫描计数0；GPU0–3各仅1个本任务compute，SM占用约19%–31%，GPU4–7无compute。当前判定为`RUNNING_STARTUP_HEALTHY_THROUGH_EPOCH_10_11`，该诊断尚未产生paper last10结果。SSH/SCP短连接全部退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

### 07:21只读心跳监控

- 直接N607预检通过；4个原queue与4个原trainer PID均持续运行，无重启或进程替换。row3、4、11、12分别完整写入epoch`33,31,33,31/200`，4份`metrics.json`均连续且无`final`字段。
- 完整读取12份当前日志，共748行/69845字节；4份训练日志均确认稳定partition、`short_stem1d`和`sgd_momentum=0.9`。`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、成功完成job=0；全量硬错误扫描计数0。
- GPU0–3各仅1个本任务compute，SM占用约29%–30%；GPU4–7无compute。当前判定为`RUNNING_HEALTHY_THROUGH_EPOCH_31_33`，不是momentum正式选型结果。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

### 07:51只读心跳监控

- 直接N607预检通过；4个原queue与4个原trainer PID持续运行，无重启或进程替换。row3、4、11、12分别完整写入epoch`119,116,124,117/200`，4份`metrics.json`均连续且无`final`字段。
- 完整读取12份当前日志，共1968行/158138字节；4份训练日志均确认稳定partition、`short_stem1d`和`sgd_momentum=0.9`。`PAPER-EVAL-SUMMARY=0`、`FINAL-TEST=0`、成功完成job=0；全量硬错误扫描计数0。
- GPU0–3各仅1个本任务compute，SM占用约22%–31%；GPU4–7无compute。当前判定为`RUNNING_HEALTHY_THROUGH_EPOCH_116_124`，不是momentum正式选型结果。SSH短连接已退出，本地`ssh.exe=0`、N607 TCP22已建立连接`=0`。

## 2026-07-15 08:21 momentum0.9完整诊断结果

- 完成性：4/4个job均自然完成epoch200；4个`QUEUE-JOB-END status=0`、4个`PAPER-EVAL-SUMMARY`和4份含`FINAL-TEST`的训练日志齐全，原queue与trainer均已退出。完整分析覆盖4份metrics的800个epoch及16份日志的3184行/250476字节，硬错误0。
- 小型证据包只含日志、metrics、manifest和scheduler TSV，不含dataset/checkpoint。远端与本地SHA256均为`506c707dd989ed1e9b4c47d28d37eb513014bf34cfacca60240c9ccbab4b6f4d`；本地路径为`analysis_tmp/paper_repro_riei_momentum09_probe_seed1337_20260715_071500/final_0821`。
- 正式比较固定epoch191–200 last10；稳定partition、short stem、mean、no-RMS、no-feature-norm、seed1337和学习率均与momentum0对照一致，唯一变量为`momentum=0.9`。

|行|论文均值±SD|momentum0 last10|momentum0.9 last10均值±SD|差值|绝对误差改善|final|source val last10|last10 loss|CE/MI/IE/FN|论文±2SD|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
|3|66.09±0.67%|68.95%|63.94±1.89%|-2.15pp|+0.71pp|63.15%|99.88%|-2.9791|0.00084/0.00093/2.48421/2.862|未命中|
|4|70.51±3.53%|79.32%|76.64±2.56%|+6.13pp|+2.68pp|78.04%|99.89%|-2.9784|0.00106/0.00093/2.48378/2.961|命中|
|11|72.05±2.71%|67.03%|63.20±0.79%|-8.85pp|-3.83pp|62.17%|99.90%|-2.9794|0.00068/0.00088/2.48429/4.075|未命中|
|12|73.46±2.00%|66.83%|63.59±1.22%|-9.87pp|-3.24pp|61.88%|99.89%|-2.9784|0.00105/0.00092/2.48380/2.956|未命中|

- momentum0.9四行MAE=`6.75pp`，高于momentum0的`5.83pp`；仅row3、4降低绝对误差，row11、12恶化。它同时未满足“MAE降低”和“至少3/4行改善”两个预注册条件，因此判定为`REJECT_MOMENTUM09`，不运行完整12行。
- momentum0.9把CE/MI压到约`1e-3`但未改善跨receiver泛化，说明剩余误差不是常用momentum缺失导致。下一项未公开因素是模型初始化seed；必须先把数据partition seed与模型seed解耦，否则直接改变`--seed`会同时改变样本划分，无法归因。

## 2026-07-15 08:30固定partition模型seed诊断设计

- 新增`--wisig_split_seed`，负值保持旧行为并复用`--seed`；paper queue显式传递`WISIG_SPLIT_SEED`。该改动只允许固定数据partition，不改变既有默认实验语义。
- 新launcher为`code/scripts/launch_riei_modelseed_probe_20260715.sh`，预定run ID为`paper_repro_riei_modelseed_probe_split1337_20260715_083000`。固定split seed1337、short stem、SGD momentum0、mean、no-RMS、no-feature-norm、200epoch和paper last10，只比较model seed0与42；row3、4、11、12各两个seed，共8个job。
- 每个候选seed分别与既有seed1337四行对照。预注册门槛为四行MAE低于seed1337的`5.83pp`且至少3/4行降低绝对误差；只有满足门槛的单一固定seed才允许进入完整Table III确认。若两者均通过，按四行MAE较低者进入确认；若均失败，则拒绝把seed作为复现修复。
- 当前状态为本地实现待验证；启动前必须完成`ssr-gpu`聚焦测试、`bash -n`、8-job dry-run、根目录非Git快照、Git提交、直接N607预检、实时容量门、SCP hash核对和远端dry-run。

### 08:33本地验证与版本准备

- `ssr-gpu`下根目录及Git镜像均完成`py_compile`和3个聚焦测试文件，结果各为`8 passed`；根目录仅出现既知`.pytest_cache`无写权warning。两份launcher均通过`bash -n`，root/Git的code、queue与launcher内容逐字一致。
- 8-job dry-run确认8个job、8个capacity gate、model seed0与42各4行、所有命令`WISIG_SPLIT_SEED=1337`、momentum0和short stem。新增测试同时验证负值split seed回退旧`--seed`语义，以及split/model seed均写入`split_info`。
- 根目录不是Git仓库；快照为`code/snapshots/paper_repro_riei_modelseed_probe_split1337_20260715_083000/`。SHA256：`cvs_data.py=e3b80d1d...`、paper queue=`d2ad5621...`、launcher=`24208548...`、test=`5daaa78f...`。
- 下一步仅在提交本任务文件、重新执行直接N607预检并确认每GPU`existing_compute+planned_peak≤2`后，才同步这3个运行文件并启动；不得根据已结束momentum run的空闲GPU假设当前容量。

### 08:32同步前容量门

- Git镜像仅提交本任务6个文件，提交为`e1e869a Probe fixed-partition RIEI model seeds`；根目录报告、traceability及运行文件与Git镜像内容一致，未纳入工作树中其他任务的修改或未跟踪artifact。
- 直接N607预检通过；实时process/CWD/cmdline检查训练进程0，GPU0–7 compute均为0，目标run/log目录不存在。计划每GPU新增1个训练，因此`existing_compute+planned_peak=1≤2`。
- 同步前远端SHA256：`cvs_data.py=a2093e0a...`、paper queue=`5a1fe1f1...`；新launcher不存在。计划同步本地`baselines/common/cvs_data.py`、`run_wisig_paper_scope_queue.sh`及`code/scripts/launch_riei_modelseed_probe_20260715.sh`到N607同路径，随后核对SHA256、远端`py_compile`、`bash -n`和8-job dry-run。
