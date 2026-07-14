# DRIFT论文v2协议修复与性能优化报告

## 任务信息

- 实验ID：`paper_repro_drift_v2_repair_20260714_141223`
- 建立时间：2026-07-14 14:12:23+08:00
- 操作者：Codex
- 目标：定位DRIFT复现性能偏低的剩余原因，在不干预N607现有任务的前提下完成本地协议修复和下一轮8候选矩阵设计，使最终五随机种子结果与当前DRIFT论文v2同协议结果一致。
- 当前状态：`V206_FIVE_SEED_CONFIRM_RUNNING_HEALTHY`
- 远端边界：DRIFT v2发现矩阵已完整退出；Phase1仍可能运行。五seed确认仅在实时容量门通过后启动，不终止、不覆盖或影响Phase1产物。

## 论文版本与成功口径

|口径|训练接收机|目标接收机|训练实现|汇总方式|论文目标|
|---|---|---|---|---|---:|
|DRIFT v1历史口径|`1-1,14-7,7-7`|七个未见接收机|batch=64|单次run最后5个checkpoint均值|75.62%|
|DRIFT v2当前主口径|`1-1,14-7,7-7`|同七个未见接收机|batch=256；随机抽取；仅信道均衡预处理|5个不同随机种子的最终epoch均值|73.54%|

当前主复现以[DRIFT arXiv v2](https://arxiv.org/html/2510.09405v2)为准；v1的75.62%只保留为历史对照，不再与v2配置混为一个成功口径。

成功条件：先在8候选单种子诊断矩阵中找到稳定且接近73.54%的配置，再固定同一配置运行5个不同随机种子；主报告使用每个seed最终epoch的七接收机联合结果，五seed均值与73.54%的绝对差不超过2个百分点，同时报告SD和每接收机结果。

## 已有结果与问题定位

`paper_repro_fixopt_riei_drift_seed1337_20260714_105000`的8个DRIFT候选均已完成。最佳最后5轮为`D02_cap3500_mse020`的66.06±1.52%，与v1目标相差9.56个百分点、与v2目标相差7.48个百分点。候选中出现单epoch 68.95%至70.96%的峰值，说明模型并非完全失效，但训练末段稳定性和协议一致性仍不足。

|优先级|现有实现|当前论文v2要求|影响判断|修复/验证|
|---|---|---|---|---|
|P0|每个TX/RX组合固定取最前800/200条|随机抽取800条训练和200条测试|样本序列可能带采集时序偏置；seed未真正控制论文子集|新增`--wisig_paper_sample_strategy random`，训练、验证、总体测试和逐接收机测试使用可复现随机抽样|
|P0|对每包I/Q额外做RMS归一化|v2声明信道均衡是唯一信号级预处理|会消除幅度相关射频指纹，且属于论文未声明处理|新增`--no-wisig_rms_normalize`，v2候选明确关闭|
|P0|DRIFT固定batch=64|v2使用batch=256|中心损失按domain统计，batch大小直接改变每批domain覆盖、统计方差和每epoch优化步数|wrapper支持`DRIFT_BATCH_SIZE`，v2严格候选使用256并保留64配对消融|
|P0|单seed最后5轮均值对75.62%|v2为5个随机种子最终epoch均值对73.54%|比较对象不一致，可能误判复现失败或成功|发现阶段`paper_eval_last_n=1`；确认阶段固定最佳配置跑5 seeds|
|P1|负MSE采用特征维求和且目标无下界，仅用cap抑制爆炸|论文公式为负平方L2，但未公开数值稳定实现|修复前已出现MSE、特征范数爆炸；cap3500虽提升到66.06%仍波动|8候选同时比较raw-sum、cap3500/4000、feature-mean和grad clip；非公式改动均标为实现假设而非论文事实|
|P2|当前没有官方代码可核对优化器细节|论文仅给结构、损失、超参数和训练设置|无法证明cap值是作者实现|论文严格候选与数值稳定候选分开命名、分开裁决|

## Traceability正向表

|需求/论文条目|代码承载|验证证据|状态|
|---|---|---|---|
|v2随机抽取每TX/RX训练800、测试200|`code/dataset_wisig.py::make_wisig_drift_day1_split`|单元测试校验同seed复现、不同seed变化、train/val无重叠，逐接收机named tests与aggregate test使用完全相同样本|本地PASS|
|v2仅使用信道均衡预处理|`baselines/common/cvs_data.py`的`--wisig_rms_normalize`|dry-run命令含`--no-wisig_rms_normalize`，split_info记录`rms_normalize=false`|本地PASS|
|v2 batch=256|`run_wisig_paper_scope_queue.sh`的`DRIFT_BATCH_SIZE`|矩阵与wrapper内层dry-run严格候选含`--batch_size 256`|本地PASS|
|v2最终epoch、5 seeds|发现矩阵`paper_eval_last_n=1`；后续确认矩阵固定最佳配置5 seeds|最终报告汇总5个不同seed的final row|发现命令本地PASS，远端未启动；确认阶段未设计|
|不超过每GPU两训练|新矩阵启动前读取N607实时process/CWD/cmdline/GPU；每GPU仅增加1个训练|capacity gate输出`current+planned_peak<=2`|远端当前繁忙，已延期|

反向审计结论：本次代码变化均可回溯到上表P0协议缺口或P1数值稳定诊断；没有修改`项目.md`定义的CVS Stage2场景，没有改变RIEI Table III路径，没有加入目标接收机训练数据，也没有把cap/mean/grad clip伪装为论文事实。heartbeat已更新为先完成fixopt，再运行v2发现矩阵，最后才运行5-seed确认矩阵。

## 计划中的8候选诊断矩阵

|候选|batch|抽样|RMS|负MSE|cap|其他|目的|
|---|---:|---|---|---|---:|---|---|
|V201_strict_raw|256|random|off|sum|0|无|v2论文字面严格基线，允许只读观察是否再次发散|
|V202_v2_cap3500|256|random|off|sum|3500|无|将当前最佳稳定器迁移到v2协议|
|V203_rms_control|256|random|on|sum|3500|无|隔离关闭RMS的贡献|
|V204_front_control|256|front|off|sum|3500|无|隔离随机抽样的贡献|
|V205_batch64_control|64|random|off|sum|3500|无|隔离batch=256的贡献|
|V206_mean_impl|256|random|off|mean|0|无|检验常见MSELoss按元素均值实现假设|
|V207_cap4000_lmse015|256|random|off|sum|4000|`lambda_mse=0.015`|检验较弱分离压力和较宽cap|
|V208_cap3500_clip5|256|random|off|sum|3500|`grad_clip=5`|检验末段波动是否来自梯度尖峰|

## 本地变更与验证

计划变更：

- `baselines/common/cvs_data.py`：增加论文抽样策略与RMS归一化开关，并只在`drift_day1`路径接入。
- `code/dataset_wisig.py`：实现DRIFT训练/验证/测试的seed控制随机抽样并写入split_info。
- `baselines/drift/train_cvs.py`：把抽样和RMS口径打印到完整训练日志。
- `run_wisig_paper_scope_queue.sh`：暴露DRIFT batch、MSE reduction、grad clip、抽样和RMS环境参数，保持旧默认兼容。
- `code/scripts/launch_drift_v2_repair_matrix_20260714.sh`：8候选顺序安全队列与容量门。
- `tests/test_baseline_training_behaviors.py`：新增协议抽样/归一化回归测试。

验证结果：

- `ssr-gpu`环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，PyTorch 2.10.0+cu128。
- `python -m py_compile baselines/common/cvs_data.py code/dataset_wisig.py baselines/drift/train_cvs.py`：PASS。
- `python -m unittest tests.test_baseline_training_behaviors.BaselineWiSigPaperProtocolTest`：6/6 PASS。
- `bash -n run_wisig_paper_scope_queue.sh`：PASS。
- `bash -n code/scripts/launch_drift_v2_repair_matrix_20260714.sh`：PASS。
- 8-job dry-run：PASS，8张GPU各`planned_peak=1`，严格v2命令包含batch256、random、RMS off、final epoch口径。
- wrapper内层dry-run：PASS，确认参数展开为`--batch_size 256 --wisig_paper_sample_strategy random --no-wisig_rms_normalize --mse_reduction sum --mse_cap 3500 --grad_clip_norm 5`。

版本说明：根目录`.git`缺少`HEAD`，不能作为有效Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`。镜像时发现承载面已有较新的IQ中心化、Meta-SSL抽样和interval checkpoint测试，已保留这些不相关改动并把合并后的文件反向同步回根目录，没有以根目录旧副本覆盖它们。

Git承载面实现提交：`f302b1f repair DRIFT v2 reproduction protocol`。提交仅包含本任务8个文件；承载面中既有的`mitigating_da_rootcause`改动和`local_artifacts`未纳入提交。

## 远端启动条件与预期产物

仅当现有fixopt队列完全退出后才允许：运行`tools/n607_ssh_preflight.ps1`；用实时进程/CWD/cmdline和`nvidia-smi`确认每GPU`existing_compute+planned_peak<=2`；本地验证完成；SCP后hash一致；远端`bash -n`与dry-run均通过；最后启动唯一新run。14:19检查显示当前8个RIEI训练仍到epoch156–157、4个仍排队，硬错误为0，因此继续延期。预期产物包括每候选完整200epoch日志、`metrics_epoch.csv`、`metrics.json`、最终checkpoint、scheduler manifest和PID表。

完成后必须完整分析全部200epoch日志，不得只读tail；主表保持同一候选的七接收机结果、aggregate final、loss/feature norm、异常和裁决在同一行。

## 2026-07-14 16:02远端同步与启动前门控

- 前序fixopt已20/20完成且全部queue退出；完整分析结论为DRIFT最佳66.06±1.52%、RIEI期刊last5均值56.07%，两者均未复现。
- 再次执行直接N607预检通过。实时GPU进程显示GPU0、2–7各有1个Phase1训练，GPU1无训练；本矩阵每GPU计划新增1个，因此峰值分别为2或1，满足`existing_compute+planned_peak<=2`。
- 已同步并核对本地/远端SHA256：`cvs_data.py=a2093e0a...`、`dataset_wisig.py=8bf22bd8...`、`drift/train_cvs.py=01f47bba...`、paper queue=`2ba90874...`、v2 launcher=`9a57d98b...`。
- 远端两个shell脚本`bash -n`通过；8-job dry-run完整展开V201–V208，确认batch、random/front、RMS、reduction、cap、lambda和grad clip均按矩阵设置。
- 计划正式命令：`bash code/scripts/launch_drift_v2_repair_matrix_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。
- 独立run/log根：`paper_reproduction/runs/paper_repro_drift_v2_repair_20260714_141223`和`paper_reproduction/logs/paper_repro_drift_v2_repair_20260714_141223`；不会覆盖fixopt或Phase1产物。

## 2026-07-14 16:04启动与健康检查

- 正式命令按计划执行成功；实际容量门：GPU0、2–7均`current=1+planned=1=2`，GPU1为`0+1=1`，未超过2/GPU。
- launcher PID：V201=`420333`、V202=`420336`、V203=`420341`、V204=`420348`、V205=`420357`、V206=`420374`、V207=`420393`、V208=`420417`；对应训练PID=`420472,420488,420512,420539,420559,420569,420575,420579`。
- 约4–5分钟健康检查时8个训练均存活，进度分别为epoch63、63、58、67、45、82、69、63/200；batch64 control较慢符合预期。
- 日志配置确认：v2主候选为batch256、random、RMS off、last1；RMS/front/batch64/mean/lambda和clip对照均正确生效。
- 8个日志未发现Traceback、RuntimeError、参数错误、OOM、Killed、NaN或Inf。GPU0、2–7各为Phase1＋本任务，GPU1仅本任务；未干预Phase1。
- 当前状态：`RUNNING_HEALTHY`，尚不是artifact-complete或复现成功。

## 2026-07-14 16:15 heartbeat只读监控

- 8个训练PID全部存活；当前进度：V201=159、V202=161、V203=140、V204=163、V205=102、V206=180、V207=163、V208=157/200。batch64 control继续最慢，符合每epoch步数更多的预期。
- 8份`metrics.json`均已持续写入，但尚无候选完成`PAPER-EVAL-SUMMARY`，完成数0/8。
- GPU0、2–7各为1个Phase1＋1个本任务，GPU1仅本任务；每GPU训练数不超过2。
- 全部当前日志未见Traceback、RuntimeError、参数错误、OOM、Killed、NaN或Inf；本轮保持只读，未同步、重启或修改远端状态。

## 2026-07-14 16:45发现矩阵完整结果

- 8个训练与queue均已退出，`metrics.json`与`PAPER-EVAL-SUMMARY`均为8/8，远端硬错误扫描为0。已拉取24个小型日志/调度文件与8份`metrics.json`，完整读取6490行日志和全部1600条epoch记录；8个候选均覆盖epoch1–200。
- 正式口径固定为epoch200 final，不使用best-val或目标域单epoch峰值。论文v2同协议目标为5seed final均值73.54%。V206在单seed final达到72.68%，绝对差0.86pp，是唯一进入±2pp的候选，因此选为五seed确认配置。

|候选|机制|final/%|相对73.54/pp|val@200/%|train loss@200|MSE|center|GRL|feature norm|z_tx norm|z_rx norm|裁决|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|V201|raw sum、无cap|64.17|-9.37|99.81|-579.50|-29020.61|0.36|0.90|113.37|84.27|97.40|未复现；MSE无界扩张|
|V202|sum、cap3500|63.35|-10.19|99.81|-69.40|-3498.50|3.70|0.53|20.48|51.19|43.91|未复现|
|V203|V202＋RMS|60.45|-13.09|99.75|-69.71|-3499.96|0.30|0.28|19.89|45.75|45.10|未复现|
|V204|V202＋front抽样|40.10|-33.44|99.22|-68.08|-3495.11|3.73|1.76|19.86|49.46|43.85|未复现；固定前段抽样最差|
|V205|V202＋batch64|69.76|-3.78|99.83|-68.85|-3499.82|0.36|1.14|33.20|71.33|50.19|接近但未进入阈值|
|V206|mean、无cap|72.68|-0.86|99.83|0.39|-35.32|0.05|1.10|35.38|54.42|1.18|单seed命中；进入五seed确认|
|V207|sum、cap4000、lambda0.015|67.54|-6.00|99.83|-60.00|-4000.00|0.08|0.00|19.42|47.86|51.14|未复现|
|V208|V202＋clip5|66.89|-6.65|99.78|-69.52|-3499.90|0.41|0.47|20.96|48.32|41.91|未复现|

|候选|1-19|19-2|2-1|2-19|20-1|7-14|8-8|七receiver aggregate final|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|V201|64.58|62.58|70.83|44.33|60.92|69.17|76.75|64.17|
|V202|64.67|69.92|76.75|49.42|49.17|82.17|51.33|63.35|
|V203|61.25|55.67|74.25|53.92|55.33|68.58|54.17|60.45|
|V204|42.58|44.08|51.42|24.42|30.42|54.75|33.00|40.10|
|V205|65.83|70.50|87.50|53.58|64.50|67.67|78.75|69.76|
|V206|70.67|68.33|95.58|67.00|57.58|83.50|66.08|72.68|
|V207|62.33|78.00|67.92|60.58|63.08|73.75|67.08|67.54|
|V208|74.00|67.25|73.83|49.17|55.17|90.33|58.50|66.89|

### 诊断结论与五seed确认设计

- 根因证据集中指向MSE实现尺度：`sum`使负MSE项到-3500 cap或无界到-29020，训练总loss被该项主导；`mean`把MSE稳定到-35.32，同时final提升到72.68%。RMS、front抽样、batch缩小、cap和梯度裁剪均不能解释或消除主要差距。
- V206仍呈receiver异质性（57.58%–95.58%），单seed不能宣称复现。确认矩阵固定V206全部协议参数，只改变seed=`1337,2024,3407,4242,7777`，正式指标为五个epoch200 final的均值与SD；成功阈值为均值相对73.54%绝对差不超过2pp且无训练崩溃。
- 本地新增`code/scripts/launch_drift_v2_confirm_v206_20260714.sh`；独立run为`paper_repro_drift_v2_confirm_v206_20260714_164900`，计划GPU=`1,6,7,0,2`各1个job，启动前必须重新执行实时容量门。
- 本地验证：`bash -n code/scripts/launch_drift_v2_confirm_v206_20260714.sh`通过；5-job dry-run完整展开且每个job严格复用V206参数，仅seed与GPU不同。

## 2026-07-14 16:54五seed确认启动前门控

- Git承载提交：`9ee5c71`；确认launcher SHA256=`4d5601a61b2b2c40c947bdf8fc3034702c467e3ffba539fd2f79ec2a3833d266`。计划同步到N607的`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_drift_v2_confirm_v206_20260714.sh`。
- 直接N607预检通过。实时GPU compute PID仅GPU2=`262967`、GPU3=`263429`、GPU4=`263892`，均属于既有Phase1；GPU0、1、5、6、7为空。本任务计划GPU1、6、7、0、2各新增1个，峰值分别为1、1、1、1、2，满足每GPU不超过2。
- 发现矩阵8个训练与queue已全部退出；未干预、终止或覆盖Phase1。下一步仅同步新launcher，核对远端hash、`bash -n`和5-job dry-run后启动独立确认run。

## 2026-07-14 16:55五seed确认启动与健康检查

- SCP后远端SHA256与本地一致，远端`bash -n`通过，5-job dry-run完整展开。正式命令：`bash code/scripts/launch_drift_v2_confirm_v206_20260714.sh --launch --gpu-ids 1,6,7,0,2 --max-train-per-gpu 2`。
- 实际容量门：GPU1、6、7、0均`current=0+planned=1=1`，GPU2为`1+1=2`，未超过2/GPU。launcher PID依seed为1337=`445180`、2024=`445183`、3407=`445188`、4242=`445193`、7777=`445200`；训练PID分别为`445317,445305,445319,445332,445333`。
- 健康检查时5个训练均存活，已到epoch15–16附近，5份`metrics.json`均持续写入；完整配置确认batch256、random、RMS off、MSE mean、无cap、lambda_mse0.020、final last1均正确生效。
- 日志未见Traceback、RuntimeError、OOM、Killed或NaN。GPU2为Phase1＋本任务，其余本任务GPU各1个训练；未影响Phase1。当前仅为`RUNNING_HEALTHY`，尚未形成五seed复现结论。
