# DRIFT论文v2协议修复与性能优化报告

## 任务信息

- 实验ID：`paper_repro_drift_v2_repair_20260714_141223`
- 建立时间：2026-07-14 14:12:23+08:00
- 操作者：Codex
- 目标：定位DRIFT复现性能偏低的剩余原因，在不干预N607现有任务的前提下完成本地协议修复和下一轮8候选矩阵设计，使最终五随机种子结果与当前DRIFT论文v2同协议结果一致。
- 当前状态：`LOCAL_REPAIR_READY_REMOTE_DEFERRED`
- 远端边界：N607上的`paper_repro_fixopt_riei_drift_seed1337_20260714_105000`与`phase1_dgleo_corepath8_20260714`仍在运行。本轮不远端同步、不启动、不终止、不覆盖任何产物。

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
