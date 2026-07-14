# phase1_dgleo_corepath8_20260714

## 启动前记录

- 时间：2026-07-14；操作者：Codex。
- 目标：在HierCore8 r3诊断基础上，直接修复“泛化依赖宽known域、层级软门近似reject-all、source overflow口径冲突、U_s利用失效、concat_sa闭集监督重复和open梯度未闭环到z_id”。
- 协议：Phase1 source-only；labeled/unlabeled/source-val=0.08/0.72/0.20，rho_label=0.10；不使用目标receiver或真实unknown训练。
- 权重：120epoch，只保留E120 final权重；held-out clean和leo_weak测试只在训练结束后执行。
- 星地视图：保留concat_sa；训练和评估均使用leo_clear_weak、leo_low_elev_weak、leo_rain_weak，因此本轮只证明增强族内压力鲁棒性。
- 对比：同seed 713301的phase1_dgleo_hiercore8r3_20260713及P0Closed8 C6。

## P0修改

|问题|直接修改|必须观察的闭环证据|
|---|---|---|
|feat_joint显式混入DAC/PA缺陷|新增ungated feat_cls身份核心，TX logits仍使用feat_joint|z_id leakage下降且overall/strict不降|
|层级概率连乘reject-all|新增smooth-min训练组合与known core TPR损失|core hard TPR>=0.85后proxy/bridge仍下降|
|动态batch门漂移|新增TX×domain×clean/sat窗口冻结reference bank|reference anchor持续非零，固定窗口内边界不漂移|
|overflow口径冲突|direct metric与source episode共用16/18度半径上限，leave-domain目标不高于半径|source_overflow_hard与legacy overflow同步下降|
|batch temporal gate与shuffle冲突|按sample identity建立跨epoch伪标签稳定记忆|temporal pass、CE selected和trusted core同步上升|
|concat_sa重复TX CE和clean KD污染|clean TX CE与sat TX CE各计一次；clean teacher只看clean|sat性能不降，closed/open梯度冲突下降|
|预算统计被closed-only路径污染|预算、分组范数和冲突投影限定到id_backbone|budget_scope_shared_zid_path=1，放大倍数<=8|
|final自注册tail reference|禁止final epoch参与reference选择|reference_epoch<120；p99 delta不是构造性0|

## 实验矩阵

|候选|GPU|机制|
|---|---:|---|
|CP_R0_R3_REPLAY|0|同seed旧机制对照，仅包含tail reference修复|
|CP_R1_ID_CORE|1|只切换ungated feat_cls|
|CP_R2_FROZEN_GATE|2|feat_cls+冻结reference+smooth-min+known TPR|
|CP_R3_OVERFLOW_ALIGNED|3|R2+18度同口径overflow/leave-domain|
|CP_R4_U_EPOCH_BANK|4|R3+U_s跨epoch temporal bank|
|CP_R5_CONCAT_DEDUP|5|R4+concat_sa监督去重与clean-only teacher|
|CP_R6_FULL_STABLE|6|全机制稳定版+z_id路径预算+最大8倍缩放|
|CP_R7_FULL_AGGRESSIVE|7|全机制激进版+16度半径+更高open预算|

所有候选每GPU一个、同seed、120epoch。R0/R1保留旧目标合同用于归因；R2-R7统一p95/p99/CVaR/proxy目标54度/70度/56度/0.35。

## 成功标准

- DG：overall>=89.45%，strict UDU>=86.18%，clean receiver floor不低于75%。
- Satellite第一阶段：sat strict mean>=74%，RX8或receiver×scenario floor>=62%，且RX9/RX10下降不超过0.5pp；长期目标仍为78%/73%。
- Known安全：hard core TPR>=0.85；只在该条件下统计proxy/bridge改善。
- Open-set代理：fixed p99<=57.29度、proxy<=0.30、bridge<=0.15、source-episode overflow第一阶段<=0.92。
- U_s：后20epoch temporal pass和trusted core>=10%，direct active>=80%，CE selected不再接近0。
- 泄漏：receiver/day/channel excess第一阶段<=0.45/0.15/0.25。
- 任何候选不能以低known覆盖换取低proxy；不能以sat平均值换取弱receiver继续恶化。

## 本地版本与验证

- Git仓库：E:\type10-7\github_publish\CVS-RFFI-repo。
- Git提交：356195bd8d8c41126bde56f035e0e2c830371c2c。
- train_ssdg.py SHA256：68F11ADFE5A680717DABF79B3233522629CF7B431FF823395F0F78A104E3FB96。
- losses.py SHA256：C74BB63DC0BE820C832D84D17874B862D4111231B136F6426FE14B38D74599A2。
- model.py SHA256：AFC6E6266A09FD5F5BE967FED85254C6C92FA0241A0336FD5FFA3EB12AA1C417。
- launcher SHA256（实验身份分组容量门控后）：5954ABAA9816C6E4ACF4971652066008B5198BA539F7E01E13E6F32FE723826D。
- dualguard16 SHA256（实验身份分组容量门控后）：A5701DB4FAD802930A6236A5BEA179E919A9511BA25F177FF4F3E28F6EF75835。
- py_compile通过。
- focused Phase1/模型/launcher测试80项通过；扩展集合中95项通过，另有5个既有federated fixture因缺少fed_fishr_bank属性失败，与本批未修改模块无关。
- dry-run生成8条唯一命令，GPU0-7各1条，checkpoint_selection=final_only。

## N607计划

- 远端根：/home/szu2070436088/2510044040/CV-SincNet。
- Python：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python。
- run：runs/phase1_dgleo_corepath8_20260714/<candidate>。
- logs：logs/phase1_dgleo_corepath8_20260714/<candidate>.out。
- 2026-07-14 09:12 CST重新按PID、完整命令行、CWD、输出目录和GPU核验：CorePath8正式实验为0个；服务器另有13个`paper_reproduction` RIEI/DRIFT训练，GPU0-4每卡2个、GPU5-7每卡1个。此前“服务器无训练”的结论是过滤范围错误，已撤销。
- 正式launcher新增全量NVIDIA compute-client资源门控：GPU0-7必须全部为0个compute进程才整体启动8个Phase1候选；等待期间每60秒写入逐卡occupancy，不把RIEI/DRIFT进程记作Phase1。该条件与底层`dualguard16`空卡合同保持一致。
- 2026-07-14 09:29 CST已启动资源等待launcher，PID 235974；命令为`python -u code/scripts/launch_phase1_dgleo_corepath8_20260714.py --run-id phase1_dgleo_corepath8_20260714 --wall-hours 10 --poll-seconds 30 --launch-settle-seconds 3 --max-total-compute-per-gpu 2 --resource-wait-timeout-seconds 10800 --resource-poll-seconds 60`。首个slot快照为GPU0-4各2个、GPU5-7各1个，因此正式Phase1训练保持0个，未抢占资源。
- launcher日志：`logs/phase1_dgleo_corepath8_20260714_launcher.out`；资源满足后才创建8个候选训练进程，训练墙钟10小时从实际候选启动后计时。
- 10:27 CST第一次资源等待结束时，外层门控允许GPU0和GPU3各保留1个复现进程，但底层`dualguard16`按空卡合同拒绝启动，报`requires empty GPUs`；8个Phase1候选均未创建。该不一致已修复为全空门控，失败日志保留为诊断证据。
- 空卡合同修复提交：`de05a6d`。远端hash和`py_compile`通过，启动前RIEI/DRIFT=0、Phase1=0、NVIDIA compute client=0。
- 正式重启launcher PID 262012，日志`logs/phase1_dgleo_corepath8_20260714_launcher_retry1.out`。候选于10:29:08-10:29:29依次落地：GPU0/262027、GPU1/262503、GPU2/262967、GPU3/263429、GPU4/263892、GPU5/264355、GPU6/264818、GPU7/265283。
- `scheduler_events.tsv`记录8条LAUNCHED，`nvidia-smi pmon`确认每卡恰好1个主训练进程。`/proc`中另外64个同命令PID为8个候选各自的DataLoader worker，不计为独立实验或GPU compute client。
- 根据后续审查，空卡不是必要条件。新门控按GPU主compute PID逐个读取`/proc/<pid>/cmdline`：同run_id目标实验已存在则阻断；PID身份不可读则fail-closed；每卡总compute client最多2个；已有1个无关实验时，仅在free memory>=10000MiB时允许再启动本矩阵的1个候选。启动前和每个候选`Popen`前均重新检查，并导出`prelaunch_gpu_snapshot.json`。
- 门控容量单位进一步由“CUDA PID数”修正为“实验身份组数”：优先按`run_id+candidate_id`或`output_dir`归组，缺少结构化参数时按Linux process group归组。同一实验的多个CUDA子进程只占1个槽位；身份不可读仍fail-closed。每卡已有1个无关实验且free memory>=10000MiB时允许启动1个目标实验；已有2个实验、已有同run目标或显存不足时阻断。
- 同步映射：`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase1_dgleo_dualguard16_20260712.py`→`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_dualguard16_20260712.py`；`E:\type10-7\github_publish\CVS-RFFI-repo\code\scripts\launch_phase1_dgleo_corepath8_20260714.py`→`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_corepath8_20260714.py`。
- Git提交：`a304d60`（实验身份分组门控）和`9fc4396`（同步映射记录）。两份launcher已通过direct SCP同步；远端SHA256与本地一致，远端`py_compile`通过。
- 本地focused测试15项通过，覆盖“1个无关实验可共享”“同一实验多个CUDA PID只计1组”“第3个实验阻断”“同run重复和低显存阻断”；`py_compile`通过。
- 10:30 CST另一个独立的`paper_repro_repaired_riei_drift_seed1337_20260714_103000`批次在CorePath8启动后落地，每卡新增1个RIEI/DRIFT实验。10:42 CST现场为每卡2个真实compute实验（CorePath8 1个+RIEI/DRIFT 1个），free memory约20GB，未发现Traceback、RuntimeError或CUDA OOM。新门控会正确阻断假设中的第3个实验；这不代表CorePath8进程混淆，也不需要停止任一现有任务。
- 远端实测快照中每卡均正确识别为2个实验身份组；以当前run_id检查时为`target=1,unrelated=1`并阻断重复，以新run_id检查时为`target=0,unrelated=2`并阻断第3个实验。SSH/SCP结束后本地`ssh.exe=0`且N607:22无残留连接。
- 10:42 CST CorePath8八候选均持续训练，`metrics_epoch.csv`已有4-6个epoch行；正式科学结论仍等待final-only训练和最终评估完成。
- 当前已运行的CorePath8是在8卡空闲时启动，不受门控代码更新影响，也不会重启或中断。新行为用于后续矩阵。
- 预计单候选训练约5.1-5.5小时，final评估/probe/diagnostic导出约0.5-1小时；8卡并行总墙钟预计6-7小时，硬上限10小时。

## 运行冒烟验证

- 2026-07-14 09:20 CST在GPU7第二个允许槽位启动`CP_R7_FULL_AGGRESSIVE_SMOKE_E2D`，PID 229198；同卡原PID 221959属于RIEI复现，二者通过命令行和输出目录隔离。
- 两个epoch完整执行，`metrics_epoch.csv`共2行，无Traceback、RuntimeError或非有限训练指标；final-only权重写入成功。终态exit code 5来自两轮诊断不满足promotion合同，不是运行时错误。
- E2冻结reference bank生效：version=1、active_epoch=2、anchor_count=672，U_s reference anchor=336；`zid_path`梯度预算标志为1；跨epoch temporal pass从0升至0.116，pseudo CE selected从0升至3/7056。
- E2仍暴露核心科学风险：dynamic DM proxy_vaccept=0.139、bridge=0.044，但known hard core accept仅0.149，legacy proxy_vaccept约0.589、bridge=1.000，source-episode overflow=1.000。因此不能把DM下降解释成拒识改善；正式120epoch必须同时检查known TPR、旧proxy与fixed endpoint。
- 前三次冒烟仅在参数预检阶段失败，依次为关闭sat评估却保留sat best metric、非法best metric名称、heavy-eval interval设为0；均未进入训练，也未占用持续GPU资源。

## 正式运行早期遥测

- 10:46 CST八候选均存活且无Traceback、RuntimeError或CUDA OOM，已完成5-7个epoch；R6 E5单epoch约219秒。按当前并行负载估算训练主体约7-8小时，final评估另需约0.5-1小时，仍受10小时硬墙钟保护。
- R6 E5固定reference bank已激活：anchor_count=672、active_epoch=2；`zid_path`预算标志=1，open/closed平衡梯度范数约13.16/54.10，梯度冲突率约0.460，冲突投影后cosine由0.049升至0.097。
- 当前不能判定open-set改善：dynamic DM proxy/bridge约0.141/0.043，但known core accept仅0.131；旧口径proxy_vaccept约0.600、bridge_accept=1.000，source_episode_overflow=1.000。动态门低值仍主要由known拒收形成，必须等待known core TPR恢复并验证旧代理同步下降。
- 早期几何中source-episode p95/p99/CVaR约58.73/66.22/63.18度，U_s temporal pass约0.121；source-val DG health best约98.63且当前回落仅0.054pp。E10前heavy evaluation尚未开始，不能外推strict UDU、satellite floor或最终拒识质量。

### E18联合快照

|候选|epoch|DM core accept|DM proxy|legacy proxy|legacy bridge|source overflow|source p95/p99|U_s direct active|source-val sat mean/floor|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|R0 replay|19|0.586|0.220|0.625|1.000|0.666|58.66/75.32|0.968|91.53/90.20|
|R1 ID core|25|0.282|0.168|0.578|1.000|0.972|49.59/56.11|0.000|90.44/88.99|
|R2 frozen gate|18|0.113|0.150|0.609|1.000|0.953|48.96/55.20|0.016|90.92/89.45|
|R3 overflow aligned|18|0.109|0.151|0.612|1.000|1.000|48.11/54.08|0.000|90.85/89.30|
|R4 U epoch bank|18|0.110|0.153|0.612|1.000|1.000|48.17/54.14|0.000|90.90/89.43|
|R5 concat dedup|18|0.112|0.155|0.611|1.000|1.000|48.13/54.01|0.000|90.85/89.35|
|R6 full stable|18|0.121|0.158|0.611|1.000|1.000|49.47/55.69|0.000|91.00/89.58|
|R7 aggressive|18|0.115|0.162|0.614|1.000|1.000|48.67/54.47|0.000|90.90/89.49|

该快照只用于训练机制诊断。R2-R7的p95/p99虽下降，但known core accept仅0.109-0.121，旧proxy仍约0.609-0.614且bridge=1，不能解释为安全接收域收紧。R3的18度同口径使overflow暴露为1.0，说明结构性local component仍未把跨receiver/day/channel样本纳入稳定核心。R4虽将temporal pass提高到约0.110，但U_s direct仍为0，表明epoch bank没有闭环到direct loss路由。表中sat为source-val增强族内结果，不是strict UDU或最终satellite test。

### E51-E73中期诊断

|候选|epoch|DM core accept|legacy proxy/bridge|source overflow|source p95/p99|radius/inter|U_s direct active|
|---|---:|---:|---:|---:|---:|---:|---:|
|R0 replay|59|0.586|0.597/1.000|0.709|64.31/78.61|1.163|0.746|
|R1 ID core|73|0.219|0.589/1.000|0.819|42.26/46.26|1.314|0.000|
|R2 frozen gate|52|0.108|0.606/1.000|0.841|44.28/49.30|1.751|0.000|
|R3 overflow aligned|52|0.104|0.607/1.000|1.000|43.11/47.65|1.695|0.000|
|R4 U epoch bank|51|0.103|0.609/1.000|1.000|43.31/47.24|1.708|0.016|
|R5 concat dedup|58|0.095|0.607/1.000|1.000|41.82/45.90|1.714|0.048|
|R6 full stable|58|0.094|0.607/1.000|1.000|42.83/47.23|1.765|0.016|
|R7 aggressive|59|0.092|0.608/1.000|1.000|41.87/45.57|1.687|0.032|

R3-R7出现明确的几何伪改善：source p95/p99下降，但overflow不动、known core覆盖继续坍缩、radius/inter显著恶化，旧proxy/bridge也未同步下降。说明当前18度半径合同与smooth-min/reference gate共同收缩了动态局部尺度，却没有学到跨receiver/day/channel稳定交集。R1单独使用`feat_cls`可将overflow降至约0.819，但core覆盖仍低且U_s direct完全失活；R0保留较高core和U_s路由，却以p99扩张和高legacy风险为代价。最终结论仍需E120、final-only测试和endpoint artifact。

## 最终状态与证据完整性

- 8/8候选均完成120epoch；每个`metrics_epoch.csv`恰好120行，final held-out evaluation、terminal、tail safety和diagnostic prototype均完整，最终选择权重均为E120`final_ssdg.pth`。
- 全量CSV无非有限值；stdout未发现fatal、Traceback、CUDA OOM、Killed或RuntimeError。`scheduler_summary.json`记录8个终态、0缺失，整组墙钟6.84小时。
- 8/8终态均为`NON_PROMOTABLE_GUARD_BLOCKED`、return code 5；这是科学/安全门阻断，不是运行崩溃。
- 每个候选的endpoint export均为false，prototype均为`DIAGNOSTIC_COMPLETE`且`diagnostic_only=true`。fusion component字段已真实导出，但没有正式硬拒识边界。
- best source-val epoch均为E112；final相对best的val gap仅0.03-0.10pp。tail reference均为E112-E118，p99 reference-to-final delta绝对值不超过0.44度，说明“final自注册reference”和训练后期tail扩张假阴性已修复。
- 本轮final测试不再冻结为相同行：overall跨度89.746-90.347、strict UDU跨度85.968-86.823、sat strict mean跨度71.973-72.444，证明此前测试结果不变的问题已修复。

## Final泛化与星地主表

|候选|overall|strict UDU|clean RX floor|sat mean/floor|sat strict mean/floor|sat RX floor|相对C6结论|
|---|---:|---:|---:|---:|---:|---:|---|
|C6基线|89.953|86.183|74.633|-|- /72.575|56.550|前序联合最优|
|R0 replay|89.746|85.968|75.650|77.859/76.750|72.336/71.165|57.267|DG回落，sat未升|
|R1 ID core|90.270|86.623|75.608|77.718/76.517|72.352/71.058|56.450|clean DG提升，sat弱点恶化|
|R2 frozen gate|90.308|86.693|75.642|77.656/76.469|72.174/70.882|56.833|DG提升，sat回落|
|R3 overflow aligned|90.347|86.823|75.608|77.603/76.375|72.217/70.905|56.842|overall/strict最佳，sat未升|
|R4 U epoch bank|90.148|86.595|75.692|77.427/76.202|71.973/70.685|56.583|U机制未带来泛化收益|
|R5 concat dedup|90.046|86.610|75.858|77.617/76.383|72.246/70.823|56.092|clean floor最佳，sat floor下降|
|R6 full stable|90.054|86.617|75.358|77.785/76.537|72.444/71.052|56.250|sat strict最佳但仍低于C6|
|R7 aggressive|90.071|86.597|75.492|77.572/76.362|72.228/70.873|55.833|激进版最弱sat floor|

相对C6，R3把overall/strict提高0.394/0.640pp，R5把clean receiver floor提高1.225pp；但最佳sat strict mean仍比C6低0.131pp，最佳sat receiver×scenario floor仅57.267%，距离阶段目标73%仍差15.733pp。当前提升只发生在clean overall/strict和部分clean receiver，不发生在satellite mean/floor，也没有修复最弱receiver。

## Final拒识潜力主表

|候选|fixed p95/p99°|fixed proxy/bridge|tail/overflow accept|clean hard core TPR|fixed min-inter/ratio|legacy overflow|legacy proxy/bridge|
|---|---:|---:|---:|---:|---:|---:|---:|
|C6基线|28.41/57.29|0.360/0.260|0.640/0.624|约0.503 core accept|-/4.38|0.956|-|
|R0 replay|42.55/69.19|0.181/0.000|0.490/0.191|0.000|25.12/1.66|0.670|0.620/1.000|
|R1 ID core|16.20/24.90|0.080/0.086|0.199/0.213|0.010|2.78/4.67|0.731|0.590/1.000|
|R2 frozen gate|16.22/25.94|0.182/0.255|0.288/0.276|0.111|2.45/5.59|0.717|0.613/1.000|
|R3 overflow aligned|13.48/21.99|0.165/0.241|0.262/0.268|0.061|1.94/5.51|1.000|0.613/1.000|
|R4 U epoch bank|13.40/21.84|0.162/0.237|0.257/0.267|0.058|1.78/5.65|1.000|0.612/1.000|
|R5 concat dedup|12.43/20.57|0.153/0.230|0.245/0.252|0.013|1.40/5.68|1.000|0.612/1.000|
|R6 full stable|13.18/21.13|0.164/0.237|0.265/0.249|0.051|1.81/5.55|1.000|0.608/1.000|
|R7 aggressive|11.87/19.22|0.145/0.214|0.238/0.247|0.034|1.81/5.95|1.000|0.609/1.000|

R1-R7的fixed p99显著下降不是有效open-set改善。clean hard core TPR只有0.010-0.111，远低于0.85；异类local component最小间隔坍缩到1.4-2.8度，radius/inter恶化到4.67-5.95。模型通过缩小/重叠局部接收组件拒绝绝大多数known，得到低p99和低proxy，而不是提取稳定RFF不变量。旧proxy仍约0.59-0.62、旧bridge仍为1.0，确认最终风险代理没有同步改善。

R1/R2的legacy source overflow从C6的0.956降至0.731/0.717，是本轮唯一真实方向性信号；但它与hard core TPR坍缩、min-inter坍缩和U_s失活同时发生，属于“压缩known但不可接收”的诊断改善。R3加入18度同口径后overflow重新变为1.0，证明简单收紧半径合同不能解决跨receiver/day/channel结构。

## U_s、梯度与泄漏

- R4-R7的epoch bank把temporal pass从约0.008提高到约0.114，但final trusted core仅0.014%-0.028%、outside为99.8%以上，U direct active只有0%-1.6%。因此“跨epoch记忆生效”没有转化为无标签open-set梯度。
- Source unlabeled本质上是known样本；当前三态把几乎全部U_s划为`outside_reject`，导致已知无标签样本被错误隔离。outside不能继续作为repulsive unknown负样本，应改为stop-gradient quarantine。
- R6/R7已证明`z_id`路径预算统计生效，但final原始open norm仅约0.90，对应closed norm约157-161；8倍缩放后`B_os_eff`仍仅0.047-0.049，低于0.16。控制器不能从饱和/reject-all损失中恢复有效梯度。
- leakage probe全部完整，但receiver/day/channel excess仍为0.609-0.682/0.166-0.233/0.405-0.451。`feat_cls`没有成为域不变身份核，说明移除显式DAC/PA拼接不足以完成身份/域解耦。
- R0保留较高U direct和`B_os_eff=0.175`，但fixed clean hard core TPR仍为0、legacy proxy/bridge仍高；单纯增加open梯度也会沿错误边界方向优化。

## 机制归因

|机制|是否生效|结论|
|---|---|---|
|ungated`feat_cls`|部分|overall/strict提升，但leakage未降，未形成invariant core|
|冻结reference+smooth-min+TPR loss|失败|reference存在且不漂移，但hard core TPR仍坍缩，loss与fixed endpoint不一致|
|18/16度overflow同口径|负例|把legacy overflow推回1.0，证明阈值收紧不能替代结构学习|
|跨epoch U bank|执行生效、目标失败|temporal pass提高，但三态边界把known U_s几乎全部判为outside|
|concat_sa监督去重|无明显收益|clean floor单点提高，sat strict/floor没有改善|
|`z_id`路径预算|统计生效、优化失败|真实open梯度随饱和损失衰减，预算控制器无法维持下限|
|tail reference排除final|成功|reference epoch<120，p99 delta非构造性0且无晚期扩张|
|final-only测试|成功|8个候选产生可区分final测试结果，不再复用冻结测试行|
|fusion/local component导出|字段完整但仅诊断|component字段存在，endpoint boundary仍未导出|

## 四象限与候选决策

|candidate|泛化结论|拒识潜力|主要风险|可否Stage2真实unknown评估|下一步动作|
|---|---|---|---|---|---|
|R0|overall/strict回落|legacy overflow下降但p99扩张|clean hard TPR=0、legacy风险高|否|仅保留旧路径和U梯度对照|
|R1|clean DG明显提升|overflow降至0.731、fixed proxy最低|hard TPR=0.010、U完全失活、min-inter坍缩|否|保留`feat_cls`诊断，不单独推进|
|R2|clean DG提升|overflow最低0.717|hard TPR仅0.111、ratio=5.59、旧proxy未降|否|作为下一轮正覆盖修复底座|
|R3|overall/strict最佳|无有效改善|overflow=1、hard TPR=0.061|否|18度合同负例，不继续单扫半径|
|R4|无额外泛化收益|temporal执行改善|U仍99.8% outside、direct idle|否|重写U三态语义后再验证|
|R5|clean floor最佳|fixed tail较短|hard TPR=0.013、sat floor下降|否|保留concat去重，不作为主候选|
|R6|sat strict组内最佳|无联合改善|sat仍低、hard TPR=0.051、梯度预算失效|否|保留z_id预算观测，改梯度归一化|
|R7|无稳定提升|fixed p99最低|ratio最差5.95、hard TPR=0.034|否|激进负例，停止加权路线|

本轮没有第一象限候选。R2是下一轮最有价值的机制底座，但只因它同时保留clean DG并把legacy overflow降到0.717；它不是可promotion候选。

## 下一轮P0/P1方案

### P0

1. 将open预算改为“known正覆盖优先”：从E1启用高权重open几何，但当任一clean/sat hard core TPR<0.85时，至少70% open预算只分配给invariant-core alignment、hard known coverage和异类component margin；proxy/bridge/outside排斥梯度暂停，杜绝reject-all。
2. 直接对冻结`endpoint_accept_v1`训练同口径surrogate：reference/threshold按前一epoch source-train episodic bank版本化并detach，loss读取与final artifact相同的component、radius、density和hard reason code；禁止当前batch动态门作为promotion代理。
3. 把`feat_cls`升级为真正invariant core：新增TX条件下receiver/day/channel adversarial confusion、leave-one-source-receiver class-center alignment和clean-sat同样本一致性；local component只表示残差/尾部，不得作为宽接收域并集。
4. 对最危险异类component直接施加angular margin/CVaR，目标min-inter>=20度且radius/inter<=3；禁止仅压半径而不推开异类中心。
5. 重写U_s三态：`trusted_core`使用teacher置信度+冻结component一致性+class-conditioned top quota；`ambiguous_tail`只做一致性/密度；`outside_quarantine`stop-gradient，不作为unknown负样本。成功标准为trusted>=10%、ambiguous>=5%、direct active>=80%。
6. 用梯度归一化而非单纯loss放大：按`z_id`路径把positive-core/inter/proxy/U四组梯度归一到目标预算，再做冲突投影；硬限制单组scale并监控实际post-projection open/closed ratio>=0.12。

### P1

1. 增加source leave-one-receiver episodic TX CE和worst receiver×`leo_weak` CVaR，直接优化弱receiver而不是只看sat平均值。
2. 保留`concat_sa`、clean/sat各一次TX CE、clean-only teacher和三类`leo_weak`评估；同时明确它仍是同增强族压力测试，不声称跨信道族泛化。
3. endpoint artifact只有在hard core TPR、legacy风险、leakage和DG/sat联合达标后导出；继续保持final-only checkpoint和三入口parity。

下一轮8候选应依次验证：R2 replay、positive-first hard coverage、invariant-core confusion、hard inter-component margin、U known-quarantine、normalized z_id gradient、worst-receiver satellite episodic、全机制稳定版。第一阶段成功标准：overall>=90%、strict>=86.5%、clean floor>=75.5%、sat strict mean>=73%、sat RX floor>=60%、clean/sat hard TPR>=0.85、legacy overflow<=0.90、legacy proxy<=0.55、legacy bridge<0.90、fixed p99<=57.29度、ratio<=3、U direct active>=80%、receiver/day/channel excess<=0.45/0.15/0.25。任一低known覆盖换低proxy、低min-inter换短p99或sat平均换弱receiver均判失败。

## 声明边界

本轮只能评价Phase1 DG、同`leo_weak`增强族压力、known几何、source-only proxy风险、U_s执行和diagnostic prototype。不得声明真实unknown FAR/FPR95、Stage2 old/new性能、endpoint部署成功或真实星地跨信道泛化。
