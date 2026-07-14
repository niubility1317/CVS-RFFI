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
- 本地focused测试15项通过，覆盖“1个无关实验可共享”“同一实验多个CUDA PID只计1组”“第3个实验阻断”“同run重复和低显存阻断”；`py_compile`通过。
- 10:30 CST另一个独立的`paper_repro_repaired_riei_drift_seed1337_20260714_103000`批次在CorePath8启动后落地，每卡新增1个RIEI/DRIFT实验。10:42 CST现场为每卡2个真实compute实验（CorePath8 1个+RIEI/DRIFT 1个），free memory约20GB，未发现Traceback、RuntimeError或CUDA OOM。新门控会正确阻断假设中的第3个实验；这不代表CorePath8进程混淆，也不需要停止任一现有任务。
- 10:42 CST CorePath8八候选均持续训练，`metrics_epoch.csv`已有4-6个epoch行；正式科学结论仍等待final-only训练和最终评估完成。
- 当前已运行的CorePath8是在8卡空闲时启动，不受门控代码更新影响，也不会重启或中断。新行为用于后续矩阵。
- 预计单候选训练约5.1-5.5小时，final评估/probe/diagnostic导出约0.5-1小时；8卡并行总墙钟预计6-7小时，硬上限10小时。

## 运行冒烟验证

- 2026-07-14 09:20 CST在GPU7第二个允许槽位启动`CP_R7_FULL_AGGRESSIVE_SMOKE_E2D`，PID 229198；同卡原PID 221959属于RIEI复现，二者通过命令行和输出目录隔离。
- 两个epoch完整执行，`metrics_epoch.csv`共2行，无Traceback、RuntimeError或非有限训练指标；final-only权重写入成功。终态exit code 5来自两轮诊断不满足promotion合同，不是运行时错误。
- E2冻结reference bank生效：version=1、active_epoch=2、anchor_count=672，U_s reference anchor=336；`zid_path`梯度预算标志为1；跨epoch temporal pass从0升至0.116，pseudo CE selected从0升至3/7056。
- E2仍暴露核心科学风险：dynamic DM proxy_vaccept=0.139、bridge=0.044，但known hard core accept仅0.149，legacy proxy_vaccept约0.589、bridge=1.000，source-episode overflow=1.000。因此不能把DM下降解释成拒识改善；正式120epoch必须同时检查known TPR、旧proxy与fixed endpoint。
- 前三次冒烟仅在参数预检阶段失败，依次为关闭sat评估却保留sat best metric、非法best metric名称、heavy-eval interval设为0；均未进入训练，也未占用持续GPU资源。

## 声明边界

本轮只能评价Phase1 DG、星地压力、known几何、source-only proxy风险、U_s执行和diagnostic prototype。不得声明真实unknown FAR/FPR95、Stage2成功或endpoint部署成功。
