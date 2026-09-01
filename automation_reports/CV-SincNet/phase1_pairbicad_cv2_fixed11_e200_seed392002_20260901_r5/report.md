# Phase1 PairBiCAD-CV2正式E200矩阵r5

## 当前状态

- 状态：`ANALYZED / PROTOCOL_NONCONFORMING_REFERENCE_ONLY`。24/24行完成训练、严格重建和四场景评估；但本run把统一`V=0.30`拆成`V_cal=0.15/V_select=0.15`，与当前`项目.md`禁止拆分`V`的规定冲突，因此只能作为已执行配置下的工程与科学参考，不能晋级为当前协议的正式冠军。
- run ID：`phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5`。
- Git提交：`5f785287935f2b58f4e7a4f95b37341de2a176a0`；已push并独立核对远端分支OID一致。
- r4已固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，输出根和partial artifact保留；r5使用全新release和不可覆盖run根。

## 故障修复

r4的两条`CV2-B0`行均完成200epoch并保存final checkpoint，但启动器在训练返回0后只检查正式artifact是否存在，没有调用正式评估流程，因此确定性缺少`checkpoint_runtime.json`、`diagnostics.json`、clean和三种LEO评估。

r5恢复完整闭合：定位非空final checkpoint；从独立`source_loro_selection.json`和200行`metrics_epoch.jsonl`绑定终止update与E200身份；严格重建模型和trainer runtime；仅使用held-out source receiver执行clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；逐场景写JSON和日志；再次验证runtime、重建键、source-only访问标记与四场景闭合，最后才写`ARTIFACTS_COMPLETE.json`。任何环节失败均写技术失败，不伪造完成。

## 冻结矩阵与协议

- 候选：`CV2-B0/B1/B2/B3/D0/D1/D2/D3/T0/T1/T2/T3`，12种。
- fold：1、8；seed：`392002`；共24行。
- 每行从头训练200epochs；终止方式仅为`epochs=200`，命令中无`--bicad_optimizer_updates`且不使用6500updates。
- ManySig源域receiver集合`[1,3,4,6,8]`；fold1训练`[3,4,6,8]`并留出receiver1，fold8训练`[1,3,4,6]`并留出receiver8；训练日为day1/day2/day3。
- 源域角色比例：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；严格Phase1 source-only，不访问Phase2、target、support、query或truth。
- 现行增强协议：`concat_sat_ce_only`、`lambda_sat_cons=0`，三种`LEO_WEAK`课程为`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`。
- GPU0—7每卡最多2个本run训练进程；16行并发、8行排队，不超过用户指定容量。

## 本地与远端验证

- 新增回归测试先复现“训练成功但未调用正式评估”的缺口，修复后worker必须实际调用`evaluate_final_checkpoint`才允许完成。
- `code/tests/phase1_bicad_xr`完整测试通过：476项通过，仅3条既存PyTorch弃用警告。
- launcher、测试和远端release入口均通过编译；`git diff --check`通过；dry-run读回24行、12候选、fold1/8、seed392002、全部E200、每GPU2槽和8行排队。
- 历史真实checkpoint无query smoke通过：严格重建missing/unexpected/shape mismatch均为0，optimizer step完成，clean及三种LEO前向均有限，所有target/Phase2/support/query/truth访问标记为false。
- r5正式release对r4的真实`CV2-B0-F1-S392002` E200 checkpoint执行独立闭合smoke，结果`PASS`：`runtime_valid=true`、四场景齐全、`source_only=true`。smoke使用复制到独立日志目录的checkpoint，不修改r4 partial artifact。
- Luna一次聚焦P0/P1审查结论：`无P0/P1`。

## Release、路径与正式命令

- release归档：`E:\\type10-7\\local_artifacts\\phase1_pairbicad_cv2_e200_5f785287.tar.gz`。
- 单一归档本地/远端SHA256：`0190c01045aac24bb903eaac036a26638a016e17464adc5e45f9a596d5e8a30b`，已独立核对一致。
- 远端release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_cv2_e200_5f785287`。
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5`。
- dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5.dispatcher.log`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD为上述release；使用普通账户`szu2070436088`，禁止管理员账户。
- 正式命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/launch_phase1_pairbicad_cv2_screen24_20260901.py --run-id phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r5 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_pairbicad_cv2_e200_5f785287 --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --gpu-capacities 0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2`。

## 预期artifact与停止规则

每行必须闭合200epoch telemetry、final checkpoint、严格checkpoint/trainer runtime重建、final/EMA/SWAD一次`V_select`选择、`checkpoint_runtime.json`、`diagnostics.json`、clean与三种LEO弱场景独立JSON/日志以及`ARTIFACTS_COMPLETE.json`。

仅允许因数据/query越权、错误candidate/fold/receiver/day/seed/epoch、输出冲突、错误release/CWD、命令无法运行、无合法prediction/artifact闭合、同一确定性异常至少重复两行或进程归属不清而停止精确run进程树。低性能、中间指标、缺少非必要receipt/hash或报告字段不得停止、重启、热补丁或选择性重跑。若出现预登记系统技术失败，必须保留partial artifact，在本地Git修复并验证后以新release、新run ID重新发布；不得原地重启。

## 正式启动证据

- 启动时间：2026-09-01 11:11 CST；dispatcher PID`3472323`。
- dispatcher父PID为1；`/proc/3472323/cwd`独立读回精确指向release`phase1_pairbicad_cv2_e200_5f785287`，cmdline精确绑定r5、ManySig、普通账户Python和GPU容量`0:2,...,7:2`。
- 直属worker为16个，16个`train.log`已创建；GPU0—7各有2个本run计算进程，共16个，启动检查时利用率6%—40%、显存679—1,271MiB，处于数据加载/早期计算阶段。
- `ARTIFACTS_COMPLETE=0`、`TECHNICAL_FAILURE=0`；在本run目录未检出Traceback、RuntimeError、ValueError、OOM或final-artifact确定性异常。
- 判定：`RUNNING`。启动绑定和资源分配符合冻结矩阵；不得因早期利用率或性能停止、重启或热补丁。

## 2026-09-01 15:43只读监控

- 直连预检通过：普通账户`szu2070436088`、项目根和8张RTX 3090可见；未使用管理员账户。
- dispatcher PID`3472323`仍存活，父PID为1，cmdline继续精确绑定release`phase1_pairbicad_cv2_e200_5f785287`、r5 run根、ManySig和GPU容量`0:2,...,7:2`。
- 24行目录全部存在；`ARTIFACTS_COMPLETE=16/24`，`TECHNICAL_FAILURE=0`。已完成16行均有200行`metrics_epoch.jsonl`。
- 8个直属worker仍绑定未完成行：`CV2-D3-F1/F8`、`CV2-T1-F1/F8`、`CV2-T2-F1/F8`、`CV2-T3-F1/F8`，seed均为`392002`，fold均为1或8。
- 运行中trainer按周期checkpoint持续增长：D3已到`u56500`，T1已到`u36500`，T2已到`u46500`，T3已到`u31000`；这些行在结束前尚未写最终metrics，符合该入口的写出方式。
- GPU2—5处于轻量计算/阶段切换，GPU6—7约90%—92%利用率；GPU0—1检查瞬间空闲，但对应T1 checkpoint刚更新，未形成无进展证据。
- 全run的`train.log`中未检出Traceback、CUDA OOM、`RuntimeError:`或`ValueError:`确定性指纹。判定保持`RUNNING`，无需用户操作，不停止、不重启、不热补丁。

## 2026-09-01 16:02进度与ETA

- `ARTIFACTS_COMPLETE=18/24`、`TECHNICAL_FAILURE=0`；D3的fold1/8已完成`63000/63000 updates`、E200和四场景闭合。
- 剩余6行均在运行，统一计划终点为`63000 updates`：T1约`42000/42500`，T2约`55500/56000`，T3约`37500/37500`。
- 相对15:43检查，T1约推进5500—6000、T2约推进9000—9500、T3约推进6500—7000 updates；checkpoint时间更新至16:01—16:02，未出现停滞。
- 按这段真实吞吐估算，T2预计约16:20前后结束；最终瓶颈为T1/T3，预计约17:10—17:25达到训练终点。D3从最后source-LORO记录到`ARTIFACTS_COMPLETE`约需100秒，因此24行全部artifact闭合预计在17:15—17:30。
- 全量读取24行日志、metrics、runtime和四场景JSON并完成分析、报告与Git发布还需约45—75分钟；若无技术异常，最终`ANALYZED`预计约18:00—18:45。

## 2026-09-01 16:25只读监控

- `ARTIFACTS_COMPLETE=20/24`、`TECHNICAL_FAILURE=0`；T2的fold1/8已完成`63000/63000 updates`、E200和四场景闭合。
- 剩余4个直属worker精确绑定T1与T3的fold1/8；T1已到`49000/50000`，T3两行均到`45500`，计划终点仍为`63000`，checkpoint和source-LORO记录持续增长。
- GPU2、3、6、7仍有本run活动；GPU0、1、4、5已随对应行完成或阶段结束释放。未检出确定性致命错误。
- 按16:02—16:25实际速率，T1预计约17:05—17:12完成，瓶颈T3预计约17:15—17:22完成；24行artifact闭合预计保持在17:20前后。

## 2026-09-01 16:47只读监控

- 直连预检继续通过；dispatcher PID`3472323`及4个直属worker仍精确绑定r5 run根、release`phase1_pairbicad_cv2_e200_5f785287`和普通账户Python。
- `ARTIFACTS_COMPLETE=20/24`、`TECHNICAL_FAILURE=0`。剩余T1 fold1/8分别到`56500/57000`，T3 fold1/8均到`52500`，统一终点为`63000 updates`；四行source-LORO记录在检查时持续更新。
- GPU0、1、2、3、6、7可见阶段性计算，GPU4、5空闲；当前仅有4个本run worker，未超过每GPU最多2个的限制，也未影响无关进程。
- 未检出Traceback、CUDA OOM、`RuntimeError:`或`ValueError:`确定性致命指纹。判定保持`RUNNING`，不停止、不重启、不热补丁。
- 按16:25—16:47实际吞吐，T1预计约17:05—17:10完成，瓶颈T3预计约17:18—17:25完成；24行artifact闭合预计约17:20—17:30。之后完整读取与分析、报告和Git发布预计仍需45—75分钟，最终`ANALYZED`预计约18:05—18:45。

## 2026-09-01 17:10只读监控

- `ARTIFACTS_COMPLETE=22/24`、`TECHNICAL_FAILURE=0`；T1 fold8和fold1分别于17:07:57、17:09:27完成E200及四场景闭合。
- dispatcher PID`3472323`继续存活，现仅有2个直属worker，精确绑定T3 fold1/8；两行均到`59500/63000 updates`，source-LORO文件继续增长。
- GPU6、7承载剩余T3计算；其他GPU已释放或处于无本run训练负载状态。未检出确定性致命错误，也未影响无关进程。
- 按最近23分钟实际吞吐，T3训练终点预计约17:21—17:23，计入最终checkpoint与四场景评估后，24行全部artifact预计约17:24—17:28闭合；最终完整分析与Git发布预计约18:10—18:40。

# 最终实验报告

## 一、结论先行

1.运行在2026-09-01 17:24 CST结束，24/24行均为`ARTIFACTS_COMPLETE`，`TECHNICAL_FAILURE=0`。每行都有200条epoch记录、非空final checkpoint、严格checkpoint/trainer runtime重建、clean和三种`LEO_WEAK`独立评估；全量日志中未检出Traceback、CUDA OOM、`RuntimeError:`或`ValueError:`。
2.按本run预登记的旧式`V_select`均值排序，`CV2-B1`第一，`V_select=54.5052%`，clean均值`61.5389%`，三种LEO均值`49.1639%`。相对同run的`CV2-B0`，分别提高`+8.7730pp`、`+1.8278pp`和`+14.6602pp`。
3.`B1`不是稳健冠军。其fold1/fold8的`V_select`分别为`62.3526%/46.6577%`，三种LEO均值为`55.3852%/42.9426%`；clean类别floor均值只有`7.4000%`，较`B0`下降`17.8667pp`。它提高总体LEO准确率的同时，把最弱clean类别压得很低。
4.若强调均值、类别下界和两折稳定性的共同改善，`CV2-T3`是本矩阵最均衡的候选。相对静态同配置锚`T0`，`T3`的`V_select/clean/clean floor/LEO mean/LEO严格类别floor`分别提高`+2.9385/+4.2250/+21.0000/+1.1833/+2.2667pp`；两折`V_select`标准差仅`0.2268pp`，三种LEO均值标准差`0.7602pp`。
5.本run不能产生正式方法冻结。原因有三：当前协议只允许单一`V=0.30`，本run却拆分了`V_cal/V_select`；`B0`为10400updates而多数CV2候选为63000updates，结构与优化暴露量没有匹配；`B3/D0/T0`是静态同配置锚，但三者仍出现最高`3.8059pp`的`V_select`差异，说明单seed候选ID相关随机性或非确定性足以覆盖部分小增益。
6.原始设计不是严格全量落地。E200修复版登记的11项运行机制均能在对应候选路径中找到运行时证据，但VICReg、pair-delta、sparse XDC、receiver tangent、接收机前端增强、hard-LEO和三视图没有在本矩阵启用；24份`diagnostics.json`的21个聚合字段全部为`N/A`。分类性能和checkpoint闭合可信，资源、probe、梯度比等设计级诊断没有形成完整数值证据。

## 二、方法基础与数据设置

模型保留ADV3B02双骨干：共享Sinc/HF前端后分为160维身份表示`z_id`和160维域表示`z_dom`。部署快路径仍是`IQ→identity backbone→z_id→TX head`；域头、对抗判别器、pair projector、tail-risk和SWAD均为训练期组件，不进入最终快速推理图。

本run使用ManySig新数据划分，source receiver集合为`[1,3,4,6,8]`，训练日为day1/day2/day3：

- fold1：训练receiver`[3,4,6,8]`，receiver1作为source-LORO留出域；
- fold8：训练receiver`[1,3,4,6]`，receiver8作为source-LORO留出域；
- seed：`392002`；
- TX类别数：6；
- 已执行比例：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；
- 合并后仍为用户要求的`L_s/U_s/V=0.07/0.63/0.30`，但角色拆分本身不符合当前`项目.md`；
- 严格source-only：全部runtime和评估记录均为`target_access=false`、`phase2_access=false`、`support_access=false`、`query_access=false`、`truth_access=false`。

所有候选都训练200epoch。`B0`使用batch size96并得到10400updates；`B1`及后续候选使用48个物理样本的batch，严格clean/LEO拼接后网络输入为96，并得到63000updates。每4步安排一次结构化batch；在本run每折4个训练receiver时，结构化batch为24个有标签样本和24个无标签样本。

正式测试均使用同一个已选checkpoint，分别评估：

- `clean`；
- `leo_clear_weak`；
- `leo_low_elev_weak`；
- `leo_rain_weak`。

每个场景每行测试18000个样本。`accuracy`表示全部样本准确率；`class floor`表示该行、该场景6个TX类别中最低的类别准确率；`LEO mean`是三种LEO场景准确率的算术平均；`LEO严格floor`是两个fold和三种LEO场景中最低的类别floor。

## 三、11项E200机制的实际落地

|ID|机制|关键设置|实际启用范围|运行时结论|
|---|---|---|---|---|
|E200-01|固定完整训练预算|全部200epoch；科学早停只作telemetry，不中断E200|全部候选|24/24行均有epoch1—200|
|E200-02|真实clean/LEO同物理样本pair|`concat_sat_ce_only=true`；三种`LEO_WEAK`|全部候选；B0从E80开始，B1+从E1开始|`satellite_tx_ce`在审计中实际调用|
|E200-03|真实CoverageLedger|U物理ID唯一覆盖；L按TX×RX×day累计暴露|全部候选|B1+的U coverage为41.6667，最小L组暴露约15577—15606|
|E200-04|coverage warmup后平台调度|warmup完成后才允许Plateau scheduler更新|B2及后续CV候选|selection/runtime状态存在|
|E200-05|不提前硬冻结|逐训练步检查`requires_grad`，E200前不冻结|B2及后续CV候选|启用行`violations=[]`、E200仍全部可训练|
|E200-06|显式双时间尺度对抗|一次backbone前向；判别器与编码器分步；`LR_D=1.5×LR_encoder`|D1/D2/D3|conditional DANN在D1+实际调用|
|E200-07|pair梯度5%上限|`pair_identity`权重0.02、`epsilon=0.05`、最大梯度比0.05、每4步|T1/T3|pair hinge和prediction JS均实际调用|
|E200-08|困难组质量上限30%|Margin-REx/CVaR；风险权重0.6/0.3/0.1；CVaR20%；hard-group cap0.30|T2/T3|`margin_tail`每结构化步实际调用|
|E200-09|四反馈动态GRL|判别器准确率、TX margin、对抗梯度比和冲突信号共同控制；局部任务保护投影|D3|`projection_triggered=true`、`task_projection_applied=true`|
|E200-10|`V_cal/V_select`物理隔离|9000/9000，物理ID重叠0；`V_select`只做最终选模|全部候选|实现行为闭合，但违反当前单一`V`协议，不能正式晋级|
|E200-11|final/EMA/SWAD一次选模|B0/B1比较final/EMA；B3及后续可比较SWAD；不反馈训练|全部候选|每行`v_select_evaluated_once=true`|

共享关键参数如下：

- `gradient_firewall_scale=0.05`，只限制共享Sinc/HF上的域梯度；
- `lambda_sat_cons=0`；B0的`satellite CE`权重固定0.68，E80开始；B1+从0.5升至1.0并从E1开始；
- `factor_interaction_dim=24`；
- `pair_projector_dim=128`、`pair_interval=4`；
- `xdc_interval=4`，但本矩阵`sparse_xdc=false`、`xdc_kd=false`；
- `lambda_cond_xcov=0.02`，仅D2/D3启用conditional cross-covariance；
- `margin_tail_cvar_fraction=0.20`、`margin_tail_ema=0.90`；
- `ReduceLROnPlateau`为`factor=0.3`、`patience=3`、`min_lr=1e-6`；
- 旧FastTrust、pseudo label、Fishr、CSD、HCF transport、generic IQ MixUp、MixStyle和receiver tangent均关闭。

## 四、12个候选的准确含义

|候选|相对上一锚点的机制|
|---|---|
|B0|同run的ADV3B02 CORE90-matched控制：batch96、E80启用`satellite CE`、固定权重0.68，无CV2增强机制|
|B1|严格pair从E1开始、batch48、因素化域头、shared gradient firewall、`satellite CE`从0.5升至1.0|
|B2|B1+CoverageLedger收敛状态、coverage warmup、no-early-freeze、ReduceLROnPlateau|
|B3|B2+SWAD|
|D0|B3的静态同配置锚|
|D1|D0+conditional CDAN、detached adversarial、双时间尺度优化|
|D2|D1+`z_dom` TX adversary、conditional cross-covariance|
|D3|D2+动态对抗剂量、局部任务保护梯度投影|
|T0|B3的另一静态同配置锚|
|T1|T0+pair identity hinge+pair prediction JS，梯度比上限5%|
|T2|T0+Margin-REx/CVaR tail guard，困难组上限30%|
|T3|T0+T1 pair机制+T2 tail guard|

`D0`和`T0`不是新算法，而是预先静态复制B3配置的分支锚。它们用于避免运行中读取“冠军”后再动态决定后续候选，也同时暴露了相同配置在候选ID相关随机性下的波动。

## 五、候选级汇总结果

下表全部为fold1/fold8均值，单位为百分比。`±`后为两折总体标准差。

|排名|候选|V_select|clean|clean floor|clear|low elev|rain|LEO mean|LEO class-floor均值|LEO严格floor|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|B1|54.5052±7.8475|61.5389|7.4000|49.7250|48.7944|48.9722|49.1639±6.2213|12.7389|4.5000|
|2|D0|48.2819±0.4722|63.6889|25.0500|36.1889|34.9556|35.4444|35.5296±1.1019|4.5556|0.9000|
|3|T2|47.5169±2.2901|61.6639|28.2333|36.7694|34.8694|35.6278|35.7556±2.9926|7.5944|2.1333|
|4|T3|47.5011±0.2268|61.5944|34.1167|35.1806|35.3389|35.7667|35.4287±0.7602|10.1278|9.3000|
|5|D3|46.3939±1.5136|62.4361|35.3333|34.2667|33.4083|34.2972|33.9907±1.2037|3.6222|2.8000|
|6|D2|46.1444±1.2107|58.5194|9.4333|36.2222|35.0056|36.2056|35.8111±1.8296|9.6444|6.5333|
|7|D1|46.0671±2.0200|61.8944|36.0000|34.4556|32.4861|33.5806|33.5074±0.7519|3.5278|0.8333|
|8|B0|45.7322±0.2644|59.7111|25.2667|34.9611|34.0750|34.4750|34.5037±1.0259|8.0333|3.6333|
|9|T1|45.5251±0.6420|56.8500|21.4500|37.1194|36.1778|36.6556|36.6509±0.5954|2.3556|1.2333|
|10|T0|44.5626±1.3649|57.3694|13.1167|34.1500|33.9028|34.6833|34.2454±1.1676|9.8667|7.0333|
|11|B3|44.4759±2.6419|59.1167|7.6833|33.4194|32.0667|33.1639|32.8833±0.5407|4.3833|2.1000|
|12|B2|42.5751±3.0589|56.0944|13.5333|32.2306|31.3944|32.0194|31.8815±2.3111|7.1778|5.2333|

## 六、24行逐行结果

单位均为百分比。`LEO floor`为该行三种LEO场景中最低的类别floor。

|row|updates|选中checkpoint|V_select|clean|clean floor|clear|low elev|rain|LEO mean|LEO floor|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B0-F1|10400|EMA|45.4678|60.9111|34.5667|33.6444|33.1611|33.6278|33.4778|11.0667|
|B0-F8|10400|final|45.9965|58.5111|15.9667|36.2778|34.9889|35.3222|35.5296|3.6333|
|B1-F1|63000|EMA|62.3526|69.6556|9.6000|55.8389|55.0611|55.2556|55.3852|4.5000|
|B1-F8|63000|EMA|46.6577|53.4222|5.2000|43.6111|42.5278|42.6889|42.9426|17.4333|
|B2-F1|63000|final|45.6340|60.2722|26.2333|34.6500|33.5556|34.3722|34.1926|5.2333|
|B2-F8|63000|final|39.5162|51.9167|0.8333|29.8111|29.2333|29.6667|29.5704|7.3000|
|B3-F1|63000|SWAD|41.8340|54.6944|3.6333|32.8278|31.2667|32.9333|32.3426|5.2000|
|B3-F8|63000|SWAD|47.1178|63.5389|11.7333|34.0111|32.8667|33.3944|33.4241|2.1000|
|D0-F1|63000|SWAD|48.7541|63.6722|35.7667|37.3500|35.6500|36.8944|36.6315|6.0333|
|D0-F8|63000|SWAD|47.8096|63.7056|14.3333|35.0278|34.2611|33.9944|34.4278|0.9000|
|D1-F1|63000|SWAD|48.0871|66.1111|47.5667|35.6611|32.6722|34.4444|34.2593|0.8333|
|D1-F8|63000|SWAD|44.0471|57.6778|24.4333|33.2500|32.3000|32.7167|32.7556|5.0000|
|D2-F1|63000|SWAD|47.3551|59.2056|0.7333|38.0556|36.5000|38.3667|37.6407|11.8333|
|D2-F8|63000|SWAD|44.9337|57.8333|18.1333|34.3889|33.5111|34.0444|33.9815|6.5333|
|D3-F1|63000|SWAD|47.9075|67.7222|49.3333|33.2056|31.8667|33.2889|32.7870|3.5000|
|D3-F8|63000|SWAD|44.8803|57.1500|21.3333|35.3278|34.9500|35.3056|35.1944|2.8000|
|T0-F1|63000|SWAD|45.9275|59.2000|14.9667|34.8222|35.2278|36.1889|35.4130|10.7333|
|T0-F8|63000|SWAD|43.1977|55.5389|11.2667|33.4778|32.5778|33.1778|33.0778|7.0333|
|T1-F1|63000|SWAD|46.1671|59.0000|23.7667|36.0667|35.6667|36.4333|36.0556|2.4333|
|T1-F8|63000|SWAD|44.8832|54.7000|19.1333|38.1722|36.6889|36.8778|37.2463|1.2333|
|T2-F1|63000|SWAD|49.8070|63.2722|40.7333|39.6000|37.7944|38.8500|38.7481|12.6333|
|T2-F8|63000|SWAD|45.2269|60.0556|15.7333|33.9389|31.9444|32.4056|32.7630|2.1333|
|T3-F1|63000|SWAD|47.7279|63.3444|43.5333|34.0611|35.0000|34.9444|34.6685|9.3000|
|T3-F8|63000|SWAD|47.2744|59.8444|24.7000|36.3000|35.6778|36.5889|36.1889|9.6333|

## 七、与ADV3B02 CORE90-matched控制B0的比较

`B0`是同一run、同一seed、同一fold、同一source receiver/day和同一测试入口下最接近ADV3B02 CORE90的控制。它不是早期其他run的历史最高checkpoint，因此本节只作同run比较。

### B1：总体LEO最高，但类别和折间稳定性不足

相对B0，B1的`V_select/clean/LEO mean/LEO class-floor均值/LEO严格floor`分别变化：

```text
+8.7730pp / +1.8278pp / +14.6602pp / +4.7056pp / +0.8667pp
```

clean class floor却下降`17.8667pp`。按两折平均的类别准确率看，B1把LEO下的TX4从B0约`14.2%`提升到`71.2%—75.0%`，但clean TX1从`50.9833%`降到`13.6833%`。这不是均匀的域鲁棒性提升，而是类别决策边界发生大幅重排。

### T3：平均提升较小，但多项下界同时改善

相对B0，T3的`V_select/clean/clean floor/LEO mean/LEO class-floor均值/LEO严格floor`分别提高：

```text
+1.7690pp / +1.8833pp / +8.8500pp / +0.9250pp / +2.0944pp / +5.6667pp
```

T3没有B1的巨大LEO均值，但它是唯一在这组指标上全部为正、两折波动又较小的CV2候选。

### 以T0为匹配锚的tail路径归因

|比较|ΔV_select|Δclean|Δclean floor|ΔLEO mean|ΔLEO class-floor均值|ΔLEO严格floor|
|---|---:|---:|---:|---:|---:|---:|
|T1−T0：pair|+0.9625|-0.5194|+8.3333|+2.4056|-7.5111|-5.8000|
|T2−T0：tail|+2.9543|+4.2945|+15.1167|+1.5102|-2.2722|-4.9000|
|T3−T0：pair+tail|+2.9385|+4.2250|+21.0000|+1.1833|+0.2611|+2.2667|

pair单独提高LEO总体准确率，却显著伤害最弱LEO类别；tail单独提高clean和均值，也没有保护LEO严格floor。两者合用时，T3恢复并略微提高LEO类别下界，说明pair与tail存在互补，而不是简单相加。

### 对抗路径归因

- D1相对D0加入conditional DANN和双时间尺度后，clean floor提高约`10.95pp`，但LEO mean下降约`2.02pp`，LEO class-floor均值下降约`1.03pp`。
- D2相对D1加入`z_dom` TX adversary和conditional cross-covariance后，LEO mean回升约`2.30pp`，但clean下降约`3.38pp`、clean floor下降约`26.57pp`。
- D3相对D2加入动态GRL和任务保护投影后，clean回升约`3.92pp`、clean floor回升约`25.90pp`，但LEO mean和LEO class-floor均值再次下降约`1.82pp/6.02pp`。

这条路径显示对抗剂量和任务保护确实改变了预期的梯度冲突，但当前控制目标偏向恢复clean身份几何，没有同时保护LEO尾部。

## 八、结果可信度与混杂因素

### 1.同为200epoch不等于同等训练量

B0为10400updates，B1+多数候选为63000updates，后者是前者的`6.0577倍`。B1还把卫星CE从E80提前到E1，并把权重由固定0.68改成0.5→1.0。B1相对B0的巨大LEO收益同时包含：

- 更早且更长的卫星监督；
- 更大的optimizer update暴露量；
- batch结构变化；
- 因素化域头与gradient firewall。

因此不能把`+14.6602pp`全部归因于某一个PairBiCAD模块。

### 2.静态同配置锚暴露了随机性

B3、D0、T0除candidate ID外采用同一静态配置，但三者`V_select`均值分别为`44.4759%/48.2819%/44.5626%`，最大差`3.8059pp`；clean最大差`6.3194pp`，LEO mean最大差`2.6463pp`。候选ID很可能参与随机种子派生，或训练仍存在GPU非确定性。小于这一级别的单seed增益不能作为稳定方法收益。

### 3.E200完成不等于设计报告所说的科学收敛

B2及后续候选在E200终点的收敛telemetry多为`NOT_CONVERGED_SAFETY_STOP`，但用户已明确要求固定200epoch，所以launcher正确地记录`stop_action=ignored_fixed_200_epochs`并继续到E200。本报告能声明“完整E200预算已执行”，不能声明“全部候选已按原始收敛条件收敛”。

### 4.协议问题

当前`项目.md`规定统一`L_s/U_s/V=0.07/0.63/0.30`，`V`可做source侧校准、阈值冻结和checkpoint选择，但不得拆成`V_cal/V_select`。本run使用两个物理不交的9000样本子集，虽然没有target/query泄漏，仍属于方法角色不合规。因此：

```text
artifact与已执行指标：有效
当前协议正式晋级：无效
目标接收机/Phase2声明：不存在
```

## 九、完整性核验

我们完整流式读取了24行`metrics_epoch.jsonl`和`metrics_epoch.csv`，而不是抽样或只看tail；逐行解析了24份`source_loro_selection.json`、`checkpoint_runtime.json`、`diagnostics.json`、96份场景JSON、全部场景日志和`ARTIFACTS_COMPLETE.json`。核验结果：

- 24/24行epoch序列精确为1—200；
- 每行终止update与自身`planned_total_updates`一致：B0为10400，其余为63000；
- final checkpoint均非空，大小约10.57—11.72MB；
- strict reconstruction、trainer runtime strict均为true；
- missing/unexpected/shape mismatch全部为空；
- 每个场景`total=18000`，场景名和checkpoint身份一致；
- 24/24行source-only访问边界全部为false；
- `ARTIFACTS_COMPLETE=24`、`TECHNICAL_FAILURE=0`；
- 全量解析器错误列表为空。

独立的`diagnostics.json`存在明显缺口：24行的21个字段全部为`N/A`。机制调用证据仍可从每个epoch的`bicad_xr_audit`恢复，但吞吐、峰值显存、GPU-hours、receiver probe、`z_dom` TX probe、gradient ratio、margin Q0.1、最差TX×RX×day×channel和XDC donor矩阵没有形成最终聚合数值。该缺口不改变准确率和checkpoint重建结果，却阻止“设计报告全部诊断已实测闭合”的声明。

## 十、设计追踪与结论边界

设计追踪记录：`analysis/phase1_pairbicad_cv2_traceability.md`。E200修复版11项实现追踪在本地代码和行为测试中为`verified=11、deferred=0、rejected=0、blocked=0`；本run进一步证明这些机制在其指定候选中可达。当前协议验收另有1项不合规：E200-10的`V_cal/V_select`拆分必须在后续正式run中改回单一`V=0.30`。

更广的原始PairBiCAD设计不属于严格全量parity：

- 已实现且本run启用：双骨干、因素化域头、shared gradient firewall、conditional DANN、`z_dom` TX adversary、conditional cross-covariance、双时间尺度、动态GRL、任务保护投影、pair identity、Margin-REx/CVaR、no-early-freeze、coverage ledger、ReduceLROnPlateau、EMA/SWAD；
- 已有模块但本run关闭：VICReg、pair-delta、sparse XDC、XDC-KD、receiver tangent；
- 明确未作为本轮正式功能：可靠packet ID跨receiver配对、receiver-front-end物理增强、hard-LEO mining、三视图小子集、完整Fishr；
- 最高风险剩余项：当前验证角色协议不合规，其次是`diagnostics.json`全N/A和候选ID相关随机性。

## 十一、最终判定

本run的工程闭合是成功的：24行完整E200、四场景评估、严格重建和source-only边界全部通过。科学结果不是“PairBiCAD全面胜出”。

- 旧式`V_select`均值冠军：`B1`，但它有严重clean类别floor下降、fold波动和训练量混杂，不冻结；
- 当前矩阵中最均衡的机制候选：`T3`，pair与tail联用同时改善clean、LEO均值和LEO类别下界；
- 对抗分支D1—D3：未获得clean与LEO尾部同时提升，暂不作为主线；
- 正式晋级：`NO_FORMAL_PROMOTION`，原因是单一`V`协议不符合、B0计算量不匹配、单seed锚点波动过大；
- 可复用研究结论：优先保留`T3`的pair+tail互补机制；若继续确认，应在单一`V=0.30`、相同update/数据暴露、候选ID不影响随机流的条件下，与Core90 matched控制做多seed比较。
