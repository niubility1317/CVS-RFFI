# phase1_dgleo_hiercore8r3_20260713

## 启动前记录

- 时间：2026-07-13 23:24 CST；操作者：Codex。
- 目标：验证Phase1层级known几何、receiver-aware local component、无标签三态路由与有效open-set梯度预算的联合修复，同时保护clean DG、strict UDU和`leo_weak`星地性能。
- 协议边界：Phase1 source-only；`rho_label=0.08`，`rho_unlabeled=0.72`，source-val 0.20；不使用目标接收机样本或真实unknown；只能评价DG、星地压力、known几何、proxy风险、无标签分支和prototype质量。
- 对比：`phase1_dgleo_p0closed8_20260713`联合最优C6和前序J5。
- 训练：120 epoch；open-set损失从E1启用，2～4 epoch短warmup；只保留E120 final权重用于最终结论。
- 评估：source-val前100轮每10轮重评，最后20轮每2轮重评；held-out clean和三种`leo_weak`named test仅在训练结束后运行。
- 星地：保留`concat_sa`；训练和评估增强族均为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`，只能证明同增强族独立随机压力鲁棒性。

## P0闭环机制

|机制|直接目标|硬证据|
|---|---|---|
|shared class core×local support|避免local component并集绕过身份核心|固定proxy/bridge/tail/overflow accept下降|
|global与local分位数联合损失|避免局部收紧但全局tail扩张|`zid_p95/p99/tail_cvar`同步下降|
|最近异类component CVaR及overlap|修复最危险局部组件碰撞|min-inter上升、radius/inter下降|
|LODO中心、leave-domain和最差component CVaR|不靠扩大18度球容纳跨域样本|legacy source-episode overflow脱离约0.97平台|
|U三态复用detach的有标签component reference|消除稀疏U-only direct空转|U geometry active和finite U-DM指标持续非零|
|finite-aware telemetry和预算聚合parity|修复NaN吞值及guard口径错位|新增component字段有限，guard读取controller post预算|
|open梯度reserve及objective分组|使拒识几何从E1获得非小权重|预算稳定且DG健康门不触发|
|阻断时仅导出diagnostic prototype|保留结构证据但不冒充endpoint|`endpoint_artifact_ready=false`|

## 实验矩阵

|候选|GPU|机制|
|---|---:|---|
|HC_H0_FULL_STABLE|0|全机制稳定版|
|HC_H1_CORE_STRONG|1|invariant core与leave-domain CVaR加强|
|HC_H2_COMPONENT_SAFE|2|最近异类component与重叠风险加强|
|HC_H3_BOUNDARY_PARITY|3|global quantile与层级边界加强|
|HC_H4_U_TRI_ACTIVE|4|U三态与ambiguous/outside配对加强|
|HC_H5_DG_SAT_FLOOR|5|dual-worst group与星地不变性加强|
|HC_H6_FULL_AGGRESSIVE|6|全机制高open梯度预算|
|HC_H7_NO_HIERARCHY_ABL|7|移除shared-class hierarchy的消融|

## 成功标准

- overall和strict UDU相对P0Closed8联合最优下降不得超过0.5pp；receiver floor不得恶化。
- sat strict mean下降不得超过0.5pp；sat receiver×scenario floor应从56.55%提升至少3pp。
- fixed p99不高于57.29度，proxy_vaccept低于0.36，bridge低于0.20，radius/inter低于4.38，且不得转移风险。
- legacy source-episode overflow必须脱离0.95～0.98平台。
- U geometry active batch rate至少0.50；pair-only不得冒充direct；U-DM关键字段必须有限。
- 总open预算和拒识几何有效梯度需同时达标；泄漏probe excess需满足0.20/0.15/0.15。
- 单指标改善不能promotion；必须同时满足DG、星地、固定endpoint代理、legacy风险和导出完整性。

## 本地版本与验证

- Git提交：`a039244`、`e622e35`、`8182e27`、`1cc44ff`；启动记录：`2387c87`。
- 文件：`code/cvsrffi/losses.py`、`code/SSDG/train_ssdg.py`、`code/cvsrffi/phase1_v2_control.py`、`code/post_stage_common.py`、HierCore8 launcher/queue和测试。
- 验证：`py_compile`通过；focused pytest 32项、20项分别通过；扩展回归84项通过。
- 远端快照：`code/snapshots/phase1_dgleo_hiercore8r3_1cc44ff_pre_sync`。
- 同步SHA256：`train_ssdg.py=d235a253c2b20c5c2e26e7e697095f2ed06e7acb67f88301a6cd8cfd4e1e2ac2`；`losses.py=a90877689599b6871fffe1d81f6f51ccff4a5d99a4c2fc9f23ca5e636954b989`。
- 远端`py_compile`通过。

## N607落地

- 直连配置和密钥有效，但TCP/SSH返回`Connection refused`；按项目规则使用已验证实验室桥接通道。
- 远端根：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- run：`runs/phase1_dgleo_hiercore8r3_20260713/<candidate>`；日志：`logs/phase1_dgleo_hiercore8r3_20260713/<candidate>.out`。
- queue PID4157572；23:25启动H3 PID4157998、H4 PID4158066、H5 PID4158527、H6 PID4158990、H7 PID4159453。
- H0～H2等待GPU0～2上的6个外部基线作业释放；外部进程未被停止、重启或修改。
- 命令：`nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/queue_phase1_dgleo_hiercore8_20260713.py --run-id phase1_dgleo_hiercore8r3_20260713 --max-concurrent-per-gpu 2 --stable-polls 2 --poll-seconds 30 --wall-hours 10 --max-wait-hours 12 > logs/phase1_dgleo_hiercore8r3_20260713_queue.out 2>&1 < /dev/null &`。
- 每候选墙钟上限10小时；120轮训练预计5.5～7小时，final全量评估、probe和导出约0.5～1小时；整组预计7～10小时，H0～H2还取决于外部GPU释放。

## 纠错与E1验证

- 原始run在E1/E2暴露budget guard口径和finite telemetry问题；r2验证U reference复用后暴露component字段未聚合。两次均只终止本批精确run ID/PID并保留产物，未触碰GPU0～2外部训练。
- r3 H4已完成E1，无fatal、traceback或CUDA OOM。
- U direct active=0.9683，selected=21.65，weighted loss=3.096；U-DM p95=48.03度，p99=53.93度，CVaR=53.05度，proxy=0.1951，bridge=0.00153，component min-inter=57.73度。
- labeled DM component min-inter=57.30度；controller post预算=0.1989，位于设定区间，guard未误触发。
- legacy source-episode overflow=0.9744，首轮仍处于原平台，必须观察全程趋势。

## 声明边界

- 训练DM门控仍是塑形代理，不能替代最终`endpoint_accept_v1`；本轮尚未证明冻结、版本化、fail-closed硬endpoint三入口一致性。
- guard阻断时prototype仅为diagnostic artifact，不是可部署endpoint。
- legacy overflow若仍接近1，不能用新DM改善掩盖结构错位。
- 同一`leo_weak`增强族训练和评估不证明跨增强族或真实未知星地信道泛化。

## 最终状态与证据完整性

- 8/8候选完成120epoch；每个`metrics_epoch.csv`包含120行，每个stdout约5403行。
- 完整日志未发现fatal、Traceback、CUDA OOM、Killed或非有限总损失。
- 终态均为`NON_PROMOTABLE_GUARD_BLOCKED`，返回码5来自安全门阻断，不是训练进程崩溃。
- 每个候选只保留E120的`final_ssdg.pth`；terminal、final eval和diagnostic prototype中的checkpoint SHA256一致。
- 最后一个候选于2026-07-14 06:00 CST结束；单候选约5.1～5.4小时，受初始GPU排队影响整组约6.6小时。
- 2026-07-14 08:47 CST重新核验时，N607训练进程、launcher进程和GPU计算进程均为0；不得把历史外部占用、queue终态和当前活动进程混为一谈。

## 泛化与星地主表

|候选|机制|overall|strict UDU|clean RX floor|sat strict mean|sat RX floor|结论|
|---|---|---:|---:|---:|---:|---:|---|
|P0Closed8 C6|前序联合最优|89.953|86.183|74.633|72.575|56.550|比较基线|
|H0|全机制稳定|89.855|86.213|75.750|72.527|57.067|strict单点+0.03pp，sat未提升|
|H1|core加强|89.741|86.043|75.817|72.453|57.267|clean floor改善，主指标回落|
|H2|component加强|89.773|86.045|75.983|72.324|57.075|sat最差|
|H3|boundary加强|89.712|85.962|75.717|72.355|57.258|DG和sat均回落|
|H4|U三态加强|89.852|86.172|76.258|72.452|57.275|clean floor最佳，联合目标未改善|
|H5|DG/sat加强|89.730|85.968|76.025|72.487|57.500|sat弱点略升但其他receiver重分配|
|H6|高open预算|89.804|86.100|75.808|72.504|57.233|提高预算没有突破|
|H7|无hierarchy消融|89.778|86.012|76.058|72.357|57.333|层级门控诊断对照|

结论：overall和strict基本保持，clean receiver floor提高约1.1～1.6pp，但`sat strict`仍为72.32%～72.53%，低于C6的72.575%；最弱receiver×scenario仍为RX8约57%，距离阶段目标73%很远。H5只把部分性能从RX9/RX10重新分配给RX8，不是最坏组风险整体下降。

## Open-set代理与known覆盖

|候选|fixed p95/p99°|fixed proxy|bridge|tail/overflow accept|known core accept|legacy source episode overflow|结论|
|---|---:|---:|---:|---:|---:|---:|---|
|C6|28.41/57.29|0.360|0.2603|0.640/0.624|0.503|0.956|前序基线|
|H0|42.35/68.95|0.181|0.0006|0.474/0.192|0.154|0.979|proxy下降但known覆盖坍缩|
|H1|40.59/67.40|0.183|0.0009|0.465/0.208|0.157|0.978|同上|
|H2|44.26/71.46|0.176|0.0004|0.502/0.194|0.147|0.979|proxy最好但p99明显恶化|
|H3|41.47/67.91|0.183|0.0005|0.481/0.195|0.157|0.980|同上|
|H4|42.93/70.91|0.180|0.0004|0.489/0.182|0.149|0.974|overflow accept最低但known覆盖过低|
|H5|38.71/65.13|0.191|0.0010|0.444/0.199|0.159|0.978|r3尾部最好，仍差于C6|
|H6|41.85/68.37|0.180|0.0003|0.483/0.199|0.153|0.979|预算提升无结构响应|
|H7|44.74/73.30|0.371|0.1076|0.835/0.549|0.412|0.976|移除层级后的对照|

层级门控确实把proxy、bridge、tail和overflow accept大幅压低，但known core accept同时由H7的0.412/C6的0.503降至0.147～0.159。其主机制是多个soft gate相乘形成近似reject-all，而不是known表征收紧。所有r3候选的p95、p99、CVaR和legacy source-episode overflow均差于C6，不能宣称open-set潜力联合改善。

## 无标签、梯度与泄漏

- H4的U direct active均值86.0%，说明“U direct长期为0”的执行问题已修复；但120epoch只产生402/846720个CE伪标签，最终仅1/7056。
- 置信度约99.87%、component一致率约98.63%，真正瓶颈是batch内temporal neighbor gate与shuffle采样不兼容，temporal pass仅约0.65%～0.74%。
- H4平均三态为trusted core 6.77%、ambiguous 4.26%、outside 88.97%；Phase1的U_s属于source known，不能把outside直接解释成unknown负样本。
- source/invariance梯度依靠约26倍norm放大达到配额，且100%step触发总梯度裁剪到5；预算在所有trainable参数上统计，不能证明有效梯度进入`z_id`主干。
- receiver/day/channel leakage excess仍约0.61/0.16～0.19/0.40，未实现身份与域解耦。当前默认`z_id=feat_joint`显式拼接并门控DAC/PA缺陷特征，是结构性泄漏来源。

## Tail与endpoint审计

- 6/8候选把E120本身注册为tail reference，再与E120 final比较，构造出`p99_delta=0`；真实final相对历史最佳窗口约扩张0.60°～1.32°。本批尚未越过2°阻断线，但现实现存在确定性假阴性。
- 动态DM使用168个即时组件，diagnostic prototype只有18～23个融合组件，两者不是同一决策器。
- 所有prototype均为`diagnostic_only=true`、`endpoint_artifact_ready=false`，不具备正式`endpoint_accept_v1`、校准、manifest和独立三入口parity证据。
- 当前不能声明真实unknown FAR、FPR95、Stage2 old/new性能或部署拒识成功。

## 四象限与决策

|candidate|泛化结论|拒识潜力|主要风险|可否Stage2真实unknown评估|下一步|
|---|---|---|---|---|---|
|H0|strict单点持平，sat未升|门控代理下降，表征尾部变差|known覆盖坍缩|否|作为同seed机制对照|
|H1|小幅回落|无联合改善|overflow与p99高|否|停止单独加core权重|
|H2|小幅回落|proxy单项最好|p99扩大、known覆盖最低|否|保留为指标错位负例|
|H3|回落|无联合改善|boundary仍靠软门|否|淘汰|
|H4|clean floor最好，sat未升|U执行修复但终局未响应|伪标签近乎停用|否|改跨epochU记忆后复验|
|H5|sat弱点单点改善|r3尾部最好但仍差于C6|receiver性能重分配|否|用于receiver×scenario CVaR设计|
|H6|无稳定提升|高预算无响应|噪声梯度放大和全局裁剪|否|改为z_id路径预算后复验|
|H7|基本持平|移除门控后风险回升|不具备拒识边界|否|仅作为层级门控消融|

## 最终判断

当前实验对Phase1的贡献是：证明层级局部门控能在不明显损伤闭集DG的情况下压低动态accept代理，并修复U direct空转；同时以同seed消融暴露了reject-all平凡解、`feat_joint`域泄漏、batch temporal gate失效和梯度预算错位。

当前不能声明的是：known表征已收紧、真实unknown拒识改善、satellite泛化突破、`endpoint_accept_v1`闭环或Stage2成功。

最主要风险是：known core覆盖只有约15%、source-episode overflow仍约0.97、p99较C6扩大8°～16°、RX8星地floor约57%、tail reference被final污染、动态DM与最终硬边界不一致。

当前没有可推进主候选。下一轮必须以同seed逐步验证ungated identity core、冻结reference bank、smooth-min层级训练门、known TPR硬约束、18°同口径overflow、跨epochU记忆、`concat_sa`监督去重和`z_id`路径梯度预算。
