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
