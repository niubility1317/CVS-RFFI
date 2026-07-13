# phase1_dgleo_hiercore8_20260713

## 启动前记录

- 时间：2026-07-13。
- 操作者：Codex。
- 目标：在不牺牲Phase1闭集DG、strict UDU和leo_weak星地性能的前提下，直接修复P0Closed8暴露的接收域碎片化、最近异类碰撞、source episode中心漂移、无标签open-set分支空转和open-set有效梯度不足。
- 协议边界：Phase1 source-only；`rho_label=0.08`，`rho_unlabeled=0.72`，source-val 0.20；不使用目标接收机样本或真实unknown；结果只能支持DG、星地压力、known几何、proxy风险和prototype质量结论。
- 对比目标：`phase1_dgleo_p0closed8_20260713`，重点对比C6以及前序J5。
- 训练长度：120 epoch；open-set相关损失从E1启用，2～4 epoch短warmup；checkpoint固定使用E120 final权重。
- 评估调度：source-val重评前100轮每10轮一次，最后20轮每2轮一次；held-out clean/leo_weak named test仅在训练结束后运行。
- 星地协议：训练保留`concat_sa`；训练与评估场景为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；该结果只证明同增强族独立随机压力鲁棒性，不等于跨增强族泛化。

## P0联合修改

|修改|直接目标|预期指标|
|---|---|---|
|shared class core×local support层级软门控|阻止local component无条件并集绕过身份核心|`proxy_vaccept`、bridge/tail/overflow accept下降|
|global与local分位数同时优化|避免local p95变好而全局tail继续扩大|`zid_p95/p99/tail_cvar`下降|
|最近异类component CVaR与radius-overlap损失|避免全pair均值稀释最危险碰撞|`component_radius_to_inter`下降、min inter上升|
|LODO中心目标、最差component CVaR和leave-domain绝对目标|不靠扩大18度接收半径降低overflow|`source_episode_overflow`及worst-domain中心偏移下降|
|无标签三态路由加ambiguous direct与clean-sat pair损失|使U_s open-set分支持续产生有效梯度|U direct active epoch比例、tri direct count上升|
|U direct复用detach后的有标签receiver-aware component reference|避免7～9个稀疏U样本重新建局部球导致几何分支失活|geometry active batch比例和finite U-DM指标上升|
|open-set预算reserve|避免浮点/控制误差导致有效梯度低于下界|实际OS梯度占比稳定在设定区间|
|guard阻断时保留diagnostic prototype|保留失败候选结构证据，不冒充可部署endpoint|导出完整性提升，promotion仍fail-closed|

## 实验矩阵

|候选|GPU|机制|
|---|---:|---|
|HC_H0_FULL_STABLE|0|全机制稳定版|
|HC_H1_CORE_STRONG|1|invariant core与leave-domain CVaR加强|
|HC_H2_COMPONENT_SAFE|2|最近异类component与重叠风险加强|
|HC_H3_BOUNDARY_PARITY|3|global quantile与层级边界加强|
|HC_H4_U_TRI_ACTIVE|4|U_s三态与ambiguous/outside配对加强|
|HC_H5_DG_SAT_FLOOR|5|dual-worst group与星地不变性加强|
|HC_H6_FULL_AGGRESSIVE|6|全机制高open梯度预算|
|HC_H7_NO_HIERARCHY_ABL|7|移除shared-class hierarchy的机制消融|

## 成功标准

- DG：overall/strict UDU不低于P0Closed8联合最优超过0.5pp；receiver floor不继续恶化。
- 星地：sat strict mean不下降超过0.5pp；sat receiver strict floor相对56.55%至少提升3pp。
- 固定source-val几何：p99不高于C6的57.29度；proxy_vaccept低于0.36；bridge accept低于0.20；radius/inter低于4.38且不能以proxy恶化换取。
- legacy诊断：source_episode_overflow必须脱离0.95～0.98平台；若仍接近1，则判定当前batch/结构修复仍未直接对齐LODO硬定义。
- U_s：`train/w_loss_u_direct_metric_accept`不再长期为0，至少80%epoch出现有效direct样本。
- 泄漏：receiver/day/channel probe excess均需下降；Phase1 promotion硬标准仍为0.20/0.15/0.15。
- 不得用单指标改善promote；需同时满足DG、星地、固定endpoint代理、legacy风险和完整导出。

## 本地变更与验证

- 修改：`code/cvsrffi/losses.py`、`code/SSDG/train_ssdg.py`。
- 新增：`code/scripts/launch_phase1_dgleo_hiercore8_20260713.py`、`code/scripts/queue_phase1_dgleo_hiercore8_20260713.py`、`code/tests/test_phase1_dgleo_hiercore8_launcher.py`。
- 更新测试：`code/tests/test_phase1_jointp0_core.py`、`code/tests/test_unlabeled_quarantine_acceptance_loss.py`。
- 验证：`py_compile`通过；focused pytest 32 passed，包含稀疏U query复用有标签component reference并产生非零梯度的回归测试。
- 本地矩阵：`E:\type10-7\local_artifacts\phase1_dgleo_hiercore8_20260713_matrix.json`。

## N607落地信息

- 首次队列包装启动在训练进程创建前因缺少共享`dual`别名退出；未创建run目录、未占用GPU。已在本地补齐兼容别名并重新验证后再发布。

- 远端根目录：`/home/szu2070436088/2510044040/CV-SincNet`。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 运行目录：`runs/phase1_dgleo_hiercore8_20260713/<candidate>`。
- 日志目录：`logs/phase1_dgleo_hiercore8_20260713/<candidate>.out`。
- 队列状态：`logs/phase1_dgleo_hiercore8_20260713_queue_state.json`。
- GPU：0～7各1个候选，每卡1个实验。
- 墙钟上限：每候选10小时；预期120 epoch训练约6～7小时，终局全量评估与导出约0.5～1小时。
- 启动命令：`nohup <python> code/scripts/queue_phase1_dgleo_hiercore8_20260713.py --max-concurrent-per-gpu 1 --wall-hours 10 --max-wait-hours 12 > logs/phase1_dgleo_hiercore8_20260713_queue.out 2>&1 &`。

## 风险

- 当前`endpoint_accept_v1`仍是最终硬边界；训练DM层级门控仅用于塑形，不能替代endpoint结论。
- diagnostic blocked export当前只用于结构诊断，`endpoint_artifact_ready=false`，不得进入runtime或promotion。
- legacy source overflow使用LODO硬定义，与batch动态soft overflow不同；若训练代理下降而legacy不变，仍判定未闭环。
- 当前星地训练按batch循环三种leo_weak场景，不是同一优化步的clean+三场景四视图worst-group；本轮H5只能验证dual-worst与配对加强，不能证明完整receiver×scenario CVaR已落地。
