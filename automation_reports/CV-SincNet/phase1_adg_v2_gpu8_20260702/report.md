# Phase1 ADG-V2八卡机制验证设计

## 基本信息

|字段|内容|
|---|---|
|run_id|`phase1_adg_v2_gpu8_20260702`|
|设计时间|2026-07-02|
|operator/agent|Codex|
|目标|用8个一卡一实验候选验证ADG-V2开集治理损失是否能在不放松`L_vaccept_CVaR`的前提下，修复ADV3暴露的`bridge_accept=1.0`、`source_overflow`高、尾部半径偏宽和低密度误接收问题。|
|协议边界|Phase1 source-only地面训练；只评估闭集DG能力、星地压力鲁棒性、known几何、proxy/virtual unknown风险和prototype导出质量；不得声明真实unknown_FAR/FPR95或Stage2成功。|
|比较目标|以已完成的`ADV3B02_CORE90_SOFT_E200`作为主参照，`ADV3B07/B08`作为gate参照，`ADV3B30`作为satellite stress参照。|

## 本轮设计原则

|原则|落实方式|
|---|---|
|`L_vaccept_CVaR`不能放松|8个候选全部保留`--proxy_unknown_vaccept_weight 1.00`、`--proxy_unknown_vaccept_cvar_alpha 0.30`，并保持direct proxy unknown loss。|
|每张卡一个实验|GPU0-7各绑定一个候选；launcher默认`STAGE2_MAX_ACTIVE_PER_GPU=1`。|
|先机制诊断再组合|GPU1-5分别打bridge、shell/low-density、energy quantile、radius ratio、tail/overflow；GPU6/7做保守/强组合。|
|不再盲扫|只围绕ADV3失败机制：bridge全接收、source overflow高、p99长尾、component半径和低密度accept。|
|不牺牲泛化|所有候选继续监控overall、strict UDU、receiver floor、satellite mean/floor和best-final gap。|

## 本地文件

|路径|用途|
|---|---|
|`E:\type10-7\code\scripts\launch_phase1_adg_v2_gpu8_20260702.sh`|8候选一卡一实验launcher；支持`--dry-run`和`--only=`。|
|`E:\type10-7\automation_reports\CV-SincNet\phase1_adg_v2_gpu8_20260702\report.md`|本设计报告。|
|Git-backed提交|已纳入Git-backed发布仓库；最终提交号以`git log -1 --oneline`为准。|

## 远端计划路径

|项目|路径|
|---|---|
|远端root|`/home/szu2070436088/2510044040/CV-SincNet`|
|远端launcher|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh`|
|远端run目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adg_v2_gpu8_20260702`|
|远端log目录|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adg_v2_gpu8_20260702`|
|启动命令|`nohup bash code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh > logs/phase1_adg_v2_gpu8_20260702/scheduler.out 2>&1 &`|

## 固定基础配置

|配置|值|
|---|---|
|数据|`ManySig.pkl`，`split_mode=tx_rx_day_1_7_2`|
|标注比例|`labeled_ratio=0.10`、`unlabeled_ratio=0.70`、`source_val_ratio=0.20`|
|epoch|`epochs=200`、`label_epochs=130`、`pseudo_epochs=70`|
|seed|GPU0-7分别为`493000-493007`|
|proxy启动|`proxy_unknown_start_epoch=45`、`warmup=25`|
|不同TX采样|`proxy_unknown_holdout_tx_per_batch=3`|
|hard unknown池|`proxy_unknown_virtual_mode=hard`、`virtual_count=48`、`virtual_detach=false`|
|core/accept阈值|默认`core_quantile=0.90`、`accept_quantile=0.85`，个别候选按机制收紧到`0.82`|
|soft unknown mixup|`lambda_soft_unknown_mixup=0.0045`、`soft_unknown_mixup_count=24`、`soft_unknown_mixup_order=3`|
|source episode mixup|`lambda_source_episode=0.0035`、`source_episode_mixup_weight=0.75`、`source_episode_mixup_hard_k=3`|
|prototype导出|启用local component fusion，`max_components=6`、`radius_cap=15deg`、`accept_policy=local_component`|

## 候选矩阵

|GPU|candidate|机制问题|关键变量|成功标准|失败判据|
|---:|---|---|---|---|---|
|0|`ADG8G0_B02_ANCHOR_E200`|新代码下的B02锚点复核|新增ADG side weights全0；保留B02 direct vaccept、core90、accept85|闭集指标接近ADV3B02；新增ADG日志字段非NaN；`vaccept_surrogate`保持有效|闭集显著低于B02或新增日志缺失，说明代码/配置偏移|
|1|`ADG8G1_BRIDGE_CVAR_E200`|bridge全接收|`bridge_accept_weight=0.003`、`bridge_target=0.15`、`energy_margin_target=0.10`|`bridge_accept_rate`相对ADV3的1.0明显下降，strict/floor下降<1pp|bridge仍>0.9或strict/floor下降>2pp|
|2|`ADG8G2_SHELL_LOW_DENS_E200`|shell/outward与低密度误接收|`shell_outward_accept_weight=0.0025`、`low_density_accept_weight=0.0025`、`shell_width=6deg`、`density_temp=2.5deg`|`shell_accept_rate`、`outward_accept_rate`、`low_density_accept_prob`下降，receiver floor不塌|只压shell不压bridge/low-density，或rx7/rx8显著下降|
|3|`ADG8G3_ENERGY_Q10_E200`|unknown能量边界低分位风险|`energy_margin_quantile_weight=0.0035`、`energy_margin_q=0.10`、`energy_target=0.10`、`unknown_margin=0.10`|`energy_margin_q05/q10`上移，`proxy_vaccept`下降，p99不升|能量边界改善但source_overflow/p99变差|
|4|`ADG8G4_RADIUS_RATIO_E200`|known半径和类间间隔比例失控|`radius_budget_weight=0.0015`、`radius_inter_ratio_weight=0.0015`、`radius_budget=9deg`、`ratio_target=0.22`|`component_radius_p95/max`和`radius_inter_ratio`下降，strict保持|半径下降但strict/receiver floor大跌，说明收缩过强|
|5|`ADG8G5_TAIL_OVERFLOW_E200`|tail/overflow继续撑宽accept域|`tail_quarantine_weight=0.35`、`source_safe_weight=0.35`、`tail_q=0.90`、`overflow_q=0.95`、`overflow_target=0.18`|`source_overflow`下降，p95/p99下降，old closed-set基本保持|overflow仍>ADV3B02或receiver floor塌陷|
|6|`ADG8G6_CONSERVATIVE_ALL_E200`|保守组合是否产生协同|bridge/shell/low-density/energy/radius/ratio小权重全开|bridge、overflow、p99、radius指标至少两项改善，strict/floor下降<1pp|指标分散无收益或闭集损伤|
|7|`ADG8G7_STRONG_ALL_SAT_E200`|强组合加satellite guard是否可用|全ADG中等权重，`lambda_zid=0.036`、`lambda_ow=0.0028`、sat schedule提前增强|open-set代理明显改善且sat strict floor接近或超过B30|closed-set或satellite floor明显崩，强组合不可主推|

## 指标读取顺序

|维度|指标|
|---|---|
|泛化|`overall_tx`、`strict_udu`、`receiver_floor`、rx7/rx8、satellite mean/floor、best-final gap|
|直接拒识代理|`proxy_vaccept`、`vaccept_surrogate_CVaR`、`hard_proxy_accept_rate`、`shell_accept_rate`、`bridge_accept_rate`、`outward_accept_rate`|
|ADG新增治理|`bridge_governance_loss`、`shell_outward_accept_loss`、`low_density_accept_loss`、`energy_margin_quantile_loss`、`radius_budget_loss`、`radius_inter_ratio_loss`|
|known域几何|`zid_compact_pos_angle_p50/p95/p99`、`zid_tail_cvar`、`component_radius_p95/max`、`radius_to_inter_ratio`|
|tail风险|`source_overflow`、`ow_feat_angle_p95/p99`、`tail_frac`、`r3sigma`|
|prototype质量|component数量、radius cap命中、fusion字段、local component accept导出字段|

## 决策规则

|结论|条件|
|---|---|
|主推进|strict UDU和receiver floor不低于ADV3B02超过1pp，同时`bridge_accept`、`source_overflow`、p99、radius/inter-ratio至少两类风险下降。|
|机制阳性但不主推|open-set代理明显改善，但strict、receiver floor或satellite floor损伤超过2pp。|
|诊断负例|只能改善单一loss值，无法改善对应accept率或几何风险。|
|淘汰|bridge仍接近1.0、overflow仍高，且闭集/星地压力同步下降。|

## 本地验证计划

|命令|预期|
|---|---|
|`bash -n code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh`|语法通过|
|`bash code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh --dry-run`|打印8个候选、8条命令、GPU0-7一一对应|
|`conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --help`并检索ADG参数|确认远端训练入口所需参数存在|

## 本地验证结果

|命令|结果|
|---|---|
|`bash -n code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh`|PASS|
|`bash code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh --dry-run`|PASS；`candidates=8`、`commands=8`、`gpus=0,1,2,3,4,5,6,7`|
|`conda run --no-capture-output -n ssr-gpu python code\SSDG\train_ssdg.py --help`并检索ADG参数|PASS；bridge、shell/outward、low-density、energy quantile、radius budget、radius/inter-ratio、holdout TX参数均存在|
|dry-run输出|`E:\type10-7\automation_reports\CV-SincNet\phase1_adg_v2_gpu8_20260702\dry_run.txt`|

## 同步与启动边界

本报告只完成本地设计。若后续启动N607，必须先执行：

1. `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`
2. 检查GPU占用和现有训练进程；若每卡已有训练且用户未要求叠加，保持monitor-only。
3. `scp`同步`losses.py`、`train_ssdg.py`、launcher脚本到N607。
4. 远端执行`bash -n`和`--dry-run`。
5. 启动后4-5分钟检查`[CONFIG-LOSS]`、`[CONFIG-ADG]`、`[PROXY-ADG]`、`[EPOCH-BEGIN]`和Traceback/OOM/NaN。

当前未执行SCP，未启动N607实验。
