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

## N607启动记录

|字段|内容|
|---|---|
|启动请求时间|2026-07-02 11:12 +08:00|
|本地Git-backed状态|`E:\type10-7\github_publish\CVS-RFFI-repo`干净，最新提交`1e35f9c Add Phase1 ADG-V2 GPU8 experiment design`|
|本地代码目录状态|`E:\type10-7`和`E:\type10-7\code`不是Git仓库；已使用`E:\type10-7\code\snapshots\phase1_adg_v2_gpu8_20260702\`快照|
|本地验证|`py_compile`通过；launcher `bash -n`通过；dry-run为8候选、8命令、GPU0-7|
|N607预检|`tools\n607_ssh_preflight.ps1`通过；远端host=`dell-DSS8440`；项目根存在；8张RTX3090可见|
|N607训练库存|`tools\n607_training_inventory.py --direct-only --pretty`显示`gpu_compute=[]`、`active_training_processes=[]`、`centralized_active=false`|
|启动策略|每GPU一个ADG实验；远端launcher默认`STAGE2_MAX_ACTIVE_PER_GPU=1`；不超过项目允许的每GPU两个训练上限|

## 同步文件与哈希

|本地文件|SHA256|远端目标|
|---|---|---|
|`E:\type10-7\code\cvsrffi\losses.py`|`6691760116019ED50159DA2C2DD6E72724CA1849BE8D5CC68A676A105315AD29`|`/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/losses.py`|
|`E:\type10-7\code\SSDG\train_ssdg.py`|`36513415781599DBD9FD56955B018AD42B85349642E41DB0397CF5416FEC9EEC`|`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`|
|`E:\type10-7\code\scripts\launch_phase1_adg_v2_gpu8_20260702.sh`|`80248591A13562D451B4E82C93ED620B8ABEFC554C43266E1F9348969176F4AC`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh`|

## 计划远端命令

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
mkdir -p logs/phase1_adg_v2_gpu8_20260702
bash -n code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh
bash code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh --dry-run
nohup bash code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh > logs/phase1_adg_v2_gpu8_20260702/scheduler.out 2>&1 &
```

## 远端同步与验证

|项目|结果|
|---|---|
|远端预同步备份|`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/phase1_adg_v2_gpu8_20260702_remote_pre_sync_20260702_111404`|
|SCP同步|`losses.py`、`train_ssdg.py`、`launch_phase1_adg_v2_gpu8_20260702.sh`已同步到计划远端路径|
|远端SHA256|三文件与本地SHA256一致|
|远端编译|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/cvsrffi/losses.py code/SSDG/train_ssdg.py`通过|
|远端launcher语法|`bash -n code/scripts/launch_phase1_adg_v2_gpu8_20260702.sh`通过|
|远端dry-run|`dryrun_candidates=8`、`dryrun_commands=8`、`dryrun_gpus=0,1,2,3,4,5,6,7`|

## 启动与健康检查

|项目|结果|
|---|---|
|scheduler PID|`3791826`|
|scheduler日志|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adg_v2_gpu8_20260702/scheduler.out`|
|启动时间|候选status记录均为`2026-07-02T11:14:56+08:00`|
|4-5分钟健康检查|8/8候选仍在运行；8/8有GPU进程；8/8出现`[CONFIG-ADG]`、`[EPOCH-BEGIN]`、`[PROXY-ADG]`|
|错误检查|未检出`Traceback`、`RuntimeError`、`unrecognized arguments`、`CUDA out of memory`、`Killed`或`NaN`|

|GPU|candidate|PID|log|
|---:|---|---:|---|
|0|`ADG8G0_B02_ANCHOR_E200`|3791902|`logs/phase1_adg_v2_gpu8_20260702/ADG8G0_B02_ANCHOR_E200.out`|
|1|`ADG8G1_BRIDGE_CVAR_E200`|3791903|`logs/phase1_adg_v2_gpu8_20260702/ADG8G1_BRIDGE_CVAR_E200.out`|
|2|`ADG8G2_SHELL_LOW_DENS_E200`|3791916|`logs/phase1_adg_v2_gpu8_20260702/ADG8G2_SHELL_LOW_DENS_E200.out`|
|3|`ADG8G3_ENERGY_Q10_E200`|3791914|`logs/phase1_adg_v2_gpu8_20260702/ADG8G3_ENERGY_Q10_E200.out`|
|4|`ADG8G4_RADIUS_RATIO_E200`|3791898|`logs/phase1_adg_v2_gpu8_20260702/ADG8G4_RADIUS_RATIO_E200.out`|
|5|`ADG8G5_TAIL_OVERFLOW_E200`|3791890|`logs/phase1_adg_v2_gpu8_20260702/ADG8G5_TAIL_OVERFLOW_E200.out`|
|6|`ADG8G6_CONSERVATIVE_ALL_E200`|3791894|`logs/phase1_adg_v2_gpu8_20260702/ADG8G6_CONSERVATIVE_ALL_E200.out`|
|7|`ADG8G7_STRONG_ALL_SAT_E200`|3791912|`logs/phase1_adg_v2_gpu8_20260702/ADG8G7_STRONG_ALL_SAT_E200.out`|

|candidate|`[CONFIG-ADG]`|`[EPOCH-BEGIN]`|`[PROXY-ADG]`|
|---|---:|---:|---:|
|`ADG8G0_B02_ANCHOR_E200`|1|15|15|
|`ADG8G1_BRIDGE_CVAR_E200`|1|15|15|
|`ADG8G2_SHELL_LOW_DENS_E200`|1|15|15|
|`ADG8G3_ENERGY_Q10_E200`|1|14|14|
|`ADG8G4_RADIUS_RATIO_E200`|1|15|15|
|`ADG8G5_TAIL_OVERFLOW_E200`|1|15|15|
|`ADG8G6_CONSERVATIVE_ALL_E200`|1|15|15|
|`ADG8G7_STRONG_ALL_SAT_E200`|1|15|15|
