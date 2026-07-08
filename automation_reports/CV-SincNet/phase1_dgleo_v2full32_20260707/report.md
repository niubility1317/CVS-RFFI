# phase1_dgleo_v2full32_20260707

## 基本信息
- 时间:2026-07-07
- operator:Codex
- 目标:设计Phase1地面source-only域泛化训练矩阵,验证v2闭环机制,直接优化open-set几何指标,同时保护跨receiver/date和LEO星地压力泛化。
- 状态:已完成本地矩阵脚本、干跑测试、N607同步和主8候选启动;当前RUN_ID为`phase1_dgleo_v2full32_main8_20260708`,8个训练进程均已进入训练循环。
- 协议边界:Phase1 source-only,训练数据限定`ManySig.pkl`;不使用真实unknown类、不使用target receiver样本、不声明真实unknown_FAR/FPR95/Stage2成功。

## 本地文件
|文件|用途|
|---|---|
|`E:\type10-7\code\scripts\launch_phase1_dgleo_v2full32_20260707.sh`|32候选启动脚本,8张GPU每张4个实验,支持`--dry-run`和`--only=`|
|`E:\type10-7\code\tests\test_phase1_dgleo_v2full32_launcher.py`|矩阵协议、资源分配、机制消融和source-only边界测试|
|`E:\type10-7\code\snapshots\phase1_dgleo_v2full32_20260707_20260707_235418\`|非Git根目录变更快照|
|`E:\type10-7\automation_reports\CV-SincNet\phase1_dgleo_v2full32_20260707\report.md`|本报告|

## 设计原则
本矩阵不是把所有候选都做成弱化/稳定/激进/激进保护四档。GPU0保留全机制参照阶梯,用于判断“所有机制全量启动”在不同强度下的泛化/拒识冲突;GPU1-GPU7用于机制消融、单机制增强、交互压力和export gate验证。所有候选仍保留`endpoint_accept_v1`、`loss_gate_exported=false`、`tail_safety_state_machine=true`、`u_tri_state_required=true`和`feasibility_stage=audit`,避免把动态DM软门控当最终拒识边界。

## 矩阵总览
|GPU|候选|组别|类型|验证目标|
|---:|---|---|---|---|
|0|DGLEO_V2FULL32_FULL_WEAK|G0_FULL_LADDER|全机制弱化|低open-set梯度能否保护DG/LEO floor|
|0|DGLEO_V2FULL32_FULL_STABLE|G0_FULL_LADDER|全机制稳定|主参照,验证平衡解|
|0|DGLEO_V2FULL32_FULL_AGGR|G0_FULL_LADDER|全机制激进|验证加大几何loss是否真实压低p99/proxy_vaccept|
|0|DGLEO_V2FULL32_FULL_AGGR_SAFE|G0_FULL_LADDER|全机制激进保护|验证KD/sat保护能否缓解激进几何损失对strict UDU和sat floor的伤害|
|1|DGLEO_V2FULL32_DM_OFF|G1_DIRECT_LOSS_ABLATION|消融|关闭`direct_metric_accept`,仅保留endpoint评估,验证DM loss真实贡献|
|1|DGLEO_V2FULL32_DM_RELAXED|G1_DIRECT_LOSS_ABLATION|探索|放松DM目标,避免“罚很多但推不动”|
|1|DGLEO_V2FULL32_DM_PROXY_ALIGNED|G1_DIRECT_LOSS_ABLATION|探索|让DM目标与旧proxy风险代理更一致|
|1|DGLEO_V2FULL32_DM_HARD|G1_DIRECT_LOSS_ABLATION|压力|直接压p95/p99/source_overflow/proxy_vaccept上限|
|2|DGLEO_V2FULL32_SOURCE_OFF|G2_KNOWN_GEOMETRY|消融|关闭source episode收紧,验证source_overflow是否恶化|
|2|DGLEO_V2FULL32_SOURCE_FOCUS|G2_KNOWN_GEOMETRY|隔离|重点收紧known core,弱化proxy/DM干扰|
|2|DGLEO_V2FULL32_SOURCE_STRICT|G2_KNOWN_GEOMETRY|压力|强压source_episode_overflow和tail expansion|
|2|DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE|G2_KNOWN_GEOMETRY|保护|receiver-aware局部component方向,兼顾DG floor|
|3|DGLEO_V2FULL32_PROXY_OFF|G3_PROXY_BRIDGE|消融|关闭proxy unknown,验证旧proxy_vaccept/bridge风险来源|
|3|DGLEO_V2FULL32_PROXY_VACCEPT|G3_PROXY_BRIDGE|探索|集中压proxy_vaccept和proxy accept CVaR|
|3|DGLEO_V2FULL32_BRIDGE_LOW_DENSITY|G3_PROXY_BRIDGE|压力|集中压bridge_accept和low_density_accept|
|3|DGLEO_V2FULL32_SHELL_TAIL_PROXY|G3_PROXY_BRIDGE|压力|集中压shell/tail/overflow accept|
|4|DGLEO_V2FULL32_U_OFF|G4_U_TRISTATE|消融|关闭无标签loss,但保留三态审计,验证U_s是否真实贡献|
|4|DGLEO_V2FULL32_U_DOMAIN_SAT|G4_U_TRISTATE|消融|只用U_s域/星地一致性,不直接做open-set U loss|
|4|DGLEO_V2FULL32_U_DIRECT_QUAR|G4_U_TRISTATE|隔离|重点验证U_s direct/quarantine是否能参与几何收紧|
|4|DGLEO_V2FULL32_U_TRISTATE_FULL|G4_U_TRISTATE|探索|三态U_s全开,观察trusted_core/ambiguous_tail/outside_reject分布|
|5|DGLEO_V2FULL32_SAT_WEAK|G5_SAT_DG_STRESS|弱化|星地增强弱约束,观测sat floor和open-set几何基线|
|5|DGLEO_V2FULL32_SAT_STRONG|G5_SAT_DG_STRESS|探索|增强sat consistency/KD,验证LEO floor提升|
|5|DGLEO_V2FULL32_SAT_DOMAIN_ADV|G5_SAT_DG_STRESS|压力|域分类/ADV与星地一致性同时增强|
|5|DGLEO_V2FULL32_SAT_OPEN_PAIR|G5_SAT_DG_STRESS|压力|强化sat pair open-set约束,观察星地视图下p99/proxy_vaccept|
|6|DGLEO_V2FULL32_BUDGET_CLOSED|G6_GRADIENT_BUDGET|消融|闭集/KD/sat占优,验证几何loss被压制时open-set指标是否不动|
|6|DGLEO_V2FULL32_BUDGET_BALANCED|G6_GRADIENT_BUDGET|稳定|平衡预算参照|
|6|DGLEO_V2FULL32_BUDGET_OS_HIGH|G6_GRADIENT_BUDGET|压力|提高open-set预算,验证直接指标响应|
|6|DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE|G6_GRADIENT_BUDGET|保护|高open-set预算叠加KD/sat保护|
|7|DGLEO_V2FULL32_EXPORT_LOCAL|G7_EXPORT_GATE|探索|local component export严格化|
|7|DGLEO_V2FULL32_EXPORT_TAIL_GATE|G7_EXPORT_GATE|探索|tail delta gate稳定性|
|7|DGLEO_V2FULL32_EXPORT_FEASIBILITY|G7_EXPORT_GATE|压力|严格目标下的feasibility/audit行为|
|7|DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE|G7_EXPORT_GATE|保护|promotion-safe综合导出候选|

## 共同机制
- 数据:`--wisig_pkl .../ManySig.pkl`,`--split_mode tx_rx_day_1_7_2`,`labeled_ratio=0.10`,`unlabeled_ratio=0.70`,`source_val_ratio=0.20`。
- 星地视图:所有候选使用`--use_concat_sat_channel_aug`和`--no_concat_sat_ce_only`;不允许退化为只用TX CE。星地损失同时包含sat CE、sat consistency、teacher sat KL、U_s sat consistency和direct metric sat pair。
- 最终拒识边界:保留`endpoint_accept_v1`,并固定`--loss_gate_exported false`;DM软门控只作训练loss和审计代理,不作为最终拒识边界。
- tail safety:`p99_delta>2.0`阻断final export,`p99_delta>3.5`阻断promotion;同时启用CVaR delta gate。
- U_s三态:保留`trusted_core/ambiguous_tail/outside_reject`审计要求;即使`U_OFF`也必须产出三态审计并fail-closed。
- known几何:保留`source_episode_density_gate=true`和`source_episode_min_local_components=4`;source消融只关闭loss权重,不关闭审计门控。
- prototype导出:`phase1_source_zid_prototypes.pt`,local component accept,禁止global ball和tail auto accept。

## 主指标与成功标准
|类别|指标|成功标准|失败判据|
|---|---|---|---|
|DG泛化|overall_tx、strict_udu、receiver_floor、per-receiver最弱点|不低于OSFIX稳定候选,且best-final gap收窄|overall提升但strict UDU/floor下降|
|星地压力|sat mean/floor、sat strict floor、sat pair z_id距离|sat floor不低于EPOC/OSFIX参照,星地视图下open-set指标同步改善|只改善clean,星地p99/proxy_vaccept恶化|
|known收紧|zid_p50/p95/p99、zid_tail_cvar、tail_frac、r3sigma|p95/p99/CVaR同步下降,且final不扩tail|p95下降但p99/CVaR/overflow仍高|
|拒识代理|proxy_vaccept、proxy_vac_rate、proxy_auc、bridge_accept_rate、low_density_accept_rate|旧proxy_unknown指标同步下降,不是只让dm_*变好|dm_*下降但旧proxy_vaccept仍高|
|source episode|source_episode_overflow/source_overflow|SOURCE_FOCUS/STRICT相对SOURCE_OFF明显下降|source_overflow仍约0.97或比参照更差|
|边界比例|radius_to_inter_ratio、tail/overflow accept|ratio下降且tail/overflow accept下降|min_inter提升但proxy/bridge仍接收|
|U_s利用|U_s active epoch、trusted_core/ambiguous_tail/outside_reject、U direct/quarantine loss非零|U_TRISTATE_FULL优于U_OFF和U_DOMAIN_SAT_ONLY|U_s selected有数但direct/quarantine为0|
|导出安全|endpoint parity、reason code、artifact字段、p99_delta gate|best checkpoint和final export均有完整字段|final export绕过tail gate或prototype缺字段|

## 本地验证
|命令|结果|
|---|---|
|`conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_dgleo_v2full32_launcher.py -q`|4 passed;仅有`.pytest_cache`权限warning|
|`bash -n code/scripts/launch_phase1_dgleo_v2full32_20260707.sh`|通过|
|`bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh --dry-run --only=DGLEO_V2FULL32_FULL_STABLE`|确认全机制稳定候选包含DM/source/proxy/U_s/sat/domain/ADV和`phase1_source_zid_prototypes.pt`|
|`bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh --dry-run --only=DGLEO_V2FULL32_DM_OFF,DGLEO_V2FULL32_SOURCE_OFF,DGLEO_V2FULL32_PROXY_OFF,DGLEO_V2FULL32_U_OFF`|确认对应消融实际置零loss权重,不是只改候选名|

## N607启动边界
启动前边界:必须先执行`tools\n607_ssh_preflight.ps1`,记录GPU占用和server时间,再sync本地脚本/测试/报告。用户原始矩阵指定每张卡4个实验,脚本默认`MAX_ACTIVE_PER_GPU=4`;本次用户要求先启动主8候选,因此实际使用`MAX_ACTIVE_PER_GPU=1`,每张GPU一个实验,避免占满全32矩阵并保留后续扩展空间。

## 2026-07-08主8候选启动计划
用户要求先启动主要8个实验。本次不启动全32矩阵,只从每个机制组选择一个代表候选,每张GPU一个实验。

|GPU|candidate|选择理由|
|---:|---|---|
|0|DGLEO_V2FULL32_FULL_STABLE|全机制稳定主参照|
|1|DGLEO_V2FULL32_DM_PROXY_ALIGNED|验证DM训练代理与旧proxy/endpoint风险代理是否同步|
|2|DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE|验证receiver-aware local component与source episode density gate|
|3|DGLEO_V2FULL32_BRIDGE_LOW_DENSITY|直接压bridge_accept和low_density_accept|
|4|DGLEO_V2FULL32_U_TRISTATE_FULL|验证U_s trusted_core/ambiguous_tail/outside_reject三态全开|
|5|DGLEO_V2FULL32_SAT_OPEN_PAIR|验证星地concat视图下open-set几何和sat pair约束|
|6|DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE|验证提高open-set预算并用KD/sat保护泛化floor|
|7|DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE|验证promotion-safe导出、endpoint_accept_v1和tail delta gate|

本地验证:
- `bash -n code/scripts/launch_phase1_dgleo_v2full32_20260707.sh`:通过。
- `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_dgleo_v2full32_launcher.py -q`:4 passed,仅`.pytest_cache`权限warning。
- 8候选`--dry-run --only=...`:确认只输出上述8个candidate,每个GPU一个。

N607只读预检:
- `powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`:通过;server time为2026-07-08 09:52:00 CST;project root存在;8张RTX3090可见。
- 启动前GPU占用:0-7号GPU显存均约10MiB,无训练显存占用。
- `runs/phase1_dgleo_v2full32_20260707`和`logs/phase1_dgleo_v2full32_20260707`启动前不存在或为空。
- 远端`code/scripts/launch_phase1_dgleo_v2full32_20260707.sh`启动前缺失,需要同步。

同步计划:
|local|remote|
|---|---|
|`E:\type10-7\code\scripts\launch_phase1_dgleo_v2full32_20260707.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_v2full32_20260707.sh`|
|`E:\type10-7\code\tests\test_phase1_dgleo_v2full32_launcher.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase1_dgleo_v2full32_launcher.py`|

## 2026-07-08主8候选N607启动记录

### 本地验证与同步
- 本地验证:
  - `bash -n code/scripts/launch_phase1_dgleo_v2full32_20260707.sh`:通过。
  - `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_dgleo_v2full32_launcher.py -q`:4 passed;仅`.pytest_cache`权限warning。
  - `conda run --no-capture-output -n ssr-gpu python -m pytest code\tests\test_phase1_v2_control.py -q`:17 passed;仅`.pytest_cache`权限warning。
- N607预检:`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`通过;server为`dell-DSS8440`;project root为`/home/szu2070436088/2510044040/CV-SincNet`;8张RTX3090可见。
- 首次同步文件:
  - `code/scripts/launch_phase1_dgleo_v2full32_20260707.sh` -> `/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_v2full32_20260707.sh`,远端SHA256:`01e569943215d67a3b79158583e81a74ca73a12d24143679b59a743b414dcbc5`。
  - `code/tests/test_phase1_dgleo_v2full32_launcher.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase1_dgleo_v2full32_launcher.py`,远端SHA256:`f14e3db8cac2d70e541c40d027940c3f2e93ff243d2ecc10ea59456cfc8162c7`。
  - 本报告 -> `/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase1_dgleo_v2full32_20260707/report.md`。

### 首次启动异常与修复
- 首次RUN_ID:`phase1_dgleo_v2full32_20260707`。
- 首次PIDs:`507343`,`507422`,`507501`,`507580`,`507659`,`507738`,`507817`,`507896`。
- 异常:8个进程均快速退出,日志报`train_ssdg.py: error: unrecognized arguments: --phase1_v2_hard_gates ...`。
- 根因:远端`code/SSDG/train_ssdg.py`为旧版本,缺少v2 parser参数;远端`code/cvsrffi/phase1_v2_control.py`缺失。
- 修复同步:
  - `E:\type10-7\code\SSDG\train_ssdg.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`,SHA256:`9e86a3c7e8b082815cfc378bf28b4f8e5fc66493d0735d2770e8a0df1f71aa7e`。
  - `E:\type10-7\code\cvsrffi\phase1_v2_control.py` -> `/home/szu2070436088/2510044040/CV-SincNet/code/cvsrffi/phase1_v2_control.py`,SHA256:`48a94e6ae1f01c18d1bc752251ee327abbeec9fad91b1a3f6ffd3bae0df862da`。
- 修复验证:
  - `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/SSDG/train_ssdg.py code/cvsrffi/phase1_v2_control.py`:通过。
  - `train_ssdg.py --help`已包含`--phase1_v2_hard_gates`,`--endpoint_accept_policy_id`,`--tail_safety_state_machine`,`--u_tri_state_required`,`--feasibility_stage`。

### 成功启动
- 成功RUN_ID:`phase1_dgleo_v2full32_main8_20260708`。使用新RUN_ID是为了保留首次失败日志,避免覆盖`phase1_dgleo_v2full32_20260707`下已有失败artifact。
- 启动命令:

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
mkdir -p logs/phase1_dgleo_v2full32_main8_20260708
RUN_ID=phase1_dgleo_v2full32_main8_20260708 MAX_ACTIVE_PER_GPU=1 LAUNCH_SETTLE_SECONDS=2 \
  bash code/scripts/launch_phase1_dgleo_v2full32_20260707.sh \
  --only=DGLEO_V2FULL32_FULL_STABLE,DGLEO_V2FULL32_DM_PROXY_ALIGNED,DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE,DGLEO_V2FULL32_BRIDGE_LOW_DENSITY,DGLEO_V2FULL32_U_TRISTATE_FULL,DGLEO_V2FULL32_SAT_OPEN_PAIR,DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE,DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE \
  > logs/phase1_dgleo_v2full32_main8_20260708/launcher_main8_20260708.out 2>&1
```

|GPU|candidate|PID|log|metrics|
|---:|---|---:|---|---|
|0|DGLEO_V2FULL32_FULL_STABLE|511004|`logs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_FULL_STABLE.out`|`runs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_FULL_STABLE/metrics_epoch.csv`|
|1|DGLEO_V2FULL32_DM_PROXY_ALIGNED|511087|`logs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_DM_PROXY_ALIGNED.out`|`runs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_DM_PROXY_ALIGNED/metrics_epoch.csv`|
|2|DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE|511172|`logs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE.out`|`runs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE/metrics_epoch.csv`|
|3|DGLEO_V2FULL32_BRIDGE_LOW_DENSITY|511586|`logs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_BRIDGE_LOW_DENSITY.out`|`runs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_BRIDGE_LOW_DENSITY/metrics_epoch.csv`|
|4|DGLEO_V2FULL32_U_TRISTATE_FULL|511998|`logs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_U_TRISTATE_FULL.out`|`runs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_U_TRISTATE_FULL/metrics_epoch.csv`|
|5|DGLEO_V2FULL32_SAT_OPEN_PAIR|512411|`logs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_SAT_OPEN_PAIR.out`|`runs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_SAT_OPEN_PAIR/metrics_epoch.csv`|
|6|DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE|512824|`logs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE.out`|`runs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE/metrics_epoch.csv`|
|7|DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE|513239|`logs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE.out`|`runs/phase1_dgleo_v2full32_main8_20260708/DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE/metrics_epoch.csv`|

### 启动健康检查
- 2026-07-08 10:13:20 CST复查:8个PID均为`RUNNING`;各候选已到`E021/200`至`E023/200`,每个`metrics_epoch.csv`已有21至23行训练指标。
- 8个日志均包含`[CONFIG-PHASE1-V2]`,并已进入`[EPOCH-BEGIN]`训练循环。
- 错误扫描:未发现`Traceback`,`RuntimeError`,`unrecognized arguments`,`CUDA out of memory`,`NaN`。早前`memory`字符串命中仅为`proto_memory`配置字段,不是OOM证据。
- GPU状态:8张卡均有训练显存占用和GPU利用率;10:10 CST约为GPU0 2313MiB/GPU1 2151MiB/GPU2 2509MiB/GPU3 2187MiB/GPU4 2405MiB/GPU5 2501MiB/GPU6 2371MiB/GPU7 2299MiB。
- SSH清理:本地已确认无残留`ssh.exe`,无到`172.31.111.215:22`的`ESTABLISHED`连接。

### 后续检查要点
- 本次只说明主8候选已落地并启动健康,不说明open-set指标已改善,也不声明真实unknown_FAR、FPR95或Stage2成功。
- 下一次监控优先读取同一RUN_ID下8个`metrics_epoch.csv`,重点看`strict_udu`,`receiver_floor`,`sat_floor`,`zid_p95/p99`,`zid_tail_cvar`,`source_episode_overflow`,`proxy_vaccept`,`bridge_accept_rate`,`low_density_accept_rate`,`tail/overflow_accept`,`radius_to_inter_ratio`以及best-final tail expansion。

## 2026-07-08主8候选完成后分析

### 证据范围
- RUN_ID:`phase1_dgleo_v2full32_main8_20260708`。
- 远端完成状态:8个候选均有200行`metrics_epoch.csv`、完整stdout和`latest_ssdg.pth`;原训练PID均已退出。
- 本地完整解析快照:`E:\type10-7\automation_reports\CV-SincNet\phase1_dgleo_v2full32_20260707\analysis_artifacts\main8_20260708`。
- 完整stdout扫描:未发现`Traceback`,`RuntimeError`,`Killed`,`unrecognized arguments`,`CUDA out of memory`或`FATAL`。
- 协议边界:仍为Phase1 source-only地面训练;不能声明真实unknown_FAR、FPR95或Stage2成功。

### 核心结论
本组主8候选不是可promotion结果,而是一个明确的诊断性负例。8个候选全部从E28开始出现`train_skipped_nonfinite_grad=1.0`,一直持续到E200,共173个epoch。E28正是`direct_metric_start_epoch`;loss本身仍是有限值,但反传后梯度非有限,`optimizer.step()`被跳过。因此E28之后没有有效参数更新,后续open-set曲线、best/final和promotion gate只能作为“失效训练后的诊断读数”,不能证明v2机制优化有效。

### 训练健康与测试异常
|candidate|epochs|first nonfinite grad|skip epochs|test finite/unique|prototype export|verdict|
|---|---:|---:|---:|---:|---|---|
|FULL_STABLE|200|28|173|32/3|blocked|E28后无效|
|DM_PROXY_ALIGNED|200|28|173|32/3|blocked|E28后无效|
|RECEIVER_LOCAL_SAFE|200|28|173|32/3|blocked|E28后无效|
|BRIDGE_LOW_DENSITY|200|28|173|32/3|blocked|E28后无效|
|U_TRISTATE_FULL|200|28|173|32/3|blocked|E28后无效|
|SAT_OPEN_PAIR|200|28|173|32/3|blocked|E28后无效|
|BUDGET_OS_HIGH_SAFE|200|28|173|32/3|blocked|E28后无效|
|EXPORT_PROMOTE_SAFE|200|28|173|32/3|blocked|E28后无效|

测试结果“不变化”的直接原因不是测试集没读到,也不是测试调度没有触发。`interval_final`实际触发了32次测试:E10,E20,E30,E40...E170,E172...E200。但每个候选测试值只有3组唯一值,变化只发生在E10、E20、E30;E30之后全部保持同一组结果。结合E28开始所有epoch梯度非有限并跳过优化器step,根因是训练从E28后实质停止更新,测试自然固定。训练loss仍在变化,是因为前向随机增强/KD/几何loss仍在计算;它不等于参数继续更新。

### 泛化指标
以下泛化表为完成后读数,但必须带上E28后无效的限制。最强单点仍来自E10/E20附近,不是后期v2机制真正训练出来的结果。

|candidate|best overall@ep|final overall|best strict@ep|final strict|final receiver floor|weakest UDU rx|final sat floor|final sat strict floor|
|---|---:|---:|---:|---:|---:|---|---:|---:|
|FULL_STABLE|89.79@20|89.21|85.94@20|84.93|71.68|rx11:71.68|75.38|69.43|
|DM_PROXY_ALIGNED|89.32@10|89.09|84.92@10|84.12|71.87|rx11:71.87|75.08|68.45|
|RECEIVER_LOCAL_SAFE|89.89@20|89.71|85.98@20|85.92|76.62|rx11:76.62|75.24|69.54|
|BRIDGE_LOW_DENSITY|89.80@200|89.80|85.69@200|85.69|74.71|rx11:74.71|75.40|69.12|
|U_TRISTATE_FULL|89.67@20|89.34|85.67@20|84.49|70.99|rx11:70.99|74.81|68.61|
|SAT_OPEN_PAIR|90.22@20|89.80|86.11@20|85.65|73.11|rx11:73.11|75.49|69.23|
|BUDGET_OS_HIGH_SAFE|90.03@10|89.33|85.52@10|84.52|73.30|rx11:73.30|75.13|68.64|
|EXPORT_PROMOTE_SAFE|89.88@200|89.88|85.57@200|85.57|76.47|rx11:76.47|75.23|68.99|

与OSFIX16同口径相比,没有形成可信提升。OSFIX16最均衡候选`DGLEO_OSFIX_PROXY_A`已有overall 90.19、strict UDU 86.40、sat floor 75.90。本组即使看最佳单点,只有`SAT_OPEN_PAIR`达到overall 90.22/strict 86.11,但它发生在E20,早于direct metric启动和梯度失稳,不能证明v2机制带来增益;final整体低于OSFIX16强候选。receiver floor仍由rx11主导,弱receiver没有真正修复。

### open-set几何/拒识代理
|candidate|dm p95|dm p99|dm cvar|dm source_overflow|src ep overflow|dm proxy_vaccept|old proxy_vaccept|old bridge|dm tail/overflow|radius/inter|p99_delta|export|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|FULL_STABLE|50.64|84.30|71.77|0.746|0.954|0.125|0.688|1.000|0.158/0.094|1.045|24.93|no|
|DM_PROXY_ALIGNED|80.17|83.67|82.16|0.751|0.982|0.112|0.588|1.000|0.176/0.107|1.072|38.50|no|
|RECEIVER_LOCAL_SAFE|82.49|86.03|84.60|0.756|0.962|0.133|0.625|1.000|0.108/0.301|1.109|40.42|no|
|BRIDGE_LOW_DENSITY|69.71|83.53|77.39|0.767|0.955|0.121|0.625|1.000|0.135/0.000|1.020|31.42|no|
|U_TRISTATE_FULL|79.48|85.44|83.07|0.802|0.970|0.118|0.625|1.000|0.200/0.007|1.138|40.30|no|
|SAT_OPEN_PAIR|79.55|85.22|83.21|0.774|0.975|0.144|0.562|1.000|0.127/0.068|1.196|39.22|no|
|BUDGET_OS_HIGH_SAFE|71.71|85.15|81.85|0.775|0.990|0.109|0.637|1.000|0.134/0.114|1.044|39.26|no|
|EXPORT_PROMOTE_SAFE|80.82|85.70|84.08|0.758|0.993|0.156|0.550|1.000|0.134/0.277|1.131|37.11|no|

拒识潜力没有闭环改善:
- `dm_proxy_vaccept`看起来低,但旧`proxy_unknown_proxy_vaccept`仍在0.55至0.69之间,旧`bridge_accept_rate`仍为1.0。这再次说明动态DM软门控不等于最终/旧proxy接收边界。
- `source_episode_overflow`仍在0.954至0.993,没有解决source-only known域太散的问题。OSFIX16报告中该项中位约0.9729,本组没有有效下降。
- `dm p99`仍为83.5至86.0,tail cvar为71.8至84.6,距离目标p99约61至64、tail cvar约48至50仍很远。
- `radius/inter`仍大于1.0,没有把known半径压回类间安全比例内。
- `p99_delta`全部远超阻断阈值2.0/3.5,所有候选final export均被阻断,prototype export skipped。

### 无标签分支
|candidate|U direct active|U direct selected|U quarantine active|trusted/ambig/outside|verdict|
|---|---:|---:|---:|---|---|
|FULL_STABLE|0.0|110|0.0|110/0/0|未发挥|
|DM_PROXY_ALIGNED|0.0|110|0.0|110/0/0|未发挥|
|RECEIVER_LOCAL_SAFE|0.0|110|0.0|110/0/0|未发挥|
|BRIDGE_LOW_DENSITY|0.0|110|0.0|110/0/0|未发挥|
|U_TRISTATE_FULL|0.0|109|0.0|109/0/0|未发挥|
|SAT_OPEN_PAIR|0.0|110|0.0|110/0/0|未发挥|
|BUDGET_OS_HIGH_SAFE|0.0|110|0.0|110/0/0|未发挥|
|EXPORT_PROMOTE_SAFE|0.0|110|0.0|110/0/0|未发挥|

U_s分支仍没有真正参与open-set几何优化。虽然`selected`约109至110,但`train_u_dm_accept_active=0`,`train_u_quarantine_active=0`,三态退化为全部`trusted_core`,没有形成`ambiguous_tail/outside_reject`隔离。由于E28后训练已经无有效step,E102以后启动的U_s direct/quarantine即使配置存在,也不可能改变表征。

### 根因定位
1. E28前梯度有限,`train_grad_total`正常;E28后8个候选全部`train_skipped_nonfinite_grad=1.0`。例如`FULL_STABLE`E27 `train_grad_total=215.66`,E28开始`train_grad_total`为空/NaN且`train_skipped_nonfinite_grad=1.0`。
2. E28正是`direct_metric_start_epoch`;E27 `train_loss_direct_metric_accept=0`,E28 `train_dm_accept_active=1.0`,raw direct metric loss为69至186,加权loss虽小但反传后梯度非有限。
3. 代码路径在`code/SSDG/train_ssdg.py`中先`scaler.scale(loss).backward()`,再`scaler.unscale_(optimizer)`,随后`_grads_are_finite(model)`失败就跳过`optimizer.step()`。所以不是日志问题,而是训练防护正确地阻止了非有限梯度污染参数。
4. 高风险算子在`code/cvsrffi/losses.py::direct_metric_acceptance_loss`:多处`acos`角度、top-CVaR、极小温度softplus、半径/类间比、source episode sigmoid共同反传;即使loss值有限,角度导数、CVaR hard topk和小温度组合也足以造成NaN/Inf梯度。当前没有按子项隔离非有限梯度来源,因此只能定位到direct metric总loss启动。

### 当前判断
当前实验对Phase1的贡献是:发现v2全机制主8在direct metric启动后发生系统性非有限梯度,从而解释测试结果E30后冻结,并证明本轮主8不能用于判断机制改善。

当前不能声明的是:open-set指标改善、unknown拒识改善、Stage2候选推进、prototype export成功或v2闭环生效。

最主要风险是:direct metric角度/接收概率损失梯度失稳;旧proxy_vaccept和old bridge没有下降;source episode overflow仍接近0.97;U_s三态没有真正落地;tail safety全部阻断final export。

下一步动作:先修复direct_metric_acceptance_loss的梯度稳定性,新增子loss级finite guard和单元测试,再用小矩阵重跑`DM_OFF`、`DM_STABLE_SAFE`、`DM_GRAD_CLIPPED`、`DM_ITEM_ABLATION`四类验证。没有通过“E28后optimizer step持续有效、测试值随checkpoint变化、old proxy/source episode同步改善”之前,不要继续扩大v2全矩阵。
