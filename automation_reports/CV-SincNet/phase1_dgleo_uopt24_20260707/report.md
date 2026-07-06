# Phase1 DG-LEO UOPT24地面训练实验报告

## 基本信息

- 实验ID：`phase1_dgleo_uopt24_20260707`
- 时间：2026-07-07
- 操作Agent：Codex Phase1地面训练实验分析/落地Agent
- 阶段边界：Phase1 source-only地面训练。训练数据仅允许`ManySig.pkl`源接收机样本，不使用target receiver、ManyTx、真实unknown query或Stage2真实unknown指标。
- 用户授权：当前每张GPU已有约2个实验，用户明确要求本批每张卡再增加3个实验，因此launcher设置`MAX_ACTIVE_PER_GPU=5`，每GPU新增3个candidate，总计24个实验。

## 目标与假设

本批实验不是单纯提高闭集准确率，而是针对上周Phase1暴露的失败模式做直接损失优化：

- 已知类接收域尾部过长：直接优化`zid_p50/p95/p99`、`zid_tail_cvar`、`source_overflow`和`tail/overflow accept`。
- proxy/virtual unknown仍被接收：直接优化`proxy_vaccept`、`bridge_accept_rate`、`low_density_accept_rate`、`radius_to_inter_ratio`。
- 无标签U_s利用不足：在伪标签CE之外，加入U_s域分类、U_s ADV域混淆、U_s clean/LEO增强视图一致性、U_s直接指标损失。
- 星地视图弱：沿用EPOC强项`concat_sa`拼接训练，并且不是只做发射机CE；同步使用域分类、ADV、教师蒸馏、sat一致性和U_s直接指标损失。

成功标准：

- 泛化侧：`overall_tx`、`strict_udu`、`receiver_floor`、`satellite mean/floor`至少不低于ADV3B02/EPOC基线，且训练后期best/final gap缩小。
- 拒识潜力侧：同候选同epoch比较下，`proxy_vaccept`、`source_episode_overflow/source_overflow`、`bridge_accept_rate`、`low_density_accept_rate`、`tail/overflow accept`、`radius_to_inter_ratio`、`zid_p95/p99`、`zid_tail_cvar`下降。
- 冲突判据：若闭集DG提高但`proxy_vaccept`或`source_overflow/p99`上升，则只作为闭集强但open-set危险的诊断候选，不推进Stage2。

## 本地修改

|文件|作用|
|---|---|
|`code/SSDG/train_ssdg.py`|新增U_s域分类、U_s ADV、U_s clean-sat一致性、U_s直接指标损失及epoch日志。|
|`code/scripts/launch_phase1_dgleo_uopt24_20260707.sh`|新增24组Phase1 DG-LEO UOPT实验矩阵，每GPU3组，保留source-only和ManySig防护。|
|`code/tests/test_phase1_unlabeled_direct_training.py`|新增TDD测试，覆盖新参数、launcher矩阵、source-only防护和关键命令开关。|

## 本地验证

|命令|结果|
|---|---|
|`python -m py_compile .\code\SSDG\train_ssdg.py`|通过|
|`bash -n code/scripts/launch_phase1_dgleo_uopt24_20260707.sh`|通过|
|`bash code/scripts/launch_phase1_dgleo_uopt24_20260707.sh --dry-run --only=DGLEO_UOPT_P0_CORE_A`|通过，命令包含`concat_sa`、U_s域分类、U_s ADV、U_s星地一致性、U_s直接指标损失和ManySig source-only声明|
|`python -m pytest -q .\code\tests\test_phase1_unlabeled_direct_training.py`|4 passed；仅`.pytest_cache`权限警告|
|`python -m pytest -q .\code\tests\test_direct_metric_acceptance_loss.py .\code\tests\test_phase1_dgleo_directmetric16_launcher.py .\code\tests\test_phase1_unlabeled_direct_training.py`|9 passed；仅`.pytest_cache`权限警告|

## 实验矩阵

|GPU|候选|优先级/组|机制|核心变量|
|---:|---|---|---|---|
|0|`DGLEO_UOPT_P0_CORE_A`|P0_CORE|U_s域/ADV/sat/direct metric均衡主线|`lambda_u_domain=0.18`,`lambda_u_adv=0.08`,`lambda_u_sat_cons=0.26`,`lambda_u_direct_metric_accept=0.0045`|
|0|`DGLEO_UOPT_P0_CORE_B`|P0_CORE|更强直接拒识指标|更高overflow/proxy/bridge/tail权重|
|0|`DGLEO_UOPT_P0_CORE_C`|P0_CORE|安全弱化版|降低U_s直接指标和proxy压力，保护receiver floor|
|1|`DGLEO_UOPT_P0_GATE_A`|P0_GATE|tail quarantine主线|提高tail/overflow accept惩罚|
|1|`DGLEO_UOPT_P0_GATE_B`|P0_GATE|bridge/low-density强压|提高bridge和low-density accept权重|
|1|`DGLEO_UOPT_P0_GATE_C`|P0_GATE|bridge安全版|降低总压强，检查strict UDU回落|
|2|`DGLEO_UOPT_P0_SAT_A`|P0_SAT|U_s星地pair主线|提高U_s sat一致性和sat pair直接指标|
|2|`DGLEO_UOPT_P0_SAT_B`|P0_SAT|satellite floor保护|更高sat教师KL、sat一致性和pair约束|
|2|`DGLEO_UOPT_P0_SAT_C`|P0_SAT|sat软约束|保留星地收益，降低拒识损失干扰|
|3|`DGLEO_UOPT_P0_BAL_A`|P0_BAL|泛化/拒识均衡|同时抬U_s、sat、direct metric和domain|
|3|`DGLEO_UOPT_P0_BAL_B`|P0_BAL|高KD高直接指标|检查强教师是否抑制后期strict UDU下降|
|3|`DGLEO_UOPT_P0_BAL_C`|P0_BAL|均衡安全版|低学习率/弱压强对照|
|4|`DGLEO_UOPT_P1_QUOTA_A`|P1_QUOTA|高置信U_s配额|`u_direct_metric_min_selected=32`|
|4|`DGLEO_UOPT_P1_QUOTA_B`|P1_QUOTA|更高置信U_s配额|`u_direct_metric_min_selected=40`|
|4|`DGLEO_UOPT_P1_QUOTA_C`|P1_QUOTA|配额安全floor|晚启动、低压强|
|5|`DGLEO_UOPT_P1_LATE_A`|P1_LATE|U_s直接指标晚启动|`u_direct_metric_start_epoch=145`|
|5|`DGLEO_UOPT_P1_LATE_B`|P1_LATE|晚启动强约束|后期拒识直接修正|
|5|`DGLEO_UOPT_P1_LATE_C`|P1_LATE|U_s domain主控|检查U_s域监督是否单独改善DG|
|6|`DGLEO_UOPT_P1_ADV_A`|P1_ADV|强U_s ADV泄漏抑制|提高`lambda_u_adv`|
|6|`DGLEO_UOPT_P1_ADV_B`|P1_ADV|ADV+sat安全|强ADV同时保护sat floor|
|6|`DGLEO_UOPT_P1_ADV_C`|P1_ADV|ADV温和对照|检查过强ADV是否伤害strict UDU|
|7|`DGLEO_UOPT_P1_STRONG_A`|P1_STRONG|拒识上界压力|最大化直接指标压强|
|7|`DGLEO_UOPT_P1_STRONG_B`|P1_STRONG|旧类保护强压版|强教师与旧类安全保护|
|7|`DGLEO_UOPT_P1_STRONG_C`|P1_STRONG|sat pair强约束|最大化星地pair几何收缩|

## N607同步计划

|本地文件|远端目标|
|---|---|
|`E:\type10-7\code\SSDG\train_ssdg.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/SSDG/train_ssdg.py`|
|`E:\type10-7\code\scripts\launch_phase1_dgleo_uopt24_20260707.sh`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/launch_phase1_dgleo_uopt24_20260707.sh`|
|`E:\type10-7\code\tests\test_phase1_unlabeled_direct_training.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase1_unlabeled_direct_training.py`|

Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`，提交主题`Add Phase1 DG-LEO UOPT24 training matrix`；实际commit以`git log -1`为准。

## 远端启动记录

### 启动前证据

|项目|证据|
|---|---|
|N607直连预检|2026-07-07 00:57 CST通过；direct `N607`配置、identity、项目根`/home/szu2070436088/2510044040/CV-SincNet`、8张RTX3090均可见。|
|启动前GPU占用|每GPU已有2个compute训练进程，显存约4.9-5.1GB/24GB；本批按用户授权将每GPU新增3个到最多5个。|
|启动前磁盘|`/home` 11T，总已用2.8T，可用7.6T，使用率27%。|
|远端原`train_ssdg.py`hash|`b11a39f6900b772e13cce0de6be9c55f70bc08a0e7a54905057da1cdf1442518`。|
|远端同步后hash|`train_ssdg.py=d394ae6fda828f4c6e4c36f9fecae8d9af5ab6b49029dcb2b6033c3194be81b0`；launcher=`50c9d7b774d5ed0599c05c9707810781bf2f999c3633809a1e45f294f937132d`；test=`1ff1b60b7c58bcba3c2f857bc2360edfa731c98bae3b2ccc7d326da9ab9c3e8b`。|
|远端验证|`python -m py_compile code/SSDG/train_ssdg.py`通过；`bash -n code/scripts/launch_phase1_dgleo_uopt24_20260707.sh`通过；单候选dry-run通过并显示24组、`ManySig_only`、`source_only=1`、`concat_sat_ce_only=0`、U_s domain/ADV/sat/direct metric参数。|
|teacher checkpoint|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，大小8.2M，hash=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。|
|teacher source-only证据|源run日志显示`baseline_ckpt=<scratch> from_scratch=1`、`split_mode=tx_rx_day_1_7_2`、`L/U/V=8400/58800/16800 ratios=0.100/0.700/0.200`、`use_concat_sat_channel_aug=1`；同步manifest历史记录该teacher为`phase1_dataset=ManySig_only`、`manytx_in_training=0`、`target_receiver_samples_in_training=0`、`real_unknown_classes_in_training=0`。|
|ManySig数据hash|远端`Dataset_WigSig/ManySig.pkl` hash=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。|

### 子agent审查结论

|审查面|结论|处理|
|---|---|---|
|launcher/协议|通过source-only、24组、每GPU3组、`concat_sa`非CE-only、U_s多损失入参；阻塞项是报告远端证据和teacher来源证据未回填。|已在本节补齐preflight、GPU、hash、dry-run和teacher证据。|
|指标/实验设计|确认直接loss覆盖`proxy_vaccept`、`source_overflow`、`bridge_accept_rate`、`low_density_accept_rate`、`tail/overflow accept`、`radius_to_inter_ratio`、`zid_p50/p95/p99`、`zid_tail_cvar`，并覆盖U_s强增强与星地增强视图。|完成后必须同口径比较U_s是否真参与，尤其`train_u_dm_accept_active`和`train_u_dm_accept_selected`。|
|代码接入|审查仍在进行。|若后续指出阻塞项，暂停追加启动或记录为启动后修复风险。|

### 实际启动

启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
MAX_ACTIVE_PER_GPU=5 LAUNCH_SETTLE_SECONDS=8 bash code/scripts/launch_phase1_dgleo_uopt24_20260707.sh
```

提交结果：24/24 landed，日志根目录`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dgleo_uopt24_20260707`，运行根目录`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_dgleo_uopt24_20260707`。launcher未生成`.pid`文件，PID以下列`nvidia-smi pmon`中新增compute PID为准。

|GPU|候选|PID|log|
|---:|---|---:|---|
|0|`DGLEO_UOPT_P0_CORE_A`|3790852|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_CORE_A.out`|
|0|`DGLEO_UOPT_P0_CORE_B`|3791255|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_CORE_B.out`|
|0|`DGLEO_UOPT_P0_CORE_C`|3791658|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_CORE_C.out`|
|1|`DGLEO_UOPT_P0_GATE_A`|3792474|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_GATE_A.out`|
|1|`DGLEO_UOPT_P0_GATE_B`|3792879|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_GATE_B.out`|
|1|`DGLEO_UOPT_P0_GATE_C`|3793282|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_GATE_C.out`|
|2|`DGLEO_UOPT_P0_SAT_A`|3793685|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_SAT_A.out`|
|2|`DGLEO_UOPT_P0_SAT_B`|3794089|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_SAT_B.out`|
|2|`DGLEO_UOPT_P0_SAT_C`|3794492|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_SAT_C.out`|
|3|`DGLEO_UOPT_P0_BAL_A`|3794895|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_BAL_A.out`|
|3|`DGLEO_UOPT_P0_BAL_B`|3795299|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_BAL_B.out`|
|3|`DGLEO_UOPT_P0_BAL_C`|3796119|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P0_BAL_C.out`|
|4|`DGLEO_UOPT_P1_QUOTA_A`|3796539|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_QUOTA_A.out`|
|4|`DGLEO_UOPT_P1_QUOTA_B`|3796963|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_QUOTA_B.out`|
|4|`DGLEO_UOPT_P1_QUOTA_C`|3797400|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_QUOTA_C.out`|
|5|`DGLEO_UOPT_P1_LATE_A`|3797820|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_LATE_A.out`|
|5|`DGLEO_UOPT_P1_LATE_B`|3798224|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_LATE_B.out`|
|5|`DGLEO_UOPT_P1_LATE_C`|3798665|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_LATE_C.out`|
|6|`DGLEO_UOPT_P1_ADV_A`|3799494|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_ADV_A.out`|
|6|`DGLEO_UOPT_P1_ADV_B`|3799934|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_ADV_B.out`|
|6|`DGLEO_UOPT_P1_ADV_C`|3800355|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_ADV_C.out`|
|7|`DGLEO_UOPT_P1_STRONG_A`|3800775|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_STRONG_A.out`|
|7|`DGLEO_UOPT_P1_STRONG_B`|3801213|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_STRONG_B.out`|
|7|`DGLEO_UOPT_P1_STRONG_C`|3801636|`logs/phase1_dgleo_uopt24_20260707/DGLEO_UOPT_P1_STRONG_C.out`|

### 启动健康检查

|检查项|结果|
|---|---|
|日志数量|24个`.out`日志存在。|
|run目录数量|24个候选run目录存在。|
|metrics|24/24已生成`metrics_epoch.csv`。|
|GPU占用|每GPU为原2个compute进程+新增3个compute进程；启动后显存约10.0-11.5GB/24GB。|
|配置marker|24/24日志包含`[CONFIG-U-DIRECT]`、`[CONFIG-CONCAT-SAT]`、`[CONFIG-DATA]`。|
|训练marker|24/24进入metrics；首次检查时21/24进入`[EPOCH-BEGIN]`，二次检查已确认24/24生成metrics。|
|错误关键词|未发现`Traceback`、`RuntimeError`、`ImportError`、`ModuleNotFoundError`、`CUDA out of memory`、`unrecognized arguments`、batch size mismatch、out of bounds、`LOSS-SANITIZE`、`skipped_nonfinite`。|
|日志缺口|U_s直接指标日志目前有`u_dm_accept_zid_p95_deg/p99_deg`及accept类指标，但未单独记录U_s侧`zid_p50_deg`和`zid_tail_cvar_deg`；loss内部仍优化对应项，完成后U_s专属p50/tail_cvar需靠总`dm_accept`字段或后处理补算。|

## 风险与监控点

- 每GPU并发从2增至5是用户本轮明确授权，但仍需在启动前记录`nvidia-smi`和进程状态；若显存不足或出现OOM，应优先停止新增启动而不是干预已有实验。
- 不能把本批Phase1结果声明为真实`unknown_FAR`、`FPR95`、Stage2 old/new/unknown成功；本批最多证明DG、星地压力鲁棒性、known几何、proxy风险和prototype导出质量。
- `proxy_vaccept`接近1、`source_overflow`升高、`zid_p99/tail_cvar`不降、satellite floor提高但最弱receiver未修复，均判为失败或诊断性负例。
- 若best强但final回落，重点检查训练后期U_s伪标签噪声、domain/ADV过强、direct metric过早压缩尾部和sat一致性与receiver判别的冲突。
