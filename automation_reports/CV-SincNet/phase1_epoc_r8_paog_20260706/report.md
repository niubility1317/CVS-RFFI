# phase1_epoc_r8_paog_20260706

## 基本信息

|字段|内容|
|---|---|
|experiment_id|phase1_epoc_r8_paog_20260706|
|timestamp|2026-07-06 07:20 CST|
|operator|Codex|
|objective|在R7+qknn8协同推理确认负证据后，设计并启动ADV3B02教师指导的source-only开集表征修复候选，优先解决LEO星地信道下未知类贴近已知类的问题，同时保护旧类floor。|
|base/teacher|`ADV3B02_CORE90_SOFT_E200`，checkpoint:`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|algorithm|`ADV3B02_PAOG`，Prototype-Anchored Open-set Guard|
|route|`source_only_adv3b02_paog`|
|status|n607_startup_pass_running|

## 协议边界

- 地面训练只使用`ManySig.pkl`源域旧类数据，`labeled_ratio=0.10`，不加载`ManyTx.pkl`。
- 不传入`--new_wisig_pkl`，不使用目标接收机`R_t`样本，不使用真实`Y_unknown`训练或阈值拟合。
- 未知训练信号只来自source-heldout proxy、synthetic empty-space/feature shell、soft unknown mixup、source episode半径约束和open-world feature shell。
- 真实`Y_unknown`只允许在后续Stage2-C qknn8协同复评中作为eval-only query。
- 本轮地面训练完成只表示产生新的底座候选；部署成功必须由后续`M=1..target receiver count`的Stage2-C qknn8协同复评证明。

## 算法设计

`ADV3B02_PAOG`的目标是把R7失败的“proxy AUC长期低于0.5、virtual unknown accept过高”转化为直接优化项：

|模块|机制|目的|
|---|---|---|
|ADV3B02 teacher KD|clean/sat logits KL + `z_id` MSE|保持旧类身份空间，避免拒识约束压坏旧类。|
|prototype compactness|`z_id`类内半径、CVaR尾部、source episode三sigma|收紧旧类接收半径，提高按类acceptance region可靠性。|
|proxy unknown shell|source-heldout proxy、legacy-hard虚拟unknown、energy margin、radius/inter ratio|把未知代理推出旧类prototype外壳，不接触真实未知。|
|synthetic empty-space mixup|低权重soft unknown mixup|覆盖类间空白区，作为辅助而非主导。|
|LEO source派生视图|`leo_clear_weak`,`leo_low_elev_weak`,`leo_rain_weak`|只用source样本派生星地信道压力，避免target泄漏。|

## 候选矩阵

|candidate|GPU|seed|机制侧重|teacher clean/sat/zid|proxy/start|soft mix|compact|source episode|phase2 radius cap|
|---|---:|---:|---|---|---|---:|---:|---:|---:|
|`EPOC_R8_PAOG_RADIUS_ENERGY`|0|706801|旧类半径保护+energy外壳；中等proxy强度|2.00/0.75/0.380|0.0080/E34|0.00008|0.052|0.0090|14|
|`EPOC_R8_PAOG_SHELL_BALANCED`|1|706811|更强proxy shell和低密度拒识；保留KD保护|1.85/0.82/0.340|0.0110/E30|0.00012|0.046|0.0080|15|

## 本地变更

|文件|目的|sha256|
|---|---|---|
|`E:\type10-7\code\scripts\launch_phase1_epoc_r8_paog_20260706.sh`|新增R8 PAOG训练启动器，source-only ManySig训练，ADV3B02教师蒸馏，导出Phase2 prototype；fail-closed拒绝`ManyTx/ManyRx/SingleDay/new_wisig/target/unknown`作为Phase1输入。|`BF90749778EF49F3EAA59CE8D1E4B94DD18ABDCEC10CA3BB39B1E62139F036F7`|
|`E:\type10-7\code\tests\test_phase1_epoc_r8_paog_launcher.py`|验证dry-run协议边界、候选数量、关键参数、无真实未知训练声明和`ManyTx.pkl`覆盖必须失败。|`17E37E5CF9F691FAAA2024F994F4EC9B2B25BBD7F527971D26F03AF471AE2C4B`|

Snapshot:

`E:\type10-7\code\snapshots\phase1_epoc_r8_paog_20260706`

## 本地验证

|命令|结果|
|---|---|
|`bash -n code/scripts/launch_phase1_epoc_r8_paog_20260706.sh`|PASS|
|`bash code/scripts/launch_phase1_epoc_r8_paog_20260706.sh --dry-run --only=EPOC_R8_PAOG_RADIUS_ENERGY`|PASS；输出确认ManySig-only、ADV3B02教师、`real_unknown_classes_in_training=0`、无`--new_wisig_pkl`。|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m pytest code\tests\test_phase1_epoc_r8_paog_launcher.py -q`|PASS:3 passed；覆盖默认dry-run和`WISIG_PKL=/tmp/ManyTx.pkl`拒绝路径；`.pytest_cache`权限warning与实验无关。|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m pytest code\tests\test_phase1_epoc_r8_paog_launcher.py code\tests\test_phase1_epoc_r7_floor_protected_shell_launcher.py code\tests\test_epoc_adv3b02_teacher_distill.py -q`|PASS:7 passed；`.pytest_cache`权限warning与实验无关。|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m py_compile code\tests\test_phase1_epoc_r8_paog_launcher.py code\SSDG\train_ssdg.py`|PASS；首次与pytest并行运行时触发Windows conda临时文件锁，已按串行重跑通过。|

## 子agent审查处理

|问题|处理|
|---|---|
|`WISIG_PKL`可被环境变量覆盖成`ManyTx.pkl`但dry-run仍声明`manytx_in_training=0`。|已修复为fail-closed：只允许`ManySig.pkl`，且路径不得包含`ManyTx/ManyRx/SingleDay/new_wisig/target/unknown`；新增pytest验证`WISIG_PKL=/tmp/ManyTx.pkl`必须失败。|
|`qknn8_stage2_required=1`可能被误读为Stage2-C已经成功。|dry-run新增`stage2_success_claim=0`、`deployment_success_claim=0`、`qknn8_same_row_eval_required=1`、`manytx_allowed_only_in_stage2_eval=1`。|
|PAOG参数偏激进，可能压旧类floor。|报告中保持为高约束诊断候选，不写成功声明；训练有joint-safe guard和ADV3B02 KD保护，后续以同row Stage2-C指标判定。|

## 远端计划

|字段|内容|
|---|---|
|remote_root|`/home/szu2070436088/2510044040/CV-SincNet`|
|python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|run_root|`runs/phase1_epoc_r8_paog_20260706`|
|log_root|`logs/phase1_epoc_r8_paog_20260706`|
|launch command|`cd /home/szu2070436088/2510044040/CV-SincNet; nohup bash code/scripts/launch_phase1_epoc_r8_paog_20260706.sh > logs/phase1_epoc_r8_paog_20260706_driver.out 2>&1 < /dev/null &`|
|GPU policy|默认GPU0和GPU1；启动前以N607预检和`nvidia-smi`确认低显存占用，启动器仍执行`MAX_ACTIVE_PER_GPU=2`slot检查。|
|startup checks|`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`、Traceback/OOM/NaN/argparse扫描。|
|expected outputs|每候选`metrics_epoch.csv`、`metrics_epoch.jsonl`、`best_joint_safe_ssdg.pth`、`phase2_zid_prototypes.pt/json`。|

## N607同步与远端验证

|字段|内容|
|---|---|
|preflight|2026-07-06 07:22 CST PASS；direct `N607`可用，project root可见，GPU0-7均约10MiB。|
|remote sync|R8 launcher、test、snapshot、report、`code/SYNC_MANIFEST.txt`已同步到`/home/szu2070436088/2510044040/CV-SincNet`对应路径。|
|remote hash|远端`sha256sum`匹配本地最终哈希：launcher `bf90749778ef49f3eaa59ce8d1e4b94dd18abdcec10ca3bb39b1e62139f036f7`，test `17e37e5cf9f691faaa2024f994f4ec9b2b25bbd7f527971d26f03af471ae2c4b`，report `fb3778cdebcdc65ac6a69c1685f68d39d4c0499b9bead75fa7c85c2d9d97aa53`，manifest `b062f206adeacaeb906b2c30fd4d2e615013cb327a9240b3424df094bdb629fc`。|
|remote verify|远端`bash -n` PASS；dry-run含`stage2_success_claim=0`和`deployment_success_claim=0`；`WISIG_PKL=/tmp/ManyTx.pkl`触发fail-closed拒绝；远端`py_compile` PASS。|
|remote pytest note|远端`CVS-RFFI`环境无`pytest`模块，因此远端用`py_compile`和dry-run/guard验证替代；本地`ssr-gpu`已完成pytest验证。|
|ssh cleanup|同步和验证后本地复查`ssh.exe=0`、N607/bridge 22端口ESTABLISHED连接均为0。|

## N607启动证据

|字段|内容|
|---|---|
|launch command|`cd /home/szu2070436088/2510044040/CV-SincNet; mkdir -p logs/phase1_epoc_r8_paog_20260706; nohup bash code/scripts/launch_phase1_epoc_r8_paog_20260706.sh > logs/phase1_epoc_r8_paog_20260706_driver.out 2>&1 < /dev/null &`|
|driver PID|`3289090`|
|candidate PIDs/logs|`EPOC_R8_PAOG_RADIUS_ENERGY`:PID`3289110`,GPU0,log`logs/phase1_epoc_r8_paog_20260706/EPOC_R8_PAOG_RADIUS_ENERGY.out`;`EPOC_R8_PAOG_SHELL_BALANCED`:PID`3289536`,GPU1,log`logs/phase1_epoc_r8_paog_20260706/EPOC_R8_PAOG_SHELL_BALANCED.out`。|
|startup health|2026-07-06 07:24 CST复查：GPU0约2241MiB、GPU1约2055MiB；两个候选均进入epoch，`RADIUS_ENERGY`到E002/200，`SHELL_BALANCED`到E001/200。日志含`[CONFIG-TEACHER]`、`[CONFIG-LOSS]`、`[CONFIG-SAT]`、`[EPOCH-BEGIN]`，未见Traceback、RuntimeError、CUDA OOM、out-of-memory、NaN、unrecognized arguments或Killed。|
|claim boundary|当前只证明远端启动和startup PASS；不得声明Stage2-C成功、部署成功或目标完成。训练完成导出prototype后，必须接`phase2_collaborative_open_set_qknn_eval.py --collab_counts all --qknn_k 8`做M=1..target receiver count同row复评。|
|ssh cleanup|启动和健康检查后本地复查`ssh.exe=0`、N607/bridge 22端口ESTABLISHED连接均为0。|

## 成功/失败判据

|阶段|判据|
|---|---|
|startup PASS|两个候选均启动，进入epoch，日志无配置、argparse、OOM或NaN错误。|
|training useful|`best_test_tx`和sat/receiver floor不低于R7同类水平，同时proxy AUC明显超过0.5、virtual accept显著下降。|
|进入Stage2-C条件|训练完成并导出prototype后，使用qknn8和协同推理`M=1..target receiver count`复评旧类、seen-new和真实unknown拒识。|
|失败判据|若proxy AUC仍低于0.5或Stage2-C old/seen-new/unknown仍远离目标，则表明当前损失族仍不足，需要新增显式Gaussian/Mahalanobis或轻量reject head，而不是继续只调壳层权重。|
