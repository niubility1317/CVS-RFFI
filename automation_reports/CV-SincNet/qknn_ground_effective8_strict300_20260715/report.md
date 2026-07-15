# effective8严格300单元实验报告

## 基本信息

- 实验ID：`qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300`
- 日期：2026-07-15
- 操作者：Codex
- 目标：按`项目.md`执行effective8版Stage2-C正式矩阵；矩阵为5个target receiver×5个seed×3个new-count×4个K-shot，共300个预测单元；每单元覆盖3个LEO_weak场景，共900行独立评分。
- 比较对象：v14历史候选`effective8-r16-e12-leoweak-v14`、distinct ADV3B02 base runtime和direct ADV3B02流。
- 根工作区状态：`E:\type10-7`不是Git仓库；本报告以`E:\type10-7\github_publish\CVS-RFFI-repo`为Git承载面，并镜像到根工作区报告目录。

## 假设与协议边界

- 假设：44,048参数、12个ground-only epoch的effective8 merged runtime配合锁定三LEO场景对称头和1→3→5六阈值TTA，可在不读取query truth/role/quota的条件下保持v14机制。
- Phase2仅接收密封LEO_weak包；clean/raw数据集、缓存构建规格、scorer truth和旧loader路径均不得进入predictor。
- 推断为逐样本、面向所有registered classes；禁止query role、真实batch class count、class quota和global assignment。
- predictor先生成不可变预测artifact，scorer在独立进程中再联结truth；scorer输出不得反馈到推断、校准、选择、回滚或排序。
- 当前完整矩阵`launch_authority=false`；仅允许一个N607 Landlock+seccomp+sealed-memfd烟测。烟测收据通过本地授权器验证并进入Git后，才可启动300单元矩阵。

## 本地实现与验证

|类别|内容|状态|
|---|---|---|
|不可变输入|request、detached seal、manifest和包成员均复制到带`F_SEAL_WRITE/GROW/SHRINK/SEAL`的memfd|PASS|
|运行时隔离|Landlock只读运行时闭包、单一输出目录写权限、seccomp拒绝网络syscall、strace实际open ledger|PASS（本地静态/单测）；待N607实机烟测|
|effective8机制|distinct base/candidate TorchScript、锁定三场景对称头、六阈值逐样本1→3→5 TTA|PASS|
|矩阵结构|25个LEO_weak缓存、75个密封包、300个预测单元、900行评分|PASS|
|授权门禁|预烟测清单拒绝完整矩阵；绑定3场景`PROTOCOL_VALID`收据后才生成授权清单|PASS|

验证命令：

```powershell
(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate ssr-gpu
python -m pytest tests/test_stage2_predictor_runtime.py tests/test_stage2_predictor_entry.py tests/test_stage2_predictor_bundle.py tests/test_stage2_metric_scorer.py tests/test_run_cvs_stage2_predictor.py tests/test_phase2_symmetric_head.py tests/test_phase2_runtime_closure.py tests/test_phase2_pre_run_evidence.py tests/test_phase2_isolated_runner.py tests/test_phase2_candidate_capsule.py tests/test_phase2_bwrap_policy.py tests/test_build_cvs_stage2_predictor_request.py tests/test_build_cvs_stage2_predictor_bundle.py tests/test_build_cvs_stage2c_effective8_strict_plan.py tests/test_run_cvs_stage2c_effective8_strict_package.py tests/test_effective8_strict_plan_authority.py -q
```

结果：`98 passed`。

## N607执行计划

- 工作目录：待预检确认，预期为N607上的CV-SincNet项目根目录。
- Conda/Python环境：`ssr-gpu`；实际`python`路径将在预检和烟测收据中记录。
- GPU分配：烟测先用一个空闲GPU；正式矩阵使用8个分片并遵守每GPU最多2个训练实验的占用上限。Phase2推断不是训练，但仍按实时GPU证据选择设备。
- 日志：`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300/logs/`。
- 状态：`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300/state/`。
- 预期输出：75个predictor/scorer包对、300个sealed prediction artifact、300个post-run runtime evidence、300个scoring receipt和900行formal rows。
- 早停：任一包/单元出现胶囊SHA不符、缓存/包seal不符、Landlock或seccomp未生效、打开禁区路径、query协议字段冲突、预测artifact或scoring receipt不完整时立即fail closed。

## 当前状态

### N607只读预检

- 预检时间：2026-07-15 22:58 CST。
- 直接目标：`N607`，主机`dell-DSS8440`，项目根`/home/szu2070436088/2510044040/CV-SincNet`。
- GPU：8张RTX3090，预检时每张约10MiB显存、0%利用率。
- 实时训练清单：`gpu_compute=[]`、`active_training_processes=[]`、`unknown_training_active=false`。
- 依赖：`/usr/bin/strace`存在，Landlock ABI=4。
- 本地测试环境：`ssr-gpu`。N607没有同名环境；现有实验解释器为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch`2.1.0+cu121`、NumPy`2.2.5`。
- v14来源artifact：`candidate_lock_v2.json`、`effective8_adapter_fp16.pt`、`training_manifest.json`、`source_validation_v2/promotion_manifest.json`、`source_validation_v2/source_joint_feature_stats_fp32.npz`和原300单元`protocol_plan/plan_manifest.json`均存在。
- v14历史锁只含3个有效TTA阈值。当前6个float32部署槽位将缺失的`base_stop_min_score`、`shift3_stop_min_score`、`fusion_std_penalty`分别锁为无操作值`-1e9`、`-1e9`、`0.0`；历史head缺失的`gram_mix`和`uncertainty_penalty`锁为`0.0`。这些兼容值不使用target query，不改变v14三阈值决策语义。
- 每次SSH检查后：本地`ssh.exe=0`，到`172.31.111.215:22`和bridge的ESTABLISHED连接数为0。

### 版本与同步边界

- 本地Git提交：`654d2f7 feat: add strict effective8 phase2 runtime`。
- N607项目根不是Git仓库。
- 远端已有且将覆盖的3个文件均与提交父版本SHA一致：`stage2_predictor_bundle.py=08efc4...`、`build_cvs_stage2_predictor_bundle.py=9b9567...`、`run_phase2_landlock_isolated.py=c127d7...`；其余同步目标远端均不存在，因此未发现需保留的远端冲突。
- 同步命令模板：`scp -F E:\type10-7\tools\n607_ssh_config <local-file> N607:/home/szu2070436088/2510044040/CV-SincNet/<relative-file>`。
- 同步范围：本提交中的`code/cvsrffi`严格runtime文件、`code/scripts`胶囊/闭包/包构建与Landlock执行文件、`paper_reproduction/scripts`严格计划/分包/授权执行文件；不传输tests、数据集、checkpoint、现有run输出或无关工作树改动。

当前完整矩阵仍未获启动权限。下一步串行SCP并逐文件远端SHA核验，然后只生成v14严格runtime artifact；在报告记录确切命令后执行一个K=1烟测。

兼容导出修复验证：`18 passed`，覆盖历史三阈值导出、胶囊和effective8 runtime/head。

### 严格runtime artifact生成命令

生成前复核：2026-07-15 23:07 CST无训练进程，8张GPU约10MiB显存；目标`runtime_artifacts_strict_v1`不存在；`/home`剩余7.6TiB。

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
RUN=/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14
ART=$RUN/runtime_artifacts_strict_v1
CKPT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth
mkdir "$ART"
$PY code/scripts/export_cvs_phase2_effective8_lock_artifacts.py --candidate-lock "$RUN/candidate_lock_v2.json" --out-dir "$ART/lock_artifacts"
$PY code/scripts/export_adv3b02_effective8_torchscript.py --checkpoint "$CKPT" --adapter-state "$RUN/effective8_adapter_fp16.pt" --input-len 256 --base-runtime-out "$ART/base_runtime.ts" --candidate-runtime-out "$ART/candidate_runtime.ts" --parity-receipt-out "$ART/parity_receipt.json" --device cuda:0
$PY code/scripts/build_cvs_phase2_effective8_candidate_capsule.py --candidate-id effective8-r16-e12-leoweak-v14 --candidate-lock "$RUN/candidate_lock_v2.json" --base-runtime "$ART/base_runtime.ts" --candidate-runtime "$ART/candidate_runtime.ts" --adapter-state "$RUN/effective8_adapter_fp16.pt" --adapter-manifest "$RUN/training_manifest.json" --source-feature-stats "$RUN/source_validation_v2/source_joint_feature_stats_fp32.npz" --head-lock "$ART/lock_artifacts/symmetric_head_lock.json" --tta-policy "$ART/lock_artifacts/adaptive_tta_policy.json" --parity-receipt "$ART/parity_receipt.json" --out-json "$ART/candidate_capsule.json"
CAPSULE_SHA=$(sha256sum "$ART/candidate_capsule.json" | awk '{print $1}')
$PY code/scripts/build_cvs_phase2_effective8_runtime_configs.py --candidate-capsule "$ART/candidate_capsule.json" --expected-candidate-capsule-sha256 "$CAPSULE_SHA" --candidate-lock "$RUN/candidate_lock_v2.json" --source-feature-stats "$RUN/source_validation_v2/source_joint_feature_stats_fp32.npz" --tta-policy "$ART/lock_artifacts/adaptive_tta_policy.json" --out-dir "$ART/runtime_configs"
$PY code/scripts/build_cvs_stage2_runtime_closure.py --source-code-root code --output-root "$ART/runtime_closure"
```

每步stdout写入`$ART/01...05_*.json`，最终把`candidate_capsule.json`外部SHA和所有artifact SHA写入`$ART/artifact_sha256s.txt`。任一步失败即停止，不覆盖或删除部分输出。

## 完成后结果表

实验完成后在本节追加逐单元同一行结果，至少包含candidate ID、机制、receiver/TX split、K-shot、seed、old/seen-new/unknown指标、coverage/rollback/defer、loss/adapter摘要和最终判定。不得用来自不同单元的独立极值替代联合行。
