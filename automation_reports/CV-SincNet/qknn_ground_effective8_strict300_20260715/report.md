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

第一次生成在第二步停止：N607缺少`code/scripts/export_adv3b02_effective8_torchscript.py`。`runtime_artifacts_strict_v1`保留第一步锁artifact和0字节第二步日志作为失败证据，不删除、不复用。依赖闭包补查发现还缺少`build_cvs_stage2_runtime_closure.py`、`build_cvs_stage2_predictor_request.py`、`score_cvs_stage2_sealed_prediction.py`、`stage2_prediction_artifact.py`、`phase2_isolated_runner.py`和`stage2_metric_scorer.py`；这些远端均不存在。远端`phase2_runtime_contract.py`对应Git历史提交`e90a52f`，属于已确认的旧受控版本，将更新为本地当前受测版本；其他已存在依赖与本地SHA一致。补齐后改用全新的`runtime_artifacts_strict_v2`，不覆盖v1。

第二次生成在candidate TorchScript的PyTorch graph sanity-check停止：ADV3B02 FFT路径在trace复跑时把数值等价常量折叠为不同内部dtype（`complex128`与`float64`），造成graph文本不一致。v2部分目录保留。导出器改为关闭该不稳定graph复跑比较，但保留并依赖导出流程原有的独立数值门禁：base eager↔reloaded TorchScript、injected↔merged以及merged↔reloaded TorchScript的feature/logit最大绝对误差均必须≤`1e-4`。修复测试`9 passed`；下一次使用全新`runtime_artifacts_strict_v3`。

第三次生成通过trace但在reloaded runtime数值检查前暴露batch冻结：ADV3B02内部把trace时batch维转为Python整数，旧wrapper固定在trace batch=2，无法处理8行parity probes。v3保留。正式wrapper改为内部固定256行部署batch、输入不足时零填充、输出按真实动态行数切回；predictor本身按≤256行分批，因此末批可变而底层ADV3B02始终看到固定batch。跨trace batch=2、实际batch=4的回归测试`2 passed`；下一次使用全新`runtime_artifacts_strict_v4`。

第四次生成的双TorchScript和实际数值parity通过，胶囊审计因仍把历史缺失阈值读成`null`而拒绝无操作槽位。v4保留。审计器现要求历史3个有效阈值逐值一致，并仅允许缺失3项精确编码为`-1e9/-1e9/0.0`；兼容审计测试`8 passed`。下一次使用全新`runtime_artifacts_strict_v5`。

第五次生成推进到胶囊candidate-lock复验后停止：历史lock包含训练时期整个工作区的代码SHA，而当前N607若干训练脚本已更新。v5保留。部署验证现默认绑定candidate-lock文件自身SHA、所有不可变训练/选择artifact SHA、checkpoint/adapter/source stats/head/TTA/双runtime SHA和数值parity；当前9文件预测runtime由独立闭包SHA绑定。对历史工作区代码的逐文件复验保留为显式可选审计，不再要求可变工作区回退到旧版本；路径穿越或非法SHA声明仍拒绝。相关测试`11 passed`。下一次使用全新`runtime_artifacts_strict_v6`。

第六次生成完整通过：candidate capsule SHA=`825312b058a43d122d4985331ce280abb95dfe6af9aa69c0f83bf1c5fe67efd6`；base runtime SHA=`f7921a6078a1fc540270a03b05473c231dc0eae6a2931614f6c2a806d2135001`；candidate runtime SHA=`ce12b7c3eed2f514113d884e085d73e05526ed44f2891c3d8ccdf4d204dd50c9`；9文件runtime closure SHA=`b7524a8f8676cb53637009fa4bffef7cbb3d28e2b969fdcb5e0f59aebc536a3d`。注入态↔合并态feature/logit最大误差分别为`2.682209e-7`和`3.814697e-6`，base/merged↔重载TorchScript误差均为0。胶囊增量持久状态109,818字节，小于256KiB上限。

回传证据位于`evidence/runtime_artifacts_strict_v6/`。严格计划CLI入口修复后测试`6 passed`；生成清单SHA=`65a494109b28ed3ed233264d20f0d2ec4469a739480c3f90bcce57ea27d6ba88`，结构为25个cache、75个sealed package、300个prediction cell、900行formal scorer row，当前`launch_authority=false`、`authority_state=N607_LANDLOCK_SMOKE_REQUIRED`。

### N607严格烟测

- Git证据提交：`dcb0aa0 feat: bind strict plan to N607 runtime capsule`。
- 清单远端路径：`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300/protocol_plan/strict_plan_manifest.json`。
- 烟测单元：receiver=`20-1`、seed=`713101`、new-count=`20`、K=`1`、3个LEO_weak场景；只允许烟测，不允许完整矩阵。
- GPU：`cuda:0`，执行前重新检查实时进程和GPU占用。
- predictor输出：`..._landlock_strict300/cells/rx_20_1__seed_713101__new_20__k_1/predictor_output/`。
- scorer输出：同一cell下`scoring_output/`；truth只存在package的独立scorer root。
- 烟测收据：`..._landlock_strict300/smoke_receipt.json`。
- 失败边界：cache/package/cell出现部分目录时不覆盖；胶囊、seal、Landlock、seccomp、memfd、open ledger、prediction seal或独立scorer任一失败即停止。

确切命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
STRICT=/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300
PYTHONPATH=code:. "$PY" paper_reproduction/scripts/run_cvs_stage2c_effective8_strict_plan.py \
  --plan-manifest "$STRICT/protocol_plan/strict_plan_manifest.json" \
  --project-root /home/szu2070436088/2510044040/CV-SincNet \
  --stage smoke --device cuda:0 \
  --log-dir "$STRICT/logs/smoke" \
  --smoke-receipt "$STRICT/smoke_receipt.json"
```

只有`smoke_receipt.json`回传本地、SHA进入Git且授权器生成`launch_authority=true`的新清单后，才可启动8个matrix shard。

为避免长驻SSH，实际启动采用等价后台包装：先`mkdir -p "$STRICT/logs/smoke"`，再执行`nohup env PYTHONPATH=code:. "$PY" paper_reproduction/scripts/run_cvs_stage2c_effective8_strict_plan.py ... > "$STRICT/logs/smoke_driver.out" 2>&1 &`，PID写入`$STRICT/smoke_driver.pid`。后台只运行上述单一smoke stage；使用短连接检查PID、日志、GPU和`smoke_receipt.json`，不会启动matrix shard。

第一次smoke attempt（PID`1867324`）完成了receiver=`20-1`、seed=`713101`的3个LEO_weak缓存文件，然后在封包前失败：v6严格清单误把`candidate_lock_v2.json`放在runtime artifact子目录。未创建package、prediction或scorer输出；原PID/日志和v6清单保留。生成器修复为指向真实v14 run根，回归测试`5 passed`；新v7清单SHA=`cf6c4572130245653e753031ff20f2d42b0511fb9b3635cc147c02e359c0ecbf`，仍为`launch_authority=false`。第二次attempt复用已验证缓存，但使用独立`smoke_driver_attempt2.pid/out`，不覆盖首次证据。

2026-07-15 23:34 CST重新执行直连只读预检：项目根可见，8张RTX3090利用率均为0、显存均约10 MiB。提交`e75df2c`中的生成器和v7清单已同步并逐项核验：生成器SHA=`88dee562ff5163db30b4603dc12dfe7563c6224cca65861b7977f8f5ac11f2c3`，远端清单`protocol_plan/strict_plan_manifest_v7.json`SHA=`cf6c4572130245653e753031ff20f2d42b0511fb9b3635cc147c02e359c0ecbf`。同步完成后本地`ssh.exe=0`、N607/桥接机TCP22连接数为0。attempt2仅运行`--stage smoke --device cuda:0`，driver日志写入`logs/smoke_driver_attempt2.out`，阶段日志目录为`logs/smoke_attempt2`，PID写入`smoke_driver_attempt2.pid`；最终receipt仍为尚不存在的`smoke_receipt.json`，不存在时才允许启动。

attempt2于PID`1871127`落地后fail closed。它完整生成并封印predictor/scorer package，但pre-run evidence拒绝`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，因为该路径是指向常规文件`python3.10`的符号链接；未生成prediction、scoring receipt或smoke receipt。修复不改变“证据输入必须为非符号链接常规文件”规则，而是在严格package runner中把`sys.executable`解析为物理解释器，再同时传给pre-run evidence与Landlock pinned runner；相关测试`5 passed`。attempt3允许复用已经完整且会重新校验seal的package，不覆盖它，也不复用任何prediction/scorer输出；使用独立`smoke_driver_attempt3.pid`、`logs/smoke_driver_attempt3.out`和`logs/smoke_attempt3`。

提交`6f8bd42`中的strict package runner已同步到N607，local/remote SHA均为`241b3a4b908a9d5c34978c78af1ae47cc406ef9014376c63d4eaa8761c5a5d14`。23:42实时清单再次确认`gpu_compute=[]`、`active_training_processes=[]`，同步与清单连接结束后无本地SSH/TCP22残留。attempt3仍仅获smoke权限，完整矩阵未授权。

attempt3以PID`1874959`运行，成功生成并封印pre-run runtime evidence，随后请求契约因`tta_policy.uses_class_quota=false`包含禁词`class_quota`而fail closed；没有prediction、scoring receipt或smoke receipt。修复仅允许该精确TTA guard以布尔`false`穿过predictor边界，`true`、字符串或其他quota键仍拒绝；同时严格计划生成器新增安全单路径suffix，使后续输出落在全新run根。18项request/runtime/strict-plan/package/CLI测试通过。v8清单使用experiment ID=`qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300_v2`，仍覆盖25 cache、75 package、300 cell、900 formal rows，`launch_authority=false`，SHA=`1f33237f8f3a0c869f2c1d61d324893fe2678656336c4210f924a81a57962f0a`。原strict300根及attempt1/2/3全部保持不动。

同步前闭包审计确认strict_v6的9文件runtime closure内含旧`phase2_runtime_contract.py`（成员SHA=`8a8e372b1cc5534b3f3926c350acd403d5ad2b7cf9759fe31f80eded03f5e8c3`）。因此v8清单不发布、不启动，标记为本地superseded fail-closed计划。下一步在全新`runtime_artifacts_strict_v7`中逐项复制v6已验证且字节不变的base/candidate runtime、capsule、parity、lock/runtime config，再从提交`710f80c`的当前代码重新构建独立runtime closure并核验全部SHA；不修改strict_v6。只有随后重生成且绑定strict_v7的新清单才可进入烟测。

strict_v7构建完成：12项复用artifact与strict_v6逐字节`cmp=PASS`，candidate capsule=`825312b0…efd6`、base runtime=`f7921a60…5001`、candidate runtime=`ce12b7c3…50c9`保持不变；新9文件runtime closure SHA=`f3cf3a694bb3613e7d1c47a66bf66fccbbbd0255d188932092508e69dfd0a6ab`，其中`phase2_runtime_contract.py`SHA=`dae574f63d1c93ba232a21059b1c1382322467320244c0e80cc6804b97ada6e4`。证据回传至`evidence/runtime_artifacts_strict_v7/`。新v9清单绑定strict_v7和全新`..._landlock_strict300_v2`输出根，覆盖25/75/300/900，仍为`launch_authority=false`，SHA=`9d8bcb47fb9f0541e80a763aff4c0977cd7f595068aac0d79b6f269c4cffebc3`。

23:51直连预检及实时训练清单再次通过：8张GPU利用率为0，`gpu_compute=[]`、`active_training_processes=[]`。v9已同步到新根`protocol_plan/strict_plan_manifest_v9.json`并核验相同SHA。v2第一次smoke只运行receiver=`20-1`、seed=`713101`、new-count=`20`、K=`1`、`cuda:0`；PID写入`smoke_driver_v2_attempt1.pid`，driver日志写入`logs/smoke_driver_v2_attempt1.out`，阶段日志目录为`logs/smoke_v2_attempt1`，最终receipt为新根下`smoke_receipt.json`。仅当这些路径均不存在时启动。

v2 attempt1以PID`1880017`运行，pre-run evidence再次成功封印，但请求契约随后因`tta_policy.uses_query_role=false`命中`query_role`禁词而fail closed；没有prediction/scoring/smoke receipt。契约allowlist现覆盖adaptive TTA要求的完整否定guard：仅允许`uses_query_role=false`与`uses_class_quota=false`，任一为`true`仍拒绝；包含完整3项TTA访问guard的13项request/runtime测试通过。由于v2 cell已有不可变pre-run evidence，下一次使用全新`..._landlock_strict300_v3`输出根，并基于修复代码构建全新runtime closure；v2根不修改。

v3修复计划只更新隔离闭包，不重导出模型：先确认GPU无任务、`runtime_artifacts_strict_v8`与v3运行根均不存在；从strict_v7逐文件复制已验证的双TorchScript、candidate capsule、parity、lock artifacts、runtime configs和步骤收据，以`cmp`逐项确认字节一致；再使用提交`af2e939`的当前`code/`在strict_v8中新建9文件runtime closure，生成全量SHA清单。旧strict_v7、v2运行根及其失败证据保持只读不变。预期新闭包唯一相关变化是`phase2_runtime_contract.py`从SHA=`dae574f6…a6e4`更新为`be2a7daf…f119`。

完成strict_v8核验并回传证据后，才允许从同一源计划生成绑定strict_v8、suffix=`landlock_strict300_v3`的新strict清单；新清单必须继续保持`launch_authority=false`并覆盖25 cache、75 package、300 cell、900 row。随后在报告记录新清单SHA和确切路径，SCP至全新v3根，并仅启动一个receiver=`20-1`、seed=`713101`、20个真实target-new、K=`1`、3个LEO_weak场景的`cuda:0` smoke。

strict_v8已从修复提交构建：12项模型/capsule/config复用artifact逐字节一致，新闭包SHA=`e04ebc9eb907f6eb144bbbd888f3d4447b9914ca94c688d41134717ea086fd1e`，闭包内契约成员SHA=`be2a7daf3b192ad7a9d26450d44eec41f76c55ff47286e4859b931515f13f119`。新v10清单绑定strict_v8与全新`..._landlock_strict300_v3`输出根，仍为25/75/300/900、`launch_authority=false`，SHA=`d748b0b558448ddf9e961023f1fb21766a6e72ae505a4bde955cab1d6597d50b`。

23:55预检/实时清单再次确认8张GPU空闲、无训练进程。v10已同步至v3新根`protocol_plan/strict_plan_manifest_v10.json`并核验SHA。v3 smoke仍只获`cuda:0`单单元权限，使用`smoke_driver_v3_attempt1.pid`、`logs/smoke_driver_v3_attempt1.out`、`logs/smoke_v3_attempt1`和新根`smoke_receipt.json`，路径不存在时才启动。

v3 attempt1以PID`1882465`结束。完整读取3,760字节driver日志并核对全部v3文件后确认：request及4项pre-run evidence已成功生成，随后外层`run_cvs_stage2_landlock_pinned.py`导入`phase2_isolated_runner`时因远端当时缺少`cvsrffi.phase2_bwrap_policy`而失败；`prediction_artifact.cvspred`、scoring receipt、cell receipt和smoke receipt均不存在。该失败未进入ADV3B02/effective8推理，不含任何`old_acc/seen_new_acc/H_old_new`性能证据。因空`predictor_output`目录已经创建，严格runner禁止在v3原地续跑或清理。

依赖闭包复核覆盖外层runner实际导入的8个`cvsrffi`文件；当前本地与N607逐文件SHA完全一致，包含`phase2_bwrap_policy.py`SHA=`a9258cbb…e3b`。在`ssr-gpu`下显式导入`phase2_isolated_runner`和`phase2_bwrap_policy`通过，27项bwrap/pre-run/isolated-runner/strict-package/strict-plan测试通过。下一次不重建strict_v8，仅生成绑定同一strict_v8、suffix=`landlock_strict300_v4`的新v11 fail-closed清单，并使用全新v4输出根；v3证据保持不动。

v3 attempt1以PID`1882465`运行，成功生成pre-run evidence与truth-free request，首次进入Landlock pinned runner前因N607工作区缺少其外层依赖`code/cvsrffi/phase2_bwrap_policy.py`而fail closed；尚未产生prediction/scoring/smoke receipt。由于空`predictor_output`目录已创建，v3不复用。下一步先同步该本地已受控依赖并在N607执行只读完整import smoke；只有import闭包通过后才生成全新v4计划/输出根。

依赖核验显示`phase2_isolated_runner.py`与`phase2_bwrap_policy.py`已存在且SHA与本地一致，实际唯一缺失项为`phase2_pre_run_evidence.py`。补齐后远端SHA=`60bdc81b3c579ba5a0d05b797dea5f686d57b3e5a0959277e5e771356f492c06`，以实际直接脚本入口执行`run_cvs_stage2_landlock_pinned.py --help`的无写入import smoke为`PASS`。v11清单继续绑定strict_v8，但输出改到全新`..._landlock_strict300_v4`根；25/75/300/900、`launch_authority=false`，SHA=`17a6de5501700b204047e1bb663af30a374c4f50e2af4914869c9f5e21c9aa96`。

23:59预检与实时训练清单为`gpu_compute=[]`、`active_training_processes=[]`，8张GPU空闲。v11已同步至v4新根并核验SHA。v4 smoke使用独立`smoke_driver_v4_attempt1.pid`、`logs/smoke_driver_v4_attempt1.out`、`logs/smoke_v4_attempt1`和新根`smoke_receipt.json`；完整矩阵继续无权限。

v4 attempt1以PID`1885770`运行，成功推进到sealed memfd snapshot创建，但N607的Python 3.10未暴露`os.memfd_create`，因此在预测前fail closed；libc实际导出`memfd_create`且主机为`x86_64`。本地实现优先使用Python API、缺失时调用libc（再保留架构限定syscall后备），写入后仍执行`F_ADD_SEALS`并复验完整`REQUIRED_SEALS`，未降低不可变性门禁；10项memfd/package/request测试通过。v4已有partial predictor output，不复用；后续需新闭包和全新v5根。

完整读取4,302字节v4 driver日志并核对cell目录后确认：prediction/scoring/cell/smoke receipt全部不存在，故该次仍无性能指标。由于`phase2_memfd_snapshot.py`属于9文件密封predictor closure，新引入的`ctypes/platform`必须同时进入runtime closure精确外部导入与成员导入白名单；否则闭包构建应fail closed。白名单已同步更新，`py_compile`及39项memfd/runtime-closure/bwrap/pre-run/isolated-runner/strict-package测试通过。下一次必须基于该提交新建runtime artifact和全新v5运行根，不能让旧strict_v8/v11继续执行。

N607独立memfd smoke确认libc后备可创建memfd且`seals=15/required=15`。strict_v9因远端尚未同步closure白名单而在构建时fail closed并保留；随后同步提交`5d87bdd`中的`phase2_runtime_closure.py`后，strict_v10完整构建通过，12项复用artifact逐字节一致，新闭包SHA=`81bbf141901043b2e0b0386f296d2873d628011c17225b7e7f1b722c7e2b2c50`。v12清单绑定strict_v10与全新`..._landlock_strict300_v5`根，25/75/300/900、`launch_authority=false`，SHA=`e13b72266aee7e3058a921438517861f631ea17b58a9d7af06c356f02c627b48`。

00:07直连预检/实时清单再次确认8张GPU空闲，`gpu_compute=[]`、`active_training_processes=[]`。v12已同步至v5新根并核验SHA。v5 smoke使用独立`smoke_driver_v5_attempt1.pid`、`logs/smoke_driver_v5_attempt1.out`、`logs/smoke_v5_attempt1`及新根`smoke_receipt.json`；矩阵仍未授权。

v5 attempt1以PID`1890476`运行：memfd snapshot、Landlock、no_new_privs和seccomp attestation均已生效，但旧seccomp把CUDA所需`AF_UNIX/SOCK_SEQPACKET`本机IPC一并拒绝，子进程返回1；无prediction/scoring/smoke receipt。完整666,665字节strace与281,640字节audit已回传`evidence/smoke_v5_failure/`。新策略只允许`socket/socketpair`的domain=`AF_UNIX`，继续以EPERM拒绝所有其他socket domain；子进程不继承网络socket，runtime attestation精确声明`network_access_allowed=false`、`ip_network_socket_creation_seccomp_denied=true`和`unix_domain_ipc_allowed=true`。同时修复audit对`expressionrawdomain/graph_drawer`的`raw`子串误报，并让失败消息保留有界stderr tail。22项Landlock/pinned/audit/isolated/strict-package测试通过。v5不复用，后续使用全新v6根。

v5 attempt1以PID`1890476`首次完整进入Landlock predictor：12个package成员均由memfd封印且每项`memfd_seals=15`，Landlock/no-new-privs/network-seccomp/pinned-input attestation均为PASS；随后predictor返回1，未生成prediction/scoring/cell/smoke receipt。完整3,939字节driver日志只包含外层`return code 1`，结构化receipt证明内层stderr为2,243字节但旧runner只保存SHA，无法定位Python异常。666,665字节完整trace的审计另暴露两项标准库文件名假阳性：`expressionrawdomain.pyc`与`graph_drawer.pyc`因子串`raw`被误判；这不是clean/raw数据访问。

外层Landlock runner修复为按路径词元精确匹配`truth/scoring/scorer/clean/raw/manysig/manytx`，仍会拒绝`clean_cache`、`truth_sidecar`等真实敏感路径，但不再命中英文单词内部的`raw`。predictor失败时最多4,000字符的truth-free stderr tail写入外层异常和driver日志，便于下一次定位；receipt中的完整stderr SHA/size不变。`py_compile`和26项Landlock/memfd/runtime-closure/isolated-runner/strict-package测试通过。该修改不改变sealed predictor closure；下一次复用strict_v10，但必须生成全新v13清单和v6运行根，v5证据不覆盖。

v13清单已在本地生成并验证：绑定strict_v10和全新`..._landlock_strict300_v6`根，25 cache/75 package/300 cell/900 row，`launch_authority=false`、`authority_state=N607_LANDLOCK_SMOKE_REQUIRED`，SHA=`fbd1fbae9107d1c2beee7edd24c1d2fe7e4ca9a849cb001b6b944e9cc2fbe2fc`。同步前N607 v6根不存在、GPU无任务；只同步提交`2af8df8`的外层runner与该清单，远端核验SHA后仍仅允许`cuda:0`的receiver=`20-1`、seed=`713101`、20新类、K1 smoke，使用独立`smoke_driver_v6_attempt1.pid/out`和`logs/smoke_v6_attempt1`。

v6 attempt1 PID=`1894028`在两份外层脚本串行SCP期间被并发启动，形成新pinned校验+旧Landlock attestation的混合版本：旧385字节attestation缺少新字段，故在`runtime Landlock/seccomp attestation failed`处停止；无prediction/scoring/smoke receipt。这是同步竞态，不是策略测试结果。两份远端脚本随后稳定核验为pinned SHA=`378decb111f1bda7ac198d27d4f31f75bcbefd514661a26ba4be7a38f5aa8b6f`、launcher SHA=`f20c071a9d4a45f60c1f389962e347a4d70edcffca316bfa81c345ddcd443173`。新v14清单绑定strict_v10与全新`..._landlock_strict300_v7`根，仍为25/75/300/900、`launch_authority=false`，SHA=`59992f970ad157b2e6465704041a0469e5c4219f55af9fbac575836926a87f9f`。

00:16预检/实时清单确认`gpu_compute=[]`、`active_training_processes=[]`。v14已同步至v7新根并核验SHA；两份外层脚本在启动前已稳定。v7 smoke使用`smoke_driver_v7_attempt1.pid`、`logs/smoke_driver_v7_attempt1.out`、`logs/smoke_v7_attempt1`和新根`smoke_receipt.json`，仍只有`cuda:0`单单元权限。

v7 attempt1以PID`1896254`运行，成功通过新版seccomp/attestation并进入密封predictor；predictor在CUDA lazy init前调用`reset_peak_memory_stats(cuda:0)`，stderr明确报`Invalid device argument 0: did you call init?`，因此无prediction/scoring/smoke receipt。修复把设备准备封装为先`torch.cuda.init()`、再重置峰值统计；CPU路径不初始化CUDA。26项predictor/runtime/closure测试及本地9文件closure构建通过（closure SHA=`785a844d49fabc230ce803db3c64dd249093fc779980a2ccec2efd5faa31d8b2`）。v7不复用，后续重建远端closure并使用全新v8根。

strict_v11远端闭包构建完成：12项复用artifact逐字节一致，闭包SHA与本地同为`785a844d49fabc230ce803db3c64dd249093fc779980a2ccec2efd5faa31d8b2`，新predictor脚本SHA=`9eaa8f73879c2266e806097dee2f6076b3f0516c892e6b61c4b1a25221b10421`。v15清单绑定strict_v11与全新`..._landlock_strict300_v8`根，25/75/300/900、`launch_authority=false`，SHA=`8c0a37bde37b7604b1517198017fac75b0c6ed3eda7a5cbcb2635e15abb4ad66`。

00:21预检与实时清单确认无GPU计算进程、无训练进程。v15已同步到v8新根并核验SHA。v8 smoke使用`smoke_driver_v8_attempt1.pid`、`logs/smoke_driver_v8_attempt1.out`、`logs/smoke_v8_attempt1`与新根`smoke_receipt.json`，仍仅`cuda:0`单单元。

v8 attempt1以PID`1899303`运行，CUDA初始化、密封memfd、Landlock/seccomp及support feature forward均已通过，但严格闭包内`torch.from_numpy(np.asarray(...))`触发`TypeError: expected np.ndarray (got numpy.ndarray)`，因此未生成prediction/scoring/smoke receipt。修复将NumPy数据先复制为连续float32字节，再通过`torch.frombuffer(...).clone()`建立自有Torch存储，完全避开PyTorch NumPy C-API桥接；CPU回归测试同时验证原数组改写不会影响tensor。23项predictor entry/runtime/closure测试通过，新9文件本地closure构建通过，SHA=`7d8247a6feaf652acdf7d84f7bcdbab6cd32ca76b80d93dc5b445eebcad5b522`。v8不复用；下一次在全新`runtime_artifacts_strict_v12`重建闭包，并使用全新v9运行根。

00:26直连预检与实时训练清单确认8张GPU空闲，`gpu_compute=[]`、`active_training_processes=[]`。提交`555eb09`中的predictor已同步，远端SHA=`3ca338b41f5433a599555949f699cce165a86ab9cfca47625b9356c76e6fd7d0`。strict_v12以strict_v11为只读来源复制12项模型/capsule/config artifact并逐项`cmp=PASS`，重新构建的9文件闭包SHA=`8c6f464b3bba7e5fc5c09f22a5cf4f4f4a4635782ee56555a3b2d7a2ce0c6403`；与strict_v11逐成员比对，仅`cvsrffi/stage2_predictor_runtime.py`发生预期变化。证据回传至`evidence/runtime_artifacts_strict_v12/`，SSH/SCP结束后本地`ssh.exe=0`、N607/桥接TCP22连接为0。

本地计划目录v16因先创建out-dir触发生成器防覆盖门禁，v17因误把generated base plan作为source plan触发schema门禁，二者均未生成strict manifest且不发布。随后从原始受控配置生成v18清单，绑定strict_v12与全新`..._landlock_strict300_v9`运行根；覆盖25 cache、75 package、300 cell、900 formal row，继续保持`launch_authority=false`、`authority_state=N607_LANDLOCK_SMOKE_REQUIRED`，清单SHA=`fe323f56087a840e7f8b9cf8102224f09b40decbcb6cf7d93108b94f9926b8e4`。只有该v18清单可进入下一次单单元smoke。

00:31再次预检并确认`gpu_compute=[]`、`active_training_processes=[]`。远端v9根在创建前不存在；v18清单已同步到`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300_v9/protocol_plan/strict_plan_manifest_v18.json`并核验相同SHA。smoke工作目录为`/home/szu2070436088/2510044040/CV-SincNet`，Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，GPU为`cuda:0`；PID、driver日志、阶段日志和receipt分别写入`smoke_driver_v9_attempt1.pid`、`logs/smoke_driver_v9_attempt1.out`、`logs/smoke_v9_attempt1/`和`smoke_receipt.json`。确切入口为`paper_reproduction/scripts/run_cvs_stage2c_effective8_strict_plan.py --plan-manifest <v18远端清单> --project-root <项目根> --stage smoke --device cuda:0 --log-dir <smoke_v9_attempt1> --smoke-receipt <v9根/smoke_receipt.json>`；仅当上述输出均不存在时以`nohup env PYTHONPATH=code:.`启动。正式matrix shard仍未获授权。

### 三轮回顾与下一路线选择（2026-07-16 00:36 CST）

按新增治理规则保守地把最近v6/v7/v8三次严格smoke视作一组探索轮，在下一次启动前完成回顾。已重新阅读当前目标、根`项目.md`、Git承载面的`AGENTS.md`和`docs/PROJECT_PROTOCOL.md`；刷新项目conversation index并检索`effective8 125 registration forgetting seen_new_acc H_old_new`。历史命中明确指出qKNNv42/effective8已有source诊断不能替代正式Phase2 target-new结果，旧support-anchor路线也必须同时验证旧类域适应、新类注册和遗忘，不能退回old-only或unknown/open-set代理指标。

完整v8 driver日志共5,741字节，SHA=`1680beac9d39b0f4857154003dfe4e8768a81143a5924974a6b23de3a926005b`，已回传`evidence/smoke_v8_failure/`并从首字节读到EOF。它只证明严格隔离链已进入support forward后在NumPy桥接处失败，没有任何可排名性能行。v6是混合同步版本诊断，v7是CUDA统计初始化顺序错误，v8是NumPy C-API桥接错误；三者都不是候选机制性能失败，不据此修改adapter/head/TTA或在query上调参。

下一路线继续锁定同一effective8 candidate，仅修复可执行隔离链并补全正式评分门禁。域适应与新类注册必须同等覆盖：每个formal row同时保留`old_acc_before_increment`、`old_acc_after_increment`、`seen_new_acc_after_increment`和`H_old_new_after_increment`；逐旧类必须同时给出before/after accuracy、forgetting及adaptation gain。路线继续满足`LEO_weak-only`、no-clean、no-query-truth、no-role-Oracle、no-class-quota及逐样本全注册类决策；scorer结果不得回流预测。缺少注册后`seen_new_acc/H_old_new`、逐类旧类准确率或遗忘证据的cell一律视为不完整，不得计入125任务正式结果。

提交`f8de85e`正好实现上述独立scorer字段和smoke/formal执行权限门禁，并为CUDA allocator建立显式device context。56项predictor/runtime/closure/scorer/strict-package/strict-plan/authority测试通过。由于该提交改变密封predictor入口和外层正式判定，已同步但未启动的v18/v9计划标记为superseded，不执行；下一次必须重建全新runtime closure和新运行根。

并行构建留下`runtime_artifacts_strict_v9`部分目录：12项不可变模型/config文件存在，`05_runtime_closure.json`为0字节且closure目录不存在，符合旧closure白名单拒绝新增`ctypes/platform`导入的fail-closed行为。strict_v9完整保留且不补写。下一次使用全新`runtime_artifacts_strict_v10`：先同步提交`764c11c`的memfd实现与提交`5d87bdd`的closure白名单，逐文件核验SHA；再在N607直接调用`_sealed_memfd`执行临时seal smoke并验证`REQUIRED_SEALS`，通过后才复制strict_v8的12项不可变artifact并新建closure。后续计划/运行根使用全新版本，不复用v4/v11。

### v8失败、注册前后指标补全与第1次三轮回顾

v8 attempt1以PID=`1899303`退出。完整读取5,741字节driver日志确认：v8已经越过CUDA初始化并进入`build_formal_support_state`，但隔离进程中NumPy模块重载造成`torch.from_numpy`类型身份冲突，报`expected np.ndarray (got numpy.ndarray)`。该次仍无prediction、scoring、cell或smoke receipt，因而仍不能报告新类注册性能。提交`555eb09`已将严格runtime的NumPy→Torch转换改为连续buffer桥接并增加存储所有权测试。

当前补充修复在CUDA初始化后显式选定设备并以1标量初始化allocator，再重置峰值统计；同时扩展独立scorer，使每个正式场景行同时物化注册前/后的旧类整体与逐类结果：`candidate_old_class_acc_before_increment`、`candidate_old_class_acc_after_increment`、`candidate_old_class_forgetting`、对应identity字段，以及注册前/后最低旧类准确率。注册前新类尚未注册，明确写`pre_increment_new_class_state=NEW_CLASSES_NOT_REGISTERED`，对应`seen_new_acc_before_increment/H_old_new_before_increment=null`；注册后写`seen_new_acc_after_increment/H_old_new_after_increment`。单package runner新增`smoke/formal`执行模式，未授权K10在任何package物化前fail closed。相关predictor/runtime/scorer/strict-plan/authority共63项测试通过；本地9文件closure验证SHA=`7d8247a6feaf652acdf7d84f7bcdbab6cd32ca76b80d93dc5b445eebcad5b522`。

2026-07-16执行第1次“每3轮探索强制回顾”：重新核对当前目标、`项目.md`、977条项目对话索引，并完整读取qKNN V92注册、Oracle禁用和v14缺少新类结果的历史摘要。

| 回顾项 | 历史教训 | 当前决策 |
|---|---|---|
| 域适应与新类注册 | source-only v21-v23只有旧类，不能证明Stage2-C收益 | 后续候选必须同时跑域适应、注册前和注册后，不再晋升old-only结果 |
| 注册指标 | v14正式target结果文件数为0；92.28%属于不同切分且含旧Oracle机制的legacy diagnostic | 只接受严格sealed predictor+独立scorer生成的同一cell正式行 |
| 开发K值 | K1只能做压力smoke，不能代替目标工作点 | smoke完成后优先运行K10/new20单cell；K10是唯一开发选参点 |
| 遗忘 | 总体旧类准确率会掩盖最低类和个别类崩塌 | 同时报注册前后old、最低旧类、逐类差值和最坏遗忘 |
| 视图与协议 | 历史固定5-view和角色/类别配额不能直接复用 | support/query仅`LEO_weak`；自适应view只依赖逐样本置信度，禁止query真值、角色Oracle和类别配额 |

回顾规则已加入根目录`AGENTS.md`并镜像到Git承载面的`AGENTS.md`。以后每完成3轮算法候选探索，在第4轮启动前必须把目标、协议、历史路径、完整日志和下一轮取舍写回本报告。

00:31提交`f8de85e`的scorer、predictor工作区副本、strict package runner和strict plan runner已串行同步至N607并逐项核验SHA：`b25d996e…8ce3`、`b7ba6e94…0ebe`、`5b77c93d…9df0`、`cfd435a7…627`；两个CLI的远端import/help smoke均为PASS。v9清单仍绑定不可变strict_v12 closure SHA=`8c6f464b…c6403`，清单SHA=`fe323f56…6b8e4`，`launch_authority=false`。下一次仅允许执行receiver=`20-1`、seed=`713101`、20个真实seen-new、K1、3个`LEO_weak`场景的smoke；完整矩阵和K10均尚未授权。

计划命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
V9=/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300_v9
PYTHONPATH=code:. "$PY" paper_reproduction/scripts/run_cvs_stage2c_effective8_strict_plan.py \
  --plan-manifest "$V9/protocol_plan/strict_plan_manifest_v18.json" \
  --project-root /home/szu2070436088/2510044040/CV-SincNet \
  --stage smoke --device cuda:0 \
  --log-dir "$V9/logs/smoke_v9_attempt1" \
  --smoke-receipt "$V9/smoke_receipt.json"
```

后台包装只写新路径`smoke_driver_v9_attempt1.pid`与`logs/smoke_driver_v9_attempt1.out`。成功后必须回传并核验`prediction_artifact.cvspred`、`formal_rows.json`、`formal_predictions.json`、scoring/cell/smoke receipt及资源收据，随后才运行官方授权器；授权后优先执行同package的K10/new20单cell，而不是启动完整300-cell矩阵。

v9 attempt1以PID=`1906162`运行，首次完整执行了ADV3B02 support/query forward、support-only新类注册和五路注册前/后预测，并生成2,658,628字节密封`prediction_artifact.cvspred`；但predictor在写资源收据时又尝试按原路径回读Landlock已禁止的`request.json`以计算SHA，因`PermissionError`停止。独立scorer、cell receipt和smoke receipt均未生成，所以该prediction不能单独晋升为性能证据。完整6,034字节driver日志SHA=`a00e50e2…0a84`；v9保持不可变。

提交`9cb6e84`改为在首次密封memfd读取request时同步计算原始字节SHA，后续资源收据不再打开request路径；常规文件与物理不可达pinned request两条回归均覆盖。40项predictor/runtime/closure测试通过，新本地closure SHA=`6f3ccbebd1aef610c1a96f4fad94703dd0ada070b927eecabd8b68e794027fc4`。N607全新strict_v13从strict_v12逐项复制12个不可变模型/capsule/config artifact且`cmp/diff=PASS`，重建closure SHA与本地一致；证据回传至`evidence/runtime_artifacts_strict_v13/`。

strict plan v19绑定strict_v13与全新`..._landlock_strict300_v10`运行根，25 cache/75 package/300 cell/900 formal row，继续保持`launch_authority=false`、`authority_state=N607_LANDLOCK_SMOKE_REQUIRED`，清单SHA=`7868eccb4cf6b639ad6c46b046a656c5ce74c9ee54760fcfe50e0549c28e91e4`。生成后只读摘要脚本曾因误用旧字段名`cache_count`返回1，但清单已完整生成并按实际`cache_steps/package_steps/cells`重新核验通过；该摘要错误不属于计划或实验失败。

00:39再次核验8张GPU空闲且无严格runner。v10根原先不存在，现仅创建`protocol_plan/`并同步`strict_plan_manifest_v19.json`；smoke将使用新路径`smoke_driver_v10_attempt1.pid`、`logs/smoke_driver_v10_attempt1.out`、`logs/smoke_v10_attempt1`和`smoke_receipt.json`。权限仍严格锁定K1/new20单cell，K10与矩阵未授权。

v10 attempt1已由并发控制流以PID文件落地并退出，driver日志6,042字节，未生成smoke receipt。完整stderr确认strict_v13密封入口仍在predictor execution audit和返回payload两处调用`sha256_file(request_path)`；因此它虽然包含`9cb6e84`的首次pinned read摘要，但仍在资源收据之后重开Landlock禁止路径，重复触发`PermissionError`。这不是算法性能证据。提交`ec86075`移除剩余两处重开并新增源码级不回读回归；30项predictor/runtime/closure测试通过，当前新本地closure SHA=`3f8a577de614666cf33eb2cdc50244045c0898cabc22d545571d229c4b87805b`。strict_v13/v10保持不动；下一次必须同步`ec86075`入口，构建全新strict_v14并使用全新v11运行根。

v11 smoke首次完整通过，status=`PASS`、cell status=`PROTOCOL_VALID`、`matrix_launch_authority_recommended=true`。密封prediction artifact为2,658,628字节，本地SHA=`1bb2faf7036980f3088c8321c4cd47da076e578285c51d7859a317c0f5ab0790`，与cell receipt一致；正式runtime evidence SHA=`d80fc9f5273ca2432ee69b42ca216a901cd169c4221ecbd49ec42c23570eb89d`，filesystem audit、pre-open audit、memfd、Landlock/seccomp和predict/score进程隔离均为PASS。证据回传至`evidence/smoke_v11_pass/`，大型prediction/formal_predictions仅存`local_artifacts/`并完成SHA核验。

| smoke场景 | K | old before | old after | seen-new after | H after | min old class | average forgetting | mean/P95 forward | 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| leo_clear_weak | 1 | 0.6917 | 0.5667 | 0.3425 | 0.4269 | 0.0000 | 0.1250 | 2.623/3 | 协议PASS，性能不达标 |
| leo_low_elev_weak | 1 | 0.6417 | 0.4833 | 0.2750 | 0.3505 | 0.0000 | 0.1583 | 2.781/5 | 协议PASS，性能不达标 |
| leo_rain_weak | 1 | 0.6000 | 0.4583 | 0.3100 | 0.3698 | 0.0000 | 0.1417 | 2.692/5 | 协议PASS，性能不达标 |

该smoke只证明正式执行闭环，不构成candidate晋升。资源同row为44,048个可训练参数、109,818字节持久状态、168,441,856字节峰值CUDA显存、平均2.699/P95=3次backbone forward。官方授权器已把v20清单与smoke receipt SHA=`6064b3113690fa5453b4f0f82b09febdfca9999fdf68828459102e0ec8c62d94`绑定，生成`launch_authority=true`、`authority_state=N607_LANDLOCK_SMOKE_PASS`的清单，SHA=`62d78cb9aa636c8e756f582473f6c9742326c220936dcd0efb6a770a4b18ae85`。按当前目标修正，先只运行同receiver/seed/new20的K10开发单元；未批准主动扩展完整矩阵。

00:50另一个并发控制流把同一授权内容同步为v11根`protocol_plan/strict_plan_manifest_v20_authorized_62d78cb9.json`并准备matrix shard命令。当前没有“8个shard已全部启动”的证据；实时清单只确认00:52:06出现`matrix_shard 0/8`。本轮不再启动其余shard，相关额外cell需按后文竞态审计和正式结果门禁处理。

v11矩阵8个shard均落地：shard0 PID=`1918624`正常运行；shard1–7在首cell一致失败并退出。完整日志显示导出的TorchScript内部参数固定在逻辑`cuda:0`，直接传`cuda:1..7`导致输入位于`cuda:i`而卷积权重仍位于`cuda:0`。7次失败均无cell receipt且保留partial evidence，v11不清理、不原地续跑；shard0继续运行且不干预。该问题属于物理GPU映射，不改变candidate/runtime artifact：正确多GPU方式是每进程设置`CUDA_VISIBLE_DEVICES=<物理i>`，再统一传逻辑`--device cuda:0`。

为保持单一完整矩阵输出，生成全新v21计划绑定同一strict_v14与全新`..._landlock_strict300_v12`根；未授权清单SHA=`7e5358e511e52c8d2b0ec421cc3228d55237d63a26b368c1b3e1c6d6c4c1f47d`。官方授权器复用v11相同candidate/closure的PASS receipt，生成授权清单SHA=`1c972769f2e248ae46d893df46193bea02a499430d648a67230a91b7aeba2c99`，仍覆盖25 cache、75 package、300 cell、900 formal row。v12的8个shard将设置`CUDA_VISIBLE_DEVICES=0..7`且全部传逻辑`cuda:0`；GPU0因此暂时有v11 shard0+v12 shard0共2个实验进程，未超过每GPU最多2个的默认上限，其他GPU各1个。

v10 attempt1以PID=`1910063`运行，再次生成2,658,628字节密封prediction，并成功生成1,779字节predictor资源收据，证明首次密封读取SHA修复生效；随后execution audit和最终stdout返回中的两处遗留`sha256_file(request_path)`仍触发Landlock拒绝。scorer/cell/smoke receipt依然不存在，v10不原地续跑。完整driver日志6,042字节，SHA=`54bb8c51…5c78`。

提交`ec86075`把execution audit与最终返回统一绑定首次密封读取所得`request_sha256`，新增源码级门禁保证严格predictor中不再出现任何`sha256_file(request_path)`。41项相关测试通过；全新strict_v14 closure在N607与本地均为SHA=`3f8a577de614666cf33eb2cdc50244045c0898cabc22d545571d229c4b87805b`，12项模型/capsule/config复用artifact逐项一致。v20清单绑定strict_v14和全新v11运行根，仍为25/75/300/900、`launch_authority=false`，清单SHA=`717bcd08dadd0dccaf2f24ae4407d64aea4cd44b32f63b423451ee8f3884a0a4`。

v11 attempt1以PID=`1913259`完整通过严格闭环：密封prediction、execution/resource audit、3场景formal rows、1,560条逐样本formal predictions、scoring receipt、cell receipt和smoke receipt全部生成；`status=PROTOCOL_VALID`、smoke=`PASS`。K1/new20实测如下，所有数值来自同一cell、同一场景行：

| 场景 | 注册前old | 注册后old | direct ADV3B02 old | 注册后相对direct | 注册后seen-new | H_old_new | 最低旧类 | 平均遗忘 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| leo_clear_weak | 69.17% | 56.67% | 74.17% | -17.50pp | 34.25% | 42.69% | 0.00% | 12.50pp |
| leo_low_elev_weak | 64.17% | 48.33% | 66.67% | -18.33pp | 27.50% | 35.05% | 0.00% | 15.83pp |
| leo_rain_weak | 60.00% | 45.83% | 70.00% | -24.17pp | 31.00% | 36.98% | 0.00% | 14.17pp |
| 三场景等权均值 | 64.44% | 50.28% | 70.28% | -20.00pp | 30.92% | 38.24% | 0.00% | 14.17pp |

K1结论为明确负例：未满足“适应后明显优于direct ADV3B02”，注册使旧类平均下降14.17pp，TX=`14-7`三场景均为0%，且20新类seen-new只有30.92%。它只用于证明严格评测闭环，不得晋升为算法候选。资源侧满足参数/状态/epoch硬上限：44,048 trainable parameters、109,818字节持久状态、12 adapt epochs、峰值CUDA显存168,441,856字节、候选query延迟0.679ms；但自适应view平均2.699次backbone forward，1/3/5-view触发率19.81%/75.45%/4.74%，说明当前门限大多触发3-view，尚未达到“默认1-view”的理想计算分布。

完整证据已回传`evidence/smoke_v11_success/`。官方授权器以smoke receipt SHA=`6064b311…2d94`生成`authorized_strict_plan_manifest.json`：`launch_authority=true`、`authority_state=N607_LANDLOCK_SMOKE_PASS`，授权清单SHA=`62d78cb9aa636c8e756f582473f6c9742326c220936dcd0efb6a770a4b18ae85`。下一步只执行同receiver/seed/new20的K10单cell，优先评估联合域适应+新类注册目标点，不启动完整矩阵。

### K10/new20单cell正式开发运行

2026-07-16 00:49 CST只读预检确认N607的8张RTX3090均约10 MiB显存、0%利用率，无其他严格runner，项目盘剩余7.6 TiB；目标K10 cell和远端授权清单均不存在。授权清单本地SHA=`62d78cb9aa636c8e756f582473f6c9742326c220936dcd0efb6a770a4b18ae85`。先SCP至`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300_v11/protocol_plan/authorized_strict_plan_manifest.json`并逐字节SHA核验；随后只允许以下一个正式cell，不启动matrix shard：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
V11=/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300_v11
PYTHONPATH=code:. "$PY" paper_reproduction/scripts/run_cvs_stage2c_effective8_strict_package.py \
  --plan-manifest "$V11/protocol_plan/authorized_strict_plan_manifest.json" \
  --package-id rx_20_1__seed_713101__new_20 \
  --project-root /home/szu2070436088/2510044040/CV-SincNet \
  --device cuda:0 \
  --k-shot 10 \
  --execution-mode formal
```

后台包装只写`formal_k10_driver_attempt1.pid`和`logs/formal_k10_driver_attempt1.out`。完成后完整读取driver日志，并回传K10 cell下的sealed prediction、execution/resource audit、formal rows/predictions、scoring receipt和cell receipt；评价必须同时给出注册前/后旧类、逐类遗忘、seen-new、H、direct ADV3B02差值和资源开销。

K10单cell以PID=`1918307`运行，完整916字节driver日志SHA=`6d3c6b2090284db164017467102c31f9c0e5d2430632578a8838afbb28cfd13a`，返回`PROTOCOL_VALID`。prediction artifact SHA=`04e6c64f95fefd404cf4ed44fc0632346b75d054a2f820b1a4c7490f600d6837`、formal rows SHA=`a6f9b58babdbfeaa31b4afbd980d80394e4ee8d2d14eb3bcbde9d801a5a926c0`、formal predictions SHA=`0638d3b3284e68fd05fdc205fc1c130434bfc325f1eb14e43c5e806b3ac4f589`、scoring receipt SHA=`d1cb88e98979f33668faa7a872af9fe9d94082097d2af31ac134f64ed6723ed2`均已本地复算，关键receipt引用完全一致。

| K10场景 | 注册前old | 注册后old | direct ADV3B02 old | 注册前相对direct | 注册后相对direct | seen-new | H | 最低旧类 | 平均遗忘 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| leo_clear_weak | 81.67% | 64.17% | 74.17% | +7.50pp | -10.00pp | 38.75% | 48.32% | 30.00% | 17.50pp |
| leo_low_elev_weak | 75.00% | 54.17% | 66.67% | +8.33pp | -12.50pp | 32.00% | 40.23% | 20.00% | 20.83pp |
| leo_rain_weak | 74.17% | 60.00% | 70.00% | +4.17pp | -10.00pp | 35.75% | 44.80% | 40.00% | 14.17pp |
| 三场景等权均值 | 76.94% | 59.44% | 70.28% | +6.67pp | -10.83pp | 35.50% | 44.45% | 30.00%* | 17.50pp |

`*`30.00%是三个场景最低类准确率的均值；真正跨场景全局旧类floor为20.00%。K10逐旧类联合均值如下，所有值均来自同一cell的注册前/后预测：

| target-old TX | 注册前均值 | 注册后均值 | 遗忘 | 三场景最低注册后准确率 |
|---|---:|---:|---:|---:|
| 14-10 | 65.00% | 35.00% | 30.00pp | 20.00% |
| 14-7 | 85.00% | 73.33% | 11.67pp | 60.00% |
| 20-15 | 80.00% | 60.00% | 20.00pp | 50.00% |
| 20-19 | 73.33% | 71.67% | 1.67pp | 55.00% |
| 6-15 | 81.67% | 41.67% | 40.00pp | 40.00% |
| 8-20 | 76.67% | 75.00% | 1.67pp | 65.00% |

K10使注册前adapt由K1的64.44%提高到76.94%，并从低于direct 5.83pp转为高于direct 6.67pp，说明10-shot下轻量特征适配已经获得明确正收益；但新类注册随后造成17.50pp旧类遗忘，使最终old降到59.44%，仍低于direct 10.83pp。seen-new仅35.50%、H仅44.45%、全局旧类floor仅20.00%，所以主要瓶颈已从“adapt是否有效”收敛为“old/new联合注册竞争与逐类塌缩”，当前candidate仍不晋升。与identity相比，candidate注册后old均值由54.72%提高到59.44%，表明effective8只缓解了4.72pp，尚未解决注册失衡。

资源仍为44,048个训练参数、12 epoch、109,818字节持久状态、168,441,856字节峰值CUDA显存；候选延迟0.802 ms/query，平均2.726/P95=5次backbone forward，1/3/5-view触发率18.91%/75.90%/5.19%。与K1一致，当前门限仍以3-view为默认实际路径，后续必须把1-view触发率显著提高，同时保留低置信度样本的3/5-view补算。

并发审计：另一个控制流在00:52:06启动了`matrix_shard 0/8`，超出本轮“只跑K10”的主动范围。它先处理new5/new10包，K10 receipt在00:52:12落盘，而该shard直到00:54:26才把new20包标为complete；strict package runner对已有`PROTOCOL_VALID` receipt只读复用，因此没有并发写坏K10 cell。当前只发现该单一shard，按安全规则不杀停、不重启、不再启动其他shard，转为只读监控。该shard产生的额外cell必须单独审计后才能进入性能分析。

该v11 shard0随后自然完成，driver完整3,198字节、状态`complete`，覆盖4组receiver-seed、3种新类数、K=1/5/10/20，共48个`PROTOCOL_VALID` cell。完整同cell联合行保存在`evidence/shard0_v11_diagnostic/cell_metrics.tsv`。聚合结果显示：K10/20的注册前adapt普遍比direct高约6.7–7.9pp，但注册后仍低8.3–14.6pp；20新类下K1/5/10/20平均遗忘分别为19.58/21.94/21.32/20.00pp，seen-new仅33.27/40.94/42.27/44.04%。所有K×新类数组合的跨cell旧类floor均为0，证明注册塌缩不是rx20-1单点异常。K20是当前相对较好的K值，但远未达到目标；不能靠继续增大K解决。

## 完成后结果表

| candidate ID | 机制 | receiver/TX split | K | seed | 注册前old | 注册后old | seen-new | H | 最低旧类 | 平均遗忘 | adapter/资源 | 判定 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| effective8-v14-strict-v11-smoke | ADV3B02+effective8+support-only注册+adaptive view | rx20-1；6 old+20真实new；3个LEO_weak | 1 | 713101 | 64.44% | 50.28% | 30.92% | 38.24% | 0.00% | 14.17pp | 44,048参数；12 epoch；109,818B；2.699 view | `PROTOCOL_VALID`但性能负例，不晋升 |
| effective8-v14-strict-v11-k10 | ADV3B02+effective8+support-only注册+adaptive view | rx20-1；6 old+20真实new；3个LEO_weak | 10 | 713101 | 76.94% | 59.44% | 35.50% | 44.45% | 20.00% | 17.50pp | 44,048参数；12 epoch；109,818B；2.726 view | adapt注册前正收益；联合注册仍严重负例，不晋升 |

后续每个完成cell继续追加同一行，不得用来自不同单元的独立极值替代联合行。

### v12完整矩阵落地（2026-07-16 01:00 CST）

直连预检确认N607可达；v12创建前远端运行根不存在。已把授权v21清单同步至`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300_v12/protocol_plan/strict_plan_manifest_v21_authorized_1c972769.json`，远端复算SHA=`1c972769f2e248ae46d893df46193bea02a499430d648a67230a91b7aeba2c99`，与本地一致。首次哈希核验因PowerShell提前展开远端`$(...)`而产生本地包装错误；未重复SCP，清理审计确认`ssh.exe=0`且无N607 TCP22残留后，使用单条短命令完成只读复核。

8个v12 shard均以原子独立锁启动，物理GPU通过`CUDA_VISIBLE_DEVICES=0..7`隔离，runner一律传逻辑`--device cuda:0`。PID与映射如下：

| shard | 物理GPU | 逻辑device | PID | state/log前缀 | 启动后验证 |
|---:|---:|---|---:|---|---|
| 0 | 0 | `cuda:0` | 1929789 | `matrix_shard_0_logical0` | LIVE；首个new5 package完成，new10运行中 |
| 1 | 1 | `cuda:0` | 1929796 | `matrix_shard_1_logical0` | LIVE；首个new5 package完成，new10运行中 |
| 2 | 2 | `cuda:0` | 1929808 | `matrix_shard_2_logical0` | LIVE；首个new5 package完成，new10运行中 |
| 3 | 3 | `cuda:0` | 1929883 | `matrix_shard_3_logical0` | LIVE；首个new5 package完成，new10运行中 |
| 4 | 4 | `cuda:0` | 1929958 | `matrix_shard_4_logical0` | LIVE；首个new5 package完成，new10运行中 |
| 5 | 5 | `cuda:0` | 1930036 | `matrix_shard_5_logical0` | LIVE；首个new5 package完成，new10运行中 |
| 6 | 6 | `cuda:0` | 1930114 | `matrix_shard_6_logical0` | LIVE；首个new5 package完成，new10运行中 |
| 7 | 7 | `cuda:0` | 1930192 | `matrix_shard_7_logical0` | LIVE；首个new5 package完成，new10运行中 |

`/proc/<pid>/environ`逐项确认上述物理映射，`/proc/<pid>/cmdline`逐项确认`--device cuda:0 --shard-count 8`及对应`--shard-index`。GPU0同时存在保留运行的v11 shard0与v12 shard0，共2个实验进程，未超过每GPU最多2个的默认上限；其他GPU最多1个v12实验。此次状态是“已落地并在运行”，不是矩阵完成或性能成功；后续必须等待8个state全部完成、完整读取driver日志并聚合300 prediction cells/900 formal rows后才能形成最终版本判定。

### new20 truth/role扫描失败与修复追踪（2026-07-16 01:04 CST）

01:03只读监控统计v12已完成20/75 packages，即80/300 prediction cells和240/900 formal rows；shard2、3、4、6父PID已退出而state仍停在`running`。完整读取4份driver日志后确认它们均在new20 predictor package构建的`_reject_predictor_truth_leaks`失败，分别命中`query_leo_rain_weak.npz`、`query_leo_clear_weak.npz`、`query_leo_low_elev_weak.npz`和`support_leo_rain_weak.npz`；new5/new10均已通过。失败包和日志完整保留，未重启、未清理。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| TR-1 | `项目.md`与根`AGENTS.md`的no-query-truth/no-role-Oracle边界 | predictor的query包不得包含真实TX、old/new/unknown role、query quota或可回流真值 | `code/scripts/build_cvs_stage2_predictor_bundle.py`、predictor bundle测试 | verified | 结构化枚举NPZ字符串数组、manifest和opaque token；真实`query_tokens`泄漏负例继续被拒绝；26项密封/package/authority测试通过 | 未放宽结构化文本、JSON、seal或成员白名单门禁 |
| TR-2 | Phase2 sealed input package与pre-open验证 | 避免把数值IQ/ZIP随机字节中对短TX标签的偶然命中误判为结构化真值泄漏，同时保持JSON/字符串/token门禁 | 同上 | verified | 4个实际失败NPZ的raw命中仅位于`float32`IQ成员，字符串字段0命中；修复后实际文件回归PASS | 数值成员仍受NPZ allowlist、member/package SHA、detached seal和runtime audit约束 |
| TR-3 | 本地优先、Git、SCP和不可覆盖 | 修复必须本地提交、验证后同步；v12 partial evidence不得覆盖或原地续跑 | 代码、测试、报告、新运行根/计划 | verified | `git diff --check`、26项pytest、目标脚本/计划远端SHA、v13根预先不存在、8分片落地 | v12保持失败证据；v13为唯一新正式根 |
| TR-4 | 125任务诉求与当前严格25/75/300/900计划 | 只有全部300 cells/900 rows完成并通过protocol receipts才可聚合性能 | 新run states、receipts、本报告 | verified | 8/8 shard complete、75/75 packages、300/300 `PROTOCOL_VALID` cell receipts、900/900同row formal rows、完整driver日志 | 严格矩阵artifact-complete；性能门槛另判且未通过 |
| TR-5 | Experiment Reporting同row联合解释 | 汇总器必须校验receipt/formal row/resource绑定，输出300-cell明细、K×new-count联合汇总、receiver分组和审计SHA，不得拼接不同cell极值 | `paper_reproduction/scripts/summarize_cvs_stage2c_effective8_strict_matrix.py`、测试、v13 summary artifacts | verified | 2项synthetic绑定/篡改测试PASS；审计`status=PASS`；300 cells、900 rows；全部输出SHA本地复算一致 | 最终解释基于同cell联合行；不把边际极值拼接为候选 |

TR-5汇总器已本地实现并提交为`7f14821`，`py_compile`与2项synthetic绑定/篡改测试通过；脚本SHA=`13607f2f320abd11697b41bb3b17ec41bc750a212de3a8434aad19e4c83d4751`。唯一同步目标为`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/scripts/summarize_cvs_stage2c_effective8_strict_matrix.py`，只读输入为v13根，唯一新输出目录为v13根下`matrix_summary_v1`。命令固定为`python summarize_cvs_stage2c_effective8_strict_matrix.py --run-root <v13> --output-dir <v13>/matrix_summary_v1 --expected-cells 300`；输出目录预先存在时fail closed。

结构化审计定位的4个raw命中分别为：shard2的`query_leo_weak_iq.npy`中`2-5`、shard3/4的同一数值成员中`1-8`、shard6的`support_pool_leo_weak_iq.npy`中`8-3`；相应NPZ全部字符串数组均无任何old/new role或TX标签。修复仅把`.npz`扫描从整包随机字节改为`S/U`文本成员扫描；其他文件仍逐字节扫描。`py_compile`通过，`ssr-gpu`下26项build/bundle/sealed pipeline/strict package/plan authority/CLI测试通过，4个实际失败NPZ回传后由修复代码复扫为PASS。TR-3仍须在Git提交、远端SHA核验和全新运行根落地后才能验证。

本地版本状态已在远端变更前冻结：Git提交=`29d0c3d43b604db05771dd8c15d841663858b5ec`；修改文件为`code/scripts/build_cvs_stage2_predictor_bundle.py`（NPZ语义文本扫描）和`tests/test_build_cvs_stage2_predictor_bundle.py`（数值短标签碰撞正例、结构化query token泄漏负例），报告同步记录证据。目标脚本本地SHA=`81efd50c9c01b4e1b8c7c8252c58731e0c603bb918131e85b615341387f7f1de`，唯一同步目标为`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/build_cvs_stage2_predictor_bundle.py`。同步后必须复算同一SHA；v12根、packages、cells、logs和states均只读保留。

同步前确认v11 shard0已`complete`、v12所有父PID均已自然结束且没有active package builder；随后SCP目标脚本并在N607复算SHA=`81efd50c9c01b4e1b8c7c8252c58731e0c603bb918131e85b615341387f7f1de`，连接结束后`ssh.exe=0`。v12最终只有shard0完整完成，shard1–7均在不同new20包触发同类数值字节误报；8份driver从首字节读到EOF，最终共41/75 packages、164/300 cells、492/900 rows，所有partial outputs只读保留。driver/state证据已回传`evidence/matrix_v12_partial_failure/`。

全新v22计划从受控源`paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json`（SHA=`ba38a470abc2024fd570ca8b6198fd5ac2ae351650781e891c2fa44b62907ab5`）重新生成，未使用v12生成物作source。它绑定同一strict_v14 closure与全新实验根`qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300_v13`，声明25 caches、75 packages、300 cells、900 rows；未授权清单SHA=`e74a64dc469eab34f7b456a121f4ca963651449db2fca958339d762f483cb52c`。授权器复用同candidate/closure的v11 PASS smoke receipt SHA=`6064b3113690fa5453b4f0f82b09febdfca9999fdf68828459102e0ec8c62d94`，授权清单SHA=`f0576cf4731b5e832af7af4f03b34babe30495d36f3b71cf0acac07e63c1954c`、`launch_authority=true`、`authority_state=N607_LANDLOCK_SMOKE_PASS`。TR-3在新根创建、计划远端SHA核验和8分片落地前保持`pending`。

v13唯一计划同步目标预登记为`/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_landlock_strict300_v13/protocol_plan/strict_plan_manifest_v22_authorized_f0576cf4.json`。启动工作目录为`/home/szu2070436088/2510044040/CV-SincNet`，Python为`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；每个shard使用独立原子锁、PID、driver、log目录和state文件，设置`CUDA_VISIBLE_DEVICES=<shard>`并执行`paper_reproduction/scripts/run_cvs_stage2c_effective8_strict_plan.py --stage matrix_shard --device cuda:0 --shard-index <shard> --shard-count 8`。只有再次预检确认旧runner为0、GPU空闲、新v13根不存在后才创建、SCP、核验SHA并启动。

01:16最终预检与实时inventory确认`gpu_compute=[]`、`active_training_processes=[]`，8张GPU空闲，v13根不存在。新根与`protocol_plan`原子创建后，授权计划远端SHA复算为`f0576cf4731b5e832af7af4f03b34babe30495d36f3b71cf0acac07e63c1954c`。8个独立分片依次落地：shard0–7的PID为`1960790`、`1960797`、`1960809`、`1960884`、`1960960`、`1961041`、`1961122`、`1961208`；物理GPU分别为0–7，逻辑device统一`cuda:0`，启动1秒后全部LIVE。SSH/SCP结束后`ssh.exe=0`且无TCP22残留。TR-3转为`verified`；TR-4仍等待300/300 cells和900/900 rows完成。

### 新一组三轮算法探索与Round1追踪（2026-07-16 01:20 CST）

当前N607只读清单确认v13的8个matrix shard正在运行，因此不启动、不同步、不覆盖远端文件。严格链修复、协议握手修复、K1/K10基线和v11/v12/v13基线矩阵不计入新的算法探索轮；新计数从真正改变算法机制的候选开始：Round1=`EvidenceNorm`零梯度类对称注册头，Round2=`JP-R4` support-only稀疏更新，Round3=`JG-R8-LOPO` support-only稀疏更新。Round3结果完成后，第4轮启动前必须执行目标、`项目.md`、conversation index、完整日志和既有路线的强制复盘。

逐项需求、公式、输入输出、资源口径和验证门记录于`analysis/qknnv42_round1_evidence_head_traceability_20260716.md`。Round1输入仅为`[3K,C,D]`密封LEO weak注册support，输出为类原型以及每类2个FP16负证据/尺度校准量；26类新增状态104B，可训练参数、epoch、optimizer step和额外backbone forward均为0，标记为`EVAL_ONLY_CLOSED_FORM_ADAPTATION`。所有类使用同一leave-one-physical/view-out、负证据分位数、support-count收缩和评分公式，不读取query真值/角色/批次类数/类别配额。

机制归因同时更正：当前锁定head为`use_alignment=false`，所以注册前后的旧类原型由相同旧类support独立计算且数值一致；遗忘来自新增类进入argmax后的竞争/hubness，而不是全局alignment坐标被重拟合。Round1直接抑制支持证据不足或对其他类support产生高相似度的prototype hub，但不会预先宣称突破ADV3B02表征上限。

实现审查后增加三个fail-closed约束：Q95固定使用NumPy`method=higher`；类gap先向全类中位数收缩再执行正下限；另用source-locked`inverse_scale_cap`限制最大逆尺度，避免小gap制造新hub。EvidenceNorm启用时禁止alignment、非恒等Gram变换和非零uncertainty penalty。TTA的1→3→5触发继续使用EvidenceNorm前的raw cosine流，EvidenceNorm只改变最终融合/分类分数，从而保持Round1与v14的View预算可比。

## effective8 v14严格矩阵最终结果（2026-07-16 01:29 CST）

v13正式根已完成75/75 packages、300/300 prediction cells、300/300 `PROTOCOL_VALID` receipts和900/900 formal scenario rows；8/8 shard state均为`complete`，所有父PID均已退出。该状态表示协议有效且artifact-complete，不表示算法性能成功。汇总审计`status=PASS`，`audit.json` SHA=`37cb9498edafa299f255b3721d38c3d1d6e254888bf07806fed77888f84eb1e3`；完整300-cell同row表见`evidence/matrix_v13_complete/matrix_summary_v1/cell_summary.md`，900-row机器可读证据见同目录`scenario_rows.json`。

### K×真实新类数联合结果

下表每行由25个receiver×seed cell聚合；每个cell内部先对`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`三场景等权平均。所有百分数仍保持同一实验行上下文。

| 真实new数 | K | 注册前old | 注册后old | direct old | seen-new | H | 平均遗忘 | 注册前−direct | 注册后−direct | 全局旧类floor | 平均view |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 1 | 76.62% | 60.21% | 77.18% | 51.24% | 54.14% | 16.41pp | -0.56pp | -16.97pp | 0.00% | 1.966 |
| 5 | 5 | 80.56% | 64.59% | 77.18% | 60.28% | 61.62% | 15.97pp | +3.38pp | -12.59pp | 0.00% | 1.981 |
| 5 | 10 | 81.96% | 67.09% | 77.18% | 60.16% | 62.83% | 14.87pp | +4.78pp | -10.09pp | 0.00% | 1.986 |
| 5 | 20 | 82.07% | 67.91% | 77.18% | 61.41% | 63.91% | 14.16pp | +4.89pp | -9.27pp | 0.00% | 1.993 |
| 10 | 1 | 76.29% | 54.19% | 77.18% | 38.95% | 44.37% | 22.10pp | -0.89pp | -22.99pp | 0.00% | 2.329 |
| 10 | 5 | 80.23% | 60.52% | 77.18% | 46.92% | 52.30% | 19.71pp | +3.06pp | -16.66pp | 0.00% | 2.330 |
| 10 | 10 | 81.77% | 62.21% | 77.18% | 48.37% | 53.84% | 19.56pp | +4.59pp | -14.97pp | 0.00% | 2.351 |
| 10 | 20 | 81.66% | 63.00% | 77.18% | 50.59% | 55.42% | 18.66pp | +4.48pp | -14.18pp | 0.00% | 2.350 |
| 20 | 1 | 76.20% | 51.53% | 77.18% | 30.83% | 38.17% | 24.67pp | -0.98pp | -25.64pp | 0.00% | 2.520 |
| 20 | 5 | 80.21% | 58.30% | 77.18% | 39.38% | 46.75% | 21.91pp | +3.03pp | -18.88pp | 0.00% | 2.512 |
| 20 | 10 | 81.66% | 59.80% | 77.18% | 41.57% | 48.66% | 21.86pp | +4.48pp | -17.38pp | 0.00% | 2.514 |
| 20 | 20 | 81.59% | 60.71% | 77.18% | 43.19% | 50.08% | 20.88pp | +4.41pp | -16.47pp | 0.00% | 2.514 |

矩阵内最佳联合组为5-new/K20，但注册后old=67.91%、seen-new=61.41%、H=63.91%、遗忘=14.16pp，仍显著低于正式门槛。H最高的单cell是`rx_8_8__seed_713101__new_5__k_20`：注册前old=89.44%、注册后old=75.28%、seen-new=71.00%、H=73.06%、全局旧类floor=45.00%、遗忘=14.17pp、注册后相对direct=-4.17pp；它也不得晋升。

### `项目.md`门槛核验

| 检查项 | 实测证据 | 要求 | 结论 |
|---|---|---|---|
| K10、5-new | old=67.09%；最低聚合旧类`6-15`=46.80%；seen-new=60.16%；H=62.83% | old≥92%；最低旧类≥88%；seen-new≥92% | FAIL |
| K10、10-new | old=62.21%；最低聚合旧类`6-15`=41.93%；seen-new=48.37%；H=53.84% | old≥92%；最低旧类≥88%；seen-new≥90% | FAIL |
| K10、20-new | old=59.80%；最低聚合旧类`6-15`=35.87%；seen-new=41.57%；H=48.66% | old≥92%；最低旧类≥88%；seen-new≥86% | FAIL |
| K1无遗忘 | 15个receiver×new-count聚合组的适应增益全部为负；全矩阵K1平均遗忘16.41%至24.67% | overall gain≥+2pp且CI下界>0；每个receiver非负 | FAIL |
| K5相对K10鲁棒性 | 225个匹配receiver×seed×scenario×new pair中仅59个同时满足四项≤3pp下降；old/min-old/H/seen分别有78/112/65/67次违规 | 每个匹配单元四项下降均≤3pp | FAIL |
| 注册后优于direct | 0/300 cells满足注册后old≥direct；平均差为-9.27pp至-25.64pp | 应达到正收益并满足绝对门槛 | FAIL |
| 旧类floor稳定性 | 仅8/300 cells的全局旧类floor≥50%；所有12个K×new组的跨cell floor均为0 | 不得逐类塌缩 | FAIL |

candidate的遗忘在12个K×new组中均低于identity对照，但仍全部为正，范围14.16pp至24.67pp；全矩阵平均遗忘19.23pp，单cell最大40.83pp。其注册前adapt在K≥5时通常高于direct，说明轻量适配本身已产生有限正收益；新增类进入全类argmax后，旧/新类竞争与prototype hubness重新造成系统性塌缩，这是当前主要瓶颈，而不是继续增大K即可解决的问题。

### 资源、缺口与最终判定

| 真实new数 | 参数/epoch/持久状态 | 峰值CUDA显存 | query延迟均值 | 平均/P95 view | 1/3/5-view触发率 |
|---:|---|---:|---:|---:|---:|
| 5 | 44,048/12/109,818B | 168,198,144B | 0.677ms | 1.982/3 | 51.79%/47.35%/0.86% |
| 10 | 44,048/12/109,818B | 168,271,872B | 0.714ms | 2.340/3 | 35.02%/62.96%/2.02% |
| 20 | 44,048/12/109,818B | 168,441,856B | 0.751ms | 2.515/5 | 27.65%/68.96%/3.39% |

参数、epoch、持久状态与1→3→5-view预算符合偏好上限，但当前receipt未提供MAC与端到端adapt latency；严格同协议的MRIOR-SDA配对基线也未随本矩阵产出。因此不能主张资源全面达标或相对MRIOR-SDA优越。

最终判定：`effective8-v14-strict300-v13`为`PROTOCOL_VALID`、artifact-complete的正式负结果；它不是当前性能最佳版本，不满足Stage2-C晋升、部署或论文成功声明条件。后续应优先修复support-only全类注册竞争与prototype hubness，并以相同密封预测/独立评分闭环验证；不得通过query真值、role、quota或global assignment调参，也不建议仅扩展K或重复当前effective8矩阵。

设计追踪最终计数：TR-1至TR-5均为`verified`，`deferred=0`、`rejected=0`、`blocked=0`。实现与当前严格协议完全对应，不是近似实现；剩余最高风险是算法层面的注册塌缩，以及MRIOR-SDA、MAC和端到端adapt latency证据缺口。

## Round1 EvidenceNorm真实注册诊断（2026-07-16）

本轮是新“三轮算法探索→强制回顾”周期的第1轮，计数为1/3。输入为v11同一ADV3B02/effective8密封package，仅使用三个`LEO_weak`场景的registered support；receiver=`20-1`、seed=`713101`、K=10、20个真实seen-new TX。预测侧先冻结1560行truth-free NPZ，scorer随后才加入标签；query truth、role、真实批次类数、类别配额和全局分配访问均为false。本地结果标记`NON_LAUNCH_DIAGNOSTIC`，不替代Landlock正式矩阵或独立确认。

|场景|注册前old|注册后old|direct old|seen-new|H|最低旧类|遗忘|平均View|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|80.83%|68.33%|74.17%|42.00%|52.02%|35.00%|12.50pp|2.654|
|`leo_low_elev_weak`|77.50%|60.83%|66.67%|33.00%|42.79%|20.00%|16.67pp|2.758|
|`leo_rain_weak`|77.50%|60.83%|70.00%|36.00%|45.23%|30.00%|16.67pp|2.769|
|三场景等权|78.61%|63.33%|70.28%|37.00%|46.68%|20.00%|15.28pp|2.727|

|同package对比|注册前old|注册后old|seen-new|H|遗忘|
|---|---:|---:|---:|---:|---:|
|v11原head|76.94%|59.44%|35.50%|44.45%|17.50pp|
|Round1 EvidenceNorm|78.61%|63.33%|37.00%|46.68%|15.28pp|
|变化|+1.67pp|+3.89pp|+1.50pp|+2.23pp|-2.22pp|

逐样本审计给出明确机制：旧类360行中救回25行、损害11行，old→new从108降至88；新类1200行中救回69行、损害51行，wrong-new从605降至531，但new→old从169升至225。旧类`8-20`提升18.33pp、`20-15`提升8.33pp，而`14-10`下降6.67pp；新类`8-3`提升15.00pp，但`4-10`下降23.33pp。EvidenceNorm压低了新prototype hub，却对低证据新类产生过度惩罚，因此只能作为Round2边界学习的组成部分，不能单独晋升。

资源按部署与formal评估分开记账：EvidenceNorm-only为104B deployment、24B before comparator、128B formal双流；计入原型/transform/bias后为14,820B、3,180B、18,000B FP16载荷，对应29,640B、6,360B、36,000B FP32实时数组。原adapter为109,818B，因此部署持久状态总计124,638B，formal双流诊断状态127,818B；0新增训练参数、0epoch、0optimizer step、0额外backbone forward。量化后最大逆尺度9.99634≤10。峰值CUDA显存72,591,360B，本地query延迟受RTX5070Ti与N607环境差异影响，不与v11远端延迟作严格归因。

自适应TTA触发逻辑使用EvidenceNorm前raw cosine流；与v11的1560行比较只有1行从1-view跨到3-view，定位为不同GPU数值阈值边界，不能声称逐行完全相同。整体仍为平均2.727、P95=5，1/3/5-view比例18.85%/75.96%/5.19%。

最终artifact：prediction NPZ SHA=`1ab79ffcb279e9580d48c72f34d04f79f5e7f28987bc7a6140a1cb045b7f325c`，prediction manifest SHA=`a035633664d21b0cc0128583acb826801446637e6f5f66c8041a1b53696c3d06`，truth sidecar SHA=`5a70620a6b90a86ca47b8be1bad83c5e881826d976cd3885b47d0fe6ffde8470`。本地`ssr-gpu`验证：30项聚焦测试、61项更广runtime/closure/diagnostic测试及1项sealed v1 pipeline E2E均PASS；独立复审未发现P0/P1。Round1结论为“方向正收益但性能不合格”，下一轮固定进入JP-R4 support-only稀疏边界学习；Round3完成前不得启动第4轮，Round3后必须按用户新增规则回顾目标、`项目.md`、历史对话、三轮完整日志和已探索方法。
