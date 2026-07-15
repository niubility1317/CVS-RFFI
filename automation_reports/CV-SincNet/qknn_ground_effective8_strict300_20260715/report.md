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

## 完成后结果表

实验完成后在本节追加逐单元同一行结果，至少包含candidate ID、机制、receiver/TX split、K-shot、seed、old/seen-new/unknown指标、coverage/rollback/defer、loss/adapter摘要和最终判定。不得用来自不同单元的独立极值替代联合行。
