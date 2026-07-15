# qKNN全量非dense adapter epoch 125任务扫描报告

## 一、实验信息

|字段|内容|
|---|---|
|实验ID|`qknnv42_nondense_adapter_epoch_sweep_20260715_104409`|
|设计时间|2026-07-15 10:44:09（Asia/Hong_Kong）|
|操作者|Codex|
|目标|在严格ADV3B02基础上，禁用dense query图、角色Oracle与类别配额，比较`adapter epoch={2,5,10,20,30,60}`的Stage2-C表现|
|任务规模|单qKNN头FFT96基线125任务；每个adapter epoch为5个target receiver×5个seed×`K={1,2,5,10,20}`，共125任务；总计875任务|
|比较对象|历史60epoch全量非Oracle结果保留`dense_transductive`，不满足当前逐样本协议；本实验用于给出合法的非dense epoch曲线|
|特征与头|单头基线为严格ADV3B02、1-view、z_id160+FFT96、无训练型adapter；六档为严格ADV3B02+`id_norm_late_feature` adapter+5-view TTA+FFT96；均使用support-only `support_diag_whiten_fisher`和逐样本argmax|
|声明边界|2/5/10/20epoch属于极轻型首选epoch上限内；30epoch属于performance-relaxed资源消融；60epoch超过40epoch绝对上限，仅作非极轻型资源对照|

## 二、协议与数据矩阵

- Stage2-C旧类为`14-10,14-7,20-15,20-19,6-15,8-20`，新类为`1-16,1-18`，unknown不参与。
- target receiver为`20-1,3-19,7-14,7-7,8-8`。
- seed为`713101..713105`，K为`1,2,5,10,20`，每类每场景query为20。
- 三个部署主视图为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- support/query必须互斥；同一receiver×seed的support按K嵌套，query固定。
- adapter只使用地面source/proxy训练池；target query不参与训练、拟合、模型选择或早停。
- 正式query同时面对全部8个已注册类别，配置固定`qknnv42_decision_mode=per_sample_argmax`、`qknnv42_labelprop_mode=disabled`、`non_deployment_oracle_diagnostic=false`。
- 预检与完成后审计同时阻断`dense_transductive`、role Oracle、class quota、query-query graph和query batch state。
- 单qKNN头基线明确包含96维FFT辅助特征；它不使用60epoch训练型adapter，也不执行5-view TTA。

## 三、本地版本与变更

根目录`E:\type10-7`不是Git仓库；本报告镜像到Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`。工作树存在其他任务的未提交变更，本实验只暂存和提交下列明确文件，不覆盖、不暂存其他变更：

- `paper_reproduction/scripts/run_cvs_publication_matrix.py`：新增非dense配置预检和完成后query图阻断。
- `tests/test_cvs_publication_matrix.py`：新增dense query配置拒绝回归测试。
- `paper_reproduction/scripts/generate_qknnv42_nondense_adapter_epoch_configs.py`：生成六档严格配置。
- `paper_reproduction/scripts/run_cvs_qknnv42_nondense_adapter_epoch_125_20260715.sh`：单个epoch的严格特征训练、导出与125任务执行器。
- 六个`cvs_qknnv42_nondense_adapter_e*_stage2c_20260715_n607.json`配置。
- `cvs_qknnv42_singlehead_fft96_nondense_stage2c_20260715_n607.json`及对应125任务启动器。
- 本报告及后续汇总artifact。

## 四、本地验证

在`ssr-gpu`环境串行完成：

```text
python -m py_compile paper_reproduction/scripts/generate_qknnv42_nondense_adapter_epoch_configs.py paper_reproduction/scripts/run_cvs_publication_matrix.py
python -m pytest tests/test_cvs_publication_matrix.py tests/test_cvs_proposed_stage2_runner.py -q
bash -n paper_reproduction/scripts/run_cvs_qknnv42_nondense_adapter_epoch_125_20260715.sh
python -m paper_reproduction.scripts.run_cvs_publication_matrix ... # 六档配置分别dry-run
```

结果：30项联合测试通过，补充fail-closed回归5项通过；Python编译通过；两个shell启动器语法通过；六档epoch和单qKNN头FFT96基线dry-run均各生成125行矩阵，合计875行。

## 五、N607启动计划

|字段|内容|
|---|---|
|远端根|`/home/szu2070436088/2510044040/CV-SincNet`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，启动前重新验证|
|feature根|`runs/qknnv42_nondense_adapter_epoch_features_20260715/E<epoch>`|
|run根|`runs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E<epoch>`|
|log根|`logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E<epoch>`|
|GPU|启动前按实时占用分配；不超过每GPU两个训练进程|
|预计输出|每档5个严格feature NPZ、125个完整任务目录、125个任务日志、matrix manifest与launcher日志|

单qKNN头FFT96基线复用已审计的严格单视图feature NPZ，run/log子目录为`singlehead_fft96`；不重新训练backbone或adapter。

## 五点一、启动前N607证据

2026-07-15 10:47–10:50直连预检通过：身份、项目根、8张RTX3090和服务器时间可见；远端Python为3.10.19，`/home`剩余7.6TB。进程盘点确认GPU0–7各有1个RIEI训练进程，PID分别为`1116006,1116023,1116045,1116040,1116057,1116061,1116071,1116076`，每进程显存624MiB。用户已明确要求启动本实验；按每GPU最多两个训练进程规则，计划把六个adapter任务放在GPU0–5，把单qKNN头FFT96矩阵放在GPU6，不干预RIEI，不使用GPU7。

同步前远端`cvs_method_runner.py`、matrix worker和adapter导出器hash均与本地已验证版本不同，因此必须先同步本地版本。同步映射如下，均为同相对路径覆盖：

## 五点二、正式启动与即时健康

2026-07-15约10:54正式启动7个独立launcher：

|分支|GPU|launcher PID|启动日志|
|---|---:|---:|---|
|adapter E2|0|1149135|`logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E2/launcher_e2.out`|
|adapter E5|1|1149137|`logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E5/launcher_e5.out`|
|adapter E10|2|1149140|`logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E10/launcher_e10.out`|
|adapter E20|3|1149144|`logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E20/launcher_e20.out`|
|adapter E30|4|1149148|`logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E30/launcher_e30.out`|
|adapter E60|5|1149151|`logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E60/launcher_e60.out`|
|单qKNN头FFT96|6|1149155|`logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/singlehead_fft96/launcher_singlehead_fft96.out`|

启动后约15秒检查：7个launcher均存活；六个adapter子进程各占约1314MiB，连同每卡原RIEI 624MiB仍远低于24GB；六档均已写出epoch1有限loss，无Traceback、OOM、Killed或NaN/Inf。单qKNN头FFT96进入125行矩阵执行，无额外adapter训练。


|本地/远端相对路径|本地SHA256|
|---|---|
|`paper_reproduction/cvs_aligned/cvs_method_runner.py`|`8f526c7821d9ebb0342a3de325d8442e0397ef3b98bb0b60a89815c4f0ec9420`|
|`paper_reproduction/scripts/run_cvs_publication_matrix.py`|`e014297a2167bfe7633010a78ab9bcc0c2ce21f711a14817ed421531493ee8a7`|
|`code/scripts/train_apply_phase1_iq_preadapter_20260703.py`|`553410ebe4f1337aa4017bae69587a14f6b8b57e746de5c67d81e8a8e8ccc979`|
|`code/cvsrffi/checkpoint_loading.py`|`c0cfda856ce707d83ecf9547c6a3bf74995a2eaa430c9da19c63c3e211115229`|
|`paper_reproduction/scripts/run_cvs_qknnv42_nondense_adapter_epoch_125_20260715.sh`|`b91d5b7b7ea7256b7c5fe85d7c52d30369e20b4bc8c2080a301de3bef2882da1`|
|`paper_reproduction/scripts/run_cvs_qknnv42_singlehead_fft96_nondense_125_20260715.sh`|`751bd58a811f1a15a6efde7445c33bca895fa3443b9eaea3ed98be9037a76fc9`|
|`paper_reproduction/configs/cvs_qknnv42_nondense_adapter_e{2,5,10,20,30,60}_stage2c_20260715_n607.json`|逐文件hash保存在Git提交与同步后远端`sha256sum`证据中|
|`paper_reproduction/configs/cvs_qknnv42_singlehead_fft96_nondense_stage2c_20260715_n607.json`|`a2ed811823333bcf77cdd49bb81cabea74a95669436fcedac618206a53b9db6b`|


单档启动命令模板：

```bash
nohup env ADAPTER_EPOCH=<epoch> GPU=<gpu> SWEEP_ID=qknnv42_nondense_adapter_epoch_sweep_20260715_104409 \
  bash paper_reproduction/scripts/run_cvs_qknnv42_nondense_adapter_epoch_125_20260715.sh \
  > logs/qknnv42_nondense_adapter_epoch_sweep_20260715_104409/E<epoch>/launcher_e<epoch>.out 2>&1 &
```

启动前必须执行N607直连预检、进程/GPU/磁盘盘点、远端hash与dry-run验证。实际PID、GPU和完整命令将在启动后回填。

## 六、完成判据与审计

1. 单qKNN头FFT96基线和六档epoch均完成125/125，合计875/875；失败、跳过均为0。
2. 每档5个feature manifest均满足严格ADV3B02加载，missing/unexpected/shape mismatch为0/0/0，并记录对应adapter epoch、5-view和FFT96。
3. 875个split manifest全部满足support/query无重叠；K嵌套和固定query审计通过。
4. 2625个场景全部满足`role_oracle_used=false`、`equal_class_quota_used=false`、`query_query_graph_used=false`、`query_batch_state_required=false`。
5. 完整扫描adapter训练日志和875个任务日志，无Traceback、OOM、Killed、NaN/Inf；保存完整adapter loss trace。
6. 主表按adapter epoch×K联合报告`old_before`、`old_after`、`seen_new_acc`、`H_old_new`、遗忘、旧新混淆；逐行表保留receiver、seed、K、同run指标与资源档。
7. 60epoch只作为资源对照，不进入极轻型或performance-relaxed正式晋升结论。

## 七、当前状态

## 八、首轮审计发现与成对基线修复

首轮本地全量审计已确认六档adapter之间完全同任务：抽样及全局检查均显示E2/E5/E10/E20/E30/E60在同一receiver×seed×scenario×K下support与query物理sample ID一致，K内support严格嵌套，query固定。

旧的`singlehead_fft96`分支复用了2026-07-14特征池。以receiver `20-1`、seed `713101`、`leo_clear_weak`、K=5为例，它与E2仅共享4/40个support和2/160个query。因此该125行结果保留为“非成对旧池诊断”，不进入adapter epoch主比较。

修复方案是不覆盖原结果，新增`singlehead_fft96_paired`：使用严格ADV3B02、无训练型adapter、1-view、z_id160+FFT96，并复用本次adapter扫描相同的export seed=4070391、每TX最多80个物理样本、receiver/角色范围和LEO场景生成规则。新feature另存于`runs/qknnv42_singlehead_fft96_paired_features_20260715`，新125任务另存于同一sweep根的`singlehead_fft96_paired`。只有修复后的基线进入最终875行正式比较。


启动前设计、配置、fail-closed验证和875行dry-run已完成；N607直连预检通过，严格单视图feature已确认FFT96、1-view及ADV3B02严格加载0/0/0。同步、正式启动、完整日志解析与最终逐K主表待执行。
