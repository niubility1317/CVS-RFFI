# qKNNV42严格ADV3B02双125重跑报告

## 实验信息

|字段|内容|
|---|---|
|实验ID|`qknnv42_strict_dual125_20260714_183556`|
|时间|2026-07-14 18:35:56（Asia/Hong_Kong）|
|操作者|Codex|
|目标|修复ADV3B02特征导出的非严格checkpoint重建，重新运行轻量单视图FFT96与完整legacy oracle两组各125任务|
|比较对象|旧兼容加载诊断：轻量组`old=75.12%,new=64.64%,H=68.56%`；完整组`old=82.93%,new=93.37%,H=87.65%`|
|声明边界|轻量组为逐样本Stage2-C诊断；完整组含角色/类别配额Oracle，仅为`NON_DEPLOYMENT_ORACLE_DIAGNOSTIC`|

## 根因与修复

旧导出器直接调用通用`build_dual_model`，没有复用ADV3B02训练时`SSDG.train_ssdg.merge_checkpoint_args`、`_apply_model_cli_args`和`build_baseline_model`完整架构路径，导致7个missing key、31个unexpected key和3个shape mismatch仍被`strict=False`兼容加载后继续运行。

本次新增fail-closed严格加载器`code/cvsrffi/checkpoint_loading.py`：

1. 从checkpoint state恢复domain head维度；
2. 从checkpoint args恢复全部训练时模型选项；
3. 调用训练原生`build_baseline_model`；
4. 完整加载state；任何shape mismatch、missing或unexpected key立即失败；
5. 导出manifest固定记录`checkpoint_load_strict=true`与键计数0/0/0。

`code/export_spaceborne_features.py`和`code/scripts/train_apply_phase1_iq_preadapter_20260703.py`均切换到同一严格加载器，避免轻量分支与60epoch adapter分支再次分叉。

## 协议与矩阵

保持`项目.md`不变：5个target receiver×5个seed×K={1,2,5,10,20}=125行；旧类6个、新类2个；support/query同属目标接收机简化LEO视图且不重叠；unknown不参与Phase2；三场景为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。

|分支|输入与方法|决策|部署边界|
|---|---|---|---|
|strict singleview FFT96|严格ADV3B02、单个LEO视图、z_id160+FFT96、无adapter|逐样本argmax|轻量Stage2-C诊断|
|strict full legacy oracle|严格ADV3B02、60epoch id_norm_late_feature adapter、5-view TTA、FFT96|角色Oracle+类别配额Hungarian|仅上限诊断|

## 本地变更与验证

Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`。根目录`E:\type10-7`不是Git仓库，本报告同步镜像到Git承载面。启动前工作树存在其他任务对SSDG/loss/runner及报告的未提交修改，本次不覆盖、不暂存、不提交这些不相关变更。

本次文件：

- `code/cvsrffi/checkpoint_loading.py`：严格重建与fail-closed加载；
- `code/export_spaceborne_features.py`：单视图导出使用严格加载并写审计；
- `code/scripts/train_apply_phase1_iq_preadapter_20260703.py`：adapter训练与导出使用严格加载并写审计；
- `code/tests/test_exact_ssdg_checkpoint_loading.py`：严格成功与missing key阻断测试；
- 两个launcher增加独立feature root/config覆盖；
- 两个strict配置指向全新输出根，保留旧诊断artifact。

验证命令：

```text
conda run -n ssr-gpu python -m pytest code/tests/test_exact_ssdg_checkpoint_loading.py code/tests/test_stage2_export_target_old_cli.py -q
conda run -n ssr-gpu python -m py_compile code/cvsrffi/checkpoint_loading.py code/export_spaceborne_features.py code/scripts/train_apply_phase1_iq_preadapter_20260703.py
```

结果：4项测试全部通过，三个Python文件编译通过。

## N607预检与资源

2026-07-14 18:36直连预检PASS：目标、身份、项目根与8张RTX3090可见。盘点显示GPU0–7各有1个RIEI训练进程，显存约470–500MiB。按每GPU最多2个任务规则，本实验计划在GPU0–3各增加1个短时导出/adapter任务，不干预现有进程，不超过上限。

## 远端路径与启动计划

|字段|轻量组|完整组|
|---|---|---|
|GPU|0,1,2并行导出；矩阵CPU|3训练adapter并导出；矩阵CPU|
|feature root|`runs/cvs_publication_adv3b02_fft96_singleview_strict_20260714_183556`|`runs/cvs_qknnv42_full_adapter5_fft96_strict_20260714_183556`|
|run root|`runs/cvs_qknnv42_fft96_singleview_strict125_20260714_183556`|`runs/cvs_qknnv42_full_legacy_oracle_strict125_20260714_183556`|
|launcher log|`paper_reproduction/logs/qknnv42_strict_dual125_20260714_183556/single_launcher.log`|`paper_reproduction/logs/qknnv42_strict_dual125_20260714_183556/full_launcher.log`|
|PID文件|同目录`single.pid`|同目录`full.pid`|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|同左|
|工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|同左|

预期artifact：三份严格单视图NPZ、五个接收机adapter NPZ、250个task目录及各自metrics/resolved_config/split_manifest/score table、launcher与逐任务完整日志。

## 成功与停止标准

- 所有feature manifest必须为`checkpoint_load_strict=true`且missing/unexpected/mismatch均为0；否则停止矩阵。
- 两分支均必须完成125/125，support/query无重叠，三LEO场景覆盖完整。
- 日志不得出现Traceback、OOM、Killed、NaN/Inf或静默兼容加载。
- 结果按同一row联合报告old、new、H、coverage/defer与Oracle边界；不以不同row极值拼接结论。

## 结果

运行中，完成后回填。
