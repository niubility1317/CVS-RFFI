# phase2_target_old_mlp_adapter_upper_bound_20260706_0345

## 基本信息

|字段|内容|
|---|---|
|experiment_id|phase2_target_old_mlp_adapter_upper_bound_20260706_0345|
|timestamp|2026-07-06 03:45 CST|
|operator|Codex|
|objective|在prototype-only和ridge linear-probe target-old上限均低于OLD80后，验证固定epoch小MLP adapter是否能在只使用`R_t/Y_old/target_old`support的情况下显著提升旧类target query准确率。|
|diagnostic_scope|`TARGET_OLD_ONLY_UPPER_BOUND_DIAGNOSTIC`/`NON_DEPLOYMENT_DIAGNOSTIC`|
|base_feature_packages|`ADV3B02_CORE90_FROZEN`、`EPOC_B_OSPR_FEATURES`|

## 协议边界

- 该实验只使用`dataset_role == target_old`且TX属于`Y_old`的feature rows。
- `target_new`、`target_unknown`、`proxy_unknown`不会用于训练、support-only标准化统计、阈值拟合、early stopping或模型选择。
- support/query按每个old TX的稳定排序切分，support取前`K`，query取剩余样本，记录support/query索引hash和重叠数量。
- 训练为固定epoch，无query early stopping。不同seed/K只作为诊断网格逐行报告，不作为部署模型选择。
- 本实验不报告`seen_new_acc`、unknown FAR、unknown reject，不声明Stage2-C成功或部署成功。

## 假设与对照

|项目|内容|
|---|---|
|hypothesis|若目标域旧类特征中仍存在非线性可分信号，小MLP adapter应显著高于prototype-only和ridge linear-probe，至少接近或超过OLD80阶段门槛。|
|comparison_target|prototype-only target-old上限和ridge linear-probe target-old上限。|
|failure_interpretation|若MLP仍低于OLD80或min_old_class_acc仍很低，说明当前ADV3B02/EPOC_B feature geometry在LEO叠加信道下对旧类目标域本身已不足，应转向底层特征修复或teacher-guided distillation，而不是继续堆协同拒识阈值。|

## 本地变更

|文件|用途|sha256|
|---|---|---|
|`E:\type10-7\code\scripts\eval_target_old_mlp_adapter_upper_bound.py`|新增target-old-only小MLP adapter上限诊断脚本；远端PyTorch/NumPy兼容性修复后使用list构造tensor；样本不足row改为fail-closed；支持显式`R_s/R_t/channel_view`协议字段并记录观测target-old接收机集合。|`C0C8A255E63088A0B06E18E54DDE5AFD835B2DEDCA0C23EF1CD747BDDC63856B`|
|`E:\type10-7\code\tests\test_target_old_mlp_adapter_upper_bound.py`|新增协议边界、smoke验证、样本不足fail-closed验证和receiver/channel字段验证。|`58CCAA7D0D07714480F56DCEF1BCD9BCF8128F75BD06F42069FCCA5B2E9EADA4`|

Snapshot:

`E:\type10-7\code\snapshots\phase2_target_old_mlp_adapter_upper_bound_20260706_0345`

## 本地验证

|命令|结果|
|---|---|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m pytest code\tests\test_target_old_mlp_adapter_upper_bound.py -q`|RED:缺少`eval_target_old_mlp_adapter_upper_bound`，符合TDD预期。|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m pytest code\tests\test_target_old_mlp_adapter_upper_bound.py code\tests\test_target_old_linear_probe_upper_bound.py -q`|PASS:4 passed；仅`.pytest_cache`权限warning。|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python -m py_compile code\scripts\eval_target_old_mlp_adapter_upper_bound.py code\tests\test_target_old_mlp_adapter_upper_bound.py`|PASS。|
|`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 conda run -n ssr-gpu python code\scripts\eval_target_old_mlp_adapter_upper_bound.py --help`|PASS。并行`conda run`曾触发Windows临时文件锁，串行重跑通过；该问题不是脚本失败。|
|远端直接测试暴露`torch.from_numpy`兼容性问题后，本地补丁并重跑`pytest`+`py_compile`|PASS:4 passed；脚本hash更新为`1588C3A3CA15065F6AA6EA2CA77C1E2785D51D1E0B1C9A8FA33AB342F91869EC`。|
|子agent审查指出样本不足row可能高估后，本地补丁并重跑`pytest`+`py_compile`|PASS:5 passed；脚本hash更新为`1321CA0675468E63AF09E5A80B233B8D32CFF1E4841F886BD4B2E3FCE3469FE2`，测试hash更新为`B05D7116C67BBD39AE0F680F9C4845105852727BDD09557096F0B1B06AECC1D9`。|
|feature manifest缺少source/target receiver列表后，本地补丁并重跑`pytest`+`py_compile`|PASS:5 passed；脚本hash更新为`C0C8A255E63088A0B06E18E54DDE5AFD835B2DEDCA0C23EF1CD747BDDC63856B`，测试hash更新为`58CCAA7D0D07714480F56DCEF1BCD9BCF8128F75BD06F42069FCCA5B2E9EADA4`。|

## 计划同步

Remote root:

`/home/szu2070436088/2510044040/CV-SincNet`

|本地路径|远端路径|
|---|---|
|`E:\type10-7\code\scripts\eval_target_old_mlp_adapter_upper_bound.py`|`N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/eval_target_old_mlp_adapter_upper_bound.py`|
|`E:\type10-7\code\tests\test_target_old_mlp_adapter_upper_bound.py`|`N607:/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_target_old_mlp_adapter_upper_bound.py`|
|`E:\type10-7\automation_reports\CV-SincNet\phase2_target_old_mlp_adapter_upper_bound_20260706_0345\report.md`|`N607:/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_target_old_mlp_adapter_upper_bound_20260706_0345/report.md`|
|`E:\type10-7\code\SYNC_MANIFEST.txt`|`N607:/home/szu2070436088/2510044040/CV-SincNet/code/SYNC_MANIFEST.txt`|

## 计划远端运行

|字段|内容|
|---|---|
|conda/python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|cwd|`/home/szu2070436088/2510044040/CV-SincNet`|
|run_root|`runs/phase2_target_old_mlp_adapter_upper_bound_20260706_0345`|
|log_root|`logs/phase2_target_old_mlp_adapter_upper_bound_20260706_0345`|
|device|CPU foreground diagnostic by default；若GPU空闲可显式`--device cuda:0`，但当前小support训练无需占用GPU。|
|target_old_tx_ids|`14-10,14-7,20-15,20-19,6-15,8-20`|
|source_receiver_ids|`1-1,1-19,14-7,18-2,19-2,2-1,2-19`|
|target_receiver_ids|`20-1,3-19,7-14,7-7,8-8`|
|target_channel_view|`satellite/LEO`|
|k_values|`1,2,5,10,20,50`|
|seeds|`1,2,3`|
|epochs|`120`|
|hidden_dim|`64`|
|lr/weight_decay/dropout|`0.01`/`1e-4`/`0.0`|

Representative server command pattern:

```bash
PYTHONPATH=code:code/scripts /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
  code/scripts/eval_target_old_mlp_adapter_upper_bound.py \
  --feature_npz <candidate features_stage2c_leo_multirx.npz> \
  --target_old_tx_ids 14-10,14-7,20-15,20-19,6-15,8-20 \
  --source_receiver_ids 1-1,1-19,14-7,18-2,19-2,2-1,2-19 \
  --target_receiver_ids 20-1,3-19,7-14,7-7,8-8 \
  --target_channel_view satellite/LEO \
  --k_values 1,2,5,10,20,50 --seeds 1,2,3 \
  --epochs 120 --hidden_dim 64 --lr 0.01 --weight_decay 0.0001 \
  --dropout 0.0 --device cpu \
  --output_json <run_root>/<candidate>/mlp_adapter_metrics.json \
  --summary_csv <run_root>/<candidate>/mlp_adapter_summary.csv
```

Feature inputs:

|candidate|feature_npz|
|---|---|
|ADV3B02_CORE90_FROZEN|`runs/phase2_adv3b02_frozen_manytx_unknown_diag_20260706/ADV3B02_CORE90_FROZEN_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz`|
|EPOC_B_OSPR_FEATURES|`runs/phase2_epoc_b_ospr_qknn_collab_20260705_retry1/EPOC_B_OSPR_QKNN8_M1_TO_ALL_R5/features_stage2c_leo_multirx.npz`|

Expected outputs per candidate:

- `mlp_adapter_metrics.json`
- `mlp_adapter_summary.csv`

## 结果

Status:remote_completed_negative_evidence。

Remote verification:

|项目|结果|
|---|---|
|N607 preflight|PASS，直连`N607`、项目根、GPU可见；GPU2/3显存最低约10MiB，本诊断使用CPU前台运行。|
|remote hash|脚本`c0c8a255e63088a0b06e18e54dde5afd835b2dedca0c23ef1cd747bddc63856b`；测试`58ccaa7d0d07714480f56dcef1bcd9bcf8128f75bd06f42069fcca5b2e9eada4`。|
|remote py_compile|PASS。|
|remote direct tests|`direct_target_old_mlp_adapter_tests=PASS`。|
|remote run mode|CPU foreground diagnostic，无GPU训练占用，无PID留存。|
|run_root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_target_old_mlp_adapter_upper_bound_20260706_0345`|
|log_root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_target_old_mlp_adapter_upper_bound_20260706_0345`|
|ssh_cleanup|每次SSH/SCP后本地`ssh.exe`无残留，到`172.31.111.215:22`和`172.31.105.18:22`无`ESTABLISHED`连接。|

Artifact hashes:

|candidate|artifact|sha256|
|---|---|---|
|ADV3B02_CORE90_FROZEN|`mlp_adapter_metrics.json`|`fffbeac8433175f2397c269dc7ccaabaf6f20aa19567d8ad48fbf408bcc4317e`|
|ADV3B02_CORE90_FROZEN|`mlp_adapter_summary.csv`|`4ab253cf3bcda7a51491c0c6d2e66937014dac31ceae941e9d4b2e3517d92626`|
|EPOC_B_OSPR_FEATURES|`mlp_adapter_metrics.json`|`819e931855c4e849866a3b4752f5185e59ce04c890cf8875df1be78cc39bfa33`|
|EPOC_B_OSPR_FEATURES|`mlp_adapter_summary.csv`|`64542558eac029995663be0072eb6be96e5f6329fa2f6575c5454f9d72fcb553`|

Protocol checks from final metrics:

|candidate|receiver_split_disjoint|observed_target_old_rx_within_target_receiver_ids|source_receiver_ids|target_receiver_ids|observed_target_old_rx_ids|target_channel_view|invalid_rows|
|---|---:|---:|---|---|---|---|---:|
|ADV3B02_CORE90_FROZEN|true|true|`1-1,1-19,14-7,18-2,19-2,2-1,2-19`|`20-1,3-19,7-14,7-7,8-8`|`20-1,3-19,7-14,7-7,8-8`|`satellite/LEO`|0|
|EPOC_B_OSPR_FEATURES|true|true|`1-1,1-19,14-7,18-2,19-2,2-1,2-19`|`20-1,3-19,7-14,7-7,8-8`|`20-1,3-19,7-14,7-7,8-8`|`satellite/LEO`|0|

Same-row result table:

|candidate|row_type|K|seed|support|query|train_acc|old_acc|macro_old_acc|min_old_class_acc|verdict|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|ADV3B02_CORE90_FROZEN|best_old|20|1|120|9480|100.00%|69.82%|69.82%|32.09%|below OLD80；min旧类远低于95%|
|ADV3B02_CORE90_FROZEN|best_floor|50|1|300|9300|100.00%|67.27%|67.27%|44.32%|best floor仍低于OLD80和min95|
|EPOC_B_OSPR_FEATURES|best_old|5|1|30|9570|100.00%|72.99%|72.99%|25.39%|below OLD80；min旧类极低|
|EPOC_B_OSPR_FEATURES|best_floor|50|2|300|9300|100.00%|67.99%|67.99%|42.65%|best floor仍低于OLD80和min95|

Interpretation:

- 小MLP adapter在support上达到`train_acc=100%`，但target query old accuracy最高只有`72.99%`，每类最低最高只有`44.32%`。
- 与prototype-only和ridge linear-probe相比，MLP没有突破OLD80，说明当前瓶颈不是线性头表达能力不足，而是LEO叠加目标域特征几何本身没有稳定分离旧类。
- 该结果仍是`TARGET_OLD_ONLY_UPPER_BOUND_DIAGNOSTIC`，不包含seen-new注册和unknown拒识，不能声明Stage2-C成功或部署成功。

## 风险与下一步

- MLP上限仍低于OLD80，继续调未知类阈值或协同投票数量不是主线；应转向ADV3B02 teacher-guided distillation、source-side LEO strong-view feature separation、support-protected geometry repair。
- 若MLP显著超过OLD80，则说明部署端轻量adapter有利用空间，但仍需回到Stage2-B/C协议，重新加入seen-new support/query与unknown eval-only拒识。
