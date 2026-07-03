# phase2_adv3b02_support_ridge_adapter_20260704

## 基本信息

|字段|值|
|---|---|
|实验ID|phase2_adv3b02_support_ridge_adapter_20260704|
|时间|2026-07-04|
|operator/agent|Codex|
|目标|评估support-only轻量ridge adapter head在ADV3B02 Stage2-C多目标接收机协同推理中的诊断价值|
|结论边界|NON_DEPLOYMENT_DIAGNOSTIC；不作为Stage2-C成功、部署成功或论文主结论|

## 假设与对比目标

假设：在冻结Phase1特征的前提下，每个未见目标接收机只用本机`target_old`和`target_new`的`K-shot support`闭式拟合线性ridge head，可能提升old和seen-new识别；但unknown query必须只用于评估，不能参与阈值、adapter或超参选择。

对比目标：上一轮qKNN/dual-route support-quality诊断在`k=5`仍只有`old_acc=0.1872`、`seen_new_acc=0.0833`、`unknown_reject_rate=0.8667`，本路线检查“known识别可恢复但unknown拒识是否失效”的折中。

成功门槛仍是用户指定目标：old 99%且每类≥95%；seen-new 97%且每类≥93%；unknown reject 99%。本报告未达到该目标。

## 协议审计

|项目|状态|
|---|---|
|Stage2-C角色|包含`target_old`、`target_new`、`target_unknown`|
|support权限|只使用`target_old`和`target_new`support拟合ridge head|
|unknown权限|`target_unknown`只作为query评估；`unknown_query_eval_only=True`|
|阈值scope|评估器安全字段为`support_known_only`；报告细分字段为`support_known_ridge_only`|
|target receiver domain|由特征NPZ读取，目标接收机数为5|
|协同数量|`collab_counts=all`，覆盖1..5|
|事件对齐|`receiver_domain_ranked`，用于多未见接收机domain ensemble诊断；不是严格same-event同步协同|
|星地视图|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|

## 本地改动与版本状态

|文件|用途|SHA256|
|---|---|---|
|`E:\type10-7\code\scripts\phase2_support_ridge_adapter_eval.py`|新增support-only ridge adapter Stage2-C open-set诊断脚本|855B308DBC201DFB5B72170EBCEC651108611928ABA1B31A204AC09CE2849FB6|
|`E:\type10-7\code\tests\test_phase2_support_ridge_adapter_eval.py`|新增协议和阈值行为回归测试|EED42706FB434D4A7BCF9509BB5C59579507F0DDB8F7A90C0C474CDBF4DB6A55|

根目录`E:\type10-7`不是Git仓库。已创建本地快照：

`E:\type10-7\code\snapshots\phase2_support_ridge_adapter_20260704\`

Git镜像仓库：

|字段|值|
|---|---|
|路径|`E:\type10-7\github_publish\CVS-RFFI-repo`|
|分支|`codex/cvs-rffi-release-20260626`|
|提交|`309b7ad Add support ridge adapter Stage2-C diagnostic`|
|最终状态|ahead origin 361；无未提交改动|

## 本地验证

|命令|结果|
|---|---|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_support_ridge_adapter_eval.py`|PASS|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_support_ridge_adapter_eval.py -q -p no:cacheprovider`|3 passed|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_support_ridge_adapter_eval.py --help`|PASS|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_collaborative_open_set_qknn_eval.py code\tests\test_phase2_collaborative_open_set_qknn_eval.py code\tests\test_phase2_support_ridge_adapter_eval.py -q -p no:cacheprovider`|119 passed|

## 本地ADV3B02诊断命令

输入特征：

`E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz`

主命令参数：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_support_ridge_adapter_eval.py --feature_npz E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz --output_json E:\type10-7\local_artifacts\phase2_adv3b02_support_ridge_adapter_20260704\support_ridge_thr02.json --output_evidence_csv E:\type10-7\local_artifacts\phase2_adv3b02_support_ridge_adapter_20260704\support_ridge_thr02_evidence.csv --collab_counts all --collab_group_policy available_up_to_k --partial_collab_min_receivers 1 --event_alignment_policy receiver_domain_ranked --k_shot 8 --query_per_class 20 --ridge_lambda 0.1 --ridge_score_threshold 0.2 --ridge_temperature 0.05 --unknown_risk_threshold 0.65 --accept_margin_threshold 0.02 --fusion_policy risk_margin --label_fusion_policy weighted_vote_margin
```

## 本地结果

`ridge_score_threshold=0.2`：

|协同接收机数|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_reject_rate|defer_rate|bytes_per_event|verdict|
|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.6043|0.3000|0.5333|0.5000|0.0000|0.0000|96.0000|known有提升但unknown完全失败|
|2|0.6791|0.3500|0.6500|0.5000|0.0000|0.0000|174.1759|known有提升但unknown完全失败|
|3|0.7380|0.5278|0.6333|0.4750|0.0000|0.0000|236.7166|known有提升但unknown完全失败|
|4|0.8342|0.6500|0.6500|0.5250|0.0000|0.0000|283.6221|达到OLD80阶段诊断但unknown失败|
|5|0.8396|0.6500|0.6667|0.5250|0.0000|0.0000|312.7036|达到OLD80阶段诊断但unknown失败|

`ridge_score_threshold=0.3`和`0.9`在当前`ridge_temperature=0.05`下仍未触发unknown拒识，说明softmax score过饱和，单ridge head不是可部署open-set解。

本地输出：

|文件|SHA256|
|---|---|
|`support_ridge_thr02.json`|4B905CAD2D8D282E9D22D591273F5A3347F7D95BDE91B26A9B31353C3873112F|
|`support_ridge_thr03.json`|1E95990AAAD8632BCD30EF55ECB10F497A18E0EB7342357DCE854FD126BCF81E|
|`support_ridge_thr09.json`|55355785AEE86CD981FD383D527F1567ED17EE83398F97333BF7F60291469CA5|

## N607同步与远端测试

### Preflight

|项目|结果|
|---|---|
|命令|`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`|
|SSH|direct `N607` OK|
|远端host|`dell-DSS8440`|
|远端项目根|`/home/szu2070436088/2510044040/CV-SincNet`|
|GPU状态|8张RTX 3090均`utilization.gpu=0`、`memory.used=10MiB`|

### 同步映射

|本地|远端|SHA256|
|---|---|---|
|`E:\type10-7\code\scripts\phase2_support_ridge_adapter_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_ridge_adapter_eval.py`|855b308dbc201dfb5b72170ebcec651108611928aba1b31a204ac09ce2849fb6|
|`E:\type10-7\code\tests\test_phase2_support_ridge_adapter_eval.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_support_ridge_adapter_eval.py`|eed42706fb434d4a7bcf9509bb5c59579507f0ddb8f7a90c0c474cdbf4db6a55|

同步命令使用直接SCP：

```powershell
scp -F E:\type10-7\tools\n607_ssh_config code\scripts\phase2_support_ridge_adapter_eval.py N607:/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_support_ridge_adapter_eval.py
scp -F E:\type10-7\tools\n607_ssh_config code\tests\test_phase2_support_ridge_adapter_eval.py N607:/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_support_ridge_adapter_eval.py
```

### 远端验证

|命令|结果|
|---|---|
|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -V`|Python 3.10.19|
|`sha256sum code/scripts/phase2_support_ridge_adapter_eval.py code/tests/test_phase2_support_ridge_adapter_eval.py`|与本地一致|
|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_support_ridge_adapter_eval.py`|PASS|
|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m pytest ...`|环境缺少`pytest`，未安装新包|
|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_support_ridge_adapter_eval.py`|3 tests OK|

远端特征按SHA256匹配到：

`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz`

本地对应hash：

`db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`

### 远端运行命令

远端环境：

|字段|值|
|---|---|
|Conda/Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|GPU上下文|`CUDA_VISIBLE_DEVICES=0`；实际脚本为CPU侧闭式ridge和离线推理，显存占用极低|
|run目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_support_ridge_adapter_20260704`|
|log目录|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_support_ridge_adapter_20260704`|

主命令逻辑：

```bash
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_support_ridge_adapter_eval.py \
  --feature_npz runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/features.npz \
  --output_json runs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr02.json \
  --output_evidence_csv runs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr02_evidence.csv \
  --collab_counts all \
  --collab_group_policy available_up_to_k \
  --partial_collab_min_receivers 1 \
  --event_alignment_policy receiver_domain_ranked \
  --k_shot 8 \
  --query_per_class 20 \
  --ridge_lambda 0.1 \
  --ridge_score_threshold 0.2 \
  --ridge_temperature 0.05 \
  --unknown_risk_threshold 0.65 \
  --accept_margin_threshold 0.02 \
  --fusion_policy risk_margin \
  --label_fusion_policy weighted_vote_margin
```

同时以同参数运行`ridge_score_threshold=0.9`作为高阈值压力点。首次远端命令的标签生成使用了不存在的`python`别名，导致临时`thr00`产物被覆盖；随后已用显式`thr02`和`thr09`标签重跑，最终结果以下列产物为准。

### 远端结果

`ridge_score_threshold=0.2`：

|协同接收机数|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_reject_rate|unknown_FAR|defer_rate|bytes_per_event|latency_ms_p95|verdict|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.6043|0.3000|0.5333|0.5000|0.0000|1.0000|0.0000|96.0000|0.0634|不达标|
|2|0.6791|0.3500|0.6500|0.5000|0.0000|1.0000|0.0000|174.1759|0.0634|不达标|
|3|0.7380|0.5278|0.6333|0.4750|0.0000|1.0000|0.0000|236.7166|0.0634|不达标|
|4|0.8342|0.6500|0.6500|0.5250|0.0000|1.0000|0.0000|283.6221|0.0634|仅OLD80诊断达成；unknown失败|
|5|0.8396|0.6500|0.6667|0.5250|0.0000|1.0000|0.0000|312.7036|0.0634|仅OLD80诊断达成；unknown失败|

`ridge_score_threshold=0.9`：

|协同接收机数|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_reject_rate|unknown_FAR|defer_rate|bytes_per_event|latency_ms_p95|verdict|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.6043|0.3000|0.5333|0.5000|0.0000|1.0000|0.0000|96.0000|0.0612|不达标|
|2|0.6791|0.3500|0.6500|0.5000|0.0000|1.0000|0.0000|174.1759|0.0612|不达标|
|3|0.7380|0.5278|0.6333|0.4750|0.0000|1.0000|0.0000|236.7166|0.0612|不达标|
|4|0.8342|0.6500|0.6500|0.5250|0.0000|1.0000|0.0000|283.6221|0.0612|仅OLD80诊断达成；unknown失败|
|5|0.8396|0.6500|0.6667|0.5250|0.0000|1.0000|0.0000|312.7036|0.0612|仅OLD80诊断达成；unknown失败|

### 远端产物

|文件|SHA256|
|---|---|
|`runs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr02.json`|bd4799ca7648f1c7a2c71122394333dbe38d7754b7ef389a11e2fb86b1b802c1|
|`runs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr02_evidence.csv`|7e343926155f2c789d9be6a95e1dcc58af5e84bf28495bf2456167fd11c9d81c|
|`runs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr09.json`|64cec3300146bd0f13de94cc949e23bcf33920c0f3ce0d0c84a9466b59820fcd|
|`runs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr09_evidence.csv`|e4ac605404845f569ee94b3b8f33953e58583fef56cad5c51d3641d40616a0c2|
|`logs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr02.log`|b7c944f801ab1010261d5fb29b88ca54a2ab7f3ab2df8c76e1708d159741d109|
|`logs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr09.log`|d16e0663fae70b350d12c2a7546621f6c59fd4942a7e40d6efefd2bde6d7c554|

本地归档：

`E:\type10-7\remote_artifacts\phase2_adv3b02_support_ridge_adapter_20260704\`

## 最终解释

support-only ridge head能够显著恢复known分类，说明当前ADV3B02特征中存在目标接收机内old/seen-new可分信号；但该head没有可靠open-set边界，unknown全部被误接收，`unknown_FAR=1.0`。因此该路线只能作为“known恢复上限诊断”和后续adapter设计的负例，不满足协同推理目标。

## 风险与后续检查

|风险|处理|
|---|---|
|ridge head提升known但无法识别unknown|标为诊断路线，不写部署成功|
|`receiver_domain_ranked`不是严格same-event协同|报告明确为多未见接收机domain ensemble诊断|
|超参来自人工诊断扫描|不作为星上自动选择策略；后续需support-only CV或固定配置|
|未达到用户目标|继续保留目标未完成状态|
