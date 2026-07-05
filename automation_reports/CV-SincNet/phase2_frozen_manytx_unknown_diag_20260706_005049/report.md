# phase2_frozen_manytx_unknown_diag_20260706_005049

## 基本信息

|字段|内容|
|---|---|
|实验ID|phase2_frozen_manytx_unknown_diag_20260706_005049|
|时间|2026-07-06 00:50 CST|
|执行者|Codex|
|目标|在ADV3B02_CORE90_SOFT_E200+qknn8基础上，增加冻结特征ManyTx未知类拒识只读诊断，评估协同接收机数量1..N下old/seen-new/unknown同row指标|
|性质|NON_DEPLOYMENT_DIAGNOSTIC，read-only evaluation|

## 假设与对照

假设：当前R3源域OOD/proxy趋势为负时，需要先用真实ManyTx冻结特征诊断确认proxy失败是否对应真实target unknown失败。该诊断不训练、不改模型、不用真实`Y_unknown`调阈值，只输出同一候选/同一协同数量下的old、seen-new、unknown指标和资源代理字段。

对照：既有`phase2_collaborative_open_set_qknn_eval.py`可直接评估协同qknn8，但缺少面向当前目标的安全封装字段、目标阈值判定表和“诊断-only”结果约束。

## 本地改动

|文件|目的|
|---|---|
|`E:\type10-7\code\scripts\phase2_frozen_manytx_unknown_diagnostic.py`|新增薄封装诊断脚本，复用现有qknn8证据构建和协同评估，固定输出`protocol_safety`、1..N协同摘要、目标达标字段|
|`E:\type10-7\code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py`|新增合成NPZ单测，验证未知query eval-only、协同数量1..N、非正参数拒绝、CSV输出和旧Stage2-C特征包manifest兼容重标|

## 协议边界

|边界|记录|
|---|---|
|地面训练接触真实未知类|禁止；本脚本不训练|
|未知query阈值拟合|禁止；`uses_unknown_query_for_threshold=false`|
|target_unknown训练/校准计数|固定为0|
|ManyTx真实unknown|仅作为query评估|
|成功声明|`stage2_success_claim=false`，`deployment_success_claim=false`|
|协同数量|默认`collab_counts=all`，输出1..target receiver count|
|资源字段|只报告`bytes_per_event`、`latency_ms_p95`等代理字段；未声明真实卫星资源达标|
|旧NPZ兼容|默认关闭；显式`--repair_legacy_roles_from_manifest`时，仅把manifest中`unknown_tx_ids`对应的旧`target_new`行重标为`target_unknown`，并写入`protocol_safety`|

## 本地验证

|命令|结果|
|---|---|
|`conda run -n ssr-gpu python -m py_compile code\scripts\phase2_frozen_manytx_unknown_diagnostic.py code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py`|PASS|
|`conda run -n ssr-gpu python code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py`|PASS，3 tests OK；负例会打印argparse usage|
|`conda run -n ssr-gpu python code\tests\test_phase2_collaborative_open_set_qknn_eval.py`|PASS，53 tests OK|

注：Windows上并行`conda run`曾触发临时文件锁，已改为串行验证；这不是实验结论。

## 同步计划

|本地路径|远端路径|
|---|---|
|`E:\type10-7\code\scripts\phase2_frozen_manytx_unknown_diagnostic.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py`|
|`E:\type10-7\code\tests\test_phase2_frozen_manytx_unknown_diagnostic.py`|`/home/szu2070436088/2510044040/CV-SincNet/code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py`|
|`E:\type10-7\automation_reports\CV-SincNet\phase2_frozen_manytx_unknown_diag_20260706_005049\report.md`|`/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/phase2_frozen_manytx_unknown_diag_20260706_005049/report.md`|

## 远端命令计划

远端只做有界验证和只读诊断，不启动训练：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile code/scripts/phase2_frozen_manytx_unknown_diagnostic.py code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py
```

若远端存在当前Stage2-C冻结特征NPZ，则执行小样本只读诊断并输出到本报告目录；若只存在旧Stage2-C特征包且manifest明确包含`new_tx_ids/unknown_tx_ids`，可用`--repair_legacy_roles_from_manifest`执行兼容诊断，并把结果标为旧包diagnostic-only；若不存在，则只记录远端测试通过和缺少feature NPZ，不伪造结果。

## 风险

|风险|处理|
|---|---|
|默认`receiver_domain_ranked`不是严格同事件协同|报告为诊断-only；若要部署证据需用`strict_event_key`重跑|
|真实ManyTx特征包路径可能不存在|远端先有界查找既有NPZ，不做大范围扫描|
|当前R2/R3训练仍活跃|本次只读测试不抢占GPU；如需真实特征导出另开报告|
|目标99/97/99尚未满足|本脚本只提供证据表，不把缺口包装成完成|

## 当前状态

## 远端验证与诊断结果

|项目|结果|
|---|---|
|N607预检|PASS，直接`N607`可用，项目根可见，GPU可见|
|远端Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19|
|活跃训练|R2仍在GPU0/1，R3仍在GPU2/3；本次未启动训练|
|远端hash|脚本`c6e6d4f7...9865a2`，测试`f1e61f82...0e9512`，报告`7d08b646...2a803`，清单`9f38af3b...45d7d`|
|远端验证|`py_compile` PASS；`python code/tests/test_phase2_frozen_manytx_unknown_diagnostic.py` PASS，3 tests OK|
|SSH清理|SCP/SSH后本地`ssh.exe`无残留，N607和桥接机22端口无ESTABLISHED连接|

### 有界特征包检查

已检查现有特征NPZ：

|候选|结论|
|---|---|
|`runs/phase2_qknn_adaptive_manynew20_20260705/.../features_leo_repaired.npz`|ADV3B02/LEO特征存在，但只有`target_old/target_unknown/proxy_unknown`，缺少`target_new`，不能作为完整Stage2-C诊断|
|`runs/phase2_adv3b02_manynew10_pairprotected_20260704/.../features_leo_repaired.npz`|ADV3B02/LEO特征存在，但同样缺少`target_new`|
|`runs/stage2_phase2_runner_20260622_120919/OA_MSE_STAGE2C_HEAD_SEEN_NEW/features.npz`|旧Stage2-C特征，含`target_old/target_new`且manifest含`unknown_tx_ids`，使用显式兼容开关执行diagnostic-only评估|

### 兼容诊断输出

远端命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_frozen_manytx_unknown_diagnostic.py \
  --feature_npz runs/stage2_phase2_runner_20260622_120919/OA_MSE_STAGE2C_HEAD_SEEN_NEW/features.npz \
  --output_json automation_reports/CV-SincNet/phase2_frozen_manytx_unknown_diag_20260706_005049/legacy_stage2c_diag.json \
  --output_summary_csv automation_reports/CV-SincNet/phase2_frozen_manytx_unknown_diag_20260706_005049/legacy_stage2c_summary.csv \
  --repair_legacy_roles_from_manifest \
  --k_shot 1 --query_per_class 5 --qknn_k 1 \
  --receiver_class_reliability_policy none --class_reliability_policy none --class_verifier_policy none \
  --candidate_set_min_receivers 1 --max_event_bytes 1024 --max_event_latency_ms 25
```

输出已拉回本地：

|文件|位置|
|---|---|
|JSON|`E:\type10-7\automation_reports\CV-SincNet\phase2_frozen_manytx_unknown_diag_20260706_005049\legacy_stage2c_diag.json`|
|CSV|`E:\type10-7\automation_reports\CV-SincNet\phase2_frozen_manytx_unknown_diag_20260706_005049\legacy_stage2c_summary.csv`|

同row结果：

|特征包|receiver_count|collab_count|old_acc|min_old_class_acc|seen_new_acc|min_seen_new_class_acc|unknown_reject_rate|unknown_FAR|known_coverage|latency_ms_p95|结论|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|旧Stage2-C兼容包|1|1|0.0000|0.0000|0.0000|0.0000|1.0000|0.0000|0.0000|1.1051|负证据：全拒识保住unknown，但旧类/新类完全不可用|

协议安全字段：

|字段|值|
|---|---|
|`diagnostic_only`|true|
|`legacy_role_repair_applied`|true|
|`uses_unknown_query_for_threshold`|false|
|`target_unknown_training_count`|0|
|`target_unknown_calibration_count`|0|
|`target_unknown_query_count`|321|
|`threshold_scope`|`support_known_only`|
|`goal_satisfied_counts`|空|

## 当前状态

本地实现、N607同步、远端hash/py_compile/测试和一个旧Stage2-C兼容只读诊断已完成。该诊断不是目标完成证据；它证明当前可用旧特征包在严格未知拒识下牺牲了全部known coverage。下一步应导出或定位真正包含`target_old/target_new/target_unknown`的ADV3B02_CORE90_SOFT_E200+LEO多接收机冻结特征包，再用本脚本跑1..N接收机完整诊断；若仍出现全拒识，则转入地面再训练的虚拟未知/EVT/反向点/负原型路线。
