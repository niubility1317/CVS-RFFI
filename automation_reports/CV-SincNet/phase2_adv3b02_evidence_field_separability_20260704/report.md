# phase2_adv3b02_evidence_field_separability_20260704

## 基本信息

|字段|值|
|---|---|
|实验ID|phase2_adv3b02_evidence_field_separability_20260704|
|时间|2026-07-04|
|operator/agent|Codex|
|目标|诊断ADV3B02+qknn8当前support-calibrated风险字段是否足以同时保known和拒unknown|
|底座|ADV3B02_CORE90_SOFT_E200|
|结论边界|DIAGNOSTIC_ONLY；扫描使用query标签估计上限，不能作为部署阈值策略|

## 动机

前两轮负例显示：

|路线|主要结果|问题|
|---|---|---|
|support-only ridge head|old可到0.8396，但unknown_FAR=1.0|known恢复但没有open-set边界|
|dual evidence safety rescue|k=4 unknown_FAR=0.025，但old=0.4583、seen-new=0.4750|安全门先拒掉大量known，后续known替换无法恢复|

因此本轮不继续盲调融合参数，而是直接扫描已有evidence字段：若在FAR≤0.01/0.05下known上限仍很低，则下一步必须改证据生成或特征训练边界，而不是继续在融合层补丁。

## 本地改动

|文件|用途|
|---|---|
|`E:\type10-7\code\scripts\phase2_evidence_field_separability_diag.py`|新增风险字段分离性诊断工具，扫描单字段和组合字段门控|
|`E:\type10-7\code\tests\test_phase2_evidence_field_separability_diag.py`|新增单元测试，验证FAR约束下的门控扫描|

## 本地验证

|命令|结果|
|---|---|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_evidence_field_separability_diag.py`|PASS|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_evidence_field_separability_diag.py -q -p no:cacheprovider`|1 passed|

## 本地诊断命令

输入evidence：

`E:\type10-7\remote_artifacts\phase2_adv3b02_collab_open_set_qknn_full_20260703\collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv`

命令：

```powershell
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe code\scripts\phase2_evidence_field_separability_diag.py --evidence_csv E:\type10-7\remote_artifacts\phase2_adv3b02_collab_open_set_qknn_full_20260703\collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv --output_json E:\type10-7\local_artifacts\phase2_adv3b02_evidence_field_separability_20260704\field_separability_event098.json --max_combo_size 2 --modes max,mean --far_targets 0.01,0.05,0.10 --max_thresholds 128
```

## 本地结果

|FAR约束|最佳字段组合|mode|threshold|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_cov|defer_rate|判定|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|≤0.01|`margin_risk,oldness_risk`|max|2.2106e-05|0.1083|0.0000|0.2150|0.2100|0.9900|0.0100|0.1550|0.6760|known几乎不可用|
|≤0.05|`margin_risk,evt_risk`|mean|3.1990e-09|0.3250|0.0000|0.2850|0.1500|0.9500|0.0500|0.3412|0.5270|unknown可控但known崩溃|
|≤0.10|`margin_risk,evt_risk`|max|3.8218e-06|0.3717|0.1200|0.3250|0.1900|0.9000|0.1000|0.3937|0.4850|仍远低目标|

结论：当前support-calibrated风险字段在低FAR约束下没有足够known保留能力。要达到old99/seen-new97/unknown99，下一步应转向证据生成阶段或轻量特征边界训练，例如support-only class-conditional negative boundary、目标域adapter的open-set margin loss、或用source/proxy non-old构造部署前open-set校准，而不是继续在现有风险字段上调融合门限。

## N607同步与远端测试

### Preflight

`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`通过。远端项目根可见；8张RTX 3090均`utilization.gpu=0`、`memory.used=10MiB`。

### 同步与验证

|项目|结果|
|---|---|
|远端Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19|
|远端脚本hash|d70ac2c9bbe1b34b4947555c2e96095161bdc2b19e312378217fc0beebcc653b|
|远端测试hash|07aa6e4f2e54d4e8c509565b28ff922e648b8f099af9d1af71a33233d5d66293|
|远端语法检查|PASS|
|远端unittest|1 test OK|

### 远端命令

```bash
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_evidence_field_separability_diag.py \
  --evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv \
  --output_json runs/phase2_adv3b02_evidence_field_separability_20260704/field_separability_event098.json \
  --max_combo_size 2 \
  --modes max,mean \
  --far_targets 0.01,0.05,0.10 \
  --max_thresholds 128
```

### 远端结果

|FAR约束|最佳字段组合|mode|threshold|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|known_cov|defer_rate|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|≤0.01|`margin_risk,oldness_risk`|max|2.2106e-05|0.1083|0.0000|0.2150|0.2100|0.9900|0.0100|0.1550|0.6760|
|≤0.05|`margin_risk,evt_risk`|mean|3.1990e-09|0.3250|0.0000|0.2850|0.1500|0.9500|0.0500|0.3412|0.5270|
|≤0.10|`margin_risk,evt_risk`|max|3.8218e-06|0.3717|0.1200|0.3250|0.1900|0.9000|0.1000|0.3937|0.4850|

远端产物：

|文件|SHA256|
|---|---|
|`runs/phase2_adv3b02_evidence_field_separability_20260704/field_separability_event098.json`|508d65023efde66f029a9a8d65e538cacf6d322042ba5afe3364fe7817cbbbc8|
|`logs/phase2_adv3b02_evidence_field_separability_20260704/field_separability_event098.log`|e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855|

本地归档：

`E:\type10-7\remote_artifacts\phase2_adv3b02_evidence_field_separability_20260704\`

## 最终解释

本轮给出比单个融合负例更强的证据：即使允许离线oracle扫描已有风险字段和二字段组合，在`unknown_FAR<=0.05`约束下，known full accuracy上限也只到`old_acc=0.3250`、`seen_new_acc=0.2850`。因此当前ADV3B02+qknn8 evidence字段不是“阈值没调好”，而是现有支持集风险字段没有形成可同时保known和拒unknown的边界。下一步应设计轻量训练/适配型open-set边界，例如：

1. support-only class-conditional negative prototype/shell训练；
2. source/proxy non-old预校准的open-set margin adapter；
3. 在ADV3B02特征上加入目标域旧类+seen-new support的compactness，同时用源/proxy unknown保持外环拒识。

该结论仍不代表目标完成；它用于收窄下一步算法方向。
