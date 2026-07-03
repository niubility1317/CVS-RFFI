# phase2_adv3b02_dual_evidence_safety_rescue_20260704

## 基本信息

|字段|值|
|---|---|
|实验ID|phase2_adv3b02_dual_evidence_safety_rescue_20260704|
|时间|2026-07-04|
|operator/agent|Codex|
|目标|验证qknn8安全门与第二路known标签证据是否可解耦，以改善卫星群协同open-set RFFI|
|底座|ADV3B02_CORE90_SOFT_E200|
|结论边界|NON_DEPLOYMENT_DIAGNOSTIC；未达到old99/seen-new97/unknown99目标|

## 算法候选

新增`dual_evidence_safety_rescue`事件级候选：

1. qknn8安全路线先对每个事件输出`accept`、`unknown_reject`、`request_more`或`defer`。
2. 只有当安全路线已经`accept`时，第二路known证据才允许替换输出标签。
3. 若安全路线`unknown_reject/defer/request_more`，第二路known证据不能救回。
4. 该规则对old、seen-new、unknown使用同一决策流程，不按真实role作弊；true label只用于最终指标统计。

该设计用于验证“open-set安全边界”和“known标签质量”是否能工程上解耦。

## 本地改动与版本

|文件|用途|SHA256|
|---|---|---|
|`E:\type10-7\code\scripts\phase2_dual_evidence_safety_rescue_eval.py`|新增双证据安全门诊断脚本|c7fa55b69ca262e77e5cada3ad9db303c3ef3c8924166953568dcf033a3b1386|
|`E:\type10-7\code\tests\test_phase2_dual_evidence_safety_rescue_eval.py`|新增事件级融合单测|9a7ac9966099c74aa5b380e2f54c8ca7bbe1a186d101d5f01b277c9cdf458813|

根目录`E:\type10-7`不是Git仓库。代码快照：

`E:\type10-7\code\snapshots\phase2_dual_evidence_safety_rescue_20260704\`

Git镜像：

|字段|值|
|---|---|
|路径|`E:\type10-7\github_publish\CVS-RFFI-repo`|
|分支|`codex/cvs-rffi-release-20260626`|
|提交|`19df4a7 Add dual evidence safety rescue diagnostic`|

## 本地验证

|命令|结果|
|---|---|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile code\scripts\phase2_dual_evidence_safety_rescue_eval.py`|PASS|
|`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest code\tests\test_phase2_dual_evidence_safety_rescue_eval.py code\tests\test_phase2_support_ridge_adapter_eval.py -q -p no:cacheprovider`|4 passed|

本地真实ADV3B02诊断覆盖协同数量1..5，结果与远端一致。

## N607验证与运行

### Preflight

`powershell -ExecutionPolicy Bypass -File tools\n607_ssh_preflight.ps1`通过。远端项目根可见；8张RTX 3090均`utilization.gpu=0`、`memory.used=10MiB`。

### 同步与远端验证

|项目|结果|
|---|---|
|远端Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Python 3.10.19|
|远端脚本hash|c7fa55b69ca262e77e5cada3ad9db303c3ef3c8924166953568dcf033a3b1386|
|远端测试hash|9a7ac9966099c74aa5b380e2f54c8ca7bbe1a186d101d5f01b277c9cdf458813|
|远端语法检查|PASS|
|远端unittest|1 test OK|

### 远端命令

```bash
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/phase2_dual_evidence_safety_rescue_eval.py \
  --safety_evidence_csv runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02_evidence.csv \
  --known_evidence_csv runs/phase2_adv3b02_support_ridge_adapter_20260704/support_ridge_thr02_evidence.csv \
  --protocol_metadata_json runs/phase2_adv3b02_collab_open_set_qknn_full_20260703/collab_open_set_qknn_candidate_set_cvs_support_calibrated_event098_adv3b02.json \
  --output_json runs/phase2_adv3b02_dual_evidence_safety_rescue_20260704/dual_evidence_safety_rescue.json \
  --collab_counts all \
  --partial_collab_min_receivers 3
```

## 结果

|协同接收机数|old_acc|min_old|seen_new_acc|min_seen|unknown_reject|unknown_FAR|defer_rate|known_cov|bytes/event|p95 ms|判定|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|1|0.0000|0.0000|0.0000|0.0000|0.5333|0.0000|0.6457|0.0000|40.0|0.1350|不可用|
|2|0.2516|0.0500|0.3878|0.2414|0.8333|0.0625|0.1111|0.4020|80.0|0.1350|FAR超限且known不足|
|3|0.2417|0.0500|0.3250|0.3000|1.0000|0.0000|0.0350|0.3625|120.0|0.1350|拒识达标但known崩溃|
|4|0.4583|0.1500|0.4750|0.4500|0.9750|0.0250|0.0100|0.5500|149.6|0.1350|最佳折中但远低目标|
|5|0.3833|0.1500|0.4250|0.4000|1.0000|0.0000|0.0050|0.4688|169.2|0.1350|拒识达标但known崩溃|

远端产物：

|文件|SHA256|
|---|---|
|`runs/phase2_adv3b02_dual_evidence_safety_rescue_20260704/dual_evidence_safety_rescue.json`|146e8937d81db54f4b65ee68152e791e7996a404613e57fd5261fda743a91494|
|`logs/phase2_adv3b02_dual_evidence_safety_rescue_20260704/dual_evidence_safety_rescue.log`|e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855|

本地归档：

`E:\type10-7\remote_artifacts\phase2_adv3b02_dual_evidence_safety_rescue_20260704\`

## 解释

该候选证明“先安全门、后known标签替换”的简单解耦仍不够。qknn8安全门为了压低unknown FAR，会大量把old/seen-new拒成unknown或defer；known路线只能替换已被安全门接受的事件标签，无法恢复被安全门拦截的known样本。继续优化应转向证据生成阶段：提高qknn8 known正证据质量，或增加support-only class-conditional negative boundary，而不是在融合后救回。

用户目标未完成，目标状态继续保持active。
