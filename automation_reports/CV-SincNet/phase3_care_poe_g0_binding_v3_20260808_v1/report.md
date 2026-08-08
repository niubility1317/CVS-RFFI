# Phase3 CARE-PoE G0物理绑定v3技术实验报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase3_care_poe_g0_binding_v3_20260808_v1`
- 状态：`LOCAL_VERIFIED`
- 证据等级：`TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT`
- 时间：2026-08-08
- 操作者：Codex主代理；N607唯一runner待交接
- Git commit：`PENDING_LOCAL_COMMIT`

## 1.目标、假设与对照

本实验验证修订后的`LocalEvidenceV3`完整技术链：先生成不含truth/role的逐reception proxy证据和独立物理binding sidecar，再由绑定器重新封存`verified_physical`证据；predictor必须核对调用方冻结的binding root；scorer必须核对prediction manifest、文件hash、行数和A/B/C/D×`N_sat=1..5`完整覆盖。

假设：伪造行内`physical_binding_receipt_id/hash`不足以通过预测入口；旧`cvs.phase3.local_evidence.v2`不能直接升级；替换预测文件不能进入评分；同一event无论含多少reception仍只计1 shot。上一版`phase3_care_poe_g0_synthetic_20260808_v1`是技术对照，本轮不读取性能数值。

N607库存审计未发现真实`emission_event_id/satellite_reception_id`或采集binding receipt。因此本实验只验证合成技术路径，不产生真实多星、same-emission、unknown FAR、安全拒绝率或旧类准确率主张。

## 2.本地变更与验证

| 文件 | 目的 |
|---|---|
| `code/cvsrffi/phase3_care_poe.py` | v3精确schema、物理binding seal/validate/root、矩阵与评分完整性 |
| `code/scripts/phase3_bind_physical_evidence.py` | proxy证据与采集binding一对一连接入口 |
| `code/scripts/phase3_care_poe_fixture.py` | 合成proxy→binding→verified fixture |
| `code/scripts/phase3_care_poe_predict.py` | 外部binding sidecar/root强校验 |
| `code/scripts/phase3_care_poe_score.py` | prediction manifest/hash/完整矩阵强校验 |
| `code/tests/test_phase3_care_poe.py` | bypass、tamper、旧v2、非覆盖与CLI闭环负测 |

本地环境：`ssr-gpu`。验证结果：Phase3与真实bundle相关测试`39 passed`；`py_compile`和`git diff --check`通过。独立复审先发现scorer可信任manifest自报的截断预算轴；强制预算精确等于`[1,2,3,4,5]`并加入自洽截断负测后，结论为`P0=0,P1=0,ALLOW_N607_SYNTHETIC_G0_BINDING_V3=YES`。

## 3.冻结矩阵与命令

冻结roster为`SAT-01..SAT-05`，arms为A/B/C/D，节点预算为1、2、3、4、5，合成event为3个，预期prediction共60行。所有入口只执行一次；retry=`NO`。

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase3_care_poe_g0_binding_v3_20260808_v1_<commit8>`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase3_care_poe_g0_binding_v3_20260808_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase3_care_poe_g0_binding_v3_20260808_v1`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：CPU技术闭环，不占用GPU

冻结执行顺序：

```text
python scripts/phase3_care_poe_fixture.py --output-dir <run>/fixture
BINDING_ROOT=$(python -c "import json; print(json.load(open('<run>/fixture/fixture_manifest.json'))['binding_root'])")
python scripts/phase3_care_poe_predict.py --base-evidence <run>/fixture/base_evidence.jsonl --new-evidence <run>/fixture/new_evidence.jsonl --physical-bindings <run>/fixture/physical_bindings.jsonl --expected-physical-binding-root "$BINDING_ROOT" --output-dir <run>/prediction
python scripts/phase3_care_poe_score.py --predictions <run>/prediction/predictions.jsonl --prediction-manifest <run>/prediction/prediction_manifest.json --truth-sidecar <run>/fixture/truth_sidecar.jsonl --output <run>/metrics.json
python scripts/phase3_care_poe_lifecycle.py --predictions <run>/prediction/predictions.jsonl --credential-template <run>/fixture/credential_template.json --fresh-support <run>/fixture/fresh_support.jsonl --output <run>/lifecycle.json --k 5
```

## 4.成功与停止条件

技术成功要求：四入口exit=0；fixture binding root与prediction manifest一致；prediction=60行且完整覆盖A/B/C/D×N1-N5×3events；全部`shot_count=1`；N1满足A=C、B=D；manifest为`truth_sidecar_opened=false`；lifecycle到达`FRESH_K_READY_FOR_STAGE2_C`。

仅在commit/hash不符、目标路径已存在、协议字段泄漏、prediction数量或覆盖不符、确定性异常、写入覆盖风险时停止。合成metrics不得用于停止、调参或晋级。预期只回收JSON/JSONL、stdout、completion和manifest，不下载数据、checkpoint或大型artifact。

## 5.运行结果

待N607唯一runner回填。正式Phase3性能矩阵仍等待真实、预标签物理binding资产。
