# phase2_adv3b02_stage2c_center_veto_probe_20260707

## 基本信息

|字段|值|
|---|---|
|实验ID|phase2_adv3b02_stage2c_center_veto_probe_20260707|
|时间|2026-07-07|
|操作者|Codex|
|目标|验证`support_center`新类注册增强是否能在rescue后二级unknown veto下保留seen-new性能|
|协议边界|K=5/K=10目标域old+seen-new support；target receiver query为LEO星地信道；unknown query仅评估|

## 本地与远端状态

|项目|值|
|---|---|
|本地脚本|`code/scripts/launch_phase2_adv3b02_stage2c_center_veto_probe_20260707.sh`|
|本地验证|`bash -n`通过；本地dry-run展开8组诊断|
|远端脚本SHA256|`cdb35ef00e52145648b6e0854b07f8b62828b3c62202008f0d66f0ff166e6c58`|
|远端runs|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_stage2c_center_veto_probe_20260707`|
|远端logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase2_adv3b02_stage2c_center_veto_probe_20260707`|
|本地summary|`automation_reports/CV-SincNet/phase2_adv3b02_stage2c_center_veto_probe_20260707/remote_artifacts/stage2c_center_veto_probe_summary.json`|

## 结果表

|variant/profile/K|old_acc|min_old|seen_new_acc|min_seen|unknown_FAR|known_coverage|defer|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|`STAGE2C_HEAD_SEP/CENTER_VETO_E95_ANY_M1/K10`|0.6190|0.1286|0.0107|0.0000|0.2393|0.3327|0.0221|旧类尚可，新类塌缩，FAR过高|
|`STAGE2C_NORM_SEP/CENTER_VETO_E95_ANY_M1/K10`|0.6452|0.0571|0.0143|0.0000|0.2393|0.3439|0.0188|旧类尚可，新类塌缩，FAR过高|
|`STAGE2C_NORM_SEP/CENTER_VETO_E92L92S92_M2/K10`|0.6310|0.0000|0.0071|0.0000|0.2125|0.3235|0.0188|退化为旧类路线|
|`STAGE2C_HEAD_SEP/CENTER_VETO_E92L92S92_M2/K10`|0.5714|0.0143|0.0036|0.0000|0.2125|0.3020|0.0221|退化为旧类路线|
|K=5四行|0.0000|0.0000|0.0000|0.0000|0.0000-0.0036|0.0000-0.0010|0.099-0.100|全拒绝/近全拒绝|

## 结论

`support_center`能在无veto时带来seen-new命中，但与`rescue_unknown_veto`叠加后仍被unknown风险同形性压垮。当前证据表明，继续调后处理阈值难以同时满足old、seen-new和unknown；下一步应改新类注册打分本身，例如按seen-new support构造类内/类间相对距离、每类最低保障或unknown对照虚拟壳，而不是只在rescue后拦截。
