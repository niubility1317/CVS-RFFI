# D99/D100 target narrow r8 handoff

- Run ID：`d99_d100_narrow_rx20_1_seed713101_k10_new20_d6efa5ad_20260721_r8`
- 状态：`FAILED_PRE_PREDICTION_BINDING_DRIFT`
- Wrapper/Python PID：`1588355/1588357`，均已退出；exit=`1`
- GPU：物理GPU1，最终`0%/10MiB`
- 唯一错误：`D99D100QueryEvaluationError: class binding TX/handle bijection drift`
- 根因：当前row的opaque handles与历史D20 binding handles不同；Phase1 TX集合、顺序、checkpoint、class index和direct-logit index均匹配。
- Candidate prediction：`0`
- Candidate score/detailed：`0`
- `narrow_receipt.json`：不存在
- 结论：这是query prediction前的接口技术失败，不是D99/D100性能负结果。
- 完整远端日志SHA256：`92cbfef67fcffb6209da60511bcae1c66b30dfe32e766b82e5f5ea8cc54b203c`
- Offline build receipt SHA256：`60d9936355eff135ac513599b25ef122b959a11d43b9cb7b9e49ffd663f6fed3`
- Registration pair SHA256：`4048db7ce58d830c0d33d9642d229e84dd26058dcd1cb4497b27dce970146525`
- 完整53文件、29,194,421B原始回收包保存在：`E:\type10-7\automation_reports\CV-SincNet\ground_prototype_da_research_20260720\artifacts\d99_d100_narrow_rx20_1_seed713101_k10_new20_d6efa5ad_20260721_r8\`
- 远端run/log均保留；原run ID不得覆盖或重试。
- 本地SSH终态：`ssh.exe=0`，N607/bridge ESTABLISHED TCP22=`0/0`。
