# D105 Phase1 R6固定批容量合同根因分析

## 结论

R6的`strict D105 tap/reference dual archive parity failed`由执行batch合同未绑定直接导致。参考dual archive以固定容量256执行：不足256行的末批先零填充到256行，完成同一次双分支前向后再切回真实行。R6以`--batch-size 128`直接前向真实切片，最后80行未补零；模型对batch形状敏感，因此这80行产生结构性`z_dom`差异。没有证据指向checkpoint、输入顺序或D105-FTU1特征语义漂移。

R6保持`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重试。修复必须使用新提交、新归档和新run ID。

## 全量8400×160差分

|分支|全局最大绝对差|元素`>1e-5`|受影响行|前8320行最大差|余弦最小值|
|---|---:|---:|---:|---:|---:|
|`z_id`|`1.9073486328125e-05`|7|1|`<1e-5`|`0.9999999999973186`|
|`z_dom`|`0.0019412636756896973`|9190|80|`1.9073486328125e-06`|`0.9999999316694641`|

`z_dom`受影响行恰为索引8320—8399，即R6最后一个80行partial batch；这80行全部超过`1e-5`，其中19行超过`1e-3`。最大差位于行8325。参考archive的runtime audit记录`batch_size=256`、`runtime_invocations=33`；8400行对应32个完整256行批次和一个208行真实数据＋48行零填充批次。R6则对应65个完整128行批次和一个80行真实批次。

## 假设判定

|假设|判定|证据|
|---|---|---|
|CPU/GPU后端漂移|排除为直接原因|参考和R6正式归档均为Torch2.1.0+cu121、GPU0、float32|
|batch形状合同差异|确认|误差严格集中在R6最后80行；旧wrapper固定补零到256，R6直接前向80行|
|D105-FTU1语义改变|不支持|`dual_feature_forward.py`与`model.py`相关blob未改变，两侧均为`feat_joint`和`feat_imp→dom_enhancer`|
|checkpoint/输入顺序漂移|强烈排除|checkpoint、cache、salt均SHA绑定；8400行labels、receiver_ids、physical_ids逐值一致|

## D105-FTU2可行性摘要

1. 只修Phase1执行合同，不改变方法、模型、特征或性能门。
2. 固定`forward_batch_capacity=256`。
3. 所有正式批次实际前向shape均为`[256,2,T]`。
4. 不足256行的末批用全零IQ补齐。
5. 前向后只切回真实行，不持久化填充行。
6. receipt记录capacity、调用数、末批真实行和填充行。
7. `tap-cache`拒绝任何非256调用自由度。
8. 257行应形成两次256前向，末批1＋255。
9. 8400行应形成33次256前向，末批208＋48。
10. 批形状敏感fake model必须证明删除padding会失败。
11. checkpoint/cache/salt/metadata和参考archive SHA门不变。
12. `z_id==ReLU(pre_relu)`同次前向字节绑定不变。
13. `z_dom`仍来自同次`dom_backbone.feat_imp→dom_enhancer`。
14. 全8400行`max_abs<=1e-5`门不放宽。
15. 本地实现、回归、独立审查、精确archive smoke全部通过后才可创建新run。

## 文件边界

- 修改`code/cvsrffi/stage2_d105_phase1_bundle.py`：固定容量分块、零填充、切回与receipt。
- 修改`code/scripts/build_d105_phase1_bundle.py`：`tap-cache`固定或强制256。
- 修改`tests/test_stage2_d105_phase1_bundle.py`：257/8400边界、敏感fake model、receipt与非256负测。
- 不修改`stage2_d105_feature_tap.py`、`dual_feature_forward.py`、checkpoint、参考archive或parity阈值。
