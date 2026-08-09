# Phase3 SCB真实N=1桥接追溯

日期：2026-08-10

范围：仅实现`phase1_single_control_bundle_v1`单条IQ到CARE/CIRF的技术桥接，不改变SCB、CARE、CIRF核心，不调用N607，不调用scorer，不创建N>1事件。

## 实现合同

|条目|状态|证据|
|---|---|---|
|外部`content_root`与`TECHNICAL_LOCAL4_CONTROL_BUNDLE`严格加载|VERIFIED|`code/scripts/phase3_scb_real_n1_bridge.py`调用`scb.load_bundle(...,expected_bundle_status=scb.BUNDLE_STATUS)`；外部根漂移测试拒绝且不创建输出根|
|单个`iq.npy`输入|VERIFIED|仅接受finite实数数组`[T,2]`或`[2,T]`，并校验预注册原始文件SHA256；错shape、NaN和SHA漂移测试均拒绝|
|context精确九字段allowlist|VERIFIED|只允许`linkage_mode,proxy_group_id,satellite_reception_id,node_id,base_manifest_id,correlation_group_id,delay_ms,deadline_ms,sealed_at_ms`；`linkage_mode`固定`proxy_unverified`；extra、truth、role测试均拒绝|
|`cvs.phase3.local_evidence.v3`|VERIFIED|调用`scb.local_evidence_from_bundle`并检查schema；只写一条本地证据|
|CARE N=1身份恒等|VERIFIED（按时）|调用`scb.care_n1_parity`，逐项检查`p_local/decision/label/reason`恒等|
|超时处理|VERIFIED（fail-closed）|`delay_ms>deadline_ms`由SCB先标记`SCB_CONTEXT_DEFER`；CARE现有合同无on-time行时桥接拒绝写出，未自创阈值或理由|
|CIRF逐字节透传|VERIFIED|canonical LocalEvidence字节经`cirf.n1_passthrough_bytes`后必须逐字节相同，并记录输入/透传SHA256|
|技术声明边界|VERIFIED|manifest固定`technical_only=true`、`performance_result=false`、`truth_sidecar_opened=false`、`same_event_claim=false`、`collaborative_gain_claim=false`、`n_sat=1`|
|输出原子性与不覆盖|VERIFIED|输出根预先存在即拒绝；四个JSON先写入进程专属staging根，再原子替换为新根；失败不覆盖已有根|
|真实N>1协同|BLOCKED_DATA|本薄层刻意只承载N=1；真实多接收节点、same-event绑定、相关性控制与协同收益证据尚未提供，不能声明N>1|
|性能结果/unknown FAR/注册授权|DEFERRED|本桥接不打开truth sidecar、不运行scorer、不生成性能指标或注册授权|

## 输出工件

成功时且仅写入以下四个文件：

1. `local_evidence.json`
2. `care_n1_identity_receipt.json`
3. `cirf_n1_passthrough_receipt.json`
4. `bridge_manifest.json`

测试fixture使用现有SCB构造规则生成带`runtime_capacity_token=1`的技术控制bundle，避免真实checkpoint/ManySig访问；这只证明桥接机械闭环，不构成Phase3性能证据。

## 验证记录

在`ssr-gpu`环境串行执行：

```text
python -m py_compile code/scripts/phase3_scb_real_n1_bridge.py code/tests/test_phase3_scb_real_n1_bridge.py
pytest -q code/tests/test_phase3_scb_real_n1_bridge.py -vv
git diff --check
```

聚焦测试结果：`10 passed`。覆盖成功fixture、外部根漂移、IQ SHA漂移、nonfinite、错shape、context extra/truth/role、输出根存在、CARE恒等、CIRF字节/SHA恒等及超时fail-closed。`py_compile`通过，`git diff --check`通过。

本次新增文件SHA256（工作树快照）：

|文件|SHA256|
|---|---|
|`code/scripts/phase3_scb_real_n1_bridge.py`|`4b91b8ba0ea899dd652e7247eb27f1039109c3915aaa7b2d73b57b1f8ac1cae6`|
|`code/tests/test_phase3_scb_real_n1_bridge.py`|`c9416a8226258ec261adeb70b5213797bc12067fc3e216196a523dd014b39eda`|

追溯文件自身SHA在交接检查中单独计算，避免自引用。

## 交接边界

本次仅新增桥接CLI、聚焦测试与本追溯文件；不提交Git、不同步N607、不修改其他代理所有权文件。若要推进真实N>1，必须先提供独立数据/事件绑定与冻结协同合同，并重新完成对应证据审查。
