# Phase2可复用单观测LEO_weak数据资产

## 结论

本轮已将正式30-cell母缓存一次性物化并准入，后续方法不再重复追溯clean/source数据或重新生成LEO观测。每个物理IQ只进入一个`leo_*_weak`场景并只产生一份星地接收观测；Phase2只能读取由该固定接收IQ密封得到的support/query capsule。

## 覆盖

|维度|覆盖|
|---|---|
|receiver|`20-1,3-19,7-14,7-7,8-8`|
|seed|`713101–713106`（1开发＋5确认）|
|场景|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|TX|6个target-old＋20个nested target-new|
|每TX×场景|40条独立观测＝20条support pool＋20条query|
|总量|30 cell、93600条物理样本、93600条观测、120个数据文件、309MiB|

固定复用规则：new5/new10/new20取同一20类注册表前缀；K1/K5/K10/K20取support rank前缀；query固定为rank20–39。切片不增加信道、不增加view计数、不改变物理样本身份。

## 一次准入结果

- 30/30 cell coverage PASS；
- 每role×TX×receiver×scenario精确40条；
- 三场景物理ID交集为0；
- `physical_sample_count == physical_sample_observation_count == 93600`；
- 禁止的clean/raw/source IQ、features、logits、prototypes成员为0。

本地审计镜像位于`E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_data_asset_audits`：

|文件|SHA256|
|---|---|
|`phase2_data_asset_summary.json`|`a4b3073472cab5fb75abc797e71fae43d5df4f219579b68df9fc91d971fcd99b`|
|`phase2_data_asset_hashes.json`|`585acc4162fb0500c806129588118b1280bf96864fbab91f3ffc2303f773b83c`|
|`reuse_slices.json`|`7416af92992d440d039a505554252147ca0e049fb5553a5b1b10556f69f249a3`|

## 首个K10/new5消费结果

首个before/after密封capsule、Ed25519 path-free authorization与D18 support-only状态已生成。authority commit为`fdedd9cfdfbb5db9f8962ba529403042b7de7011570dff514e9a629a44695147`。D18保持0参数、0epoch和3168 MAC/sample，但3候选×3场景×5fold的最佳support均值仅old70.00%、seen-new62.00%，旧/新floor均出现0，最大遗忘25%，所以安全门回退`D18_Z0`并保持`query_opened=false`、`formal_metric_claim_allowed=false`。下一候选必须直接优化floor与旧类保护，不能绕过support gate获取query指标。

## 相关提交

- `86e8d0de`：可复用签名Phase2 capsule与path-free runtime；
- `ec659303`：single-observation cache role schema对齐；
- `d2dda6be`：可恢复的30-cell cache launcher；
- `60db8794`、`4065e58a`：全矩阵审计与精确signal-member guard；
- `df838d3f`：正式row-pair薄CLI。
