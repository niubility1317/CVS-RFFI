# Phase1双读出开放世界deployment bundle v2冻结设计

版本：2026-08-08

目标模式：`ACTIVE`

设计状态：`DESIGN_FROZEN`

候选：`P1-DUALREADOUT-BUNDLE-V2`

证据状态：`TECHNICAL_BUNDLE_NOT_PERFORMANCE_PROMOTED`

## 1.经验继承与唯一职责

GeoSat Lite四臂证明：C的clean→LEO一致性提供最强LEO鲁棒类别路径；B的known-only角度几何提供相对更好的source held-TX连续陌生度信号；把二者放在同一训练路径或作为硬一致性门都会损害known覆盖。因此v2不再训练新模型、不做特征对齐、不做hard disagreement gate：C负责registered类别、`z_id`和`z_dom`，B只提供连续JS分歧。

Phase1只输出不可变本地deployment bundle和当前received-IQ的本地证据。它不得读取target query真值、真实role、多节点消息、anonymous track、credential或授权状态，也不得把source proxy写成真实unknown。

## 2.bundle成员

```text
manifest.json
runtimes/angular.ts
runtimes/robust.ts
calibration/calibration.npz
calibration/receipt.json
```

两个TorchScript runtime都只接收`[B,2,L]` received IQ并返回`z_id,z_dom,tx_logits`。bundle禁止包含`.pth`、raw IQ、样本/member ID、source cache、role、truth、proxy rows或可逆样本索引。manifest记录每个成员SHA256、大小、checkpoint lineage、class handle顺序、公式版本和content root；加载时验证exact allowlist、hash、shape和finite。

## 3.source-only聚合几何

C的source-known `z_id`按registered class形成单位球面中心与Q0.95角半径；C的`z_dom`按receiver×day形成域中心与Q0.95角半径。每个类/域至少2个独立physical sample。B/C logits只在两者均正确的source-known样本上校准JS尺度，任何proxy/held/target行都不得进入fit。

对单条received IQ：

\[
d_c=\theta(z_{id},\mu_c)/R_c,\quad a_c=\exp(-d_c^2/2)
\]

\[
a_{dom}=\max_g\exp[-(\theta(z_{dom},\nu_g)/S_g)^2/2]
\]

\[
j=\operatorname{clip}(JS(p_B,p_C)/J_{src},0,1)
\]

\[
e_{unknown}=1-(1-[1-\max_ca_c])(1-j)
\]

\[
q=\sqrt{a_{dom}(1-H(p_C)/\log C)}
\]

`p_local`由`(1-e_unknown)p_C`与unknown质量`e_unknown`拼接后归一化。registered label始终来自C，不允许B覆盖类别结果。

## 4.三态与校准

只在B/C同时正确的source-known物理样本上冻结：`tau_q=Q0.01(q)`、`tau_margin=Q0.01(margin_C)`、`tau_unknown_low=Q0.99(e_unknown)`、`tau_unknown_high=max(Q0.999(e_unknown),tau_unknown_low+0.05)`并裁剪到1。

- `registered(c*)`：`q>=tau_q`、margin≥`tau_margin`且`e_unknown<=tau_unknown_low`；
- `unknown`：`q>=tau_q`且`e_unknown>=tau_unknown_high`；
- 其余为`defer`。

这些阈值只保护source-known覆盖，不宣称达到真实unknown FAR。proxy/held审计失败不得修改bundle。

## 5.Phase3接口

每个接收节点把v2输出封装为`LocalEvidenceV2`：`p_local/q/local_decision/local_label`来自bundle；event/reception/node/correlation/deadline来自标签可见前的采集元数据。bundle runtime不得自行构造same-event关系。缺少合法事件绑定时只允许逐reception或`proxy_unverified`技术审计。

## 6.发布矩阵

1.本地toy runtime完成bundle build/load/tamper/no-role tests。
2.N607用B/C真实checkpoint并行导出TorchScript，进行eager↔TorchScript parity。
3.复用已封存B/C actual-IQ feature NPZ并补充C `z_dom` source-only导出，构建不可覆盖bundle。
4.真实received-IQ no-query smoke输出本地证据；proxy/held只在bundle封存后审计。
5.独立P0/P1后发布；不增加签名权限层作为本轮前置门。

