# ADVB02 NTRS Adapter-Only实验设计

## 目标

本设计严格落实用户提供的《ADVB02 NTRS-V2阶段一实验深度分析》。目标是在成熟ADVB02 checkpoint上冻结身份骨干与共享CosFace头，仅训练可学习物理上下文和类共享低秩残差，验证仅由nuisance context预测的小修正能否改善三种LEO_WEAK场景，同时保持clean和raw身份空间。

本轮只形成Phase1 source-only、ManySig地面代理数据和LEO_WEAK物理启发代理信道证据，不声明真实在轨、Phase2适配、真实unknown拒识或多节点协同结果。

## 冻结协议

- 数据角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- seed：`392034`；A0重复实验仍使用相同seed并用独立run ID测量运行噪声。
- 训练和最终测试信道：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- clean只作为无信道增强对照；禁止`mixed_orbit`。
- 最终checkpoint固定为E200；最终独立测试逐场景报告，不用LEO聚合代替场景结果。
- adapter候选从成熟D1 E200 checkpoint初始化；不得读取target receiver、support、query或truth。

## 模型

新增`ntrs_variant=v3_adapter`：

```text
z_anchor = stopgrad(E_frozen(x))
q = Q_psi(descriptor_40(x))
a = coefficient_head(q)
delta = U a
delta = bound(delta, alpha_max * ||z_anchor||)
z_rob = z_anchor - delta
robust_logits = H_frozen(z_rob)
```

约束如下：

- 残差只读取`q`，禁止读取`z_anchor`，避免TX identity shortcut。
- `Q_psi`在A1及后续可训练；A1-R中冻结为固定随机投影。
- `U`是所有TX共享的rank-8低秩basis；不建立独立分类头，不使用LayerNorm。
- A1/A2的`alpha_max`分别比较0.02和0.05时由profile冻结；首轮默认A2使用0.05。
- adapter-only阶段冻结身份骨干、domain骨干和共享CosFace头；优化器不得包含raw参数。
- 训练主路径、FISHR、prototype、open-world geometry和伪标签都继续读取raw输出；只有显式NTRS adapter损失读取robust输出。
- 最终评测比较raw和always-on robust；在残差净救回为正前不训练learned gate。

## 训练目标

Adapter-only目标为：

```text
L = L_sat_CE
  + lambda_KL * KL(H(z_clean) || H(z_sat - delta_sat))
  + lambda_margin * L_margin
  + lambda_clean0 * mean(||delta_clean||^2)
  + lambda_relative * mean(||delta_sat||^2 / (||z_sat||^2 + eps))
```

- `L_sat_CE`只作用于卫星样本robust logits；禁止对clean robust logits使用同权重CE。
- A1-R/A1只启用sat CE、clean-zero和relative correction。
- A2增加teacher KL与margin。
- clean和satellite损失必须分开统计。
- 伪标签来自raw/frozen teacher，不得来自fused或robust输出。

## 矩阵

|行|初始化|可训练参数|目标|用途|
|---|---|---|---|---|
|A0|从头|原Core90|原Core90|同release运行噪声基线|
|A0-B|从头|原Core90|严格NTRS旁路|旁路端到端等价|
|A1-R|成熟D1|低秩adapter，q冻结随机|sat CE＋clean-zero＋relative|随机q阴性对照|
|A1|成熟D1|q＋低秩adapter|sat CE＋clean-zero＋relative|可学习nuisance context|
|A2|成熟D1|q＋低秩adapter|A1＋teacher KL＋margin|类共享修正完整首版|
|A3|A2晋级后|q＋低秩adapter＋source support gate|A2＋支持门|未见域可校正性|
|A4|A3晋级后|A3＋极小core，head冻结|teacher raw保持＋联合微调|极小联合更新|

A0/A0-B/A1-R/A1/A2属于首轮可证伪矩阵。A3只有A2通过后启动，A4只有A3通过后启动；profile与代码可同时实现，但不得绕过前序科学门槛。对1–2pp效应，首轮通过后至少完成3个独立重复再晋级。

## 机制证据和晋级门槛

必须保存：

- raw、robust、fused/always-on准确率及rescued/harmed；
- relative correction p50/p95；
- rotation angle p50/p95；
- clean correction p95、satellite correction p95；
- q encoder梯度范数与参数更新证据；
- adapter-only raw backbone/head最大参数漂移，要求为0；
- clean和三种LEO_WEAK逐场景最终指标。

晋级要求全部满足：

```text
raw逐样本预测与冻结基线一致
delta LEO mean >= +1.0 pp
delta clean >= -0.5 pp
每个LEO场景下降不超过0.5 pp
delta strict UDU >= 0
rescued > harmed
q_gradient_norm > 0
raw_parameter_max_abs_drift == 0
```

## 错误与安全边界

- adapter-only checkpoint缺失、结构重建不一致、raw参数进入优化器、q应训练却无梯度、错误seed/角色/场景、输出覆盖或最终prediction不闭合属于技术失败。
- 低性能不停止合法运行；只触发同row分析和下一候选。
- 不覆盖或删除D0、D1、D2、D3、V2-1及本轮任何checkpoint、日志或评测结果。

