# D6a纯support安全选择门追踪

日期：2026-07-17
状态：实现、单测、support-only选择与fresh holdout验证完成
声明边界：只允许注册support证据；不读取query IQ、feature、prediction或score；不提交Git。

| ID | Source section | Requirement | Target files | Status | Verification | Notes |
|---|---|---|---|---|---|---|
| D6A-01 | 用户要求 | candidate overall LOO相对base下降不得超过2pp | `code/cvsrffi/stage2_support_safe_selector.py` | verified | `test_overall_drop_gate_rejects_anonymous_degraded_candidate` | baseline按同support物理ID匹配 |
| D6A-02 | 用户要求 | 逐state×scenario×class floor使用Wilson下界与Beta-Binomial平滑 | 同上 | verified | Wilson/Beta边界测试与实际support审计 | 不再用裸0.10对0劫持排序 |
| D6A-03 | 用户要求 | 增加leave-two-physical-sample-out或分层support subsampling稳定性 | 同上 | verified | leave-two fold覆盖测试；实际K10六slice LTO | 每个物理support恰好被holdout一次 |
| D6A-04 | 用户要求 | 并列时偏好单view，不按候选名硬编码排除 | 同上 | verified | 匿名退化候选与同分复杂度测试 | 排除仅由证据门决定 |
| D6A-05 | `项目.md`7.1.1/7.2 | 单物理样本单LEO观测；view不增加K；API无query/role/quota接口 | 同上、`tests/test_stage2_support_safe_selector.py` | verified | 接口签名、跨scenario复用阻断、view-as-K阻断 | before/after允许复用同row旧support，跨scenario禁止复用 |
| D6A-06 | 用户要求 | 仅在全新独立development切片存在时才可prediction后评分 | `automation_reports/CV-SincNet/d6a_support_safe_selector_20260717/fresh_holdout_10_per_class` | verified | fresh hash/token与原query、K10 support零重叠；prediction COMMIT后首次truth join | development-only，不是确认矩阵 |

最高风险：如果在已经看过query结果的row上重新设计选择规则并把结果解释为性能验证，会形成隐性query调参。本轮即使复算该row support，也只能用于门逻辑审计，不能产生独立性能结论。

## 实际support-only选择

只打开before/after的`enrollment_only`包，未打开apply/query、prediction、score或truth。结果：

- baseline `base`：LOO overall `0.7235`，leave-two-out overall `0.7196`，worst LOO Wilson `0.0226`，worst LTO Wilson `0.0226`。
- `view3_base_rms_cfo`：LOO overall `0.5784`、LTO overall `0.5588`，由2pp overall双门排除。
- 匿名CFO单view：LOO overall `0.2176`、LTO overall `0.2275`，四个非劣门全部失败；排除不检查候选名。
- `view3_base_rms_spec15`：overall与LTO通过，但worst LOO Wilson为`0`，相对base `0.0226`超过预登记2pp floor容忍度，因此被保守排除。
- 最终只保留并锁定单view `base`。

support-only COMMIT SHA256：`981acd9683edb70f54ce2f1db52a27a29ebcdf94cc58574a94c71bdbd3d08ed6`。

## Fresh holdout

每个注册TX×scenario原缓存有40个物理样本。保持原K10 support不变，严格排除原20/query类后，剩余10/类作为从未评分的fresh query：

- after：11类×3场景×10=`330`条；
- before：同一fresh集合中的6个旧类×3场景×10=`180`条；
- fresh query与原query token、post-channel IQ SHA均零重叠；
- fresh query与K10 support post-channel IQ SHA零重叠；
- predictor包不含标签，predict CLI无truth参数。

prediction COMMIT SHA256：`f61889aee012779cd2a6a425d3d73e719ede87e29c2bb09d656e6583aa544c98`。其后隔离scorer首次连接truth，得到：

|指标|fresh结果|
|---|---:|
|old before|0.8278|
|old after|0.6333|
|old floor before|0.5333|
|old floor after|0.4667|
|seen-new|0.6667|
|H|0.6496|
|forgetting|19.44pp|

该结果说明D6a成功避免了CFO选择灾难，但保守回退base仍无法解决注册后old→new侵入与floor问题。它是独立fresh development holdout，不是125确认或门槛通过证据。
