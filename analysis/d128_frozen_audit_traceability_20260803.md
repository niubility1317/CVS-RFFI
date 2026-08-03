# D128冻结审计路径追踪

来源：`58ee10f5`的§5.1“连续技术停止后的单候选one-shot”。本记录只覆盖训练可微路径与冻结asset外层审计的分离；不改变FSRG、RDHA或D92-lite数学、数据协议、候选矩阵或one-shot入口。

| ID | 来源要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|
| D128-FA-01 | 默认训练replacement必须保留caller graph，拒绝`requires_grad=False` | `code/cvsrffi/stage2_d127_checkpoint_hooks.py`、`tests/test_stage2_d127_checkpoint_hooks.py` | verified | `test_frozen_audit_replacement_is_strictly_separate_and_same_downstream` | 不以`requires_grad_(True)`伪造图。 |
| D128-FA-02 | 明确冻结审计入口允许无图replacement，仍走同一checkpoint downstream并校验tensor/shape/device/tap/finite | `code/cvsrffi/stage2_d127_checkpoint_hooks.py` | verified | A/C同downstream数值一致、NaN拒绝 | 仅内部审计使用，不替换训练入口。 |
| D128-FA-03 | `_fsrg_audit`、`_rdha_audit`及冻结评估logit/changed检查走冻结入口 | `code/cvsrffi/stage2_d127_phase1_release.py`、`tests/test_stage2_d127_phase1_release.py` | verified | `test_internal_real_shape_bridge_path_runs_outer7_final14_without_public_injection` | 训练callback保持严格可微。 |
| D128-FA-04 | A/B/C训练路径仍向临时状态参数产生非零梯度 | `tests/test_stage2_d127_checkpoint_hooks.py` | verified | `test_phase1_bridge_uses_real_downstream_int8_qknn_and_keeps_outer_gradients` | 不改变优化器或候选数学。 |
| D128-FA-05 | 同参数有图与冻结无图downstream的detach数值一致 | `tests/test_stage2_d127_checkpoint_hooks.py` | verified | A/C严格与冻结entry逐tensor`rtol=0,atol=0`比较 | 只比较相同replacement值。 |
| D128-FA-06 | 最小真实checkpoint型A/C模拟回归，不要求SSH | `tests/test_stage2_d127_checkpoint_hooks.py`、`tests/test_stage2_d127_phase1_release.py` | verified | A/C hook模拟及outer7/final14 release夹具共21 passed | 不触碰one-shot入口。 |

## 反向审计

验证：`conda run -n ssr-gpu python -m py_compile ...`通过；`conda run -n ssr-gpu python -m pytest -q tests/test_stage2_d127_checkpoint_hooks.py tests/test_stage2_d127_phase1_release.py`为21 passed（仅既有AMP弃用warning）；最终签名清理后，`tests/test_stage2_d127_phase1_release.py`为15 passed。

反向审计：6项`verified`，0项`deferred/rejected/blocked`。最高剩余风险是实际N607 checkpoint上的A微episode smoke尚由主流程在本地review后执行；本实现没有SSH、实验发布或性能声明。
