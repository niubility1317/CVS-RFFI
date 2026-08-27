# SF-TAPFT S15+取证记录

## 报告确定的主假设

- S15的300步沿用4500步参考时钟，225步处于warmup；NLL恶化可能来自短程schedule不匹配。
- 目标head+time norm是独立query最优结构；Adapter和深层解冻不再进入主线。
- OOF正标量温度不改变argmax，因此可在保持BA/floor的同时优化NLL。
- S02仅训练`t3.norm`，结构最轻但必须等待新的独立query后才能替代M02。

## 当前Git状态

- 分支：`codex/meta-adapter-tri-r4-v1-20260824`。
- HEAD：`072d64df0ab3b40e45c9d369ee151e6894570ff2`。
- 既有`conversation_index/`和`local_artifacts/`目录不属于本次改动，不得stage。

## 初步代码定位

- `AdaptationConfig`已支持`norm_scope/norm_affine/scheduler_reference_steps/warmup_ratio`，S15短程schedule可复用现有时钟接口。
- `leave_one_out_prototype_logits`当前确实为sample×class双层Python循环，符合报告提出的向量化对象。
- `_fit_single`当前无条件`deepcopy`teacher，并在每步验证时计算`_checkpoint_distance`；KD=0和稀疏validation尚未优化。
- `CheckpointAverager`已改为许可delta平均，但snapshot保存面、温度、deployment-only入口和混合norm集合仍需精确审计。
- 核心测试面集中在`test_target_only_progressive_adapt.py`、`test_target_only_progressive_runner.py`、`test_sf_tapft_slim_matrix.py`和prediction测试。
- `ProgressiveTrainabilityPolicy`当前只支持单一`norm_scope`和统一`norm_affine`，无法表达S16-A/B的分层weight/bias规则。
- `SFTAPFTSelectionResult`尚未保存OOF logits、labels和fold provenance，无法严格拟合OOF温度或绑定OOFKD教师。
- 最终平均已有许可参数约束，但训练期snapshot仍需确认并收缩为head与许可delta。
