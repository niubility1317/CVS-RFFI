# CORE90 V2矩阵、回放与独立分析验收

本次仅实现本地工具和合成source训练功能测试，没有启动新正式训练、加载目标模型或执行目标推理，没有提交或推送（由主Agent统一集成）。既有`report_core90_target_test.py`未接管。

## 已实现接口

- `build_core90_game_matrix.py --rows v2`生成V2_A—F、默认建议seed392005/392006/392007的配置；均为scratch E200、旧fixed课程、审计关闭，B8使用`head_grad_only`。默认旧矩阵及显式旧行仍保留。生成不代表seed派发授权或实验通过。
- `build_core90_game_replay.py --version 2 --recipient-config ...`要求独立seed、完整source donor、v2实现、scratch/no resume、匹配非空`game_data_contract`和epoch/split契约；旧默认v1兼容保留。请求动作包的类型及head k联合多重集合严格匹配，不宣称recipient实际接受数或墙钟匹配。
- `build_core90_game_exposure_replay.py`提供`validate_exposure_schedule`、`permute_exposure_schedule`及`build_exposure_schedules`。schema为`core90_exposure_schedule_v2`，逐step记录epoch、opaque sample_ids、真实selected_mask、scenario和channel_seed。count模式在同batch长度/E80前后分层置换实际扰动包，保留recipient样本流；exact模式只在同ordered IDs之间置换。输出实际场景计数、matching_scope、exact_sample_assignment和nontrivial；没有可用非平凡置换不能作为已完成顺序实验。完整donor生成要求显式共享data-order seed。runtime消费由主Agent集成验证。
- `analyze_core90_conditional_robustness.py`只读固定预测。支持既有`core90_target_frozen_15_v1`及其`predictions_complete.json`、源域manifest和显式通用closure manifest。打开truth前验证声明的全部模型及四场景ID闭合；A/B必须opaque ID和类数一致，已记录augmentation seed必须一致。随后独立连接truth，报告共同clean-correct集合规模、占比、TX/RX覆盖、三场景错误率和错误互转，保留overall。结果明确为回顾性已接触基准分析，不能回流拟合/选择。
- `analyze_core90_game.py`保留每row/seed完整读数，新增逐seed因子交互及均值/样本标准差；补齐封闭source预测的RX×TX、dense confusion和预测直方图；缺预测产物时不伪造。新增v2探针状态/原因、实际暴露、effective-policy计数及请求/决定/实际更新字段。旧记录没有原始请求时请求计数为null。
- B4/B7诊断仅使用完整日志可见证据：缺训练预测直方图时首次塌缩为UNKNOWN；历史使用、方向/历史预测余弦缺失时保持UNKNOWN。dominant class≥0.99只是描述性观测阈值，不宣称科学塌缩或根因已证明，不产生修复实验。

## 实际验证

先新增测试并观察缺失API/版本参数失败，再实现。最终执行：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 -m pytest code/tests/test_game_tracking_analysis_v2.py code/tests/test_game_tracking_analysis.py code/tests/test_game_tracking_replay.py code/tests/test_game_tracking_matrix.py -q
49 passed
```

覆盖：六行因子语义/配置parse、旧矩阵兼容、动作联合多重集合、v2 donor拒绝旧版及契约不匹配、真实mask与分阶段暴露计数、exact无可用置换、预测缺行/重复/坏manifest/错ID/真值泄露/缺声明模型时truth provider从未打开、shared-clean正确集合和RX/TX覆盖、source dense confusion及最弱RX×TX、B4/B7缺证据不推断。

## 仍需主流程闭合

真实完整source donor及新增seed尚不存在于本次验证中；严格动作/暴露对照的正式实验deferred。runtime真实mask/seed/原始requested_action日志及日程消费需由主Agent验证。历史B4/B7日志若没有所需梯度或直方图信息，工具不能逆向恢复，根因仍UNKNOWN。没有做性能结论或速度承诺。

## 后续接线与补充验收

主Agent续派后，本协作者已完成V2 runtime与prepare_context接线：真实mask/channel seed、连续policy、独立clock/coordinator checkpoint、state_checked replay质量保护及实际曝光文件消费。runtime_v2先6项通过，阶段变化失效补丁后另1项聚焦通过，详见`control_review.md`。旧integration/resume fixtures显式version1，兼容检查通过。

新增`--rows v2_strong_source`仅生成3个LR候选（1e-4/2e-4/4e-4）；manifest预冻结四场景source Macro-F1均值最大、并列较低lr、固定final E200的一次选择规则。`--strong-source-matrix`在全部3个合法source候选完成后只选一次，再给全部确认seed同一配置；缺完整候选返回DEFERRED，不读target。强候选/确认与核心因子矩阵分开。

已生成`configs/core90_game_v2_planned_20260911`18行及`configs/core90_game_v2_strong_source_planned_20260911`3行，21份持久化JSON独立parse通过，均scratch E200、fixed、审计关闭、未派发。B8长期轨迹尚待验收，核心manifest明确`PENDING_B8_LONG_TRAJECTORY_ACCEPTANCE`，保留head_grad_only仅作planned配置。
