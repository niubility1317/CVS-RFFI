# SF-TAPFT F2+OOF温度真实query评估

## 预登记

- run ID：`stage2_sf_tapft_f2_oof_temp_query_rx20_1_s392002_20260827_r1`。
- 当前状态：`LOCAL_VERIFIED_PENDING_RELEASE`。
- 候选：F2=`S15-SCHED300+OOF全局温度`；固定300步完整cosine、30步warmup、`head+all time norm`、无Adapter，strict bundle中的推理温度为`T=1.07144044514`。
- 任务：不注册新类，只比较同一旧6类query上的`DA0_REG0`与`DA1_REG0`；REG0的新类准确率、old/new harmonic和注册效应均为`N/A`。
- 数据：`p2_min_v1/VALIDATED_ONCE`；receiver=`20-1`；场景=`leo_clear_weak`；K=10×6=60条support；query为rank10–19旧6类各10条，共60条。
- query capsule：`sf-erbt-oldonly-rx20-1-s713101-clear-k10-holdout10-v1`。
- query split：`p2_min_v1-rx20-1-s713101-clear-old6-k10-rank0_9-holdout-rank10_19`。
- 证据边界：该query是真实received-IQ独立holdout，但其truth已在历史16行闭合后揭示；F2在本次预测前未读取query/truth且未由该query选择，结果标记为`REUSED_VALIDATED_HOLDOUT_NOT_NEW_PROSPECTIVE_QUERY`，不得声称全新前瞻确认，不得反馈调参或选择性重跑。
- predictor边界：query NPZ只允许`received_iq/query_ids`；先生成不可覆盖的`DA0_REG0/DA1_REG0`prediction，完整闭合后才由独立scorer打开truth。
- 指标：accuracy、balanced accuracy、class floor、NLL、逐类准确率、`DA1_REG0-DA0_REG0`差值；另与同holdout历史M02结果作上下文对比。
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet/<release-checkout>`。
- bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_s15plus5_rx20_1_s392002_20260827_r1/F2/output/sf_tapft_clean_single_bundle.pt`。
- 数据根：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1`。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_f2_oof_temp_query_rx20_1_s392002_20260827_r1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_f2_oof_temp_query_rx20_1_s392002_20260827_r1`。
- GPU：物理GPU0。
- 预期artifact：`prediction/{da0_reg0.npz,da1_reg0.npz,prediction_receipt.json}`、truth-last`score.json`、GNU time和GPU采样。
- 技术停止：仅限bundle/data绑定错误、query truth/role进入predictor、support/query重叠、错误checkout、输出碰撞、无法加载真实checkpoint、无prediction闭合或确定性系统异常；不得因性能低停止。

## 预注册判断

- 主要问题：F2的快速适配在真实holdout上是否相对自身`DA0_REG0`提升BA、floor并降低NLL。
- 快速档参考门槛：相对历史M02同holdout锚点BA≥86.17%、floor≥60%、NLL≤0.5394；因为本次holdout已复用，该门槛只作同row诊断，不构成新独立晋级。
- OOF温度只改变logit尺度，不改变argmax；因此它可改变NLL但不能单独改变accuracy、BA、floor或逐类准确率。

