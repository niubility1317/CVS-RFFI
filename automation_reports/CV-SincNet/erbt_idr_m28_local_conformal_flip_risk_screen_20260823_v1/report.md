# ERBT-IDR M2.8局部共形翻转风险screen预登记报告

## 当前状态

`LOCAL_VERIFIED_RELEASE_PENDING`

本报告预登记一个不可覆盖的M2.8 screen。它不覆盖M2.7原始证据，也不把screen结果外推为完整125结论。

## 候选与因果问题

主基线固定为去RF32 D92 E0（B0），性能分支固定为M2.5 B3。M2.8仅判断B3相对B0的单条query翻转是否可由target support的局部证据支持。

- C1：`M28-C1-B3-MGD-PAIR-POSTERIOR`
- C2：`M28-C2-B3-MGD-LOCAL-CONFORMAL-RECALL`

候选只读取当前`p2_min_v1`已接收IQ导出的IF256/FFT96与已注册support标签；query truth、query角色、批量类别数和全局重分配均不可用。query不更新任何状态。

## 冻结矩阵

- matrix kind：`screen`
- receiver：`3-19`、`8-8`
- method seed：`7282101`
- 条件：`K5/new20`、`K10/new5`
- arm：B0、B3、C1、C2
- 配对identity：4
- 方法row：16
- 场景单元：48

## 冻结方法

1. B0/B3完全复用既有实现，RF32保持移除。
2. 从FFT96构造MGD96；以旧类support逐类中位中心的类平衡均值估计目标域中心。
3. 通过严格support leave-one-out生成类别条件非一致度、目标类稳定度和top1/top2类别对事件。
4. 采用global→destination→pair的Beta-Binomial收缩，避免稀疏pair直接过拟合。
5. C1仅接受高置信top1；C2允许满足更严格后验与共形条件的top2。
6. 每条query输出必须精确等于完整B0或完整B3分数行。
7. `K<5`、全局事件不足或任何风险条件失败时回退B0。

## 技术停止规则

仅在协议/query泄漏、错误matrix/checkout、输出碰撞、运行命令不能执行、进程归属不清、prediction不闭合、scorer连接错误或至少两行出现相同确定性prediction前异常时停止。不得因低性能停止。

## 晋级门槛

候选必须同时达到`ΔH vs B0≥0.002`、`ΔH vs B3≥0.0002`、`N_help>N_harm`、`Δmin_old≥-0.005`、`Δmin_new≥-0.005`，才运行完整125。未达门槛则发布`SCREEN_NEGATIVE_NO_FULL125`。

## 发布字段（实现提交后回填）

- 实现Git commit：`e6eb5dc7a63b79cc70811302ff9f84f72da382b0`
- 实现远端OID：`e6eb5dc7a63b79cc70811302ff9f84f72da382b0`（独立核对一致）
- 本地环境：`ssr-gpu`
- N607 Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- N607 CWD：release内`code`目录
- prediction设备：CPU；`max-workers=2`
- feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features`
- supplemental feature root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features`
- scoring root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars`
- supplemental scoring root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- checkpoint SHA-256：`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`
- release root：`/home/szu2070436088/2510044040/CV-SincNet/releases/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1`
- 预期artifact：`matrix_index.json`、16个`row_execution_receipt.json`、16个prediction、独立score root、`results_summary.json`

2026-08-23直连只读preflight通过：普通账户、项目根、Python、checkpoint、feature/scoring roots均存在；release/run/log及release archive目标均不存在。GPU0–2有其他负载，GPU3–7空闲；本轮CPU预测不占用GPU。

## 预登记执行命令

prediction：

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/run_m28_local_flip_risk_matrix.py --run-id erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1 --matrix-kind screen --feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features --supplemental-feature-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/erbt_idr_m24_d1_refit_full125_20260820_v1_supplement_features/artifacts/features --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/predictions --device cpu --max-workers 2

truth-last scorer：

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/score_m28_local_flip_risk_matrix.py --matrix-index /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/predictions/matrix_index.json --scoring-root /home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437/artifacts/sidecars --supplemental-scoring-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m24_d1_refit_full125_20260820_v3/scoring_root_repaired_v3 --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/scores --bootstrap-repeats 2000

汇总：

    PYTHONPATH=. /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u scripts/summarize_m28_local_flip_risk_matrix.py --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/predictions --score-root /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/scores --output /home/szu2070436088/2510044040/CV-SincNet/runs/erbt_idr_m28_local_conformal_flip_risk_screen_20260823_v1/results_summary.json

## 本地实现与验证

变更文件包括M2.8局部风险模型、通用row executor接入、screen/full125 runner、独立truth-last scorer、汇总器、聚焦测试、实现追踪和本报告镜像。

- RED：2个测试模块因`cvsrffi.stage2_m28_local_flip_risk`不存在而按预期collection失败。
- GREEN：M2.8聚焦测试`8/8`通过。
- 相邻回归：M2.5/M2.7/M2.8共`40/40`通过。
- 编译/集成回归：M2.4与M2.8共`29/29`通过。
- 入口smoke：runner、scorer、summarizer的正式模块入口均返回帮助并退出0。
- `git diff --check`：通过。
- 独立P0/P1审查：`PASS`，没有会直接导致真实实验跑错、越权、覆盖输出、不能启动或不能产生合法prediction的问题。

首次直接执行`code/scripts/*.py`的帮助命令因Python模块根不在`sys.path`而失败；按正式入口从`code`目录使用`python -m scripts.<module>`复验全部通过。该事件是错误调用方式，不是实现缺陷。
