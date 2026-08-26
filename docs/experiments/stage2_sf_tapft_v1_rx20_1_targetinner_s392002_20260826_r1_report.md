# SF-TAPFT V1目标域内部性能筛选预登记

## 实验身份

- run ID：`stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1`
- 候选：`SF_TAPFT_V1_REPORT_DEFAULT`
- 权限：`DIAGNOSTIC_NON_FORMAL`
- Git commit：`bd58c27d962393fcd1e7efb6a518ccc59b9de5ee`
- 数据绑定：`protocol_schema=p2_min_v1`、`phase2_data_status=VALIDATED_ONCE`、`capsule_id=d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`、`split_id=stage2b-rx20-1-seed713101-before-support-prefix`

## 可证伪矩阵与停止规则

- 单seed：`392002`。
- 单目标接收机：`rx20-1`。
- `K=10`旧类target support，4-fold target-inner选择。
- frozen与SF-TAPFT在相同OOF fold上比较balanced accuracy、NLL和true-class margin。
- 只有多数fold不下降、平均NLL改善且accuracy或margin改善时选择`adapted`；否则选择`zero_adapt`并停止该候选，不进入query性能验证。
- 技术停止仅限协议/query越界、错误checkpoint/split/GPU、输出碰撞、错误checkout、确定性重复异常、无法产生`selection.json`或进程归属不清；不得因中途指标低而停止。

## 版本与命令

- 本地Git工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\meta-adapter-tri-r4-v1-20260824`
- N607 release目标：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1`
- N607 CWD：上述release目录。
- smoke命令：`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -X utf8 code/scripts/run_target_only_progressive_adapt.py --config configs/stage2_sf_tapft_v1_rx20_1_clear_smoke_s392002_20260826.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_smoke --device cuda:0`
- 性能筛选命令：`CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -X utf8 code/scripts/run_target_only_progressive_nested.py --config configs/stage2_sf_tapft_v1_report_default_s392002_20260826.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1 --device cuda:0 --folds 4`

## 环境、输入与输出

- N607 GPU：物理GPU0；preflight时利用率0%、显存1MiB。
- 服务器解释器：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，PyTorch`2.1.0+cu121`，CUDA可用。服务器不存在`ssr-gpu`环境，本次不伪造环境名。
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- support：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2b_sclba_a_t5t25_s713101_20260824_v1/input/support_rx20_1_k10_clear_smoke.npz`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_smoke`
- 性能筛选输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1.log`
- 预期artifact：smoke的`sf_tapft_bundle.pt/smoke.json`；性能筛选的`selection.json`，仅当选择`adapted`时另有`sf_tapft_bundle.pt`。

## 声明边界

该实验只产生target-inner OOF筛选证据，不读取query，不产生正式Phase2 prediction，也不连接truth。`RUNNING`、smoke通过或OOF选择均不得表述为正式最终性能；只有后续独立query prediction和truth-last scorer闭合后才可声明query性能。
