# SF-TAPFT+D92 E0去RF32旧类诊断预登记

## 实验身份

- run ID：`stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1`
- 状态：`LOCAL_VERIFIED`
- Git commit：`f085b9d40c008aee68bc59c6f7007c6b4b8dd629`
- 方法锁：`D92-E0-NORF32`，使用`identity160+FFT96(A4)`，`rf32_used=false`。
- 状态：只比较`DA1_REG0/SF_HEAD`与`DA1_REG0/SF_D92E0_NORF32`；不注册新类，REG0的新类准确率及old/new harmonic均为`N/A`。
- receiver：`20-1`；seed：`713101`；场景：`leo_clear_weak`。
- support：6个旧类各10条，共60条；holdout：同源pool中从未参与SF适配的rank10–19，6类各10条，共60条。

## 数据绑定

- 数据源：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_three_da_leoweakonly_20260715_v1/phase2_predictor_packages/rx_20_1/seed_713101/support_leo_clear_weak.npz`
- SF support完整SHA256：`f5591fa081b197c90969095faba1ff88a3360c4fab1c719cc2014316d13e9c9f`，与pool的rank0–9逐字节相同。
- 评估`capsule_id=sf-erbt-oldonly-rx20-1-s713101-clear-k10-holdout10-v1`。
- 评估`split_id=p2_min_v1-rx20-1-s713101-clear-old6-k10-rank0_9-holdout-rank10_19`。
- SF bundle适配绑定：`capsule_id=d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`、`split_id=stage2b-rx20-1-seed713101-before-support-prefix`。
- builder必须验证K10×6、support/holdout物理ID零交集，并分别输出有标签support、无标签query和独立truth sidecar。predictor不得打开truth。

## 版本、路径与命令

- 本地工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\meta-adapter-tri-r4-v1-20260824`
- N607 release：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1_f085b9d4`
- N607数据导出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1`
- N607 run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1`
- N607 log：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1.log`
- SF bundle：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1/sf_tapft_bundle.pt`
- GPU：物理GPU0；preflight时利用率0%、显存1MiB。
- predictor命令：`CUDA_VISIBLE_DEVICES=0 <python> -X utf8 code/scripts/run_sf_erbt_oldonly.py predict --bundle <sf_bundle> --support <data>/support.npz --query <data>/query.npz --data-handle <data>/data_handle.json --output-root <run>/prediction --seed 713101 --device cuda:0`
- scorer命令：`<python> -X utf8 code/scripts/run_sf_erbt_oldonly.py score --predictions <run>/prediction/predictions.npz --truth <data>/truth.npz --prediction-receipt <run>/prediction/prediction_receipt.json --data-handle <data>/data_handle.json --output <run>/score.json`

## 预期artifact、停止与判断

- 预期：`data_handle.json`、`support.npz`、`query.npz`、独立`truth.npz`、`prediction/predictions.npz`、`prediction_receipt.json`、`score.json`。
- 技术停止仅限错误K/split/bundle绑定、query truth/role进入predictor、输出碰撞、错误checkout、无法加载真实bundle、无法产生完整prediction或进程归属不清；不得因指标低停止。
- 主判断：`SF_D92E0_NORF32-SF_HEAD`的旧类balanced accuracy和class floor；任一下降则不认为叠加有用。单seed单场景只作诊断，不晋级为完整Phase2结论。
