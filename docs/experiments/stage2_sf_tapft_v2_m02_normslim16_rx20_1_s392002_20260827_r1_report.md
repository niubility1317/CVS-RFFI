# SF-TAPFT M02 norm逐级瘦身16行实验

## 预登记

- run ID：`stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827_r1`
- 状态：`RUNNING`
- 固定提交：`9c9ae29a65cb019d6a6fd30c613ecc8b470f8cc8`
- 目标：以独立query最优M02（完整target head+全部time norm）为锚，先缩减norm范围、norm affine和训练步数；本轮不压缩target head，不重新引入Adapter、完整`t3`或B/C阶段。
- 数据：`p2_min_v1/VALIDATED_ONCE`；capsule=`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`；split=`stage2b-rx20-1-seed713101-before-support-prefix`；receiver=`20-1`；旧6类K=10，共60条support。
- query边界：所有16行只运行4折support-inner OOF和全60条support refit；不读取query、query truth或query role，不重复使用已经参与M02筛选的rank10–19 truth。
- checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 优化seed：`392002`；4折定义、target head、rho=`0.5`、loss和学习率均与M02一致。

## 16行矩阵与GPU

|GPU|row|候选|唯一变化|
|---:|---|---|---|
|0|S00|M02锚点|完整复现`head+all norm`|
|0|S01|head-only|删除全部norm更新|
|1|S02|t3 norm|只训练`t3.norm`|
|1|S03|t2+t3 norm|只训练`t2/t3.norm`|
|2|S04|backbone norm|训练`t1/t2/t3.norm`，删除`time_fuse norm`|
|2|S05|fuse norm|只训练`time_fuse.1`|
|3|S06|t1 norm|只训练`t1.norm`|
|3|S07|t2 norm|只训练`t2.norm`|
|4|S08|t3+fuse norm|训练`t3.norm+time_fuse.1`|
|4|S09|t2+t3+fuse norm|删除`t1.norm`|
|5|S10|all norm weight|全部norm只训练weight|
|5|S11|all norm bias|全部norm只训练bias|
|6|S12|late norm weight|`t3+fuse`只训练weight|
|6|S13|late norm bias|`t3+fuse`只训练bias|
|7|S14|600步|固定4500步LR时钟，截断至600步|
|7|S15|300步|固定4500步LR时钟，截断至300步|

## 命令与路径

- N607环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v2_m02_normslim16_20260827_9c9ae29a/checkout`。
- 矩阵：`configs/stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827.json`。
- 命令模板：`CUDA_VISIBLE_DEVICES=<gpu> python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix <matrix> --row-id <Sxx> --output-dir <run-root>/<Sxx> --device cuda:0 --folds 4`。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827_r1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827_r1`。
- release归档计划：本地Git归档上传到`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v2_m02_normslim16_20260827_9c9ae29a/release.zip`，只比较该归档一次本地/远端SHA。

## 发布与启动证据

- release归档本地/远端SHA256均为`24e1ae0be752c6b5289b19d5f7dda34b586aefa4a892bda84025a1ebb3cc6f72`，远端编译通过。
- 真实checkpoint无query smoke：`SMOKE_PASS`；`query_opened=false`、`source_opened=false`、support物理样本数60、更新参数张量数2。
- 2026-08-27正式启动16/16行；wrapper PID为107874–107903的16个已登记离散进程，GPU0–7均严格2个训练进程，单进程初始显存约652–670MiB。
- 启动只读回查确认所有命令均使用冻结checkout、矩阵、row ID和各自不可覆盖输出目录；launch receipt和GPU采样器已生成。

## 判定与停止规则

- 相对S00/M02的结构非劣门槛：`BA_new≥BA_M02-0.5pp`、`floor_new≥floor_M02`、`NLL_new≤NLL_M02+0.03`。由于60条support的分辨率较粗，BA实际上需要基本持平。
- 同时报告：可训练参数数、实际变化元素、bundle大小、wall-clock、最大RSS和GPU显存采样峰值。
- 通过门槛后按“参数最少→步数最少→NLL最低”确定唯一最小候选；本轮不接入query truth。
- 技术停止仅限协议/query泄漏、错误checkpoint/capsule/split/K、错误checkout、输出碰撞、不能产生selection/bundle、确定性重复异常或进程归属不清；不得因低性能停止。

## 本地验证与审查

- TDD：先观察14项能力负测和矩阵模块缺失负测失败，再实现最小功能。
- 聚焦回归：79项通过；Python编译通过；CLI help通过；16行严格解析和GPU每卡2行检查通过。
- P0/P1定点审查：发现新增字段会拒绝旧M02bundle；已修复为仅允许`norm_scope/norm_affine/scheduler_reference_steps`以默认值缺省，未知字段继续拒绝。修复负测与query闭合回归通过；未发现其他会使本次真实实验跑错、越权、覆盖输出、不能启动或不能产生合法selection的问题。

## 预期artifact

每行必须生成`selection.json`、`sf_tapft_clean_single_bundle.pt`、完整stdout/stderr日志和`runtime_time.txt`；矩阵级生成launch receipt、GPU采样记录和完成汇总。只有16/16行artifact完成并解析后进入`ARTIFACTS_COMPLETE`，完成同row分析后进入`ANALYZED`。

## 完成结果

### 运行中快照

- 2026-08-27 12:48 CST：S15（300步上限）率先闭合；其余15行继续运行，异常指纹为0。
- S15 support OOF：BA=`0.8611111939`，fold floor=`0.7777778506`，NLL=`0.5148361325`，4折均不退化；最终全support refit采用203步。
- S15资源：可训练参数元素1584，实际变化元素1584，bundle 4,292,510B，wall-clock 39分17.91秒，最大RSS 1,829,304KiB；exit status=0。
- S15权限：`query_opened/query_truth_opened/query_role_opened/source_opened/target_eval_opened`全部为`false`，`nonpermitted_changed_names=[]`，support_count=60。
- 4500步结构行会对4个OOF fold分别完整执行4500步，再进行一次选中步数的全support refit；不存在训练早停。因此本轮预计还需约8–9小时。该估计不构成结果或停止条件。
- S15必须等待同run S00锚点后才能判定非劣，目前保持`PENDING_ANCHOR`。
