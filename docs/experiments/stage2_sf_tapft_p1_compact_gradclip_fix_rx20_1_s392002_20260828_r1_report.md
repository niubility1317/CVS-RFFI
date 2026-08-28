# SF-TAPFT P1 Compact梯度裁剪定点修复D0/D4实验报告

## 1.最小预登记

- run ID：`stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r1`
- 当前状态：`LOCAL_VERIFIED`
- 实现commit：`7ffef5703aa536ac0f3a29474eea3261ba2f8f5f`
- 触发原因：旧compact训练的梯度裁剪集合从完整`student`取参数，遗漏独立`CompactH6Suffix.t3.norm`；历史P0C与旧D0在Q120出现2条argmax漂移，因此旧D0/D4标记为`METHOD_MISMATCH_COMPACT_GRADCLIP_OMISSION/NO_VALID_P1_RESULT`，保留但不得作为矩阵基线
- 修复：梯度裁剪直接遍历`optimizer.param_groups`中的实际可训练参数；reference H6与compact原位训练在强裁剪回归中以`atol=1e-7`保持`t3.norm.weight/bias`和target head等价
- 聚焦验证：118项测试通过；独立P0/P1定点审查PASS

## 2.冻结行与数据

|行|方法|步数|cache|GPU|目的|
|---|---|---|---|---:|---|
|D0|P0C H6 Compact|300/150/70|FP32 storage/compute|0|恢复严格H6工程基线|
|D4|H6+head-only class-CVaR|300/150/100，末30步Top2，权重0.03|FP32 storage/compute|1|在正确compact裁剪路径上验证floor增强|

- 复用冻结config：`configs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828.json`
- 数据：`p2_min_v1/VALIDATED_ONCE`；capsule=`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`；split=`stage2b-rx20-1-seed713101-before-support-prefix`
- receiver/scene/seed：`rx20-1/leo_clear_weak/392002`；旧6类K=10，共60条support；不注册新类
- query：Q60+Q120零重叠并集，共180条；只报告`DA0_REG0`与`DA1_REG0`
- 科学边界：该query truth已被历史实验使用，本run仅为工程回放和方法等价修复，不得用于正式晋级或重新选择超参数

## 3.N607发布

- 环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r1`
- 命令：`python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix configs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828.json --row-id <D0|D4> --mode deploy --deployment-inplace --delta-only --output-dir <run-root>/support/<row> --device cuda:0`
- 预计artifact：每行`selection.json`、`sf_tapft_delta_bundle.pt`、GNU time；随后Q60/Q120各自`da0_reg0.npz`、`da1_reg0.npz`、`prediction_receipt.json`和truth-last score
- release只进行一次本地到远端归档SHA核对和一次远端编译

## 4.停止与判定

只在协议/query泄漏、错误数据行、输出碰撞、错误checkout、确定性异常、无prediction闭合、scorer连接错误或进程归属不清时停止；不得因低性能停止。D0/D4全部4份prediction闭合前不得读取truth。

最终矩阵仍相对修复后D0判断：BA不下降、floor不下降、任一类别不低于D0-5pp、NLL不高于D0+0.02；可训练元素不超过1584、delta不超过10KB、适配墙钟不超过20秒。通过者仅记为`ENGINEERING_REPLAY_PASS`。

