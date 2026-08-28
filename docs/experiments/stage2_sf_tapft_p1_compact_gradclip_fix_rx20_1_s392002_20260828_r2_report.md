# SF-TAPFT P1 Compact梯度裁剪定点修复D0/D4实验报告

## 1.最小预登记

- run ID：`stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r2`
- 当前状态：`LOCAL_VERIFIED`
- 实现commit：`b17c8fe18c1d7b2165819194a347d294f14b9423`
- r1停止原因：真实checkpoint无query smoke发现FP16 storage cache未一次性materialize到FP32 compute cache；r1无正式适配或性能结果，禁止覆盖
- 本run修复：compact梯度裁剪覆盖优化器全部实际参数；smoke按storage/compute新契约一次性materialize
- 本地验证：118项compact/runner/query聚焦测试通过；smoke脚本编译和20项部署/benchmark测试通过；独立P0/P1定点审查PASS

## 2.冻结行与数据

|行|方法|步数|cache|GPU|目的|
|---|---|---|---|---:|---|
|D0|P0C H6 Compact|300/150/70|FP32 storage/compute|0|恢复与reference H6等价的工程基线|
|D4|H6+head-only class-CVaR|300/150/100，末30步Top2，权重0.03|FP32 storage/compute|1|验证不增加suffix计算的floor增强|

- config：`configs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828.json`
- 数据：`p2_min_v1/VALIDATED_ONCE`；capsule=`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`；split=`stage2b-rx20-1-seed713101-before-support-prefix`
- receiver/scene/seed：`rx20-1/leo_clear_weak/392002`；旧6类K=10，共60条support；不注册新类
- Q180：Q60与Q120各自先truth-blind形成D0/D4共4组prediction，全部闭合后才连接truth；只报告`DA0_REG0`与`DA1_REG0`
- 科学边界：query truth已有历史暴露，本run仅为工程回放；不得据此正式晋级或重新拟合方法

## 3.N607发布

- 环境/CWD：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r2`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r2`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r2`
- 命令：`python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix configs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828.json --row-id <D0|D4> --mode deploy --deployment-inplace --delta-only --output-dir <run-root>/support/<row> --device cuda:0`
- release只比较一次归档SHA并远端编译一次；smoke PASS后立即并行D0/D4
- 预期artifact：每行`selection.json`、delta、GNU time、Q60/Q120 prediction三件套与truth-last score

## 4.停止与判定

仅协议/query泄漏、错误数据行、输出碰撞、错误checkout、确定性异常、无prediction闭合、scorer连接错误或进程归属不清可停止；不得因低性能停止。

相对修复后D0：BA不下降、floor不下降、任一类别不低于D0-5pp、NLL不高于D0+0.02；可训练元素不超过1584、delta不超过10KB、适配墙钟不超过20秒。通过者仅记为`ENGINEERING_REPLAY_PASS`。

