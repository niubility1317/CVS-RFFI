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

## 5.发布和运行闭合

- release归档SHA256：`aa4972aeb62e80d5057576490c73baa905f6a997fbd01ca6a9deec77897601eb`，本地与N607远端一致；
- 远端编译：PASS；
- 真实checkpoint无query smoke：最大logit差0、最大gradient差0、prediction完全一致、FP16 storage有限值检查通过；
- support样本数：60；smoke确认query未打开；
- D0/D4均为`DEPLOY_ADAPT_COMPLETE`；Q60/Q120各自先形成prediction，随后独立scorer连接truth；
- r1保留为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`，没有复用或覆盖其run ID。

## 6.Q180性能

共同`DA0_REG0`为130/180，BA 72.2222%，floor 10.0000%，NLL 0.870038，ECE-10 0.130062。

|行|正确数|BA|floor|NLL|ECE-10|DA BA变化|DA floor变化|
|---|---:|---:|---:|---:|---:|---:|---:|
|D0|150/180|83.3333%|56.6667%|0.500754|0.074766|+11.1111pp|+46.6667pp|
|D4|150/180|83.3333%|56.6667%|0.500055|0.074701|+11.1111pp|+46.6667pp|

两行逐类正确数均为`[24,28,27,17,27,27]`，逐类准确率均为`[80.0000%,93.3333%,90.0000%,56.6667%,90.0000%,90.0000%]`。D0逐类NLL为`[0.444777,0.222214,0.384456,1.223761,0.312601,0.416717]`；D4为`[0.441803,0.223113,0.384388,1.222054,0.311762,0.417211]`。

D0/D4的180条argmax完全相同，混淆矩阵均为：

|true\pred|0|1|2|3|4|5|
|---:|---:|---:|---:|---:|---:|---:|
|0|24|6|0|0|0|0|
|1|0|28|0|2|0|0|
|2|0|0|27|0|2|1|
|3|1|11|0|17|1|0|
|4|0|1|0|2|27|0|
|5|1|0|0|0|2|27|

同row配对相对DA0均为错→对24、对→错4、都对126、都错26，精确McNemar `p=0.000180`。

## 7.Q60/Q120稳定性

|行|Q60 BA/floor/NLL|Q120 BA/floor/NLL|
|---|---|---|
|D0|86.6667%/70.0000%/0.460847|81.6667%/50.0000%/0.520708|
|D4|86.6667%/70.0000%/0.460481|81.6667%/50.0000%/0.519842|

Q60与Q120的opaque query ID交集为0，Q180每类30条。两个分区中D4均未改变argmax，只产生极小的概率校准变化。

## 8.资源与Class-CVaR效果

|行|可训练/变化元素|步骤|完整backbone forward|cached suffix forward|cache|snapshot|delta|墙钟|最大RSS|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
|D0|1152/1152|300/150/70|0|450|929520B|4608B|5305B|10.80s|1611424KiB|
|D4|1152/1152|300/150/100|0|450|929520B|4608B|5305B|12.89s|1648460KiB|

GPU显存峰值为`NOT_CAPTURED`。D4末30步Top2 Class-CVaR损失从0.466826降到0.464892，但最终只相对D0改善NLL 0.000699和ECE-10 0.000066，墙钟增加2.09秒。

## 9.结论

D0和D4都通过预登记门槛，但D4没有任何分类边界收益，资源反而更差，故判定为`PASS_NO_INCREMENTAL_VALUE`。推荐D0作为当前最小平衡工程工作点；本次仍是truth-exposed rx20回放，不能替代新未暴露capsule的正式确认。
