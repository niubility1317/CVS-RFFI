+# SF-TAPFT H6 P0原位适配与轻量部署实验报告

## 1.预登记状态

- run ID：`stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828_r1`
- 当前状态：`LOCAL_VERIFIED/N607_RELEASE_PENDING`
- 实现commit：`11f19f17a7ab54989c3781f23b1f1b952394e548`
- 方法边界：旧6类K=10，共60条support；不注册新类；仅报告`DA0_REG0`和`DA1_REG0`
- 数据句柄：`p2_min_v1/VALIDATED_ONCE`
- capsule：`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`
- split：`stage2b-rx20-1-seed713101-before-support-prefix`
- receiver/scene/seed：`rx20-1/leo_clear_weak/392002`
- 基础模型：ADV3B02 CORE90 checkpoint
- query边界：support适配、FP16安全复核和fallback均不读取query；Q180 prediction完整后才由独立scorer连接truth

## 2.本轮修改

|文件|作用|
|---|---|
|`target_only_progressive_adapt.py`|显式原位训练、最小许可参数锚点、优化器可达性检查、稳定H6前缀/后缀API、FP16 support安全复核和FP32 fallback|
|`target_only_progressive_runner.py`|原位部署入口、delta-only输出、delta v2、资源字段|
|`stage2_sf_tapft_query_closure.py`|checkpoint+delta v2直接形成DA0/DA1 prediction，兼容完整bundle|
|`sf_tapft_deployment_benchmark.py`|单进程资源采样，预热3次、正式10次，median/P90/max|
|`run_sf_tapft_slim_matrix_row.py`|增加原位和delta-only发布参数|
|`run_sf_tapft_deployment_benchmark.py`|矩阵行隔离资源基准入口|
|`stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828.json`|冻结P0A/P0B/P0C|

本地`ssr-gpu`验证：5个相关测试文件共99项通过；修改模块编译通过；三行矩阵严格解析通过。一次独立P0/P1审查发现delta-only尚未进入query closure，已定点修复并复审通过。

## 3.冻结矩阵

|行|所有权|cache|部署输出|唯一目的|GPU|
|---|---|---|---|---|---:|
|P0A|复制完整student|FP16|兼容完整bundle+delta v2|当前实现锚点|0|
|P0B|原位|FP16+support FP32安全复核|delta v2 only|目标轻量部署候选|0|
|P0C|原位|FP32|delta v2 only|数值fallback对照|0|

三行沿用H6固定300/150/70步、相同目标函数、相同温度、相同seed和相同许可参数集合。资源基准逐行串行独占GPU，不与性能任务或其他基准并行。

## 4.N607命令与路径

- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828_r1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828_r1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828_r1`
- release归档：本地与远端仅比较一次归档SHA；远端解压后执行一次Python编译

support适配命令模板：

```text
python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix configs/stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828.json --row-id <P0A|P0B|P0C> --mode deploy --output-dir <run-root>/support/<row> --device cuda:0 [--deployment-inplace] [--delta-only]
```

资源基准命令模板：

```text
python code/scripts/run_sf_tapft_deployment_benchmark.py --matrix configs/stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828.json --row-id <row> --output-root <run-root>/benchmark/<row> --device cuda:0 --warmup-runs 3 --measured-runs 10 [--inplace] [--emit-clean-single-bundle]
```

P0A不加原位参数并生成兼容bundle；P0B/P0C使用原位和delta-only参数。query closure复用既有最大Q180固定received-IQ输入，输出到不可覆盖的`prediction/<row>`；scorer只在三行prediction完整后读取truth。

## 5.预期artifact

- `support/<row>/selection.json`
- `support/<row>/sf_tapft_delta_bundle.pt`
- 仅P0A：`support/P0A/sf_tapft_clean_single_bundle.pt`
- `benchmark/<row>/benchmark.json`及3个warmup、10个measure不可覆盖子目录
- `prediction/<row>/da0_reg0.npz`
- `prediction/<row>/da1_reg0.npz`
- `prediction/<row>/prediction_receipt.json`
- truth-last score和最终同row分析

## 6.停止与晋级规则

只在协议/query泄漏、错误stage/receiver/seed/K/scene/split、输出覆盖、错误checkout、重复确定性异常、无prediction闭合、scorer连接错误或进程归属不清时技术停止。不得因低性能停止。

沿用H6 Q180门槛：BA不低于83.3333%，class floor不低于56.6667%，任一类别准确率下降不超过5pp，NLL不高于0.521858。工程门槛：delta不超过10KB，P0B适配额外峰值显著低于P0A，wall-clock不劣于P0A的10%，support安全复核通过。满足性能和工程门槛的最小候选晋级P1新未暴露capsule D0–D4。

## 7.结果占位

待N607闭合后追加：PID/CWD/cmdline/GPU/log增长、三行selection、FP16安全复核、delta bytes、cache bytes、10次资源统计、Q180总体/逐类/NLL/ECE/McNemar和晋级结论。
