# SF-TAPFT H6 P0原位适配与轻量部署实验报告

## 1.预登记状态

- run ID：`stage2_sf_tapft_h6_p0_inplace_delta_s392002_20260828_r1`
- 当前状态：`ANALYZED/P0C_PROMOTED`
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

## 7.最终结论

本轮P0已完成实现、N607真实checkpoint适配、三行隔离资源基准、delta-only最大Q180 prediction和truth-last评分。最终状态为`ANALYZED`。

- **P0C（原位+FP32 cache+delta-only）晋级为P1工程底座。**它与P0A的180条argmax完全一致：BA=83.3333%、floor=56.6667%、NLL=0.500739、ECE-10=0.074764；median墙钟缩短1.85%，median进程RSS下降0.83%，CUDA allocated峰值下降6.85%，CUDA reserved峰值下降4%。
- **P0B（原位+FP16 cache）不晋级为默认点。**它保持180条argmax一致，并把CUDA allocated峰值降低7.10%，但median墙钟从9.858秒增至11.687秒，变慢18.56%，超过允许的10%劣化。
- **delta-only闭合成功。**P0B/P0C均未生成完整clean-single bundle；三行都只用4628B delta v2从ADV3B02 CORE90 checkpoint重建DA1并形成Q60/Q120 prediction。P0A完整bundle为4,292,190B，delta缩小99.8922%。
- 三行均为150/180正确，逐类正确数均为24/28/27/17/27/27。相对`DA0_REG0`的130/180，DA1有24条错转对、4条对转错，精确McNemar p=0.000180。
- 当前晋级只表示`rx20-1/leo_clear_weak/K=10/seed392002`上的工程和性能闭合，不构成多receiver、三scene、低K或多seed泛化完成。

最终判定：`P0C_PROMOTED_AS_P1_ENGINEERING_BASE`、`P0B_NOT_PROMOTED_WALL_CLOCK_GATE`、`DELTA_ONLY_QUERY_CLOSURE_VERIFIED`。

## 8.实现落地

### 8.1原位模型所有权

新增`fit_sf_tapft_inplace`。部署调用方把本轮独占模型实例交给适配器，适配器不再深拷贝完整student；研究入口`fit_sf_tapft`继续保留复制语义。

原位路径只保存许可模型参数和target head的训练前CPU锚点。优化器可达参数ID必须与许可模型参数+head参数集合完全相等；冻结BatchNorm buffer训练前后逐值比较。三行最终`nonpermitted_changed_names=[]`。

### 8.2稳定H6前缀/后缀接口

实现`encode_h6_prefix`、`forward_h6_suffix`和引用型`H6SuffixTrainer`。trainer只引用一个model、一个head和一个cache。FP16 cache为464760B，FP32 cache为929520B；基础模型参数+buffer tensor为4,199,312B。

### 8.3FP16支持集安全复核

每次cache训练结束后，用同一合法support执行一次FP32 full-path前向，核对有限性、argmax、真实类别margin符号和逐类recall。P0A/P0B均为0条argmax不一致、0个逐类recall不一致；最大logit差分别为0.001573和0.001574，未触发fallback。P0C的FP32 cache最大logit差为0。

若未来FP16不通过，代码恢复许可参数锚点并以同seed、同日程和同目标函数执行FP32 cache重训；整个过程发生在query prediction之前。

### 8.4delta v2与真实加载

delta直接由训练前许可参数锚点计算。v2保存许可参数FP16差量、target head FP16权重、scale、class IDs、基础checkpoint、adapter rank，以及DA0重建所需rho和prototype scale。严格loader检查schema、target binding、参数名和shape；历史delta v1继续兼容。query closure按schema自动选择full-state或delta loader。

## 9.support适配闭合

|行|所有权/cache|完整bundle|delta|cache|安全复核|max logit差|可训练/变化元素|
|---|---|---:|---:|---:|---|---:|---:|
|P0A|复制/FP16|4,292,190B|4628B|464760B|PASS|0.001573|1152/1152|
|P0B|原位/FP16|不生成|4628B|464760B|PASS|0.001574|1152/1152|
|P0C|原位/FP32|不生成|4628B|929520B|PASS|0|1152/1152|

三行均固定300/150/70步，`research_selection_executed=false`、`folds=0`、`validation_forward_steps=[]`、`query_opened=false`、`query_truth_opened=false`、`source_opened=false`。完整stderr未发现Traceback、RuntimeError、OOM、Killed、NaN或Inf。

## 10.隔离资源基准

每行在GPU0单进程串行运行，预热3次后正式测量10次。

|行|wall median/P90/max|CPU RSS median/P90/max|CUDA allocated median|CUDA reserved median|
|---|---|---|---:|---:|
|P0A|9.858/10.007/10.037s|1601.48/1601.80/1601.80MiB|171.47MiB|300MiB|
|P0B|11.687/11.767/11.796s|1592.47/1592.47/1592.47MiB|159.29MiB|288MiB|
|P0C|9.676/9.758/9.779s|1588.17/1592.83/1592.83MiB|159.73MiB|288MiB|

|行|wall median|wall P90|RSS median|CUDA allocated|CUDA reserved|
|---|---:|---:|---:|---:|---:|
|P0B相对P0A|+18.56%|+17.58%|-0.56%|-7.10%|-4.00%|
|P0C相对P0A|-1.85%|-2.49%|-0.83%|-6.85%|-4.00%|

同一行的13次运行位于一个长寿命Python进程。CUDA allocator在正式测量前可能已有reserved池，因此`cuda_reserved_adaptation_extra_peak_bytes=0`只表示未超过既有池，不代表物理显存零增量；本报告以绝对峰值为主。

## 11.最大Q180总体性能

6份prediction全部先闭合，receipt均为`PREDICTIONS_COMPLETE`、`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。之后独立scorer才连接Q60和Q120 truth。

|行/状态|正确|BA|floor|NLL|ECE-10|
|---|---:|---:|---:|---:|---:|
|DA0_REG0（三行相同）|130/180|72.2222%|10.0000%|0.870038|0.130062|
|P0A DA1_REG0|150/180|83.3333%|56.6667%|0.501585|0.074851|
|P0B DA1_REG0|150/180|83.3333%|56.6667%|0.501584|0.074854|
|P0C DA1_REG0|150/180|83.3333%|56.6667%|**0.500739**|**0.074764**|

P0B相对P0A有0条prediction变化，最大logit差0.000826；P0C相对P0A也有0条prediction变化，最大logit差0.175272。DA1相对DA0共同提升BA11.1111pp、floor46.6667pp，并降低NLL约0.3693。

## 12.逐类准确率与NLL

|类别|DA0正确/30|DA0准确率|DA1正确/30|DA1准确率|变化|
|---:|---:|---:|---:|---:|---:|
|0|18|60.0000%|24|80.0000%|+20.0000pp|
|1|30|100.0000%|28|93.3333%|-6.6667pp|
|2|26|86.6667%|27|90.0000%|+3.3333pp|
|3|3|10.0000%|17|56.6667%|+46.6667pp|
|4|29|96.6667%|27|90.0000%|-6.6667pp|
|5|24|80.0000%|27|90.0000%|+10.0000pp|

|类别|DA0 NLL|P0A NLL|P0B NLL|P0C NLL|
|---:|---:|---:|---:|---:|
|0|1.738265|0.445622|0.445602|**0.444718**|
|1|0.329098|0.222507|0.222539|**0.222228**|
|2|0.696159|0.388492|0.388490|**0.384456**|
|3|1.309860|1.223498|**1.223468**|1.223741|
|4|0.133540|0.313165|0.313179|**0.312582**|
|5|1.013308|**0.416227**|0.416229|0.416713|

## 13.混淆矩阵

三行`DA1_REG0`混淆矩阵完全相同，行为真值、列为预测：

|true\\pred|0|1|2|3|4|5|
|---:|---:|---:|---:|---:|---:|---:|
|0|24|6|0|0|0|0|
|1|0|28|0|2|0|0|
|2|0|0|27|0|2|1|
|3|1|11|0|17|1|0|
|4|0|1|0|2|27|0|
|5|1|0|0|0|2|27|

类3主要混淆为类1（11/30），是P1需要用统一公式保护的困难边界；不得为具体类别ID硬编码专属分支。

## 14.分区稳定性

|行|Q60 BA/floor/NLL|Q120 BA/floor/NLL|Q180 BA/floor/NLL|
|---|---|---|---|
|P0A|86.6667%/70%/0.460952|81.6667%/50%/0.521902|83.3333%/56.6667%/0.501585|
|P0B|86.6667%/70%/0.460944|81.6667%/50%/0.521904|83.3333%/56.6667%/0.501584|
|P0C|86.6667%/70%/0.460842|81.6667%/50%/0.520688|83.3333%/56.6667%/0.500739|

Q60比Q120高5pp BA、20pp floor，因此晋级以最大Q180为准。

## 15.发布与下一阶段

- 实现commit：`11f19f17a7ab54989c3781f23b1f1b952394e548`
- 预登记commit：`084b1c96864e32b2115994a5e590f7330e6465e7`
- release归档本地/远端SHA256：`db3a90462e05944608136075ae90eb4def68003cf500d3c678048536461ecf6d`
- 本地`ssr-gpu`相关99项测试通过；远端6个模块编译和矩阵解析通过
- N607不存在`ssr-gpu`环境；远端沿用既有`/home/szu2070436088/.conda/envs/CVS-RFFI`

下一阶段以P0C执行引擎进入新未暴露合法capsule的P1 D0–D4：D0 H6、D1 Q2A、D2 Q2B、D3 R1-T、D4 head-only class-CVaR。P0B只保留为显存较小但时间未过门的工程对照。
