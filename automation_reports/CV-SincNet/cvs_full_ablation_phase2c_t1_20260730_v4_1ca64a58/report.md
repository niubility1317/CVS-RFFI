# CVS-RFFI Phase2-C T1完整消融v4运行报告

## 实验身份

|字段|值|
|---|---|
|实验ID|`cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|时间|2026-07-30|
|operator|Codex主代理；N607 sole launch owner=`stage2_t1_n607_release`|
|目标|修复v3的NumPy 2.x/PyTorch 2.1转换故障后，完整执行Stage2-C全部1425个logical row，不按中间性能缩小范围|
|正式代码commit|`1ca64a586b85c97fbaa2a677a6ca5776ffd239b3`|
|状态|`LOCAL_VERIFIED / INDEPENDENT_REVIEW_P0_0_P1_0 / REMOTE_NOT_LANDED / NO_PERFORMANCE_RESULT`|

## 假设与比较目标

v3的停止根因仅位于`P2-BASE-ADAPTER-HEAD`训练适配器的NumPy→Torch输入桥。v4以`torch.frombuffer(...).reshape(...).clone()`替代`torch.from_numpy`，并以`detach().cpu().tolist()`替代`Tensor.numpy()`输出桥；方法、超参数、数据权限、1425行矩阵、GPU预算和评分规则保持不变。比较目标仍为设计报告冻结的19个Stage2-C消融arm。

## v3关闭与复用边界

v3固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得resume、覆盖或补写。v4只读复用已经完整闭合的输入，不复用v3的run、request、row log、driver或部分预测/评分：

|复用输入|只读路径|闭合证据|
|---|---|---|
|Package|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_package_t1_20260730_v2_0903163e`|45/45；summary SHA256=`ed61d7c8403c203f4b1c73d4dd87251a67cf01b7da7f6f580143dfa852b026cb`|
|Feature|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437`|75/75 identities；225/225 scope cache；summary SHA256=`423ea50e5f74a1dcc055f605d093d699865f371b92dd0ee7960a31993c94a91d`|
|Formal sidecar|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_sidecar_t1_20260730_v3_47212437`|30/30；60/60文件；summary SHA256=`7d7de4a6dbb22caa5ec4cde594cdf66dc678423c8639882af99cbfce88bb5c45`|
|Phase2 states|`cvs_full_ablation_phase2_states_t1_20260730_v14_cuda_init`|325/325；失败0|

v4仅重新生成与新commit和新run ID绑定的75-entry cache index、1425-entry registry和sealed plan。该操作不重建或重验数据集。

## 固定矩阵与服务器路径

|字段|值|
|---|---|
|source plan|`stage2c_screening_plan_1ca64a58.json`|
|source plan SHA256|`d8c420c22a0a775bc7d6c79af513d79adf78d29a25713e03591a4a79e9136fd7`|
|矩阵|19 arms×75 identities=1425 logical；预计1350 physical|
|环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU并发|8张GPU，每卡总compute进程最多2个|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|input/seal|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|request|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|run|`/home/szu2070436088/2510044040/CV-SincNet/runs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|row log|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58`|
|driver|`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58_driver`|

## 本地修改与验证

|文件|目的|
|---|---|
|`code/cvsrffi/stage2_trainable_lowrank_support_adapter.py`|移除该模块全部`torch.from_numpy`与`Tensor.numpy`桥，保留float32/int64、shape、device和clone隔离|
|`tests/test_stage2_trainable_lowrank_support_adapter.py`|模拟两个旧桥均抛出N607兼容异常并执行真实`fit_locked`|

`ssr-gpu`环境验证：focused 11项通过；相邻Stage2-C执行、工厂、row、input-release、scorer和sidecar链路96项通过；compileall通过。独立复审`P0=0 / P1=0`，允许commit和fresh v4 release。

## 启动和健康门

`reuse_seal_template.txt`先验证三类复用输入的闭合summary，再在fresh v4 input根生成index、registry和sealed plan。`launch_template.txt`验证release HEAD/clean、Phase2 states、source/sealed plan和三类输入后，才创建fresh driver并启动完整矩阵。

启动后立即记录main PID、CWD/cmdline、16个worker、GPU0–7每卡2个槽、日志增长、首个row和首个worker波次。只在P0协议/安全违规，或两个不同row在prediction前出现同一确定性异常指纹时停止v4精确进程树；不得按性能停止、选行或缩小矩阵。完成后必须闭合1350个physical状态、1425个logical score output、全部prediction/score/log和runner summary，再进入结果分析。
