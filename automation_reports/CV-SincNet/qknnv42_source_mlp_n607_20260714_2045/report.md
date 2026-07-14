# qKNNV42源域教师蒸馏轻量MLP实验报告

## 实验信息

|字段|内容|
|---|---|
|实验ID|`qknnv42_source_mlp_n607_20260714_2045`|
|时间|2026-07-14|
|操作者|Codex|
|目标|在ADV3B02参数完全冻结的前提下，只训练qKNN后置低秩残差特征映射，逼近历史60epoch adapter教师特征；随后在严格125行Stage2-C矩阵上检查是否把完整历史性能差距压到3pp以内|
|比较对象|严格历史125行：old=84.0667%、new=93.2400%、H=88.2270%|
|协议|5个target receiver×5个seed×K={1,2,5,10,20}；旧类6个、新类2个；三种LEO场景|

## 假设与边界

- 训练输入只包含冻结ADV3B02源域`z_id160`与历史adapter60对应源域教师特征，共1440个严格键对齐样本。
- 不使用target support、target query、target标签或unknown样本拟合MLP；ADV3B02梯度更新为0。
- MLP为`LayerNorm+160→rank→160`低秩残差映射，候选`rank={32,64,128}`、`alpha={0.25,0.5,1.0}`，源域physical-key holdout选择；最终候选用全部源域行重训200epoch。
- 本实验属于qKNN后置适配，不是再次训练ADV3B02；若最终125行仍未通过3pp门槛，必须记录为负诊断，不能晋升。

## 本地版本与验证

|文件|用途|SHA256|
|---|---|---|
|`paper_reproduction/scripts/fit_apply_qknnv42_source_teacher_mlp.py`|源域教师蒸馏、MLP选择、目标cache离线映射|`D0987630507B05B3026FE08ADB2050CFE89165588D5EACBE1A4F32229A86A224`|
|`paper_reproduction/scripts/fit_apply_qknnv42_source_teacher_adapter.py`|严格源域键对齐、教师/源与目标冻结cache哈希和TTA策略校验|`E4336BAE9727ABA5F243D9717237F48016E7FA750A50B095F792E7F56FCCD574`|
|`paper_reproduction/cvs_aligned/cvs_method_runner.py`|把后置适配器参数、support/query MAC和状态量计入qKNN资源账|`3446213E5736A8762179FA12CCA8EC518186A45130707DB3CBB42EEF62E01D1A`|
|`paper_reproduction/scripts/benchmark_qknnv42_feature_adapter_sweep.py`|严格校验映射cache来源、TTA策略、有限值和运行网格并执行125行评估|`EB16D17C425EC352183A93FF27449272A87D74890246692F1EF762AF74819819`|

本地验证：`ssr-gpu`环境下`py_compile`通过；36项定向测试通过；Bash启动脚本语法检查通过。修复后CPU真实cache smoke完成，1440/1440源域行严格对齐，源与5个target cache逐一通过checkpoint、frozen、`none/1-view`策略校验，输出5个target cache，rank32映射参数10560、估算10240MAC/sample。smoke仅1epoch，不作为性能结果。教师cache固定SHA256为`DEAC6F96D68E788050579819B43392BDC98C2281C4F5BDAAE49A59CECE5AC727`，基础checkpoint固定SHA256为`2699EEDCAFE8CEC880828592D2D65BA3781A9948939DA5CF5C82B47143D59C98`；脚本启动时会强制复核两者。

## N607资源与启动计划

2026-07-14 20:45 CST直接SSH preflight通过。GPU6为`0%/10MiB`，其他GPU存在活动Phase1训练；本实验只使用GPU6，不干预现有进程。服务器允许每GPU最多两个训练进程，本实验使GPU6最多新增1个进程。

|字段|计划值|
|---|---|
|远端工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|Conda/Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|GPU|物理GPU6；`CUDA_VISIBLE_DEVICES=6`，进程内`cuda:0`|
|输入根|`runs/qknnv42_source_mlp_20260714/input/`|
|输出根|`runs/qknnv42_source_mlp_20260714/output/`|
|日志|`logs/qknnv42_source_mlp_20260714.out`|
|预期输出|`adapter_summary.json`、MLP权重NPZ、5个映射后的target feature cache|

精确服务器命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && CUDA_VISIBLE_DEVICES=6 nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m paper_reproduction.scripts.fit_apply_qknnv42_source_teacher_mlp --frozen-source-root runs/qknnv42_source_mlp_20260714/input/frozen_source --frozen-target-root runs/qknnv42_source_mlp_20260714/input/frozen_target --teacher-cache runs/qknnv42_source_mlp_20260714/input/teacher/FULL_RX_20-1/ADV3B02_FULL_ADAPTER5_FFT96/features_full_adapter5_fft96.npz --expected-teacher-sha256 DEAC6F96D68E788050579819B43392BDC98C2281C4F5BDAAE49A59CECE5AC727 --expected-checkpoint-sha256 2699EEDCAFE8CEC880828592D2D65BA3781A9948939DA5CF5C82B47143D59C98 --out-root runs/qknnv42_source_mlp_20260714/output --policies none --rank-grid 32 64 128 --alpha-grid 0.25 0.5 1.0 --epochs 200 --device cuda:0 > logs/qknnv42_source_mlp_20260714.out 2>&1 &
```

## 完成后检查

1.检查进程、GPU6、完整日志、Traceback/NaN/OOM和`adapter_summary.json`。
2.拉回输出cache，在本地运行严格历史Oracle125行和单qKNN125行；逐行保持同一split。
3.完整历史门槛要求old/new/H三个矩阵均值相对84.0667/93.2400/88.2270下降均不超过3pp。
4.报告同一候选整行指标、参数量、MAC、状态和最终判定；不使用边际最大值拼接结论。

## 当前状态

`PREPARED_LOCAL_VERIFIED_PENDING_SYNC_LAUNCH`。
