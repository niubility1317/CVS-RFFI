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

代码与首版实验报告已提交至Git提交`f0ac9b6`（`Add strict lightweight qKNN adapter evaluation`）；提交后工作树仅保留与本实验无关的既有修改。

|文件|用途|SHA256|
|---|---|---|
|`paper_reproduction/scripts/fit_apply_qknnv42_source_teacher_mlp.py`|源域教师蒸馏、MLP选择、目标cache离线映射；兼容Torch2.1+NumPy2.x状态持久化|`822A61B54B0A66896074C08F48B327FF0CC944C30E122320FB1148BA8C1F2FA0`|
|`paper_reproduction/scripts/fit_apply_qknnv42_source_teacher_adapter.py`|严格源域键对齐、教师/源与目标冻结cache哈希和TTA策略校验|`827122377EE012E59B6D8281F56119C8E0C65865AE13D9C34FDE39A9264743DA`|
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
|输出根|`runs/qknnv42_source_mlp_20260714/output_v2/`|
|日志|`logs/qknnv42_source_mlp_20260714_v2.out`|
|预期输出|`adapter_summary.json`、MLP权重NPZ、5个映射后的target feature cache|

## 同步记录

2026-07-14 21:18 CST完成直接SCP；同步后远端SHA256与本地一致，GPU6仍为0%/10MiB，且无同名MLP进程。

|本地来源|N607目的地|验证|
|---|---|---|
|`paper_reproduction/scripts/fit_apply_qknnv42_source_teacher_adapter.py`|`paper_reproduction/scripts/`|SHA256=`E4336BAE...CCD574`|
|`paper_reproduction/scripts/fit_apply_qknnv42_source_teacher_mlp.py`|`paper_reproduction/scripts/`|SHA256=`D0987630...6A224`|
|`local_artifacts/qknnv42_frozen_source_pair_local_20260714_2008/FULL_RX_20-1`|`runs/qknnv42_source_mlp_20260714/input/frozen_source/`|4个文件|
|`local_artifacts/qknnv42_frozen_features_local_20260714_1958/FULL_RX_*`|`runs/qknnv42_source_mlp_20260714/input/frozen_target/`|20个文件|
|历史adapter60教师NPZ|`runs/qknnv42_source_mlp_20260714/input/teacher/FULL_RX_20-1/ADV3B02_FULL_ADAPTER5_FFT96/`|SHA256=`DEAC6F96...C727`|

精确服务器命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && CUDA_VISIBLE_DEVICES=6 nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m paper_reproduction.scripts.fit_apply_qknnv42_source_teacher_mlp --frozen-source-root runs/qknnv42_source_mlp_20260714/input/frozen_source --frozen-target-root runs/qknnv42_source_mlp_20260714/input/frozen_target --teacher-cache runs/qknnv42_source_mlp_20260714/input/teacher/FULL_RX_20-1/ADV3B02_FULL_ADAPTER5_FFT96/features_full_adapter5_fft96.npz --expected-teacher-sha256 DEAC6F96D68E788050579819B43392BDC98C2281C4F5BDAAE49A59CECE5AC727 --expected-checkpoint-sha256 2699EEDCAFE8CEC880828592D2D65BA3781A9948939DA5CF5C82B47143D59C98 --out-root runs/qknnv42_source_mlp_20260714/output_v2 --policies none --rank-grid 32 64 128 --alpha-grid 0.25 0.5 1.0 --epochs 200 --device cuda:0 > logs/qknnv42_source_mlp_20260714_v2.out 2>&1 &
```

## 完成后检查

1.检查进程、GPU6、完整日志、Traceback/NaN/OOM和`adapter_summary.json`。
2.拉回输出cache，在本地运行严格历史Oracle125行和单qKNN125行；逐行保持同一split。
3.完整历史门槛要求old/new/H三个矩阵均值相对84.0667/93.2400/88.2270下降均不超过3pp。
4.报告同一候选整行指标、参数量、MAC、状态和最终判定；不使用边际最大值拼接结论。

## 当前状态

`COMPLETED_NEGATIVE_DIAGNOSTIC`。首次启动已landed但在训练完成后的权重持久化阶段失败：N607为Torch2.1.0+NumPy2.2.5，Torch零拷贝数组触发`np.savez`的`__array_function__`兼容错误；无GPU进程残留，GPU6恢复0%/10MiB，失败输出根保留且不覆盖。兼容修复提交为`9b6e92a`，修复后脚本SHA256为`822A61B5...F2FA0`，再次直接SCP并验证远端哈希一致。

v2于N607物理GPU6成功完成，PID=`622549`；观察到显存约355MiB，任务退出后GPU6恢复0%/10MiB。完整v2日志已拉回`E:\type10-7\automation_reports\CV-SincNet\qknnv42_source_mlp_n607_20260714_2045\qknnv42_source_mlp_20260714_v2.out`，无Traceback、NaN或OOM。输出包括`adapter_summary.json`、43,516B适配器NPZ和5个target cache；最终适配器为`local_artifacts/qknnv42_source_mlp_n607_20260714_v2/adapters/source_teacher_residual_mlp_none.npz`，SHA256=`DCB7ACA5BBE38CE5BB6AB26C14245199AE13A44E70A3407A5002338A208F565E`。日志只写最终候选摘要，没有逐epoch loss；因此本报告能确认完成状态、holdout指标和最终artifact，但不能从日志重建200个epoch的收敛曲线，这是本次训练可观测性的明确限制。

## 方法、输入与输出

MLP为`LayerNorm+160→rank→GELU→160`残差映射：

`z'=normalize(normalize(z)+alpha·W_up·GELU(W_down·LayerNorm(normalize(z))))`。

每个minibatch优化`L=(1-cos(z',z_teacher))+0.2·MSE(z',z_teacher)+0.01·MSE(z',normalize(z))`，使用AdamW、学习率`1e-3`、权重衰减`1e-4`、batch size≤256、梯度裁剪5.0和200epoch。候选为`rank={32,64,128}`与`alpha={0.25,0.5,1.0}`。输入是1440个源域严格键对齐的冻结ADV3B02`z_id160`与adapter60教师`z_id160`；1140行用于候选训练，300行physical-key holdout用于选择。target cache只在模型选择和全源域重训后执行映射，未参与拟合。

输出保持FFT96和split元数据不变，只把主特征替换为`qknn_post_adapter_z_id160`。所有输出manifest均记录父cache、基础checkpoint、TTA policy、target不参与拟合、ADV3B02梯度更新为0、参数量、MAC和状态量。

## N607训练结果

9个候选中，`rank32/alpha0.25`取得最高源域holdout cosine=`0.918858`、MSE=`0.001014`，随后用全部1440个源域样本重训。最终适配器含10,560个参数，推理为10,240MAC/sample，FP32持久状态42,244B；相比历史adapter60的289,685个可训练参数减少96.35%。全搜索加最终重训的前向计算约51.98G MAC；反向计算未用硬件计数器测量，不能与MAC精确相加。该训练直接消费预计算160维cache，不执行ADV3B02前向或反向。

## 严格历史Oracle矩阵

以下每行均为同一候选的5个接收机×5个seed×5个K-shot=125行联合均值；差值顺序为old/new/H相对严格历史84.07/93.24/88.23的百分点。所有候选split与历史逐行一致。

|候选|old|new|H|差值pp|含MLP的head MAC/场景|状态|门槛|
|---|---:|---:|---:|---|---:|---:|---|
|`support_center+FFT0.70`|82.00|83.07|82.06|-2.06/-10.17/-6.17|24.986M|77.01KB|FAIL|
|`support_diag_whiten_fisher+FFT0.70`|82.00|82.73|81.92|-2.07/-10.51/-6.31|24.986M|77.01KB|FAIL|
|`none+FFT0.70`|79.56|84.68|81.60|-4.51/-8.56/-6.62|24.986M|77.01KB|FAIL|
|`support_diag_whiten+FFT0.70`|81.36|82.16|81.33|-2.70/-11.08/-6.90|24.986M|77.01KB|FAIL|
|`support_center+FFT0.34`|78.48|84.09|80.78|-5.58/-9.15/-7.45|24.986M|77.01KB|FAIL|
|`support_diag_whiten_fisher+FFT0.34`|78.53|83.87|80.66|-5.53/-9.37/-7.57|24.986M|77.01KB|FAIL|
|`support_diag_whiten+FFT0.34`|78.14|83.37|80.21|-5.92/-9.87/-8.02|24.986M|77.01KB|FAIL|
|`none+FFT0.34`|75.02|84.31|78.72|-9.04/-8.93/-9.51|24.986M|77.01KB|FAIL|

## 角色分支Oracle矩阵

|候选|old|new|H|差值pp|含MLP的head MAC/场景|状态|门槛|
|---|---:|---:|---:|---|---:|---:|---|
|`support_role_center+FFT0.65`|82.19|85.23|83.25|-1.87/-8.01/-4.97|15.220M|81.01KB|FAIL|
|`support_role_center+FFT0.70`|82.18|85.03|83.16|-1.89/-8.21/-5.07|15.220M|81.01KB|FAIL|
|`support_role_diag_whiten_fisher+FFT0.65`|82.13|84.84|83.01|-1.94/-8.40/-5.22|15.220M|81.01KB|FAIL|
|`support_role_diag_whiten_fisher+FFT0.70`|82.11|84.71|82.95|-1.95/-8.53/-5.27|15.220M|81.01KB|FAIL|
|`support_role_center+FFT0.75`|81.79|84.83|82.84|-2.28/-8.41/-5.39|15.220M|81.01KB|FAIL|
|`support_role_diag_whiten_fisher+FFT0.75`|81.59|84.43|82.53|-2.48/-8.81/-5.70|15.220M|81.01KB|FAIL|
|`support_role_diag_whiten+FFT0.65`|81.47|84.32|82.39|-2.60/-8.92/-5.84|15.220M|81.01KB|FAIL|
|`support_role_diag_whiten+FFT0.70`|81.47|84.24|82.36|-2.60/-9.00/-5.86|15.220M|81.01KB|FAIL|
|`support_role_center+FFT0.80`|80.84|84.41|82.13|-3.23/-8.83/-6.10|15.220M|81.01KB|FAIL|
|`support_role_diag_whiten+FFT0.75`|80.79|83.91|81.88|-3.27/-9.33/-6.35|15.220M|81.01KB|FAIL|
|`support_role_diag_whiten_fisher+FFT0.80`|80.50|83.99|81.76|-3.56/-9.25/-6.47|15.220M|81.01KB|FAIL|
|`support_role_diag_whiten+FFT0.80`|79.82|83.73|81.27|-4.25/-9.51/-6.96|15.220M|81.01KB|FAIL|

最佳角色分支候选为`support_role_center+FFT0.65`，old=82.19%、new=85.23%、H=83.25%。old差距已压到1.87pp，但new仍低8.01pp、H低4.97pp，未通过三指标均不低于历史3pp的门槛。相对此前无MLP的最佳角色分支81.84/85.05/83.00，本次只提高约0.35/0.17/0.25pp，源域教师holdout的0.9189 cosine没有转化为足够的target seen-new恢复。

## 单qKNN确认

可部署单qKNN使用单视图、逐样本argmax、dense LP关闭、无角色/配额Oracle和零decision workspace。在独立seed713106-713110的125行确认中，`Fisher+FFT0.70+bias-0.08`得到old=71.06%、new=74.00%、H=72.01%，head与MLP合计5.079M MAC/场景、状态77.01KB。相同确认网格的无MLP结果为70.98/74.69/72.33；MLP变化为+0.08/-0.69/-0.32pp，没有改善H。

## 结论与后续边界

该MLP在计算上属于轻量后置适配：单样本10,240MAC仅为identity-only ADV3B02单视图8.912M MAC的0.115%，状态约41.25KB；N607实测显存约355MiB。它在性能上未替代历史adapter60。最好的Oracle诊断仍依赖old/new角色与类别配额，不能作为星上可部署算法；最好的单qKNN虽然可部署，但H仅72.01%。因此本路线最终标记为`NEGATIVE_DIAGNOSTIC_NOT_PROMOTABLE`，不进入部署主线。

后续若继续追求历史≤3pp，不能再只拟合源域教师特征。需要引入不使用query标签的target support条件化映射、类原型对齐或更强的源域跨接收机训练目标，并在独立seed上预注册选择规则；任何使用角色真值、类别配额或完整query batch的方案仍只能列为Oracle诊断。
