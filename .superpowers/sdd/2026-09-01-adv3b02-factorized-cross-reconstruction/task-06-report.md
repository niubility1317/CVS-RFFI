# ADV3B02-FCR Task6报告：冻结物理特征、Fisher门控和重构误差

## Status

Task6拥有的两个模块和两个聚焦测试已完成。本任务只提供模块级实现与本地证据；没有进行完整FCR接线、真实checkpoint、训练、N607或独立P0/P1审查。因此FCR-08、FCR-09和FCR-19为`implemented`，不构成端到端、跨天或部署结论。

## Files

- `code/cvsrffi/phase1_fcr_physics.py`
- `code/cvsrffi/phase1_fcr_losses.py`
- `code/tests/test_phase1_fcr_physics.py`
- `code/tests/test_phase1_fcr_reconstruction_losses.py`
- `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`仅FCR-08/FCR-09/FCR-19行
- 本报告

## Red/green evidence

1. 红测命令：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_physics.py code/tests/test_phase1_fcr_reconstruction_losses.py -v`。
   - 预期失败：`ModuleNotFoundError: No module named 'cvsrffi.phase1_fcr_physics'`和`ModuleNotFoundError: No module named 'cvsrffi.phase1_fcr_losses'`，确认新增接口不存在。
2. 初版实现暴露两个直接实现边界：幅度条件残差的255/256长度错配，以及NaN SNR未严格降为零置信度；均由同一聚焦测试捕获后以最小修正解决。
3. 绿测同一命令：`7 passed in 5.48s`。
4. `conda run --no-capture-output -n ssr-gpu python -m py_compile code/cvsrffi/phase1_fcr_physics.py code/cvsrffi/phase1_fcr_losses.py`退出0；`git diff --check`退出0。

## Frozen physical features and Fisher gate

`FrozenFingerprintFeatureBank`没有参数或可训练统计量，确定性输出八个具名块：`iq_non_circularity`、`am_am`、`am_pm`、`memory_residual`、`spectral_shoulder`、`phase_noise_psd`、`amplitude_conditioned_residual`和`cyclostationary`。测试核对参数总数为0、块名完整、同输入结果一致及全部有限。

`FisherIdentifiabilityGate`的质量输入是响应基Gram有效秩、激励覆盖、PAPR、SNR质量和噪声地板质量，全部在计算前`detach()`。权重有界于`[0,1]`且非负。PA权重同时受Gram/SNR/噪声质量、PAPR和激励覆盖控制；固定其余质量时，低PAPR常幅输入的`pa`权重低于高PAPR输入。退化Gram或非有限SNR显式降置信度，不产生NaN或放大任何物理块。

此外，模块提供逐样本指纹能量上界、响应平滑和参数边界罚项，供后续损失组合接线使用。

## Reconstruction losses

- `heteroscedastic_complex_nll`将方差先限制在`FCRConfig.variance_floor..variance_ceiling`，对相同有界方差验证精确条件均值的NLL低于坏均值。
- `mrstft_loss`使用长度256可用的32、64、128三窗，对数幅度中显式加噪声地板；完全相等小于`1e-6`，零输入有限。
- `phase_increment_loss`比较归一化共轭相邻乘积，按target相邻幅度加权；`+pi`和`-pi`等价环绕夹具小于`1e-2`，不对wrapped phase直接L1。
- `physical_feature_loss`只比较冻结的同名物理特征块，并只乘对应Fisher权重。测试确认全零权重精确归零、非零输入有限且梯度到预测IQ存在并有限。

## Trace and scoped self-review

只修改FCR-08/FCR-09/FCR-19三行，证据严格限定为本地模块测试。自查确认：特征库无分类器、可学习特征统计或身份/query输入；Fisher质量路径无梯度；FFT、归一化和对数均有显式下界；相位通过单位共轭增量处理环绕；NLL方差不会逃逸配置区间；所有公开损失返回标量且零/近零IQ有限。

## Commit and publish

本报告随Task6提交。提交和push后的本地HEAD/远端OID独立比对在任务完成回执中给出；将最终提交OID写回同一提交会改变其对象，不能形成稳定自引用。

## Interfaces for Tasks7/9/10

- Task7可在`phase1_fcr_losses.py`扩展交叉、共享、cycle和监督项，保持本任务的四个重构函数与Fisher权重接口不变。
- Task9可把Task5`FCRDecodeOutput(mu_iq,log_variance,delta_f)`接入`heteroscedastic_complex_nll`，把Task4的响应Gram/质量统计接入门控；当前模块不创建模型接线。
- Task10应仅在合法Phase1训练输入上使用本模块；`FrozenFingerprintFeatureBank`和Fisher质量均不读取身份标签或任何Phase2 query信息。
