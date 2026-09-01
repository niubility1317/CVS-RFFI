# ADV3B02-FCR Task4报告：激励条件化发射机响应算子

## Status

本地模块实现和聚焦验证已完成。未执行完整FCR接线、真实checkpoint、训练、N607或独立P0/P1审查；因此FCR-04和FCR-05仅标为`implemented`，不构成端到端、跨天或部署验证。

## Files changed

- `code/cvsrffi/phase1_fcr_factors.py`
- `code/cvsrffi/phase1_fcr_fingerprint.py`
- `code/tests/test_phase1_fcr_fingerprint.py`
- `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`
- 本报告

## Red/green evidence

1. 红测命令：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_fingerprint.py -v`
   - 初次运行因测试文件括号错误而未收集；仅修正测试语法后重新运行。
   - 预期红测：收集时`ImportError: cannot import name 'excitation_features' from 'cvsrffi.phase1_fcr_factors'`。这是新增固定激励接口尚不存在的正确接口失败。
2. 绿测命令：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_fingerprint.py -v`
   - 结果：`5 passed`。

## Operator contract and limits

`excitation_features(s_hat)`固定输出`[B,256,4]`的`(|s|,|s|^2,|s|^3,slew)`；`fixed_response_basis(s_hat)`固定输出`(s,conj(s),s|s|^2,s[n-1]|s[n-1]|^2)`。这些基函数没有可学习替代路径。

`FingerprintFactorEncoder`保留输入`id_feature_raw:[B,160]`，仅加入规范化canonical/residual/excitation摘要，输出单位范数`z_f_id:[B,160]`和独立`z_tx_state:[B,16]`。该模块不改变上游`id_feature_raw`；跨天一致性约束的接线仍属于后续损失任务。

算子计算：

```text
delta_physical=einsum(fixed_basis,response_coef)
delta_small=bounded_residual(excitation,z_tx_state)
delta_f=limit_energy_and_bandwidth(delta_physical+delta_small,s_hat,rho_max)
```

`bounded_residual`是rank=4、kernel=3的局部卷积加状态投影，仅接收四维`excitation`和`z_tx_state`；它不接收raw、canonical或residual IQ。其输出先经`tanh`限制，再由统一逐样本上限处理。默认`rho_max=0.10`，测试使用`0.08`和`0.12`夹具。

为使包含`conj(s_hat)`的固定IQ项仍满足公共相位等变，算子使用由最大幅度激励点确定的相位参考，对共轭基乘以该参考的平方；其余PA和memory项本身同相旋转。最终使用固定三抽头低通并按每个样本的`rho_max*||s_hat||/||delta||`缩放。零激励时相位载波和输出均为零，所有除法均采用`clamp_min(1e-8)`。

## Module evidence

- 固定输入`[1,2]`夹具验证激励的幅度、二次、三次和slew值，以及PA直通、IQ共轭和一阶memory基值。
- `z_f_id`形状为`[3,160]`且逐样本L2范数为1；`z_tx_state`形状为`[3,16]`且有限。
- 对相同因子输入，将`s_hat`旋转`0.4rad`后，`delta_f`与原输出乘同一复相位在`atol=rtol=1e-4`下相等。
- 零输入返回complex64零`delta_f`且`response_quality`有限。
- 每个样本均验证`||delta_f||/||s_hat||<=residual_ratio_max+1e-5`；受限残差的直接调用只允许`excitation,z_tx_state`两个输入。

## Trace changes

FCR-04和FCR-05更新为`implemented`，证据严格限定为本地模块测试。追踪表没有声称跨天稳定、完整模型可达、端到端重构或N607结果。

## Scoped self-review

- `bounded_residual`签名和内部计算只读取激励与状态，没有raw/canonical/residual IQ参数或引用。
- 响应基是纯固定函数；可学习部分只产生系数及短感受野、低rank幅度残差。
- 公共相位等变、零输入有限性、complex64输出和逐样本能量上限均由实际模块测试覆盖。
- 没有改动Task3 canonicalizer、模型接线、decoder、loss、训练、数据或launcher。

## Commit and publish

本报告随Task4提交写入。提交OID、push结果和远端分支OID由提交后的独立读回在任务完成回执中给出；将这些最终OID反写到同一提交会改变提交对象，不能形成稳定自引用。

## Interfaces for Tasks5/9

- Task5可消费complex64`delta_f:[B,256]`、`response_coef:[B,4]`和有限的`response_quality`。
- Task9可将`FingerprintFactorEncoder`连接到既有160维`id_feature_raw`、Task3的`canonical_iq/residual_iq/s_hat`；本任务未创建完整wrapper或训练路径。
