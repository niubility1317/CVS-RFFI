# ADV3B02-FCR Task3报告：保守Canonicalizer与时序内容因子

## Status

本地实现和聚焦验证已完成；本报告写入提交前，提交、推送与远端OID读回将在完成回执中记录。未执行端到端模型接线、真实checkpoint、N607或独立P0/P1审查。

## Files changed

- `code/cvsrffi/phase1_fcr_canonicalizer.py`
- `code/cvsrffi/phase1_fcr_factors.py`
- `code/tests/test_phase1_fcr_canonicalizer.py`
- `code/tests/test_phase1_fcr_content.py`
- `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`
- 本报告

## Red tests

1. `conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_canonicalizer.py -v`
   - 预期失败：两个测试均因`ModuleNotFoundError: No module named 'cvsrffi.phase1_fcr_canonicalizer'`失败。
2. `conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_content.py -v`
   - 预期失败：三个测试均因`ModuleNotFoundError: No module named 'cvsrffi.phase1_fcr_factors'`失败。

## Green tests and compile

- `conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_canonicalizer.py code/tests/test_phase1_fcr_content.py -v`：5 passed。
- `conda run --no-capture-output -n ssr-gpu python -m py_compile code/cvsrffi/phase1_fcr_canonicalizer.py code/cvsrffi/phase1_fcr_factors.py`：exit0。
- `git diff --check`：exit0。

## Canonicalizer

`ConservativeCanonicalizer`仅从IQ估计公共`log_gain`、`phase0`和`omega`，并采用解析逆变换。`log_gain`限制为`[-log(4),log(4)]`，`phase0`使用`π*tanh(·/π)`，`omega`使用`0.20*tanh(·/0.20)`；除以增益前使用`clamp_min(1e-4)`。模块无自由FIR、共轭校正、残差网络、身份标签或TX分类器。

本地合成unit-carrier测试显示，加入gain=1.4、phase0=0.5、omega=0.03后的规范化NMSE低于扰动IQ的NMSE。独立的时变相位加`0.18*conj(clean)`IQ imbalance夹具保留了大于`1e-4`的复数残差能量。

## Content factor

`ContentSequenceEncoder`以kernel7、stride4的局部卷积和短kernel3局部卷积产生默认`[B,64,32]`的`z_s`，生成token前不做全局池化。`ContentGenerator`用线性局部上采样、短卷积和`tanh`边界输出complex64`[B,256]`的`s_hat`。置信度来自token后的摘要并经sigmoid限制在`[0,1]`。

掩码输入置零后，掩码位置的复数重构损失可反向传播到内容模型参数。`ContentFactorEncoder.identity_input(...,detach_identity_input=True)`默认返回detach后的token摘要；身份CE反向传播时内容模型全部参数的`grad`保持`None`。

## Trace changes

FCR-03和FCR-15更新为`implemented`，限定为本地合成和模块测试证据；独立probe、端到端接线和N607证据仍未取得。

## Self-review

- 参数估计仅含三个公共标量，并具有显式数值边界；没有可学习波形修正路径。
- 默认长度256、stride4时token数为64；输入、mask和token形状均显式校验。
- IQ输入和输出在模块边界为实数`[B,2,L]`，重构`s_hat`为complex64`[B,L]`。
- 身份摘要默认detach；masked reconstruction路径未detach。

## Commit and publish

本报告随任务提交写入。最终Commit OID、push结果和远端分支OID由提交后的独立Git读回在任务完成回执中给出；将OID或push结果反写入本报告会改变该提交对象，不能作为同一提交的稳定自引用。

## Concerns and interfaces for Tasks4/5/9

- Task4可消费`ContentOutput.z_s`和complex64`ContentOutput.s_hat`；本任务未实现`z_f`、响应算子或TX分类头。
- Task5可消费`CanonicalOutput.canonical_iq`、`eta_hat`和`residual_iq`，其IQ张量均为`[B,2,256]`。
- Task9负责把模块接到完整FCR模型；本任务没有创建完整wrapper，也没有改变既有模型、训练或checkpoint路径。
