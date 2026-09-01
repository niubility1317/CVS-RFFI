# ADV3B02-FCR Task5报告：结构化nuisance与物理顺序概率Decoder

## Status

本地模块实现和聚焦验证已完成。实现只覆盖Task5拥有的nuisance和Decoder模块；未执行完整FCR接线、异方差NLL、真实checkpoint、训练、N607或独立P0/P1审查。因此FCR-02、FCR-06、FCR-07仅更新为`implemented`，不构成端到端或部署证据。

## Files changed

- `code/cvsrffi/phase1_fcr_nuisance.py`
- `code/cvsrffi/phase1_fcr_decoder.py`
- `code/tests/test_phase1_fcr_nuisance.py`
- `code/tests/test_phase1_fcr_decoder.py`
- `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`的FCR-02/FCR-06/FCR-07行
- 本报告

## Red/green evidence

1. nuisance红测：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_nuisance.py -v`
   - 预期失败：收集时`ModuleNotFoundError: No module named 'cvsrffi.phase1_fcr_nuisance'`。
2. nuisance绿测：相同命令结果为`2 passed`。
3. Decoder红测：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_decoder.py -v`
   - 预期失败：收集时`ModuleNotFoundError: No module named 'cvsrffi.phase1_fcr_decoder'`。
4. 聚焦绿测：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_nuisance.py code/tests/test_phase1_fcr_decoder.py -v`
   - 结果：`5 passed`。
5. `conda run --no-capture-output -n ssr-gpu python -m py_compile code/cvsrffi/phase1_fcr_nuisance.py code/cvsrffi/phase1_fcr_decoder.py`退出0；`git diff --check`退出0。

## Nuisance module

`StructuredNuisanceEncoder`只消费`x:[B,2,256]`和`eta_hat:[B,3]`的全局统计量。共享低容量statistics encoder输出四段二维代码：`z_ch:[B,16]`、`z_rx:[B,8]`、`z_sync:[B,6]`和`z_gain:[B,3]`，每例总维度为33，小于原始IQ维度512。模块没有`skip`命名模块，不输出任何时序latent，也不接受身份标签。

`z_sync`依次有界表示公共相位、CFO、Doppler rate、STO、SFO与同步残差；`z_gain`为有界AGC/幅度项；`z_ch`和`z_rx`分别为有界短通道与接收机残差参数。零和近零IQ夹具中所有输出有限。

## Decoder order and variance

`PhysicsOrderedDecoder.forward`只接受`(s_hat,delta_f,nuisance)`，不接收原始`x`。其严格顺序为：

```text
u_hat=s_hat+delta_f
linked=apply_short_channel(u_hat,z_ch)
linked=apply_rx_residual(linked,z_rx)
mu=apply_sync_and_gain(linked,z_sync,z_gain)
```

测试确认`call_trace=("content","fingerprint","channel_receiver")`、`mu_iq:[B,2,256]`、`delta_f`逐元素不变，并且content、fingerprint及全部四段nuisance输入均可反向传播。方差由全部nuisance代码条件化，使用`FCRConfig`的`variance_floor`和`variance_ceiling`在sigmoid区间参数化并再次clamp；测试夹具`[0.02,0.20]`和默认范围均保持`variance_floor<=exp(log_variance)<=variance_ceiling`。零内容输入同样有限。

## Trace and self-review

FCR-02/FCR-06/FCR-07已更新为仅由本地模块测试支持的`implemented`状态。自查确认：没有raw-waveform Decoder输入或旁路；nuisance没有skip或时序复制路径；顺序为content→fingerprint→channel/receiver→sync/gain；方差受config上下界限制；实复IQ与complex64接口、默认长度与所有latent梯度均经过聚焦测试。

## Commit and publish

本报告随Task5提交写入。提交、push与远端OID独立读回由提交后的任务完成回执记录；不在提交前把未来OID写入本报告，以免改变同一提交对象。

## Interfaces and concerns for Tasks6/7/9

- Task6可消费`FCRDecodeOutput.mu_iq/log_variance/delta_f`实现异方差NLL与物理重构项；Task5未实现损失。
- Task7可使用`NuisanceOutput.eta_pred`和四段代码接入已知nuisance监督与cross-cycle；本任务未进行类别泄漏probe。
- Task9应从Canonicalizer、Content、Fingerprint、Nuisance和Decoder组装完整FCR；仍需保持`use_fcr=false`旧模型逐元素兼容。
