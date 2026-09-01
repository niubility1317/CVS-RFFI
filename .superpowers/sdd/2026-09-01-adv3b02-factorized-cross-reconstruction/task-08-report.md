# ADV3B02-FCR Task8报告：定向指纹移植与necessity

## Status

本地模块语义实现与聚焦验证完成。FCR-18更新为`implemented`，不标为`verified`：FCR-13的真实WiSig严格Fingerprint Pair能力仍为`blocked`，本任务没有也不得把合成夹具写成真实strict-pair、端到端、真实checkpoint、训练或N607证据。

## Files changed

- `code/cvsrffi/phase1_fcr_transplant.py`
- `code/cvsrffi/phase1_fcr_losses.py`中的移植损失入口
- `code/tests/test_phase1_fcr_transplant.py`
- `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`仅FCR-18行
- 本报告

## Red/green evidence

1. 红测：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_transplant.py -v`。
   - 实现前收集失败：`ImportError: cannot import name 'compute_transplant_losses' from 'cvsrffi.phase1_fcr_losses'`，证明任务接口尚不存在。
2. 增加drop-f内容/nuisance梯度隔离测试后，红测仅该项失败：删除指纹分支仍向`source_factors.z_s`回传梯度。
3. 最终绿测：同一命令`9 passed in 4.33s`。
4. 共享损失回归：`conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_reconstruction_losses.py code/tests/test_phase1_fcr_cross_losses.py -v`，结果`9 passed in 4.53s`。
5. `conda run --no-capture-output -n ssr-gpu python -m py_compile code/cvsrffi/phase1_fcr_transplant.py code/cvsrffi/phase1_fcr_losses.py`退出0；`git diff --check`退出0。

## Strict pair and factor routing

`compute_transplant_losses`只接受`pair_valid_mask["fingerprint"]`为真、`fingerprint_pair_index`在范围内、源/目标标签均可见且源目标TX标签不同的Task2严格行。无有效行时`active_pairs=0`，`target_id/preserve_s/preserve_n/same_f/drop_f/total`都是有限精确零，Decoder、re-encoder和分类器均不调用。

对每个有效定向行，生成调用严格为：

```text
decode(source.z_s,target.z_f_id,target.z_tx_state,source.z_n_parts)
```

重编码结果比较分离的源内容、源nuisance和目标`z_f_id`。同TX控制单独以源`z_s/z_f_id/z_tx_state/z_n_parts`重建，检查源身份分类与源指纹恢复；跨TX不会退化为随机shuffle。

## Frozen classifier and gradients

分类器在移植操作中被置为`eval()`，其所有参数为`requires_grad=False`。它只接收生成IQ执行正常身份前向，pair元数据不会传给该前向；可见目标标签只在其后用于CE。测试确认分类器参数无梯度，同时CE梯度到达生成IQ、源内容与目标指纹因子。

## Five components and necessity route

输出组件严格为`target_id`、`preserve_s`、`preserve_n`、`same_f`和`drop_f`。`drop_f`把目标`z_f_id/z_tx_state`全部置零；正确定向路径的指纹残差误差为stop-gradient参考，删除路径未将误差增加到`necessity_margin`时施加正罚项。

删除路径的源`z_s`和每个`z_n_parts`均detach，避免内容或nuisance支路代偿缺失指纹。`freeze_decoder=True`在每次Decoder调用期间关闭Decoder参数梯度、但不detach其输入；聚焦测试确认Decoder参数不获得梯度，而主定向路径仍可回传至目标指纹因子。Fisher门控由显式`fingerprint_residual_error`回调在训练接线时提供；本模块不引入可学习物理特征或shuffle-gap验收。

## Self-review

- 隐藏/无效目标标签、越界索引、同标签cross-TX映射均不能启动生成链。
- 无有效严格pair不会构造替代行或调用Decoder、re-encoder、分类器。
- 分类器冻结不切断生成IQ输入梯度；pair/标签没有进入分类器输入。
- 定向路径为源内容/源nuisance加目标指纹；同TX为独立控制。
- drop-f正确路径只作stop-gradient参考；删除路径不更新源内容/nuisance；`freeze_decoder=True`防止Decoder放大必要性。
- 未把随机shuffle gap或合成夹具视为真实FCR-13严格pair证据。

## Commit and publish

本报告与Task8拥有文件以`feat:add-FCR-directed-transplant`提交。提交后的本地HEAD、push状态和远端OID独立读回在任务完成回执中报告；不将未来OID回写入同一提交，避免制造不稳定的自引用。

## Interfaces for Tasks9/10

Task9需提供把实际FCR Decoder、FCR re-encode和冻结ADV3B02身份前向包装为本接口的薄适配层；不改变本模块的严格pair筛选。Task10只应在有合法`L_s`可见标签且FCR-13能力实际成立的行启用`target_id/same_f/drop_f`，并为`fingerprint_residual_error`接入Fisher门控后的冻结物理指纹误差；缺少真实严格pair时保持本任务的精确零语义。
