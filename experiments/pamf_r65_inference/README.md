# R65 冻结权重推理

仅适用于原 R65 `prune_after10` E20/4940 完整 checkpoint。恢复器仍为 273765 参数、三个 prior、六次 CG，无新增训练或权重。

```python
from experiments.pamf_r46_delayed_prune.runtime import load_checkpoint
from experiments.pamf_r65_inference.engine import InferenceEngine

model, metadata = load_checkpoint(checkpoint_path, "cuda:0")
model.eval().requires_grad_(False)
del metadata

# 默认保持原计算顺序；首次调用为该批量预热并捕获 CUDA Graph。
restore = InferenceEngine(model)
restored_iq = restore(received_iq)[0]
```

`received_iq` 必须是该设备上的 float32 `[B,2,256]`。每次调用复制当前输入，返回独立存储；不同批量分别捕获，首次预热/捕获成本与累计图池显存另计。同一个 engine 的调用按单一执行流串行使用；并发请求使用各自的 engine 和输入输出缓冲区。

`reuse=True` 启用窗内常量方差残差复用，FFT/IFFT 从 63 次减为 39 次，存在 FP32 舍入差异。两版的正式结果分别见[运行报告](../../automation_reports/CV-SincNet/20261021-phase1-restoration-R65-inference-opt-s2026092701-m2-r01/report.md)；尚未评分时不能假定识别决策不变。

`graph=False` 可用于 CPU 或 eager 调试。接入原冻结 F50 时传入 `InferenceEngine(model, original_reader)`，返回 `(restored_iq, variance_mean, features, logits)`；默认头与全部 F 分支保持。正式预测和测速均使用这个入口，不能通过只替换 `model.forward()` 来假定旧 `training_outputs()` 路径也已优化。

图捕获失败或数值异常直接报错，不自动退回原实现。恢复器单独部署不需要 F、教师、源缓存或 clean 输入。独立加载/预处理/传输、服务排队与正式产物写入应另计时。
