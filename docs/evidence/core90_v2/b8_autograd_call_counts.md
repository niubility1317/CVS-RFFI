# B8独立非计时调用计数

实际运行`count_b8_autograd_calls.py`，在CPU使用同一合法真实source batch90、scratch初态、固定E1/E80/E131上下文，逐一执行三个实现各一步。包装torch.autograd.grad和Tensor.backward API，记录直接调用栈、create_graph及输入参数张量数。未修改正在进行的CUDA计时，没有复用时间计数替代实际API观测。

|阶段|实现|solver field_evaluations|autograd.grad API总数|create_graph=True|objective内部grad API|额外显式HVP|FISHR loss|
|---|---|---:|---:|---:|---:|---:|---:|
|E1|reference|2|2|0|0|0|0.016725262627005577|
|E1|head_grad_only|2|2|0|0|0|0.016725262627005577|
|E1|graph_reuse|2|2|0|0|0|0.016725262627005577|
|E80|reference|2|2|0|0|0|0.016725262627005577|
|E80|head_grad_only|2|2|0|0|0|0.016725262627005577|
|E80|graph_reuse|2|2|0|0|0|0.016725262627005577|
|E131|reference|2|2|0|0|0|0.016305819153785706|
|E131|head_grad_only|2|2|0|0|0|0.016305819153785706|
|E131|graph_reuse|2|2|0|0|0|0.016305819153785706|

三阶段均观测到reference请求185个参数张量→185个；D1请求4个head参数张量→185个；graph同样4→185，第二次直接对保留图与隔离head求梯度。这里185/4是张量对象数，不是标量参数数量。Tensor.backward API总数均0。采样分项遥测本次关闭，开启时的额外API调用单列，先前telemetry smoke已验证。

当前`legacy/losses.py:fishr_logit_gradient_variance_loss`是已有解析logit-gradient proxy，使用softmax(logits)-one_hot及域内方差。其loss非零且原梯度保留，实际没有内部autograd.grad/create_graph调用；不能把它误称为漏计的高阶API调用，也不能因API数量0而声称该loss被删除。

`field_evaluations`是求解器场评估数量；本次独立测得的autograd.grad API总数恰好也为2，两个指标含义不同。2不是所有自动微分节点、CUDA kernel或底层backward算子总数，后者本次未计量。没有额外显式HVP。

模型前向：E1/E80 solver reference与D1各2，graph为1；E131分别4/4/2。JSON原`model_forward`读自ctx.forward_calls，E131包括prepare_context的一次teacher前向；新增prepare_teacher_forward和solver_model_forward明确拆开。真实source自然伪标签选择为0，非空U与prototype覆盖由已有独立单测承担，本计数不夸大其激活。

命令与证据：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 local_artifacts/core90_v2/count_b8_autograd_calls.py
# exit=0，9行；详细调用栈与loss见b8_autograd_call_counts.json。
```
