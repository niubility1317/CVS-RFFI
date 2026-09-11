# B8 V2本地实现与验证

最终状态：VERIFIED。确定性CUDA真实source固定上下文benchmark已完成E1/E80/E131×reference/D1/graph全部9组，保留每组20步warmup＋100步计时×3次原预算（2700个计时更新，含warmup共3240个更新）。每阶段计时前单步等价检查通过；D1与graph共18个120步重复端点均通过原atol=1e-6、rtol=1e-5的模型、optimizer、RNG、mask、prototype全状态比较。端点检查在计时区域外进行。该结论不替代正式训练、跨数据泛化或长期收敛验收。

支持范围：本轮CORE90数值与成本验收为FP32，scratch-only、合法source。graph_reuse与启用GradScaler或CPU/CUDA autocast组合显式拒绝，不静默采用未经验证的AMP路径；拒绝检查在benchmark启动后补入，FP32数值路径不变。本轮也不声称D1完整CORE90 AMP数值等价已验收。未访问target、远端或旧checkpoint，未运行正式训练。

## 原预算CUDA成本

设备为RTX5070Ti，PyTorch2.10.0+cu128；启用deterministic algorithms、cudnn.deterministic，禁用cudnn.benchmark，CUBLAS_WORKSPACE_CONFIG=:4096:8；matmul TF32关闭、cudnn TF32开启。batch=90，同一合法WiSig source batch、阶段、scratch初态和RNG，每个repeat重置。source角色计数L_s=6300、U_s=56700、V=27000。以下为CUDA同步后的每步墙钟中位数及全部3次范围，telemetry_interval=0。

|阶段|实现|秒/步中位数|3次范围|allocated峰值MiB|增量峰值MiB|reserved峰值MiB|模型前向/步|头前向/步|
|---|---|---:|---|---:|---:|---:|---:|---:|
|E1|reference|1.264744|1.197943–1.389722|643.585|546.503|710.000|2|2|
|E1|head_grad_only|1.179021|1.155499–1.179179|640.637|543.555|710.000|2|2|
|E1|graph_reuse|0.766708|0.755705–0.769824|640.525|543.443|710.000|1|2|
|E80|reference|3.531112|3.091145–3.708622|643.585|546.503|712.000|2|2|
|E80|head_grad_only|2.928275|2.891230–3.073936|640.637|543.555|712.000|2|2|
|E80|graph_reuse|2.060336|1.936565–2.082198|640.525|543.443|712.000|1|2|
|E131|reference|3.895713|3.539948–3.927557|909.440|811.987|928.000|4|4|
|E131|head_grad_only|3.106831|2.992918–3.198053|906.491|809.039|928.000|4|4|
|E131|graph_reuse|1.924628|1.909467–1.929544|906.379|808.927|928.000|2|3|

E1：D1观察耗时下降6.78%，graph观察耗时下降39.38%。

E80：D1观察耗时下降17.07%，graph观察耗时下降41.65%。

E131：D1观察耗时下降20.25%，graph观察耗时下降50.60%。

墙钟归因保持UNKNOWN_SHARED_LOAD：设备运行桌面和其他应用，重复顺序固定，期间有未计时CPU检查；观察耗时差不能直接归因为纯算法提速。allocated为本进程PyTorch分配峰值，incremental扣除相应warmup后的baseline；reserved受跨组缓存历史影响，不能等同活跃tensor需求或全系统GPU显存。原始JSON保留每个repeat的baseline、峰值、计数与耗时，未用CPU smoke代替本表。

每步field_evaluations均为2。独立CPU API拦截另证实9种阶段/实现组合各调用torch.autograd.grad两次，Tensor.backward零次、create_graph=True零次、显式额外HVP零次；reference目标参数张量数185→185，D1/graph为4→185。FISHR沿用解析logit-gradient-variance proxy，未在objective内部另调用autograd.grad。这里统计API调用，不能解释成引擎节点数或CUDA backward kernel数；详见[b8_autograd_call_counts.md](b8_autograd_call_counts.md)。graph减少完整模型前向，仍保留完整目标及两次场求导。

所有9组FISHR为正；自然伪标签选择为0且prototype空状态，不能用该benchmark声称活跃U/prototype的实际成本。E131保留强视图前向路径；有效mask与非零prototype数值覆盖来自独立受控单测。E1/E80/E131是同初态的固定阶段上下文，不是连续训练轨迹。低频分项telemetry额外reverse成本单列，主表未启用；CPU telemetry smoke只证明计量路径可运行，不给出稳定GPU遥测开销估计。

非确定性原始两行结果保留为PARTIAL，不并入本表。同实现负对照首次出现后端数值漂移，具体算子UNKNOWN；确定性ref/ref、ref/D1、ref/graph在E1和E131各8步全状态逐位检查通过，随后完成本表原预算。详细首差和停止证据见[b8_cuda_determinism_diagnosis.md](b8_cuda_determinism_diagnosis.md)。

正式证据：[完整9组JSON](b8_benchmark_cuda_deterministic.json)、[逐步确定性诊断](b8_cuda_determinism_diagnosis.md)、[独立API成本](b8_autograd_call_counts.json)。

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 code/scripts/benchmark_core90_b8.py --device cuda --wisig-pkl E:/type10-7/local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl --implementation all --stage all --warmup 20 --steps 100 --repeats 3 --batch-size 90 --output local_artifacts/core90_v2/b8_benchmark_cuda_deterministic.json
```

## 实现与原始CPU功能验证记录

下面保留实现说明和既有CPU验证，CUDA最终状态与成本以上文为准。

实现文件：`code/cvsrffi/game_tracking/solvers.py`、`head_lookahead.py`、`code/tests/test_game_tracking_b8_fastpath.py`、`code/scripts/benchmark_core90_b8.py`。runtime/config/step_context由主任务集成。

API：`GameSolver(..., b8_impl='reference'|'head_grad_only'|'graph_reuse', telemetry_interval=0)`；`step(..., predictor_grad_scope=None|'all'|'head_only')`。默认reference保持旧路径；checkpoint保留实际b8_impl并拒绝实现不一致恢复。`Core90ReusableGraph(objective,ctx)`是完整目标callable；graph模式必须传入该明确契约，无契约显式NotImplementedError。

D1仅限制原点autograd目标为adv_head，完整前向、拼接batch、Dropout、BN和U强视图不变。虚拟预测仍使用原optimizer的head-only裁剪、AdamW矩和weight decay；回滚后正式更新一次。零对抗测试明确检查头无梯度/无optimizer状态/无weight decay。

图复用保留原完整目标图，捕获实际adv_head输入与头前RNG，深拷贝头及其所属optimizer状态，虚拟更新隔离副本；替换原clean对抗CE，保留所有其他loss；对在线非头与隔离头分别求梯度并按身份映射。活跃图期间不修改在线参数。测试没有切eval、删除FISHR/U/prototype或添加展开高阶求导。现有FISHR实现是logit-gradient-variance proxy，不误称其为新HVP。

低频`result.telemetry`包括各参数角色origin/raw/clipped梯度范数、正式/虚拟位移、formal_clip_ratio；使用wrapper时增加完整当前目标的online/predictor对抗梯度余弦、非对抗梯度变化。分项诊断额外reverse次数单列；零对抗时没有对抗梯度的余弦为None。普通lambda缺少分项契约时明确UNAVAILABLE。`solver.last_trace`保存采样步逐参数梯度、裁剪、虚拟/正式位移，供测试对照，不写入checkpoint。

实际命令与结果：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 -m pytest code/tests/test_game_tracking_b8_fastpath.py -q
# TDD初次：3 failed，全部因未实现b8_impl构造参数。
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 -m pytest code/tests/test_game_tracking_b8_fastpath.py code/tests/test_game_tracking_solvers.py code/tests/test_game_tracking_field.py -q
# 最终聚焦测试51项全部通过（B8=16、solvers=19、field=16），exit=0。仅既有autocast弃用警告。
git diff --check
# exit=0，仅Git换行转换提示。
```

覆盖：小模型BN/Dropout多步AdamW精确状态、逐参数raw/clip/update；真实CORE90网络和完整实际objective在受控合成输入E1/E80/E131下，两种新实现×零/非零对抗；FISHR、prototype明确非零，E131强视图mask显式固定选中；参数、BN、optimizer、prototype、EMA、CPU RNG和mask检查。D1及零对抗要求逐位一致；graph非零对抗使用事先给定atol=1e-6、rtol=1e-5。另含graph三步已有AdamW矩、分项梯度遥测、注入异常后参数/矩/BN/RNG完全回滚。阶段状态为功能检查，未伪装成从E1训练到E131。

CPU脚本功能smoke：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 code/scripts/benchmark_core90_b8.py --synthetic --device cpu --stage 131 --warmup 1 --steps 1 --repeats 1 --batch-size 12 --output local_artifacts/core90_v2/b8_benchmark_cpu_smoke.json
```

结果：reference=1.0797s、head_grad_only=0.8802s、graph_reuse=0.5895s；模型forward=4/4/2，头forward=4/4/3，field backward=2/2/2。该smoke仅验证脚本可运行与计数，不用于速度结论：单次单步、合成数据、CPU共享负载，墙钟归因UNKNOWN；该smoke的自然伪标签选择0、prototype空状态，完整有效U/prototype覆盖来自上述独立单测。

另实际执行同命令加`--telemetry-interval 1`，输出`b8_benchmark_cpu_telemetry_smoke.json`，三实现均通过；分别1.3478/1.2545/0.9575s，每步额外telemetry backward=4。它验证独立计量路径，仍不构成稳定开销估计。
