# B8 CUDA首差定位与确定性后端验收

原始完整预算执行到E1/reference与head_grad_only两行后，发现两条120步终点TX loss分别5.844169与5.926722。虽然初始单步参数容差预检通过，但不能据此宣称完整轨迹一致，也不能直接归因为D1机制差异或“噪声”。按主任务指令停止本任务PID4320；独立Get-Process确认不存在，原exec退出。保留原JSON及副本`b8_benchmark_cuda_partial_nondeterministic.json`，均只有2行，不是完整速度验收。

新增`code/scripts/diagnose_core90_b8_cuda.py`。在同一真实source batch、scratch初态和每步共同RNG下，逐步比较：模型参数/BN、optimizer矩/step、原点head raw梯度、predictor裁剪、corrector raw梯度、正式裁剪、虚拟与正式位移、Python/NumPy/Torch CPU/CUDA RNG、mask及prototype。容差始终atol=1e-6、rtol=1e-5，没有事后放宽。报告同时记录逐位差异和超容差差异，首个超限保存完整tensor artifact。

未开启确定性时：

- reference对reference，第0步已出现逐位差异：`dom_backbone.dac_b2.res_proj.0.weight`最大差1.862645149e-9。RNG、原点head梯度、head predictor裁剪与位移、mask和prototype完全一致。
- 同一负对照第1步超过原容差：`dom_backbone.f1.dw.weight`的corrector raw梯度差1.100823283e-6；`dom_backbone.time_fuse.0.weight`正式位移差1.072883606e-6。
- reference对D1第2步corrector raw梯度超过原容差。因为同实现负对照更早失败，不能把它解释为D1本身已证不等价。

确定性模式：CUDA初始化前`CUBLAS_WORKSPACE_CONFIG=:4096:8`，`torch.use_deterministic_algorithms(True)`、`cudnn.deterministic=True`、`cudnn.benchmark=False`。E1与E131分别执行reference/reference、reference/D1、reference/graph，各8步（每组累计16步），所有比较项**逐位一致**，不只是满足浮点容差。E131真实source自然伪标签mask、强视图和现有全部阶段目标纳入比较；非空U选择与prototype额外由前述合成功能测试覆盖。

结论限定为：`non-deterministic backend path inferred from same-impl negative control; specific operator UNKNOWN`。已定位首差位置和可复现的后端条件，尚未证明具体CUDA kernel或单个算子负责，不能凭参数名反推责任算子。

实际诊断命令：

```text
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 code/scripts/diagnose_core90_b8_cuda.py --wisig-pkl E:/type10-7/local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl --steps 8 --output local_artifacts/core90_v2/b8_cuda_first_divergence.json
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 code/scripts/diagnose_core90_b8_cuda.py --wisig-pkl E:/type10-7/local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl --steps 8 --determinism on --right graph_reuse --output local_artifacts/core90_v2/b8_cuda_graph_deterministic.json
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 code/scripts/diagnose_core90_b8_cuda.py --wisig-pkl E:/type10-7/local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl --stage 131 --steps 8 --determinism on --right both --output local_artifacts/core90_v2/b8_cuda_e131_deterministic.json
C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8 code/scripts/diagnose_core90_b8_cuda.py --wisig-pkl E:/type10-7/local_artifacts/cvs_publication_inputs_20260713/ManySig.pkl --stage 131 --steps 8 --determinism on --right graph_reuse --output local_artifacts/core90_v2/b8_cuda_e131_graph_deterministic.json
```

四次命令均结束exit=0；首个命令的未确定性失败是被完整记录的诊断结果，不是“测试通过”。两份首差`.pt`与四份JSON均保留。

benchmark已默认采用主任务`runtime.configure_determinism`实现，记录完整后端配置；显式`--allow-nondeterministic`仅用于复现诊断。新增定时区外每repeat终点对照，模型、optimizer、RNG、mask、prototype再次超过原容差时，保存失败artifact并退出。CPU endpoint smoke三实现通过。新的正式预算执行写独立`b8_benchmark_cuda_deterministic.json`，不覆盖旧partial；预热20、计时100步×3、E1/E80/E131、三实现、CUDA同步保持原计划。设备含桌面负载，墙钟纯算法归因仍UNKNOWN，计数可直接报告。
