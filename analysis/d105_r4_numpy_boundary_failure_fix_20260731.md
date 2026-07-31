# D105 R4 NumPy/Torch边界失败修复分析（2026-07-31）

## 结论

D105 R4在Phase1严格特征抽取的首个批次触发NumPy到PyTorch的运行时边界异常，未生成严格tap、预测或评分工件。R4唯一有效结论仍为STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT；它不构成模型、方法、Target、Target25或性能结论。

本次仅替换D105中审计到的三处NumPy/PyTorch数组桥接点，并补充定向失败注入与闭包测试。未登录N607、未同步文件、未启动或重启运行、未修改R4原始报告。

## 证据与根因边界

|证据|已核验事实|
|---|---|
|R4交接记录|automation_reports/CV-SincNet/d105_phase1_sourceheld_asset_20260731_r1/retrieved_d105_phase1_sourceheld_d23469ba_20260731_r4_STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE_NO_PERFORMANCE_RESULT/handoff.md；SHA256=f362f5051a71d0dd88552a815c2a82680b6157a537ef1ebc36b1d8e720a3811a。|
|原始阶段日志|remote_recovery/pipeline_stage1.log；992B；SHA256=9aa65a3e034283ee481b0bb12792ac6fe8fd6e51fb3b71470abe259de49ffe10。|
|启动前闭包|N607记录为torch 2.1.0+cu121，54/54个runtime文件、legacy checkpoint精确SHA和195个tensor均通过。|
|首个异常|stage2_d105_phase1_bundle.py:1658的torch.from_numpy(batch)抛出TypeError: expected np.ndarray (got numpy.ndarray)。|
|失败后边界|严格tap、预测、评分工件均为0；Target访问为false；运行时文件仍为54/54匹配。|

最窄且有证据支持的根因是N607上torch 2.1.0+cu121与NumPy2.x之间的C扩展数组桥接不兼容。错误显示对象名称仍是numpy.ndarray，但桥接内部缓存的NumPy类型或ABI边界拒绝当前对象。D105原特征回传还使用Tensor.numpy()，属于同一风险面。

静态审计未发现应用代码层面的第二份NumPy或模块重载：启动脚本只暴露一个项目代码根；D105生产链无sys.modules清理、importlib.reload、动态zip/archive导入或运行时替换NumPy模块；启动前后runtime哈希均闭合。本地ssr-gpu为torch 2.10.0+cu128和NumPy2.2.6，不能自然复现N607异常，因此本地验证使用正常桥接字节对照和注入R4同文案异常的方式确认新路径不再调用故障接口；不替代N607技术健康验证。

## 修复范围

|位置|修复|保持的不变量|
|---|---|---|
|code/cvsrffi/stage2_d105_phase1_bundle.py|新增私有IQ桥接helper。它只接受精确np.ndarray、float32、[N,2,T]、C连续且有限的输入；通过torch.frombuffer、reshape、clone和to建立独立tensor并复核输出。|源侧输入形状、dtype、有限性、设备和严格tap流程不变。|
|code/cvsrffi/stage2_d105_query_evaluation.py|Target25批次先规范为C连续float32，再复用同一helper。|Target25输入校验、查询只读和预测流程不变。|
|code/cvsrffi/stage2_d105_feature_tap.py|不再调用Tensor.numpy()；对已验证的有限float32 CPU连续tensor使用tolist，再用np.asarray和np.ascontiguousarray重建数组。|输出[N,width]、dtype、有限性及字节表达保持。|

IQ桥接中的clone是刻意的所有权边界，输出tensor不与输入NumPy数组共享可变存储。输出路径只接受有限float32，Python浮点中间值可精确表示这些值，测试额外覆盖带符号零。

没有修改Phase1选择规则、checkpoint、模型结构、特征hook、K-shot定义、支持/查询物理ID边界、p2_min_v1协议、查询状态更新计数、Target访问权限、方法超参数或四臂比较定义。

## 本地验证

|测试|验证目标|结果|
|---|---|---|
|test_d105_iq_tensor_bridge_is_detached_byte_exact_and_bypasses_from_numpy|与正常from_numpy(...).clone()字节一致；输入后续修改不别名；注入R4异常后helper仍可用。|通过。|
|test_d105_iq_tensor_bridge_rejects_noncanonical_inputs|拒绝float64、错误通道数、非C连续数组和非有限值。|通过。|
|test_target25_tap_rows_bypass_rejected_torch_from_numpy|Target25路径在from_numpy被注入R4异常时仍产出受控特征与receipt。|通过。|
|test_d105_feature_tap_output_bridge_bypasses_tensor_numpy_type_failure|Tensor.numpy()被注入R4异常时，输出桥接仍字节一致并保留带符号零。|通过。|
|6项直接桥接窄测|全部通过。|通过。|
|10个D105测试文件统一回归|216/216通过，约67秒；仅见已有torch.cuda.amp.autocast弃用警告。|通过。|
|54个受控runtime文件py_compile|全部通过。|通过。|
|runtime manifest与method lock加载闭包|验证54个受控文件及两层SHA关联。|通过。|

## 哈希闭包与后续边界

|文件|SHA256|
|---|---|
|code/cvsrffi/stage2_d105_phase1_bundle.py|43145a76fa0780be421601033b95726e1abce886eac7a3727ffe67286bbd23a2|
|code/cvsrffi/stage2_d105_query_evaluation.py|a96fcb37a64d1c967959423ce8e13f4889e19d87b23ebbd5fe966b691cf324ee|
|code/cvsrffi/stage2_d105_feature_tap.py|4aab4febf63fe93fff73e9f78acf240db44cb086e71a7c368e9467d814c78a56|
|configs/d105_candidate_runtime_manifest_20260731.json|8940e05f9fdf92d7735bba1570bb3239ee210313ecbbeffa3511b62e21685425|
|configs/d105_candidate_method_lock_20260731.json|f36a0c6c4ee832b34cd98ed7664ec87707a4dbb1559c7c9b4b05dd13fbf4864e|

两份JSON均为显式UTF-8规范序列化，末字节均为}（125），没有末尾换行。runtime loader实际核验54个受控文件并得到runtime SHA 8940e05f9fdf92d7735bba1570bb3239ee210313ecbbeffa3511b62e21685425；method lock loader得到SHA f36a0c6c4ee832b34cd98ed7664ec87707a4dbb1559c7c9b4b05dd13fbf4864e，且内嵌runtime引用匹配。

本地修复验证不授权重新运行R4，也不把R4转换为成功运行。若主代理批准新的N607技术验证，仍须先经独立审查、Git提交、不可覆盖的新run ID、更新本地报告及专属runner交接；只有完整冻结矩阵的预测和评分工件才可进入性能分析。
