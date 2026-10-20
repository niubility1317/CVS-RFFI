# R58 冻结推理的系数缓存原型

状态：LOCAL_PROTOTYPE，21 项检查通过，尚无 GPU 提速或正式预测一致性结论。当前 R58/R59/R60 发布代码、权重、矩阵、进程和默认推理路径保持原计划；本文件不新增训练候选或实验启动。

已测得原 R58 恢复器在同卡、64 条 source V、三轮计时的 B1 中位耗时约 120～121 ms，指定 350521 父约 45 ms；接原 F50 后约 131～133 ms，父约 56 ms。这是成本证据，不是正式识别结果，也不能单凭代码推断某个算子占用了多少 GPU 时间。

源码显示每个 `StatisticGroupNorm.forward` 都对冻结的 `stat_parameters[[0,2]]` 执行索引、clamp 和 exp。168 个位置重复计算不依赖当前 IQ 的斜率；mean/variance、中心和对数方差仍依赖输入，不能缓存。本原型只缓存原表达式产生的斜率，余下算术顺序与公式逐行保持原样，不使用近似、原生 GN 替换、低精度、蒸馏或新训练。

可选入口为 `prepare_frozen_inference`，默认复制模型以保留原实例。须先用原 R58 `runtime.load_checkpoint` 严格加载，并在最终 device/dtype 上调用 `.eval().requires_grad_(False)`，然后构建缓存。正常 `load_state_dict`、参数复制或 device/dtype 改变后须刷新缓存；训练状态拒绝使用。缓存来自权重，新增 IQ、source/target/F 读取、校准和 optimizer 更新均为 0。

完整参数仍为 276453。FP32 额外缓存为 168 个非持久 Tensor，共 1344 个标量、5376 字节，另有 Python 模块元数据；不把缓存隐藏成零运行状态成本。缓存不进入 `state_dict`，原 checkpoint 字段和原类严格重载保持兼容；加载或设备变换后按权重重新建立，不能只搬运旧设备算出的缓存而假定跨设备逐位一致。

5376 字节只描述缓存 Tensor。默认 `clone=True` 会同时保留原实例和一份完整模型；另一份 FP32 参数为 1105812 字节，还须计入其原有 buffer、Python 元数据和准备时临时分配。`clone=False` 在原实例上准备缓存。上述数字不是实际 GPU 驻留或峰值显存，后续计时须按真实持有模型数量报告。

最初 21 项本地 CPU 检查使用 Torch 2.10.0+cu128、2 线程和合成 IQ，未读取真实 IQ、checkpoint、source/target 分数或 F。层级 FP32/FP64 输出及输入梯度、完整图输出、portable state 与缓存刷新等检查通过。CPU dispatcher 记录显示，完整前向的 `aten::index` 从 168 到 0、`aten::clamp` 从 337 到 169、`aten::exp` 从 339 到 171；这些是高层算子调用数，不是 CUDA kernel 数或加速倍数。

R58 正式评测完成并交付权重后，另对 calibrated、identity_control 两套真实源训练权重执行 CPU no-query 验证：严格加载固定 E20/4940，使用零输入 B1、合成高斯 B2 和小幅合成高斯 B1。两套模型的三种输入均有限且与原实现逐位相同，portable state 和原模型权重均保持一致；实际 IQ、F、TX 标签与源／目标分数读取均为 0。这里验证的是实际权重在合成输入上的 CPU 一致性，不是实际测试 IQ 上的一致性。

即使 CPU 逐位相同，也不能据此宣称 CUDA 逐位相同、正式准确率已验证或实际提速。GPU 计时和适用的正式预测对照仍须按最小流程固定预算，等待现有 GPU 任务自然完成；原性能、70%/5pp 和 TX 细节保真目标保持 ACTIVE。
