# R65 原权重推理优化

状态：VERIFIED / WAITING_GPU_CAPACITY。归档一致、远端编译、控制器/两worker的PID/CWD/命令/CUDA/CPU线程和日志增长均已核实。当前等待GPU2/3容量，真实2Ls/F50 GPU smoke及新七景结果尚未产生。

固定用户指定 R65 `prune_after10` 的原273765参数，新增训练/optimizer/TX标签/F优化均0。原checkpoint完整来源沿用原7/63/30物理源契约、源E20/4940冻结，加载时严格核对原R65元数据和共享权重别名。当前是已暴露工程基准的用户指定后续优化，不称独立盲测。

|固定行|作用|FFT/IFFT|prior/CG|GPU|
|---|---|---:|---|---:|
|static_graph|保持原FP32运算顺序，图执行与device有效性检查|63|3/6|2|
|residual_graph|同图执行，再复用原始残差和删除最后无用CG状态更新|39|3/6|3|

两版均完整source1800×7和正式168000×7，原F50 seed2026092701/defaulthead，B128、源尾批8/正式尾批64。每行七预测完成后才独立CPU truth-last；复用原R65预测作IQ/features/logits/argmax比较，不重跑旧目标。无源速度或准确率筛选，负结果保留，测试结果不回流调参。

统一 InferenceEngine 同时用于真实预测与测速；每次重放复制当前输入，输出clone，数值有效性在device累计后一次主机判断。原F权重/defaulthead不更新。真实2Ls/F50首步smoke覆盖B1/8/32/64/128、不同输入替换、旧输出所有权和状态不变，通过后立即继续。

同卡source-only测native/static/residual完整G与G+F，B1/8/32/128、3轮交替，P50/P90/P95、吞吐、setup与显存分列。包括copy/clone和有效性检查，预加载GPU输入，不含host传输/loader；正式预测另报告完整产物墙钟。独立G+F五批量累计graph池内存单列，不能冒充单B1部署内存。ConvLinearMAC仍30761512/IQ，不含物理/归一化/F。

物理GPU2/3每卡最多一个本run worker、总2；shared容量锁下allowner<2且free≥18GB才占GPU，CPU线程2、独立scorer最多1。健康任务不变。新不可覆盖run/output/release；技术失败保留影响行、不自动重试、不nativefallback、不停止其他owner。

42项相关检查、CLI及原权重本机合成GPU检查通过，唯一一次独立P0/P1审查PASS。本机合成G B1约76.03→3.23ms仅为本机执行诊断，不替代N607同卡G+F或正式性能。原输入统计、原权重不变，残差复用非逐位一致。

方法commit `780615368d7675331f3d42367b62bf02ea7031b1`。命令、环境/CWD、输入输出路径、六类seed、逐行矩阵、停止规则和预期产物见[experiment.json](experiment.json)。验收分别报告R65对照、固定原R17逐景1pp容忍、lowurban70%及其他五景距rawclean≤5pp，不把容忍/压缩当作全部目标完成。
