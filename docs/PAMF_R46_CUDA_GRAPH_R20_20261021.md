# R20：固定权重的CUDA Graph执行优化

状态：LOCAL_VERIFIED，未发布，无新版正式性能结论。

源计时显示R18两版和R19两版的参数减少均未转化为B1加速。本组保持用户原350521父版及R18两个278180版本的全部权重与数学运算，不另训、不调rank，检验固定计算图重放。三行在读取R19成绩前依据源计时固定，不进行目标排序。主对话研究相位/通道结构，本组处理执行开销。

|固定行|完整恢复器参数|继承源最终checkpoint|新增训练|GPU|
|---|---:|---|---:|---:|
|R46_350521_graph|350521|prune_refine E20/4940|0|2|
|R18_factor625_stage_graph|278180|factor625_stage E20/19700|0|2|
|R18_factor625_cond_graph|278180|factor625_cond E20/19700|0|3|

原R46来源commit bbb8daf817de050b958d217b70d4c33f4aefe341，R18来源aca99499d3f0170df71380f6429ac9a01ef412f8。旧权重均按原7/63/30物理源契约、0TX/F/V/target恢复器训练读取核实；图运行不改变其来源与成本。R46各组件累计80轮/串行120，R18各组件100轮/串行140，本组均新增0。F50原seed2026092701/defaulthead只作冻结评测，不进入恢复器训练。既有READY/EXACT_MATCH缓存复用，不重验数据。

每行首步实际2Ls noquery smoke，检查输入替换、旧输出不会被后续重放覆盖、原函数与图函数的IQ/variance/features/logits误差以及状态不变。程序复用原模块参数、反射边界、真实共轭伴随和每stage两步CG，仅把CUDA数值Python检查移到重放后。捕获缓冲区长期保留，每次copy新输入，输出clone保证所有权。原native/其他owner代码保持。

每行固定source1800×7，再完整formal168000×7，B256。所有行无源准确率或速度晋级门槛；每行全部七预测完成后独立CPU truth-last，R06共享score.lock/CPU2/max1ownscorer。对照复用原native七景预测，不重复旧target前向；完整比较IQ/features/logits及argmax。原batch128与本组256可能产生数值差异，逐景报告，不预先承诺目标逐位相同。逐景accuracy/MacroF1、TX/RX/day、净纠错、clean损伤、70%/5pp目标和父0.8pp/原R17历史1pp均单列；已暴露单seed工程基准，不称独立盲测或逐细节TX证明。

同卡64sourceV，对原父和当前候选分别测native/graph完整G及G+F，B1 warm16/64次、B32 warm4/16次，三轮正反顺序。图计时包含新输入copy和输出clone；捕获/预热单列。报告median/p90/p95/逐轮中位/吞吐、完整参数/文件字节/ConvLinearMAC、匹配suite显存及独立单模型G+F五批量graph池内存。输入已在GPU，不含传输/loader/预处理；MAC不含FFT/CG/norm/激活/访存/F，物理3stage/6CG完整执行。共享suite驻留不冒充独立部署内存。

GPU2串行原父与stage，GPU3条件版，本组最多2own/每卡1；实际R19owned句柄退出后、sharedlock下free≥18GB/allowner<2才取得GPU。普通账户、新不可覆盖run/output/release，一次preflight/一次归档两端SHA/一次compile。技术失败保留影响行及缺失测试，无nativefallback/自动重跑/热改或停止其他任务；健康行继续。实际方法commit、命令、环境/CWD、路径、停止规则与产物见experiment.json。

图接口依据[PyTorch2.1 CUDA Graph文档](https://docs.pytorch.org/docs/2.1/notes/cuda.html#cuda-graphs)实现。28项本地聚焦检查通过，其中六项使用真实R46/R18架构零/随机IQ，对原函数逐位比较，并禁止Tensor转Python bool；这些CPU检查不替代真实GPU捕获。唯一独立P0/P1审查发现的主机同步问题已修复，原问题一次定向复核PASS；未重复全审查，真实GPU捕获仍待launcher首步smoke。
