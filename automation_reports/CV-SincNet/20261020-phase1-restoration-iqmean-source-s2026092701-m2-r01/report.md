# 冻结恢复器的固定 IQ 平均

状态：PLANNED。用户目标保持为目标测试集 low-urban ≥70%，其他五环境接近原 clean 85.113889%（Agent 暂按差距≤2个百分点解释）。分别报告恢复 IQ 与同条 clean 的差距，以及接入冻结原 F/default head 后的真实 TX 识别。源分数和波形误差不能替代目标证据。

源依据是已完成 posterior03 与两条 joint30 的错误互补。1800 条源 V 的 low-urban，joint30 final 相对 posterior03 纠正/损坏为 50/49，layerwise 为 47/50；mid-urban 分别为 29/36、31/36。单独替换抵消了纠错，未通过旧规则。固定平均 IQ 是否保留互补收益尚未验证；真值选择上界不是部署收益。没有根据目标分数选择这些成员或权重。

两个新候选 mean_final/mean_layerwise 均在同一 received IQ 上计算 0.5×posterior03 输出＋0.5×joint30 final/layerwise 输出，保留原 IQ 坐标，不作 oracle 相位、增益或时延对齐。权重在本轮任何预测前固定，不拟合门控、选择器或分类头，不平均分类分数/teacher 坐标，不投票。只把新的 IQ 输入原默认头。单个失败成员保留原失败状态，新组合独立预测和源筛选。

成员是已完成、来源同一合法63000条 L/U 的 checkpoint：posterior03 为 de3118c724026869581a9cf1975b0d1314c96d6e 的固定 E10/2470，joint30 为 672a36de61da32f3974472db97448b8977bbae71 的固定 E30/7410。两者均无目标训练/选模或 L/U TX 真值读取，原 F/default heads 冻结，来源/预算/角色在一次预检核实并由原严格 loader 加载。原 VALIDATED_ONCE 数据、F 原生等价和来源证据复用，不重验数据、不增加哈希链。F 提供既有监督知识，不能称整个系统完全无标签。

本轮新增训练轮次、参数更新、L/U 优化曝光及新增可训练参数均为0，不生成新 IQ。每个组合需两个冻结的2140840参数恢复器，合计4281680参数；模型推断次数由1增至2。成员各自历史携带60/80组件epoch，扣除共享原coarse20后为120个唯一历史组件epoch；各成员串行继承50pass，每个模块不超过50轮。不是免费模型或等成本单网对照。实际评估耗时/显存包含R03基线和原头对照前向，不能把评估总耗时称作纯部署延迟。

只使用同一1800条源V、原F1、clean＋六practical。每臂七视图预测和恢复IQ固定后，独立无CUDA CPU scorer才读取V标签及paired clean评分。相对已完成posterior03，要求low/mid urban各提高至少0.5个百分点、其他四环境最多下降0.5个百分点、相对原clean下降最多0.1个百分点；六环境平均NMSE≤1.15×原feature参考0.028326367493718863。界限不复利。通过者按源urban均值最高、NMSE更低、登记顺序选定；都失败则不访问新目标。

仅选定通过臂自动测试原168000条不可变目标、clean＋六practical、三套原F50 CROSS_RX seed2026092701/02/03。三默认头独立argmax，非投票。全七视图恢复IQ/物理ID/决策完成后，独立无CUDA CPU scorer才接入目标truth和评分clean。practical推断只见当前received IQ，不读取paired clean/其他视图/目标真值，不更新状态。报告accuracy/Macro-F1、逐F/TX/RX/day、修复/损坏/净修复和距原clean差值，以及full256/prefix232/tail24 NMSE、log FFT power误差的均值/中位数/P90。一个继承训练seed、两个恢复成员、三套既有F读出，不称三恢复器种子、新盲测或在轨验证。

28项聚焦检查和唯一一次独立P0/P1审查PASS，无发现。N607普通账号，一次预检时GPU4/5各24124MB空闲、无PID。固定GPU4 mean_final/GPU5 mean_layerwise，每卡最多一条本run worker，获取时计入所有owner，进程<2且空闲显存≥12GB；CPU2线程，源scorer最多1个。不停止/重启/热改其他健康任务。新run/output/release不可覆盖，一个代码配置commit、一个发布归档本地/远端SHA比较和远端compile。launcher第一步两条真实L_s IQ＋全部成员＋三套原F的no-query smoke，验证精确IQ平均、全参数冻结和原默认头有限输出，0V/0TX/0target/0update，PASS立即继续预测。随后核实实际PID/CWD/命令/GPU/日志增长。非法输入/来源/代码、输出冲突、非有限、OOM或预测缺失按预登记技术停止规则保留产物并停止，不自动重试；具体命令、环境、路径和预期产物见experiment.json。
