# 浅中层特征自挑战：预登记

当前目标是固定三seed clean平均准确率达到84%。本对话独立探索特征自挑战；另一个活动对话继续matching/sampling/update/transform/directed/sets，不复用或修改其新机制。

三臂CE30、RANDOM_MASK、GRAD_MASK各用2026092701/02/03，从零30epoch/6660optimizerupdates，普通cos160身份头，固定末轮。逐行训练→freeze→168000clean预测→独立truth-last评分→汇总。全部9行预先固定，V不筛选，结果不调参、重排或选择性重跑。此用户明确要求覆盖默认source筛选；不把单seed结果当84%三seed成功。

浅层time_down实际48通道；中层t2.pool实际96通道。每次只遮一层，按optimizerstep交替；每包移除25%通道的完整时间轴。GRAD以正确类别logit对该特征的梯度与激活绝对乘积时间均值排序；RANDOM用同通道数和相同梯度计算后丢弃排序的对照。前5epoch关闭，随后10epoch遮挡CE权重从0增加到0.5，clean CE权重保持1。第二前向重放clean dropout RNG并恢复全局状态，不改变其他行训练轨迹；推理无mask、无排名、无增加参数。

方法受RSC启发，来源：https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3018_ECCV_2020_paper.php 。这是射频浅中层改编，不宣称论文复现或已消除RX信息。训练阶段额外5550排名和前向/模型；纯CE6660前向/模型，遮挡臂12210前向/模型；成本实测。

数据沿用父CE30的实际ManySig7/63/30及physicalIDs：L6300、U56700关闭、V27000仅健康。身份训练无checkpoint/teacher/EMA/optimizer继承，无显式CFO标量、卫星增强、辅助网络。原测试已暴露，报告保留背景，不称新的独立盲测或实际卫星部署。

代码/config身份是一个Gitcommit，发布package从release_commit.txt记录execution_commit。精确矩阵、六类seed、数据路径、输出和命令在experiment.json。CWD=独占N607release root；环境=现有CVS-RFFI Python、fullFP32；owner=本goal。GPU由全owner容量锁选择，每GPU总进程最多3；目前只读preflight普通用户/主机/项目/磁盘/GPU已核实，正式归档一次SHA对比和远端compile后运行一次真实checkpoint无querysmoke，PASS直接dispatch。

技术停止：run自身data/query越界、输出冲突、错误checkout、nonfinite/不可启动或预测评分闭合失败；首错停止新调度，健康row完成，保留partial，不性能早停、不盲重试、不干预其他owner。

预期产物：sourceidentity/freeze/config/完整step和epoch日志；固定prediction和complete；同rowaccuracy/F1/confusion/RX/day/TX；runanalysis、finalreport和completion。未全部评分前不能声明完成或84%达成。

本地真实CUDA检查通过：浅层/中层有限非零梯度、精确遮挡数、随机对照复现、CPU/CUDA RNG恢复、逐包独立推理、推理无hook/状态改变、非法U和配置拒绝。唯一独立P0/P1审查未发现问题，依赖发布范围齐全。
