# Sinc 原始 Hz 与居中缩放坐标对照

状态：LOCAL_VERIFIED，尚未发布，尚无本轮真实训练或目标性能结果。唯一 run：`20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01`；完整矩阵和配置以 [experiment.json](experiment.json) 为准。

## 源证据与假设

已读取九个真实源训练模型：LEGACY50 的 32＋18 配方、OLD_GATE_COS50 和 TIME_MS50 的 33＋17 配方各三种子，分别核验来源和固定终轮。其学生 Sinc 滤波核初末相对 L2 不超过 2.40711699e−8，48 个 Sinc 参数都包含在优化器中且累计更新计数 11100，47 个参数有非零一阶矩。最终保存矩的自适应更新项代数估计小于各坐标半个 float32 ULP；这不是每一步真实尝试更新的日志。固定学生的原生 EMA 数值重演能够精确复现九个保存教师的滤波核。证据、边界和历史结果见主报告 `docs/CVS_PHASE1_FEATURE_REVIEW_20261014.md`。

假设：直接以大幅 Hz 数值存储可训练参数使普通 AdamW 的更新难以改变真实滤波器；居中缩放坐标可能恢复其学习。滤波器实际变化只证明机制工作，不能自行证明泛化提升或 Sinc 是瓶颈。方案根据上述源证据选择，未依据 TIME 目标分数，也不继承任何 TIME 权重。

## 唯一改动与公平对照

RAW_HZ 继续使用实际原始 Hz 参数；CENTER_25K 使用 `physicalHz=initialHz+25000*coordinate`，初始 coordinate 为 0。两个方案调用原生 SincConv1d 的合成、FP32、采样率 25 MHz、24 通道、79 点核、边界和卷积路径。25000 Hz 在发布前固定，依据源频率单位及原生 50 Hz 最小带宽的尺度选择；不扫描该参数。

居中缩放同时改变 AdamW 权重衰减作用的坐标和 EMA 的数值算术。实验检验的是这一完整参数化方案，不能将差异全部归于学习率缩放或精度单一原因。RAW_HZ 和 CENTER_25K 都保留 48 个 Sinc 可训练参数，并各增加 48 个不可训练 float32 原始值 buffer（192 字节）。两个方案的原始滤波器、整体 joint160、默认 logits、其他初始参数和构建后的随机状态完全一致。缓存按真实 leaf 参数和 buffer 版本刷新，避免计算得到的 Hz 张量版本恒为 0 时复用旧滤波器。

身份网络参数保持 177025，删除旧类别 W 后实际 fθ 为 176065；没有新分支、额外损失或 TIME 改动。原始时域、频域、PA、参考响应、归一化、融合 gain、NoExplicitCFO 和逐包内部补偿保持。新增坐标进入真实 `joint160/id_feat_joint`。默认 ResidualPhysicalHead 的 6×160 类别 W、scale 30 和联合 CE 学习保持；无单独头训练、冻结换头、RX 分类器、原型或支持集优化。

三组各三种子：OLD_GATE_COS50、RAW_HZ、CENTER_25K。OLD_GATE_COS50 来自原 M12 run 的对应源 33＋17 scratch50 checkpoint 和原始默认头决策，严格核验后复用，仅用于对照及无查询 smoke，不能加载为新模型初始化。主候选是 CENTER_25K；三组固定配对差为 RAW−OLD、CENTER−OLD、CENTER−RAW。不能在结果返回后改换主候选。

## 数据、训练与源诊断

源契约保持 L_s=6300、U_s=56700（TX 真值隐藏）、V=27000，split392005、源 RX 1/3/4/6/8、源 day1/2/3、原六 TX 和全部物理 ID 不变。L_s 占训练池 10%；V 无梯度，也不更新 EMA 或选模。复用既有 VALIDATED_ONCE 数据，不新增数据检查或 hash 链。

每个新模型从零训练 50 轮、11100 次更新，33 label＋17 pseudo；历史 batch_neighbor 门控、原生 EMA 0.999、普通 cosine、AdamW lr0.0002→0.000001、weight_decay0.0001、label_smoothing0.01 保持，训练无星地增强。嵌套历史参考配置中的旧轮数仅为来源元数据，实际扁平 args 和固定预算必须核验为 50/33/17。终轮固定，不按 V 或目标选 checkpoint。

每轮从实际学生参数计算 Sinc 物理边缘和滤波核变化，不读取 IQ。第 33/34 轮分别追加一次源 V 分 RX 健康诊断，恢复模式及 Python/NumPy/Torch/loader RNG。伪标签 RX/day/预测类直方图与强学生—弱 EMA 一致性，仅从已有门控前向取值，不额外执行 U 前向，也不读取 U TX 真值。上述诊断无参数拟合。

先运行种子 2026092701 的两个新模型。两者最终 222 次更新的 labeled CE 均 ≤0.5，且 CENTER_25K 初末实际 Sinc 核相对 L2 ≥1e−4 后，才释放另两个预登记模型种子。阈值以源历史近静止核为机制反证依据，固定后不改。若失败，保留首种子全部实际训练、自动评分和缺项，停止释放后续种子；V 和目标分数不参与扩种子、排序、停止或选择性重跑。所有已训练且合法冻结的 row 自动测试。

## 默认头预测与独立评分

clean-only 固定 ManySig 168000 包，旧六类、七个目标 RX、四日，既有 opaque ID 和 IQ。每个 row 先严格加载自己的固定终轮 checkpoint，确认 source_frozen、来源和原始执行配方，通过零 IQ smoke，冻结模型和 buffers，直接使用 checkpoint 内默认头逐包 argmax。不训练新头、不使用目标统计、不按批次类别配额或全局重排。预测和来源产物全部固定后，独立 scorer 才连接 truth。

计划 9 个默认读出、1512000 个决策、162 条 overall/RX/day/TX 混淆矩阵指标。报告三种子均值和样本标准差、逐种子配对差及正差种子数、Macro-F1、最差 RX/day、修正错误与引入错误。84% 主目标仅由 CENTER_25K clean 默认头三种子均值验收；未完成三个种子时不声称完整目标结果。此为已暴露研究基准，不声称新盲测、卫星部署或新 TX 能力。

## 成本、验证与执行

6 个新模型合计 300 model-epochs、66600 次更新；历史三个对照 150 model-epochs 单独列为复用成本。额外源 V 健康诊断 12 次、324000 包；每模型 50 次只读核计算，U 额外前向为 0，单独分类头更新为 0。实际时间和吞吐在完成后报告。

本地检查已 PASS：初始全网函数和参数/RNG 等价，48 参数实际优化器成员、联合 CE 梯度、CENTER 的数值更新和 EMA/缓存刷新，删除/改变 W 不改变 z，feature-only 导出逐元素一致，strict load 后缓存恢复，源健康诊断恢复 RNG/模式，以及配置/矩阵/冻结/源条件负例。合成更新只证明接线和算术，不冒充真实训练结果。一次独立 P0/P1 审查已 PASS，未发现直接影响下次运行的缺陷；不新增设计审批、seal、签名或报告审查。

代码/配置以发布归档实际 Git HEAD 为唯一身份，由 `release_commit.txt` 和 dispatcher 登记；无需自引用文件 SHA。提交、push 后独立比较上游 OID。精确入口：`python -u -m experiments.cvs_sinc_coordinates.dispatch`；CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_sinc_coordinates_20261014_r01`；环境：普通 N607 用户、既有 CVS-RFFI Conda Python、full FP32，Torch OMP/MKL/OpenBLAS 线程 2。

输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-sinc-coordinate-oldgate-cos-e50-clean-manysig-m9-r01`。每 row 独占根/owner，已存在时核实而不覆盖。共享锁按所有 owner 统计，每 GPU 最多 3 个实验进程且至少 12000 MiB 空闲，具体 GPU 由 launch.json 记录。发布一次归档 SHA 本地/远端比较、远端 compile 和真实历史 checkpoint 无查询 smoke；PASS 后立即 dispatch。随后独立核验 PID/CWD/argv/GPU/log 增长。

所属输入/来源、隐藏 U、预算、非有限值、预测或评分技术失败会关闭新任务调度，保留正在健康执行的 row 和所有产物，不自动重试、不操作其他 owner。首种子源反证失败同样保留实际结果并明确 PARTIAL，不能标完整矩阵。运行后在同 run 更新实际状态、曲线、数值参数/核/EMA变化、joint160 边界及正负结果，全部相应实现和聚合报告提交并推送 Git。权重、原始日志、IQ、预测留在既定远端。
