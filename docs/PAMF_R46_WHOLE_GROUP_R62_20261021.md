# R62：保留归一化边界的整组压缩

状态：VERIFIED / TRAINING，已发布并实际训练，无新识别性能或 GPU 时延结论。完整 56 项针对性检查、CLI 与一次独立 P0/P1 审查通过；另已严格加载指定真实父权重，用合成 IQ 在 CPU 验证两组全部组件梯度有限且非零、父权重不变。实际 IQ、F、TX 标签和源／目标分数读取均为 0。真实两条 L_s 与 F50 的 no-query smoke 在启动器首步执行，PASS 后自动继续。

本轮继续压缩既有 U-Net 接收链恢复器。直接初始化与冻结 IQ 教师都固定为用户指定 R46 `prune_refine350521`，方法 `bbb8daf817de050b958d217b70d4c33f4aefe341`，源域固定 E20/4940。禁止继承 R48/R49/R53/R57/R58/R60 训练后的学生。父权重原物理划分与 original7/63/30、0TX/F/V/target 训练来源已由既有严格 loader 核对。该父版由用户在已暴露的历史工程基准上指定，不称新独立盲测。

现有按通道删减会改变每个 GN4 组内的统计总体。本方法只在 coarse spectral、coarse temporal 和 action 三个上下文 U-Net 中删除完整的一个原 GN 组，保留三组原通道；完整 prior、三阶段、六次 CG 和原有物理约束均保持。组选择只按父 encoder 两个残差块中四个长卷积的输出通道均方权重归一化后相加，再对原四组求和，稳定选前三组并按原顺序排列。初始化不读取 IQ，不拟合参数，也不依据源验证或目标分数选组。

两行具有相同通道索引、初始参数、参数量和训练预算，仅上下文归一化分组不同：

|行|上下文归一化|上下文参数|prior 参数|完整参数|
|---|---|---:|---:|---:|
|group_preserve|GN3，保留原每组通道宽度|136234|112594|248828|
|regroup_control|GN1，将相同保留通道重新合组|136234|112594|248828|

两行的上下文 temporal/action 宽度均为 9/15/24/33，spectral 为 6/9/15/24；prior 宽度为原 12/20/32/44。完整参数较指定父版减少约 29.01%，较现有 273765 参数图再少 24937 个参数；后者只是参数比较，并非初始化来源。prior 的 28 个共享长卷积权重、各 stage 独立 bias 与归一化 affine 均完整移植，仍参与后续联合优化。

对相同的输入特征值，GN3 保留子组的数学映射与原 GN4 对应组相同；已经用 FP32/FP64 的独立层级检查验证。上游卷积仍因删通道而改变，所以不保证整个恢复器等价，也不假定 CUDA 逐位相同。GN1 对照使该归一化差异可在同参数与同初始化下检验；两行归约方式与实测开销不同，不称完全计算匹配。

训练固定单 model seed 2026092701，E20/4940、global256/micro256、direct float32 buffer；重置 AdamW、LR 2e-4 至 2e-5、247 updates warmup、weight decay 1e-4、梯度裁剪 1、FP32/TF32 off。沿用 R07 已生成的 20 个受损视图，6300 条 L_s 与 56700 条 U_s 共 63000 条原训练记录；每行曝光 L126000/U1134000，父／学生 G 各 1260000 次输入，不访问 TX 标签、F 权重／输出、V 或 target 训练。

损失保持原 paired clean IQ NMSE＋0.05 三尺度相对功率 STFT 误差＋线性 1→0.1 的三阶段冻结父 IQ 教师误差。coarse/action/prior 全部联合优化，不新增可学习参数、校准 IQ、分类头、推理分支或额外 G 前向。旧参数继承 80 轮，本轮追加 20 轮；每组件累计 100 轮、组件和 300、串行累计 140 pass，F50 训练成本另计。该训练使用合法源域 clean 配对，属于 TX 无标签源域自监督。

同增量预算的未裁剪控制只读复用 R49 `retain_refine`，方法 `e96397bc3c7036f085317f9cc0e59074fb6ee24f`，不重复训练或加载为教师。它与新行拥有相同父、E20/4940、micro256、IQ 教师日程和原缓存，但通道、参数及归一化不同；只用于已登记的同预算比较。主对照为两行 GN3/GN1，另完整报告相对原 R17、指定父与 R44 的逐景差。

每行固定最终 E20 后，自动执行源域 1800×7、同卡完整成本测量、正式 168000×7 预测及独立 truth-last。正式七景是 clean、郊区 high/mid/low、urban high/mid/low；每行全部预测固定后才连接真值。源分数只作诊断，不据此过滤任一行。原 F50 使用原 jointly trained 默认头，同一个恢复 IQ 交给其 FFT/PA/时域/统计分支，无另训头。所有正负结果、逐 TX/RX/day、Macro-F1、修复／损伤和完整 20 轮曲线均保留，不反馈选模、参数或选择性重跑。

成本分 G 与 G＋F，B1/B32、64 条固定 source V、三轮交替顺序，报告中位/P90/P95、吞吐、驻留／增量峰值显存、训练时间与文件大小。计时为同步预加载 GPU 输入，不含传输、前处理及 loader。Conv/Linear MAC 不含 FFT、CG、norm、activation 和 F，不从参数减少推定实际加速。

资源为 N607 既有环境、物理 GPU 3/2 各一个本任务 worker，总数最多两张；每卡全 owner 少于 2 个已有 GPU PID 且空闲至少 18 GB 才分配。CPU 控制器与 scorer 为 2 线程，R41 score.lock 保证最多一个本任务 scorer。R60 本任务 GPU 进程已全部退出；其他 owner 的健康任务保持运行。正式发布使用独立不可覆盖的 `releases/pamf_r46_group_drop_20261021_r62` 和 `runs/20261021-phase1-restoration-R46-whole-group-s2026092701-m2-r62`。

固定命令入口为 `python -u -m experiments.pamf_r46_group_drop.runner run`；完整参数、Git method commit、环境/CWD、数据与输出路径写入发布登记的 `experiment.json`。启动器第一步执行真实父权重、F50 和两条 L_s 的 no-query smoke，PASS 后立即继续，无额外审批。预期产物为 resolved/comparison plan、smoke、workers/stages、逐行 initialization/training 20 records、固定最终权重、source/target 七景 predictions 与独立 summary、matched profile、assessment、完整报告及独立权重说明。

技术失败保留受影响行产物，不自动重试、停止或热修改健康行；原父来源／数据权限不匹配、非有限损失或输出、严格权重与分组合同不匹配均为实际技术失败。性能优先：原 R17 每景最多下降 1 个百分点只是退化上限，减参、层级归一化等价、源重建或容差通过不等于性能提升。low urban 70%、其他五受损场景距 raw clean ≤5 个百分点及 TX 细节真实目标均保持 ACTIVE。

流程边界：`REJECTED_EXTRA_GATE`——通用 brainstorming 中额外的设计／计划审批不属于项目最小流程白名单，按已有用户授权继续；不新增设计 approval、签名、成员哈希或收据链。完整流水线实现与必要验证完成前不宣称正式发布。


VERIFIED / TRAINING：固定方法 7a8fa7bd836946c3493635cbe0966591ecf14e1e 已发布，普通账户、单归档 SHA 比对和远端 compile 通过；真实指定父/F50/两条 L_s no-query smoke PASS。两固定行物理 GPU3/2 的实际 PID/CWD/argv/CUDA/CPU2、GPU PID、日志增长和至少一轮有限损失已核实。每卡一条本任务训练，总数2；其他 owner 不变。整组 GN3/GN1 同 248828 参数及初值，全 prior、原波形目标、E20/4940 后两行均自动七景正式测试/独立 truth-last/同卡成本，无源分数筛选，无新识别收益结论，完整目标 ACTIVE。
