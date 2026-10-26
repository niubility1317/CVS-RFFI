# 单视图 REPL：用 L 真标签约束 U 星地特征的类别关系

当前状态：真实 E20 联合 backward 检查已通过，唯一源学生已在 GPU7 开始训练。以下初次登记和容量排队段落是历史状态；最新实际证据见文末。源收益及目标收益均尚未确认。

本源域小试验只增加一项：用已有两次 L raw 前向的 joint160 特征作停止梯度的真实类别锚点，约束原有两次 U 星地 physical 半批的 joint160。高置信 U 的伪 TX 与 L 真 TX 相同、RX 不同时构成正对；其他 L 真 TX 为负对；同 TX 同 RX 排除。低置信 U 不构造 TX 类别关系，原低置信 H 同包项保留。全 L 真标签 CE、As KL0、原 ordinary 浅中/joint 跨 RX、原默认头联合训练与 U raw95/权重6/cap1 保持。

该方向来自已完成源域分析：提高统一权重常被 cap1 限幅，低置信 H 视图存在共同错误；新增 pair 本身没有独立源收益证据。这里检验让高置信 U/H 直接使用可靠 L 类别关系，能否在跨 RX 拉近同类时保留其他 TX 的分离。它不是声称这些原因已经解释目标集表现，也不把 source V 精度当成实际 U 正确率。

采用 L2 特征归一化、温度0.2；对有效query计算logsumexp正负分母减平均正对分数及log正对数，按整个physical半批平均，再对两H平均，乘6×0.05/1.25加入原pending U loss，继续原完整L梯度保护。L raw keys detach，新增loss仅对U/H部署特征回传；新增损失对最终分类权重的直接导数应为0；原head内部融合norm/gain仍可接收特征梯度，原CE/pair继续训练既有head。两份L128只是同一批真L的两次既有特征曝光，不是256个新增物理样本。复用原前向，不增加网络、分类头、EMA或推理参数。

原完整 PAMF、单 SIG_TIME、seed01，仅从自己的同契约普通 PL scratchE20 开始+10+20到E50/11100（13L37LU）。匹配参照为已完成H2LU source4view，只读其配置/分数，不读它的权重来初始化，不重训对照。V27000×raw/lowSub/midUrban/lowUrban冻结评估，不使用G或目标输入。源门槛在运行前固定为raw≥26568、H3≥62109：较H2LU的26554/62190，raw至少多14，H3最多少81（0.1pp）。未通过不扩种子或目标。通过也不等于达到目标，后续需单独登记同一方法三种子确认。

当前三种子确认的两名健康学生保持原配方、预算和自动测试计划。本试验独立run/output/owner，按全部卡的全球CUDA+pending最多2且free≥10000MiB调度；不停止其他任务。具体代码身份、环境/命令/CWD、数据与祖先、唯一矩阵、资源/停止规则、产物见[experiment.json](experiment.json)。33项focused检查及编译已通过，唯一独立P0/P1及一次直接问题定向复核均PASS；真实原E20联合backward smoke是launcher第一步，PASS立即训练，当前尚未实际启动。


本地验证完成，见[验证范围](local_verification.json)。初次P0/P1发现原头中的base_norm/pa_norm/gain也是部署特征融合参数，不能要求整段cls_head辅助梯度为0。已将零导数检查限定到最终cls_head.weight，特征融合参数归入other_features，并增加真实可微融合回归；唯一一次定向复核通过。生产损失、标签/mask/forward/预算/源门槛没有改变。

停止Lkey特征梯度只限制新增项通过这些特征张量的回传；原L全部CE和骨干联合训练继续。分类权重不直接参与新增特征loss，不意味着经过原L/U组投影后的实际参数更新与参照完全相同。当前未启动；不能把本地通过写成真实权重检查或源收益已完成。


## 首次发布容量检查未通过，改为显卡空位队列

R01 发布在GPU容量预检处因无进程空位退出。已独立核实run目录、release目录和controller提交文件均未创建；没有启动训练或模型检查。已上传R01 archive及本地package/runtime保留，没有重复覆盖或把上传当成执行成功。见[失败层与后态](prelaunch_capacity_evidence.json)。

R02使用新的release/archive目录，仅允许不占GPU的CPU控制器先进入队列；每个真实smoke/训练worker仍要求全球CUDA+pending<2且free≥10000MiB才启动，不放宽并发、不停止现有任务。科学配方、源门槛和预算不变，原计划run root从未创建，仍沿用该唯一root。新增无空位场景回归已证明控制器写QUEUED且不launch任何GPU子进程，6项源入口定向检查和编译PASS；累计34项不同focused检查已有通过证据。原P0/P1及定向分类权重问题复核保持记录，没有新增报告审批或重复审查门槛。

当前尚未执行真实E20 smoke或训练。R02提交后需独立核实controller/队列和后续实际GPU状态；队列成功不等于模型运行成功。


## R02实际队列发布

VERIFIED_QUEUED。执行控制器commit `26876bca7a75e14690f8af038863c3c8988209f2`，发布时本地与上游OID一致；新archive传输比较VERIFIED、远端compile PASS。独立核实N607普通账户controller的PID/CWD/完整命令身份，controller不在GPU计算进程中，操作系统子进程为空；两job均QUEUED、GPU worker为0。源模型尚未创建，真实E20联合backward smoke和训练均等待空位，不能把队列成功当作smoke或训练成功。

资源guard保持全球CUDA+pending<2且free≥10000MiB，包含其他owner；有空位后自动先smoke，PASS立即启动唯一源学生。原三种子确认的两个健康学生继续原E50计划。没有目标分数反馈、G读取或现有训练变更。证据：[实际队列与零GPU子进程](queue_evidence.json)。


## 实际 E20 检查通过，唯一源学生训练中

VERIFIED_RUNNING。执行 commit `26876bca7a75e14690f8af038863c3c8988209f2`。真实同契约单 SIG_TIME E20 联合 backward smoke 为 PASS，检查后没有 optimizer 更新；U 真值、目标和恢复器读取均为 0。随后同一控制器自动启动唯一源学生，未操作其他 owner。最新独立观察核实控制器及 GPU7 学生的操作系统身份、CWD/完整命令、GPU 归属，训练日志和完整记录持续增长。

新增 L 锚点项实际捕获两次 raw L128 特征（同一批真 L 的 256 次特征曝光），两次 U/H 物理半批共 256 个 query 曝光；其中 236 个满足原高置信和跨 RX 正负支持条件，8284 个正对、50248 个负对。L keys 停止梯度，没有新增 F 前向或推理参数。该项单独对 SIG_TIME / 最终分类权重 / 其他部署特征的梯度范数分别为 0.05818485 / 0 / 1.32771531；其他部署特征包括原 head 内的融合 norm/gain。该分组与旧整体梯度保护的 other 分组不同，不能直接做比值归因。

当时全部 336 条完整可读训练记录已独立检查：L768、U6、双 H、实际捕获数、全物理半批归一化、有效正负对、有限损失及原 cap1/权重6/style0 均符合。后续记录增长是运行证据，不把 336 条审计扩大称为全 E50 审计。首个新增 epoch 实测约 231.39 秒。

最新观察（2026-10-26 19:23:48 UTC）：累计 E29，完整可读新记录 2076 条。该计数不等于 E50 完成；完整源四景评估和预登记门槛仍待执行。原 source gate raw≥26568/27000、H3≥62109/81000 不变，不依据已完成主实验的目标负结果修改参数、门槛或候选。证据：[实际检查、梯度与训练增长](launch_evidence.json)。
