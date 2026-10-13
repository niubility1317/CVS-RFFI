# PA 跨延迟特征探索最小预登记

run_id：20261014-phase1-pa-cross-delay-legacycos-clean-manysig-m9-r01。状态 ANALYZED；全部训练、固定预测与独立评分已经完成。主目标是 PA_CROSS_DELAY 用原默认分类头获得 clean 三种子均值 ≥84%。本轮方案源于用户设计路线二，与另一个对话的 GN 信息恢复/分支竞争抑制不同。

|结构|model seeds|训练|默认头测试|
|---|---|---|---|
|LEGACY50 原旧 PA|2026092701/02/03|复用已核实同 seed 固定 final50，不训练|复用既定 clean 默认头决定|
|PA_CONV 同容量普通卷积|2026092701/02/03|各从零 50 轮 / 11100 更新|自动 clean 默认头推理与 truth-last 评分|
|PA_CROSS_DELAY 因子化三阶交互|2026092701/02/03|各从零 50 轮 / 11100 更新|自动 clean 默认头推理与 truth-last 评分|

6 个新模型共 300 model-epochs / 66600 更新；历史基线 150 model-epochs 单列。全部 9 行固定完成，没有按 V 或目标分数晋级、早停、重排或选择性重跑。每行直接用原默认 ResidualPhysicalHead（6×160，scale30），仅随骨干联合训练；单独头训练/适应预算为 0，没有新分类器、原型或 source 换头。预测模块不打开 source IQ、不抽取 L 特征、不导入 optimizer 或 fit；target 只用于逐包独立推理。

结构：原 MemoryPolynomialLift 的 4 延迟与 1/3/5 阶、原 EnvelopeGate、后续 PA 卷积/池化/融合和全部时频/reference 路径保留。新残差加在 PA lift 输出与既有 EnvelopeGate 之间，因此改变浅层 PA 观测而非分类器。在 component clip±2 的 IQ 上，8 组 a,b 滤波器覆盖延迟 0 至 8，各按 row L2 eps1e-4 归一化。PA_CROSS_DELAY 的基本项为 uI/Q[n]=sum_m a_m xI/Q[n-m]，v[n]=sum_l b_l |x[n-l]|²，输出 uI/Q*v；允许 m≠l，不只固定同延迟基底。PA_CONV 使用相同参数的 Conv(I,a)、Conv(Q,b)，共同拼成 16 通道，再经过 SiLU、Conv1d16→24/k1、24 通道 tanh gain（参数初值 .05）。总新增参数 576（a72+b72+projection384+bias24+gain24）；二者 state_dict 和 seed+51507 私有初始化完全一致，主初始化 RNG 不被消耗。

三阶与共同相位等变只针对裁剪不触发时的交互基本项；SiLU、component clipping 及整网不保证同一性质。该模块处理收到的 IQ，不宣称唯一 TX 硬件参数恢复、任意 RX/FIR/CFO 消除或新 TX 迁移已有效。旧 EnvelopeGate 在三项中均保留，本轮检验给既有 PA 增加可学习交叉延迟形式的增量效果，不宣称彻底隔离包络门控的全部贡献。

配方：旧 batch_neighbor 门控、EMA .999 versioned/no startup average、rx_day_quantile .86、tau .92 至 .97、neighbor window2/minconf .8、strong agreement、lambdaU .16/entropy .01、label smoothing .01、普通训练增强保持。32 label＋18 pseudo，AdamW LR .0002→1e-6 cosine50、WD .0001，无 satellite、无新辅助损失。只保存固定 final50。

数据：ManySig raw/post_sync/noeq，source split392005，L6300/U56700/V27000（7/63/30），source RX 索引1/3/4/6/8、days1/2/3、旧6 TX。U 真 TX 标签隐藏，V 仅健康诊断，不反向传播/选模。source 实际物理角色逐 row 比对既有 contract；基线 checkpoint 已核实来源为相同物理角色、无祖先、固定 final50，仅用于 smoke 和对比，新学生/teacher 各从零，不继承基线。复用匹配 VALIDATED_ONCE 数据，不新增数据 hash/revalidation。

本轮按已授权 clean-only 范围：168000 物理包/row，旧6 TX、7 target RX、4 days；仅默认读出共1512000决定，其中baseline504000复用、新推理1008000。逐row prediction固定后，独立scorer才连接truth，输出18条整体/RX/TX/day准确率、macro-F1、混淆矩阵；全部9行共162条，三seed均值/样本SD及PA_CONV−LEGACY50、PA_CROSS_DELAY−LEGACY50、PA_CROSS_DELAY−PA_CONV配对差完整保留。不是独立盲测、新类注册、Phase2 或 LEO 部署验证。

发布：本地 Git HEAD 同时承载实际代码/配置和本报告，由 publisher 写入 release_commit.txt、package.json，dispatch 用相同 commit 写入执行记录。没有额外 code/config/design/report SHA。环境 ordinary N607 / CVS-RFFI Python / fullFP32；CWD 为项目 releases/cvs_pa_cross_delay_20261014_r01；命令 python -u -m experiments.cvs_pa_cross_delay.dispatch；owner codex/root/pa-cross-delay-20261014。每 GPU 总实验进程最多3，含所有其他owner，free≥12000MiB，共享容量锁。run根不可覆盖，路径见 experiment.json。

首步真实基线 checkpoint 零 IQ no-query smoke、数学/模型/默认头/pipeline negatives；PASS 即自动训练。发布只做一次 release archive 本地/远端 SHA 比较、一次 remote compile/preflight，再检查 PID/CWD/argv/GPU/log growth。训练后每行自动接续默认头预测、独立评分；全部报告/Git交付完成前不声称实验完成。

系统技术停止规则：本 owner 的数据角色、来源、更新预算、非有限 loss/gradient、默认头冻结或 prediction/score闭环直接失败时关闭新调度，保留失败产物和健康行，不自动重试；不触碰其他 owner 的健康任务，没有性能 stop。expected artifacts：preflight/launch/queue、每新row source final checkpoint/source freeze/完整epoch与step logs、每row default_frozen/predictions/complete、scores、全部analysis/completion和结果报告。

当前 focused 证据：显式跨延迟展开最大误差2.22e-16；m1/l3 off-diagonal、因果零填充、交互基本项三阶缩放及共同相位等变、逐包独立、零/极小/裁剪输入有限、8项输入容量negative PASS。两真实骨干的全部新增参数CE梯度非零、默认头联合CE保持、gain0精确恢复原特征、默认logits逐元素相同、7项配置negative PASS。新预测/评分freeze21 negatives与完整九行合成finish（162记录/3summary/3paired/1512000决定）PASS，没有source或target IQ读取。唯一一次独立下一run P0/P1审查进行中，remote真实checkpoint smoke随publisher执行。

REJECTED_EXTRA_GATE：通用design/spec审批、额外签名或hash不在用户Exclusive Minimal Experiment Workflow白名单内；按已有授权执行本固定矩阵。

唯一一次独立下一真实run P0/P1审查 PASS：scratch来源/旧配方、同初始化、仅pa_lift改动、默认头联合训练、EMA与strict checkpoint重建、单ORIGINAL_HEAD预测/LEGACY50完整复用、独立truth-last、九行162指标/3paired、发布依赖和GPU容量规则无直接缺陷。审查只读，无真实数据或远端动作；不增加重复全review或报告审查。

独立发布后读回 VERIFIED：实际执行代码1f8814c046a8c68eba8a334e66a0aceb76e491e0，单archive SHA传输比较、remote compile、真实默认checkpoint no-query preflight PASS。6新worker实际PID/CWD/argv/visibleGPU/模式/commit一致，epoch由0增长到4至5且stdout增长；3baseline默认决定已复用并独立评分，failure0/pending0，每GPU总进程最多2（含其他owner）。当前RUNNING，无新目标成绩或收益声明。

最新用户再次明确默认分类头、禁止单独训练分类头。本 run 始终为 ORIGINAL_HEAD 单读出，无独立头训练。

结果：PA_CONV 80.7738%，PA_CROSS_DELAY 80.8133%，旧门控＋cos 80.6454%；均为三 seed clean 原头均值。完整核验 VERIFIED，84% 未达到，结论与缺项见 final_analysis.md / results.json。
