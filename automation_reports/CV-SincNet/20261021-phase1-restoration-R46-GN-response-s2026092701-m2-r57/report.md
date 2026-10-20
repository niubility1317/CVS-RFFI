# R57：既有 27.38 万参数恢复器的源 GN 响应校正

状态：**VERIFIED / RUNNING**，GPU3/2 两行实际训练。33项聚焦检查、CLIhelp及一次独立P0/P1审查通过。性能优先；主目标仍为 low urban ≥70%、其余五个受损场景距 raw clean ≤5 个百分点，以及输入 TX 细节保真。相对 R17 每景 1 个百分点是退化容差，不代表性能提升。指定 R46 `prune_refine350521` 仍为唯一直接权重与冻结 IQ 教师来源，采用原 7/63/30 物理划分。属于有 clean 模拟配对、使用 U_s、训练不读 TX 的源域自监督，不能称无 clean 盲学习。

本固定方案在读取 R55 正式分数前形成，只针对既有剪枝图的 GroupNorm（GN）响应失配，不建立其他恢复网络家族。两行均为完整 273765 参数、context 185615、原 3 prior/6 CG、GN4、权重筛选通道图和 full-fold 输入权重。候选 `gn_full`/`gn_half` 只分别将源域拟合的 GN 校正按 1/0.5 写入既有 168 个 GN 的 weight/bias；不增加推理参数、分支或校准输入。固定匹配控制复用 R55 `fold_full` 的同宽度、同输入折合及同训练预算结果，不重复训练或继承该训练后的权重。

复用 R55 已有的源通道投影与同一 128 条 source 样本（64 L_s、64 U_s，seed 2026102156，received epoch01/20）；复用的 256 次冻结父 G 前向及其成本单列。本轮额外在冻结、未校正的 full-fold 子图和指定父图各前向 256 条 received IQ，共 512 次 G 输入，一次 CPU2 校准供两行共用。0 clean/TX/F/V/target 读取，0 optimizer 更新；只读取源 received 和角色，保留父、初始子图不变后再应用校正。训练仍完整使用 paired clean 及 U_s，不混淆校准与训练权限。

每个 GN 按原通道图保留的输出通道及 16 个固定不同空间位置配对，拟合父 GN 的 post-affine、pre-SiLU 输出 `y=a*x+b`。斜率使用固定 identity-centered ridge：`a=clip((cov(x,y)+λ)/(var(x)+λ),0.25,4)`，`λ=1e-3*max(mean_channel(var(x)),1e-12)`，`b=mean(y)-a*mean(x)`。全部拟合都在同一个未校正子图上收集；随后把 `1+α*(a-1)` 和 `α*b` 合并进原 GN 的缩放、偏置。先后作用的下游输入会变化，因此局部拟合误差不增不能证明完整 G、身份细节或识别率改善。共享 prior 的长卷积仍共享，各 stage GN 保持独立。

两行均重置 AdamW，原 E20/4940、global256/micro256/direct float32 buffer、paired IQ NMSE＋0.05 relative-power MRSTFT＋冻结指定父的三阶段 IQ 教师（线性 1→0.1），coarse/action/prior 全部联合训练。每行新增 L126000/U1134000、1.26M student 和 teacher 训练输入；继承组件80＋新增20＝100、组件和300、串行140 pass；F50训练另计。GN统计校准是额外的现有参数初始化拟合，不冒充 data-free 或零初始化成本。

每行固定最终权重后自动执行 source1800×7、同卡配对 G/G+F 的 B1/B32 三轮时延/吞吐/显存和正式168000×7：clean、郊区高/中/低及 urban 高/中/低。逐行全部预测固定后独立 truth-last；不按源 F、正式分数或身份诊断筛行、调参、另选父或选择性重跑。恢复器训练/校准不加载 F；真实 no-query smoke 与冻结后评测可用原 F50 默认联合训练头，不重训或替换头。输入同一 restored IQ，由原 F 一致生成 FFT/PA/时域/统计分支。

GPU2/3 每卡至多一个本任务训练/预测 worker，R55 两行实际退出后才开始本轮 GPU 工作，跨两轮最多两条 GPU worker。CPU smoke、校准和独立评分使用 CPU2；正式评分复用唯一 R41 score.lock。普通账户、独立不可覆盖 run/output/release；一次预检/单归档 SHA/compile、真实 checkpoint no-query smoke 通过后继续，核实实际 PID/CWD/argv/GPU/log。精确命令、commit、环境及路径在 experiment.json 预登记后发布。

技术停止规则：错误输入/祖先/物理 ID、源校准非有限或不完整、输出冲突、训练或预测非有限及运行技术失败时保留产物，无自动重试；不终止、热改或重启健康任务。所有正负正式结果保留。单 seed、已暴露工程基准及人选父背景完整披露；关联和识别率均不能单独证明逐个细节属于输入 TX，原完整目标继续 ACTIVE。

REJECTED_EXTRA_GATE：泛化 brainstorming 的另行设计审批与固定 planning 审批不适用；当前用户明确的八项最小流程和既有压缩优化授权优先。本报告直接记录方法、固定矩阵及停止规则，不另设审批、签名、成员哈希、报告哈希、重复审查或数据重验。

方法commit `38a1b4da6dd2964967cebb67935ef877df11c837`；精确命令/环境见[登记](experiment.json)。


VERIFIED/RUNNING：同128源received样本额外256父/256初始子G输入的一次GN校准完成（8.0063s），168原GN拟合的斜率/截距/岭系数/局部MSE独立NumPy复算通过；源calibration 0clean/TX/F/V/target/optimizer，父及未校正子图保持不变后才应用GN初始化。两行实际R46/F50/2Ls noquery smoke PASS、全组件有限非零梯度及严格独立权重重载一致。单归档SHA/compile、controller与两worker实际PID/CWD/argv/CPU2/GPU及首轮日志增长核实，GPU3gn_full/GPU2gn_half各一条；R55两行实际退出保护已通过。保持273765参数，不根据已完成R55正式分数修改本固定方案；仅说明实现运行正常，无R57正式识别收益结论。E20/4940后自动全七景/独立truth-last，原完整目标ACTIVE。
