# R32：原 35.05 万父版上的 F-free 统计梯度协调

状态：LOCAL_VERIFIED。63 项聚焦检查 PASS，唯一独立 P0/P1 审查 PASS，无 P0/P1；尚未发布，无新增正式分数。

本轮固定原 350521 父版，继续保通道 SVD625 压缩和性能联合优化。R31 已完成的合法源梯度诊断在固定 R22 恒等版 received01/20 panel 中观察到，base 与原父统计目标分别有 8/16、10/16 批次反向；coarse/action/unique prior 在 32 批次中各有 18/17/13 次反向。这是固定源侧反事实梯度证据，不是实际 R30 优化轨迹、目标 TX1 退化的唯一原因或正式识别替代指标。新对照在任何 R30 目标结果揭示前固定。

|版本|完整参数|统计参考|实际应用梯度|GPU|
|---|---:|---|---|---:|
|moment_teacher_grad_sum|278180|原 350521 对同 received 的最终 IQ|base＋统计梯度|2|
|moment_teacher_symproj|278180|相同原父 IQ|逐模块对称移除相互反向分量后求和|3|

两行唯一初始化及 IQ 教师均为原 R46 prune_refine350521、source-final E20/4940、bbb8daf817de050b958d217b70d4c33f4aefe341；从同一原父各自 data-free SVD625 初始化，不加载 R18/R22/R23/R30 学生作为父或教师。原 R44/R38/R17/R07/R04 的 7/63/30、READY/EXACT_MATCH 数据与完整祖先权限保持。训练仅合法 L_s/U_s 模拟配对 IQ，0TX 标签/F 权重或输出/V/target。原 F50 seed2026092701/defaulthead 仅冻结评测及 2Ls smoke 使用，无新头。

两版完整 coarse106658/action80045/prior91477＝278180，较指定父减参20.6381%。保留全部通道/GN4/skip/独立bias和norm、三stage/六CG，全部联合训练。基础 IQ+.05功率STFT+三stage原IQ教师1→.1+.02分组复FFT+.25G(clean)恒等，与 R30 teacher 行源目标相同；新增统计系数保持1.0。描述为 R29 原算术的11维归一化四阶统计，lag1/2/4/8，逐窗口中心化、powerfloor1e-4；clean/原父参考全部 detached。两版均计算相同的三个描述，只有原父统计参考进入新增目标；不增加 G 前向。

每次对 batchmean base 和 weightedmoment 分别 autograd.grad，共2次，保留同一前向图。全部可训练 unique 参数按 coarse/action/prior 三个互斥完备组计算；共享 prior 权重只计一次。令 a、b 为两个组向量，c=a·b：c<0 且两者 L2>1e-12 时，应用 (a−c/||b||²·b)＋(b−c/||a||²·a)，否则 a＋b。点积、范数与组合用 FP64，cast 到原 FP32 后计算实际应用梯度遥测；两行均计算 raw/proposed，只有选中应用向量不同。两个目标均无梯度的参数保留 grad=None，AdamW 跳过，不用零梯度引入 weight decay。

对称投影的非反向性质仅针对 clip/AdamW 前的梯度；优化器动量、预条件和 weight decay 可改变实际更新，不能保证真实损失或准确率单调改善。严格反向的向量可投影为0，余弦未定义记null并排除均值，不当成无冲突成功。记录每组每轮985更新的 conflict/projected 次数、raw/applied norm/dot均值和有效余弦均值。原生单联合 backward 与两次 FP32 backward 的求和顺序不同，raw 控制只作数值近似检验，不声称历史 R30 逐值优化轨迹相同。

固定每版 E20/19700、B64/micro64、received01..20、L126000/U1134000；teacher1260000、received学生1260000＋clean学生1260000＝2520000，三个描述共3780000，0额外G；每版39400次梯度求值。ResetAdamW2e-4、warm985、cos2e-5、WD1e-4、clip1、FP32/TF32off。三组件均继承80至累计100，和300，串行140；额外 backward/FP64/遥测成本据实测量，不承诺同壁钟。

固定最终权重，不按源收益筛掉任何行：每版自动 source1800×7、完整G/G+原F B1/B32同卡成本、独立源评分、formal168000×clean/三郊区/三urban，全七景预测固定后独立CPU truth-last。报告accuracy/MacroF1、逐TX/RX/day、混淆/净纠错/clean损伤、波形/F几何、全部20轮损失与梯度统计、参数/MAC/文件/实际延迟/峰值。指定父逐景0.8pp/1344条、原R17历史1pp、70%/5pp和强身份真实性单列；历史R30 teacher及R22恒等只作结果参考，不作为父、不重跑目标、不称新随机重复。目标结果不用于选模、调参、重排或选择性重跑。

GPU2/3各一own、合计最多2，分别等待 R30 同卡 worker 终态且实际PID/module退出，前驱workers为空则继续等待；之后共享容量锁free≥18GB/allowner<2。CPU两线程/R06score.lock、最多一own scorer；其他owner和健康任务不停止、不热改。普通账户、独占新run/release；一个方法commit、一次preflight/归档两端SHA/compile。launcher首步真实原父/F50/2Ls检查新梯度路径，PASS直接排队。系统技术失败保留影响行，健康行继续，不自动重试、回退或覆盖。精确命令、数据/六类seed、环境/CWD、路径和预期产物见experiment.json。

63项检查覆盖双梯度完整模型/共享别名/参考detached/父不变、解析投影与冲突/零/None/无穷/分组负例、梯度与AdamW数值参考、完整985更新遥测、来源/预算/配置负例及完整七景truth-last/旧输出保护。修复了旧行名的extra_clean计数字典和configuration全局可变别名；config返回deepcopy。唯一独立P0/P1审查PASS，无P0/P1，通用额外审批和全库测试为NONBLOCKING/REJECTED_EXTRA_GATE。联合优化目标仍ACTIVE。
