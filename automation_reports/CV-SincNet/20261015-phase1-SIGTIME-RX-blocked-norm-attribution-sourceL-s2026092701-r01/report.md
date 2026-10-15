# 冷初值 RX 分块与辅助梯度范数归因

状态 **ANALYZED / VERIFIED**。80 个源域临时条件已完成；无新主训练晋级。下文保留预登记。上一轮完整默认头三种子证据责任路由未达到总体 84% 目标。本审计依据其源诊断结构而设计：原 INITIAL 资格使用五个 RX 的当前训练包；ROUTED/JOINT 的梯度大小也不同，因而尚不能区分路由与强度。目标测试结果不用于设置本审计的参数或排序。

三条可执行方向：①本轮先做 RX 分块、同标量且匹配辅助 F 梯度范数的一步归因；②独立研究用真实源身份梯度校准 U 的 AdamW 更新方向，须先建立从零训练且完整排除 heldRX 的源折，不能沿用已见该 RX 的终点；③真实跨 RX 判别关系继续作为独立训练方向，先核对既有 PA/identity-difference/集合/观测负结果，不换名重跑。当前只实现①，不额外发布②③。

固定 source-only：5个留出RX×4个原L180cohort×4模式，共80cell。每cell当前训练只用其余4RX的144包，评价仅用原heldbank中的该RX36包；各包均原合法L_s，V/U/target不用。初始化只读新实际LS0基线R02 seed01的scratch0快照，来源/数据角色在加载前检查；不加载训练终点或旧optimizer moments。所有cell从同一冷初值/空AdamW状态独立构造，不串接。

ZERO为原联合CE；JOINT和ROUTED沿用相同同包损失结构、权重.1、容差1e-6、当前4×144固定分母576。JOINT_NORM令loss数值保持原值，仅把其梯度乘以由当前144包计算的stop(norm(ROUTED_F)/norm(JOINT_F))。heldRX的标签/特征不参与因子、梯度或训练状态。W/三个gain的辅助梯度停止，主CE仍联合训练；不拟合任何身份/RX诊断头，actualjoint160保持部署路径。范数控制保持JOINT方向，匹配的是辅助参数梯度范数，不是AdamW实际位移；后者必须单列，不能据此宣称完全消除了优化尺度影响。

每临时步保留原188参数的AdamW lr2e-4/wd1e-4/clip5，主头正常更新。ACTUAL使用afterW和afterfeatures，PRE_W仅替换为beforeW、保留同一afterfeatures和aftergains。真实优化器更新0，160次临时CPUAdamW、900L载入、5942packet网络前向（含2zero-smoke），不保存训练checkpoint；活模型/optimizer/grad/RNG保持。读出只含defaulthead CE、错误、margin和实际F变化，全部80cell/20paired/5RX汇总，不挑优组、不以此自动晋级主训练。

严格边界：这是冷初值一步RX-blocked更新诊断，**不是完成E20的真正留一RX泛化实验**，不能据源CE下降宣称DG。原数据角色与物理ID保持；bank中的评价原始包仅作独立读出，不进入当前训练forward/梯度/校准或模型持久统计。未训练模型不携带训练RX知识，但该有限诊断不替代原要求的整折训练、U/teacher/cache完全排除。

run `20261015-phase1-SIGTIME-RX-blocked-norm-attribution-sourceL-s2026092701-r01`，代码/config由本次Git提交固定，owner `codex/root/RX-blocked-norm-attribution-20261015`。CPU2threads/CUDA_VISIBLE_DEVICES空，普通N607；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_RX_blocked_norm_attribution_sourceL_20261015_r01`，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_rx_step_attribution20/runner.py resolved_request.json`。唯一不可覆盖输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-RX-blocked-norm-attribution-sourceL-s2026092701-r01/source_diagnostic`，source依赖为已完成R02不可变release `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02`。完整数据/六类seed/物理索引继承字段及单row内80cell矩阵见[experiment.json](experiment.json)和config，均不新增未知seed扫描。

聚焦协议负例和范数/同值检查；唯一独立下一run P0/P1审查；launcher首先严格加载真实cold checkpoint做no-query smoke，PASS直接继续；单archive比较/remotecompile/preflight、PID/CWD/argv/CPU与loggrowth核实。停止：Protocol/role/provenance mismatch, nonfinite or missing native component, wrong norm control/budget, output collision or unlaunchable source execution: preserve outputs, no automatic retry. No performance stop and no healthy-process intervention.。本审计无实际训练，故无最终target预测/评分；整体84%任务仍active，历史正收益和完整负结果保留。

聚焦本地验证 PASS：32 个 source 角色/RX 排除/物理 ID/缺项及范数因子负例；三个辅助条件数值相同、范数控制保持 JOINT 方向并匹配 ROUTED 辅助 F 范数；W/三个 gain 辅助梯度 None。编译 PASS。唯一直接 P0/P1 审查最初发现既有失败根可能被 runner 异常处理覆盖；先复现 FAIL，再将失败处理移到成功新建根后的所属调用中。3 项输出归属回归 PASS，定向复查已解决 P1，保留初始 FAIL 记录。当前 source 梯度和临时读出均使用 eval 模式，不更新 normalization/dropout 随机状态；不等价于真实 20 轮 train 轨迹。真实 CP no-query smoke 仍在 launcher 首步，不以本地合成检查替代。


## 全部源域结果

执行代码 `3e5ee1ffc2d610c6fce3c8d153ed5b08492839d8`。真实 cold checkpoint no-query smoke、archive 单次比较、远端编译、PID/CWD/命令/日志增长及 CPU 无 GPU 分配均 VERIFIED。独立读回全部 80 个 cell 文件与 summary 逐项一致，20 个配对 cohort 完整，进程自然退出，无失败；耗时 30.22 s，5942 次 packet 前向、160 次临时 AdamW，0 次真实模型更新/另训头/U/V/target 访问。live 模型、optimizer、grad 和 RNG 保持。

| ROUTED 比较对象 | ACTUAL CE 平均差 | PRE_W CE 平均差 | 两种 CE 均改善 | 错误数改善/持平/变差的 cohort |
|---|---:|---:|---:|---:|
| ZERO | -0.00142918 | -0.00145569 | 20/20 | 2/18/0 |
| JOINT_GAP | -0.00086725 | -0.00094911 | 20/20 | 2/18/0 |
| JOINT_NORM | -0.00114540 | -0.00119855 | 20/20 | 2/18/0 |

差值为 ROUTED−控制，负值表示 CE 或错误数下降。5 个 RX 各有 4/4 组 CE 改善，两个读出结论一致；相对每个控制均仅 2 个 cohort 少错 1 包，其余 18 个持平，改善均在 RX8。评价只有 180 个不同物理包，每个 RX 的同 36 包被 4 个 cohort 重复读出，不能把 20 次比较当 20 个独立种子，也不能把累计少错 2 次当两条唯一错误被修正。

| 条件 | 辅助 F 梯度范数均值 | 辅助与主 F 梯度余弦均值 | 实际 F 位移范数均值 | 相对 ZERO 的 F 增量范数均值 |
|---|---:|---:|---:|---:|
| ZERO | 0.00000000 | — | 0.08018824 | 0.00000000 |
| JOINT_GAP | 1.16003156 | 0.97535967 | 0.08018820 | 0.00784108 |
| JOINT_NORM | 0.54231042 | 0.97535967 | 0.08018846 | 0.00539780 |
| ROUTED_GAP | 0.54231042 | 0.93256530 | 0.08018841 | 0.00789961 |

辅助 F 梯度范数匹配后，ROUTED 相对 JOINT_NORM 的 ACTUAL CE 仍在 20/20 组下降，均值 −0.00114540；PRE_W 为 −0.00119855。当前训练完全排除评估 RX，因此冷初值优势不只来自更小的辅助梯度范数，也不是必须使用该 RX 的当前训练包才能出现。不过 AdamW 相对 ZERO 的实际 F 增量分别约 0.00789961 与 0.00539780，并未匹配；不能据此宣称只剩方向的独立因果效应。默认 W/三个 gain 的辅助梯度均 None，主 CE 仍联合训练；PRE_W 仅是既有 W 的无拟合公式读出，未产生新分类头。

这一结果与完整主训练负结果并存：R02 默认头三种子 ROUTED 均值 81.120635%，比匹配 SSL 低 0.054167 个百分点，仅 1/3 为正；次要最差 RX 均值高 0.245833 个百分点、3/3 为正仍保留。冷初值 CE 的微小一致收益不能替代持续学习、完整 leave-RX 或目标测试收益。本审计没有预登记主训练资格，故不补设通过阈值、不重跑当前完整配方、不增加权重/频率或扫日程。

对已有完整训练日志的补充解释（不新增 IQ/模型读取）：ROUTED 三 seed 的 180 次辅助调用均非零；step16 的辅助 F 范数为 0.2183/0.2918/0.3095，至 step2880 为 0.00206/0.00442/0.00690。它仍在作用但强度明显变化。此处不能据此断言“信号变小”是总体未获益的原因，也不以简单放大梯度作为下一轮方案。

下一优先研究方向是**用真实源身份风险约束 U 在优化器中的特征更新方向**：保留原 U 教师、隐藏标签和默认联合头，分别计算当前 L-only 与 L+U 的临时 AdamW 更新，用各训练 RX 的合法 L 身份风险梯度约束两者的特征增量差；另设实际增量范数匹配控制，检验是否获得独立的跨 RX 学习内容。这是待验证的具体机制假设，不声明全局首创或已实现。它与已失败的“仅过滤历史 first moment、当前梯度不变”不同，不能复用该旧实验的资格。

该方向先建立从零开始的真实源 RX 留出训练：heldRX 必须从 backbone、U、teacher、初始化来源、缓存和统计中全程排除，保留现有默认头，使用当前 7%/63%/30% 数据角色。已核对旧 complement-CNN 的真实整 RX 留出实现可供过滤逻辑参考，但其旧网络/30 轮/数据契约不作为本轮权重或新控制。首个小矩阵的 RX、epoch、源阈值和控制须在新数据结果前固定；当前冷快照不冒充这种已训练折，也未自动启动该矩阵。真实跨 RX 判别关系仍为另一独立方向，继续保留已关闭 PA/difference/set/observational 的证据边界，不换名重跑。

历史 CENTER25K、结构正收益及完整负结果继续保留。当前 84% 默认头 clean 三种子目标未达到，goal 保持 active。完整 scalar 证据与全部配对见 [result_evidence.json](result_evidence.json)、[attribution_analysis.json](attribution_analysis.json)、[80 条件](source_cells.csv)、[60 配对](paired_cells.csv) 和 [15 个 RX/控制汇总](source_RX_summary.csv)。source-only 诊断无额外 target 预测/评分。
