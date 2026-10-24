# 完整原生 As 配方的 U 梯度范数帽对照

LOCAL_VERIFIED_NOT_LAUNCHED：29 项 cap2／梯度检查、119 项受影响流程检查、一次独立 P0/P1 检查全部通过。真实原 E20 检查待首个匹配释放的 NativeAs 卡位；通过后丢弃探针、重载原父权重继续，当前没有本候选 GPU 训练。

原同 seed F21 E20/4440，各自独立执行 native10/2220＋20/4440，累计 E50/11100＝13L＋37LU；三个 seed 为 2026092701/02/03。只加载已核实契约及完整继承来源的原 E20，不加载候选权重或原 E30。L6300/U56700/V27000 与 F0 未见 RX 物理划分不变；U TX 真值隐藏，V 仅诊断，不更新模型、归一化、EMA、原型或头。N607 普通账户、CVS-RFFI、FP32/TF32off、每 worker CPU2；命令、CWD、路径、GPU 和停止规则见 experiment.json，发布后核实执行 commit。

固定源计划 44d38cc9、完整源结果与设计 28431a31：只对 Joint 全部 19980 条既有源梯度记录计算预定 cap1／cap2，不读新 IQ、权重、U 真值或目标数据，不新增模型前向。三个 seed 均通过预登记规则；t2 平均 appliedU/L 算术提高 34.88% 至 51.62%，全参数组提高 27.83% 至 29.36%，计算出的 combinednorm>5 频率增加 0 至 0.1352 个百分点。该结果属于相同当前梯度的算术，不是新训练轨迹、实测 clipping、AdamW 更新或目标收益；Joint 不含完整 nativeAs，向 NativeAs 转移必须独立训练和测试。源计划、完整摘要及预先设计随报告保留。实际 U 正确率 UNKNOWN，不能用源 V 的高精度代替。

相对固定 NativeAs cap1，只将 t2、cls_head、other 三组 U/L 范数帽从 1 改为 2。仍以完整当前 L 目标为 anchor，对负点积 U 方向投影，tiny-L 平方范数阈值 1e-12，完整 U 梯度只保护一次并接回原 native backward。保持 raw argmax／0.95、U 权重 6、六学生、原生 As 半批、LS0.01、ordinary/raw RDrop1、As KL0、浅中层跨 RX0.1、joint160 跨 RX0.05、共享 style、CE 分母1.25、原 L／默认头联合训练和 nativeAdamW／clip5／decay。无额外 autograd、教师、学生或既有头前向，无新头或推理参数。投影后的非负点积不保证优化器更新、L 损失或目标性能单调。

每模型新增 6660 步，U 曝光 1701000、学生 U10206000；累计教师2097900、学生 U10602900。As 生成850500、物理学生1701000，有效接受 As CE 视图为2×rawacceptedhalf，raw 回退0；接受集合及全部接受样本保留。源算术 CPU1／65.676 s 成本单列，不称等实测时长。

GPU5/6/7 固定对应三个 seed，每卡最多两个本任务，计外部 compute 进程小于3、空闲显存至少10000 MiB。等待同 seed NativeAs 全部 GPU 预测完成且 BalancedCE 已派发，两侧 pending CUDA PID 均预留。首个匹配卡位真实原 E20 检查核对 L768／U1536／rawteacher256／As128／物理学生256／Practical192；逐组 U≤2L、负点积投影、tiny-L、完整 L 显式参数梯度、一次 guard 和临时 hook 退出全部符合，且至少一组实际 U>L，证明 cap2 确实生效。唯一 CPU scorer 接 BalancedCE 终态。两个相同系统性技术错误只停止新派发，保留健康任务和产物；不按目标分数停止、调参、重排或选择性重跑。

每 row 最终默认学生冻结后自动预测 raw 及固定 DURA 各 clean＋六 practical，168000/view，三 seed 共42 cell；整 row 预测固定后独立 truth-last。报告逐 seed／景、SD、MacroF1、TX／RX／day／RX×day、净纠错及实际时间。NativeAs cap1 为预定匹配比较，F20/F21 为原主控制；raw clean 对两者均至少高1个百分点、raw 六景平均对两者均严格高于1.5个百分点才达到目标，DURA 单列。固定 DURA v1.0 R69／266951参数／FP32/TF32off／reuse=True/graph=True/native=False，不用于新 F 训练，不套用旧 F50 时延，不替换部署默认。此前完整目标曝光如实保留；当前设计未读取 RXE／NativeAs／BalancedCE 新目标分数，亦不作独立盲测声明。目标保持 ACTIVE。


发布后独立核实：VERIFIED，执行 commit `a2a03f6ccf166b8abba34ca1f7862caa03277ba0` 的控制器存活，CWD 和完整命令匹配；三行实际 QUEUED，新增 GPU worker 为 0，真实原 E20 检查 PENDING。等待对应 seed NativeAs 全部 GPU 预测结束且 BalancedCE 已派发；独立核实两侧控制器及现有运行行的 PID/CWD/命令/GPU/日志增长。保留全部健康任务；排队不代表训练成功或达到目标。


完整原生 As 梯度贡献分析已固定：NativeAs/cap1、BalancedCE/cap1、NativeAsCap2/cap2 各三个 seed，全部 E21 至 E50 共59940条闭合记录。只读源配置、完成元数据和日志，不读 IQ、权重、U 真值或目标；记录权重6后的实际 U/L、保护后 U/L、范数保留、投影/帽计数及实际超过旧上限的步骤数。30项计算检查通过；CPU1，待三方法全部源 E50 后执行，不阻塞训练或测试，不新增候选及晋级门槛。当前训练与固定矩阵保持，实际优化器更新及目标收益仍需独立判断。


原 E20 分支梯度诊断已预登记：三个原同 seed F21 E20，各一次影子 NativeAs cap1 的 L128/U256 步。仅开 L/U，不开 V 或目标，U 真值隐藏；复用原4个L pair、6个U logits，量测普通/raw非信道、As及辅助分支，再逐参数核对其相加保护梯度与原完整L/U guard一致。10项定向检查、一次独立P0/P1 PASS；真实原E20无query检查作为诊断第一步。每seed9次autograd，累计7686条源前向packet，CPU2/noCUDA；没有backward、optimizer、参数或持久缓冲更新。训练模式影子缓冲变化记录后丢弃。仅一个源步骤，不冒充完整轨迹、实际优化步、clean测试收益或目标因果解释，不自动派发新候选，现有任务不变。


分支诊断技术修复：两个失败尝试均未接受为结果，现场保留且失败进程已独立核实退出。首个旧逐元素检查触发于 Sinc low_coordinate，最大绝对差2.6154e-6，两侧最大值约0.00537036。现在直接观察原 guard 自身两次完整 L/U 导数作为所有完整梯度统计基准，重新投影必须与原输出逐参数 torch.equal；组件和另按dtype默认检查及每组相对L2误差0.1%上限核对并报告，仍严格区分None和零向量。13项检查、仅针对该问题的一次P0/P1定向复查PASS。原前向/9次autograd/源样本矩阵及原训练算法均不变，未把失败诊断当作科学结论；等待同一矩阵真实复跑。


原 E20 分支诊断完成：三种子各 1 个 L128/U256 源步，原完整 L/U 梯度重新投影与原 guard 逐参数完全一致；分支求和误差另行核查并完整报告。最终成功尝试 CPU2 耗时 42.975487 s、7686 源视图、27 次 autograd，参数与 .grad 均无写入，影子缓冲变化已丢弃，无 V/U 真值/目标/G 训练访问。两次旧失败保持独立负技术记录；详见 source_branch_gradients.md 与 summary JSON。单步源域方向不能当作完整训练轨迹或目标性能因果解释，不修改当前候选及训练。


真实原 E20 检查及首批启动已独立核实：VERIFIED，执行 commit `a2a03f6ccf166b8abba34ca1f7862caa03277ba0`，检查 seed 2026092702，L768/U1536/rawteacher256/As128/物理学生256/Practical192，完整 L 显式梯度、joint 签名梯度和临时 hook 清除通过。探针丢弃，正式重新加载原 E20。目前实际 1 个正式 GPU worker 已有正源步骤，并核实 PID/CWD/命令/GPU/日志增长；其余仍排队。完整事实见 first_source_start.json；首个启动不能当作三种子全部启动或测试成功。


固定源域分支约束比较已登记：同原 E20 三 seed、每 seed 一个 L128/U256 影子步骤，复用现有 6 个分支导数，在完全相同原完整梯度上比较完整 L cap2 和完整 L／非物理 L／物理 L 同时约束 cap2。完整 L 不变，只约束 U 方向；源规则固定为所有 seed 的整体 U 范数至少保留原 cap2 的 50%，且至少移除 1 个实质分支冲突。若失败则否决，不在结果后改规则。32 项检查及一次独立 P0/P1 检查和一次定向修复复查 PASS；初查发现旧 release 不支持 cap2 形参，现单独加载已提交的纯 cap2 投影函数，原 cap1 桥及其冻结依赖保持。真实源矩阵尚未执行，无自动训练或健康进程修改。详见 source_branch_anchor_counterfactual_plan.json。


分支同时约束源域反事实完成：原 E20 三种子各一个步骤，所有原桥／角色／预算／数值检查通过；相对完整 L cap2 的最低整体 U 范数保留率 99.744189%，移除 9 个实质分支冲突，冻结源规则 PASS。CPU2 42.328271 s、7686 源视图、27 autograd，无额外前向/导数、参数更新、V/U 真值/目标访问。这是原导数的几何计算，不是训练或目标收益；结果未改变现有模型和进程，也未自动启动新训练。详见 source_branch_anchor_counterfactual.md。


源支持的下一训练对照已固定，尚未实现或启动：`20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01 / HPL_E20_AS_BG2`。只将原 NativeAs cap2 的 U 方向改为完整 L／非物理 L／物理 L 同时约束，仍从原同 seed E20 独立 own10+20/E50/11100，原接收集合、0.95/权重6/六学生/所有原损失/defaulthead/优化器与 As 预算保持。每步多2次L分支导数，每模型13320次，无额外前向，实测时间单列；不能称相同计算成本。固定 NativeAs cap2 主匹配、cap1 次匹配、F20/F21 raw 双目标及 DURA 参考42cells。生产实现、独立训练 P0/P1 和首释放卡位真实原 E20 检查待完成；source PASS 不冒充训练验证。方案在三组待收目标数值读取前冻结，现有健康任务不变。详见 native_as_branchguard_followup_design.json。


原生As cap2真实检查：逐组U范数≤2L、负点积投影、一次完整L显式梯度均符合，至少一组真实U>L，无额外autograd或前向。固定半批生成128，两个原mixed学生实际使用256条物理视图，rawAs回退0；原rawteacher256，无增强视图evalteacher或额外学生前向，原接受mask和CE1.25保持。原生As输入及非物理ordinary输入匹配，生成成本与有效As CE视图2×rawacceptedhalf分别记录；没有用排队或预期计数代替实际训练证据。


## 真实原 E20 检查通过并实际启动三种子

VERIFIED：发布commit `a2a03f6ccf166b8abba34ca1f7862caa03277ba0` 的真实原E20 GPU source smoke PASS。实际joint160两次捕获512条已有Upacket，跨RX有效anchors=478；实际签名sequence_pool辅助梯度范数=0.370686237且有限非零，原L768／U1536／rawAs0／U As生成128／物理学生256／LPractical64＋U128、共享style计划、完整L guard参数梯度与显式期望相同及临时wrapper退出均符合。探针丢弃，正式行重新加载原E20。

三个seed已按GPU5/6/7启动并实际产生原生joint／guard源步骤；独立核实进程／CWD／完整命令／GPU及日志增长，逐行现有U复用计数见source_activation.json。只按原矩阵接替释放卡位；当前是训练前缀，不代表全部源训练结束或目标收益。训练、raw＋固定DURA七景及truth-last继续自动执行，goal ACTIVE。


## 固定源域优化器首步诊断

已预登记 source_optimizer_counterfactual_plan.json：仅使用三个原同 seed F21 E20 的固定 L128/U256，各复用原9次导数；L-only、cap1、cap2 与当前 QR 三锚点分别在独立参数容器上执行一次真实 CENTER_UNITS／physical clip5／AdamW，累计12次额外可丢弃更新。原模型、梯度和健康训练不变，无 V、U 真值、目标或 G 训练读取。比较相对 L-only 的增量下降方向，负点积只表示局部一阶变化较差，不等于实测损失上升或目标准确率下降。只解释新 native10 首步的 fresh moments，不外推整条训练。19项检查与一次独立 P0/P1 审查通过；真实 E20 两包 no-query smoke 是每个源模型的第一步，通过立即继续诊断。无科学晋级门槛、自动新候选、父模型或测试矩阵调整；已暴露历史目标如实保留。


首批源优化器诊断在读取旧 combination 字段时发生 KeyError，尚未执行任何副本 optimizer step，原模型与正式训练保持。失败执行 commit e31bd0fd、日志及终止进程证据保留；技术修复改用实际 f_train 的固定 CENTER_UNITS 路径，保留模型物理单位检查与原 cosine／clip5。新增1项针对性检查及唯一一次针对性 P0/P1 复核均 PASS，随后只重执行原固定三种子源诊断；不改变健康任务、数据、方法、种子或测试矩阵。


固定源域优化器诊断已完成，详见 source_optimizer_counterfactual.md／JSON。81组原梯度与实际首步增量度量完整；12次副本更新另计，原模型、健康训练和测试计划不变。此项不是目标效果验证或新候选晋级。


## 实际更新增量保护的固定源域验证

已预登记 source_step_projection_plan.json。固定原E20三个源批次，仅对实际 BG 更新相对同参数 L-only 更新的增量作三锚点投影，保留真实 L-only FP32 参数；检查最终量化后参数差的方向与范数。薄 QR／小 SVD 及最多4次固定量化修复；失败组只允许明确记录的 L-only 数值回退。源规则为每 seed 整体增量范数至少保留50%、至少移除1个明显冲突，并满足全部实际约束和非扩张检查；不得用全零增量冒充成功。复用原27次导数／7686包视图／12次可丢弃 optimizer step，无额外前向、原模型更新、V、U 真值或目标读取。先完成实现、一次独立 P0/P1 审查及真实 E20 smoke，再直接执行；本登记不启动新 GPU 模型，不改变任何在训配方。


实际参数增量源算子已实现，21项检查通过；一次独立 P0/P1 审查 PASS，覆盖快照顺序、FP32 参数差约束、不扩张、显式量化回退、回调恢复与七模块接口。方法及源规则保持原 a23b0b9c 预登记；真实原 E20 smoke 是执行第一步，随后立即继续原三种子源矩阵，无新增审批或 GPU 训练。
