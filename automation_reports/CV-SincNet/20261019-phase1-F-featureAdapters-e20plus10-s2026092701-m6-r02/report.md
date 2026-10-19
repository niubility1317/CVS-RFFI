# 纯 F：浅中层 LoRA 与时序残差特征适配

状态：ANALYZED / VERIFIED，六行训练、全七景预测、独立truth-last评分及完整回放已完成；总体性能目标未完成。下文保留事前方案与运行记录。

## 当前授权与固定范围

2026-10-19 新附件《当前最值得检验的判断是：PAMF 的问题已经不只是……》更新本轮范围：允许在原 F 的浅中层增加特征适配空间，首轮固定原类别权重，并保留全量特征微调控制。原任务的最终 low urban>60%、clean 相对同预算 NoAs 可重复正收益、逐 suburban 接近自身 clean 目标保留；不是把适配器能运行当作目标完成。

用户已明确选择：新增适配 E10 **仅使用 L 的 clean CE、As CE、clean 保持三项**，不继续 U 半监督。U TX 仍隐藏；源 L6300/U56700/V27000 角色、RX1/3/4/6/8、day1/2/3、split392005 不变，不把 U 改成 L。原父模型和健康旧实验的半监督流程不改。

共同父状态固定为原纯 F B0 NoAs scratch FINAL20，run `20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01`，不按正式分数选父。全部新行继承同一合法父 F，新增10×222=2220次更新，累计 E20+10=E30；不是从零连续cosine30。原父成本只计共享一次，但每行明确继承20轮。三种更新空间 FULL/LORA/TEMP 各有 NoAs/As，六条新行，统一冻结既有 cosine6×160 scale30 类别权重，保留物理分支与实际 joint160；无新分类头。

LORA 在 t1/t2 pointwise 层加入 r=8、alpha=8 的线性 BA 增量，B零、A正常初始化，原参数固定。TEMP 从各块的原输入 h 计算 B SiLU(K_depthwise(Ah))，r=8/k=5，在原 pw 输出后、GroupNorm/ReLU/Pool 前相加；只有 B 零初始化，无第二个零门。t1实际48→72、t2实际72→96，预期LoRA2304新增参数、TEMP2384新增参数，以实际构建统计为准。

代码核验发现现有 CRRA 基础类是 Cin深度卷积→降维→SiLU→同宽升维，与附件提出的 A→低维时间卷积→SiLU→B 不同，且不能直接处理48→72/72→96。新模块将明确实现附件的矩形输入残差公式，复用零升维初始化和残差实现原则，不把它谎称为原 CRRA 原样启用；不引入其白化、条件编码、支持度门控或延迟日程。当前纯 F 未启用 CRRA，不把旧默认E17/30日程说成已确认失败原因。

新增阶段使用 raw clean IQ（无 ordinary 随机增强），所有学生与固定父参考都 eval 模式但学生保留 autograd，以免 clean 保持项把 Dropout 差异误当适配漂移。无 BatchNorm 状态拟合。所有六行同一输入/目标/预算；不把它写作原 native 训练的无改动续训。

固定目标为 CE0(clean)+CE0(second)+1×mean_all128[parentCorrect×KL(parent_clean||student_clean)]，温度1、真实标签优先、parent错误包不施加保持KL。As条件second为完整practical(raw)，NoAs条件second为同一raw clean，保持两份CE总份额和两次学生前向；不是削弱NoAs对照。parentCorrect只作用于合法L的clean保持项，所有As无置信/正确门、正真实CE。参考父只读、不进部署、无星地教师。新阶段不拟合U、不更新EMA。

计划共享一次预生成的完整2220步L/As观察流，全部六行逐步消费相同物理ID和实际IQ，不按模型/质量重抽。每步128 L，循环均匀shuffle完整L6300，222步/epoch；即使本阶段不用U，也不缩成每轮50步。传播→新增噪声→接收状态→同步/补偿/AGC完整保持，receiver session不含TX，无EQ，正式困难条件不改。

源诊断预先固定：L每TX×RX×day前20包共1800，保存其首次实际训练As IQ及真实元数据（包括NoAs的shadow As）；并测同L的新mid/low两draw和既有V1800两draw/完整V27000 clean。不能把源训练拟合、源V和未见RX混用。首批训练As保存真实IQ，不再把重建等同历史字节存档。

源条件在正式分数前固定：各As相对自身同空间NoAs的mid/low均值≥+.5pp，fullVclean净差≥−.2pp；LORA/TEMP进入后续优先路线还须urban不低于FULL_As。按合法源urban优先，完全并列后才比clean和成本。失败不扫rank/kernel/lambda/seed救回；raw IQ旁路与放开原W仅在后续证据支持时另立对照。本轮不自动扩E50；若源支持，后续最多再20轮达总E50，并固定匹配NoAs/FULL等控制及优化器/采样/流计数衔接。

固定 FINAL30 后逐行自动预测原clean+六practical，每view168000，全row prediction固定后独立truth-last。源决定先固定，不用正式结果选父、rank、lambda、epoch、seed、晋级或选择性重跑。报告all7 accuracy/macroF1/最差RX、RX/day/TX、修复/新增错误、suburban与自身clean差距，比较每空间As−NoAs与LORA/TEMP−FULL；一个model seed不证明可重复正收益。

## 设计逐项追踪

| ID | 附件来源 | 要求 | 目标文件 | 状态 | 验证 | 说明 |
|---|---|---|---|---|---|---|
| A01 | 一、九 | 合法固定clean F父、完整角色、不按target选父 | design.py/data.py/model.py | verified | strict own_model/role handles | 原B0 FINAL20，非旧高分父 |
| A02 | 五、七 | 标准浅中层LoRA，r8/alpha8、B零 | layers.py/model.py | verified | local_checks.json：真实F/BA merge/零函数PASS | 仅t1/t2 pw |
| A03 | 七 | TEMP读取块输入、A→Krank5→SiLU→B，preGN/prePool | layers.py/model.py | verified | 实际层hook及t2 signature梯度 | 不仅joint160末端 |
| A04 | 六 | 复用CRRA原则但不混入白化/门控/延迟 | layers.py/report | verified | 已读两个CRRA副本及实际DSConvBlock | 原类顺序/矩形差异显式保留 |
| A05 | 七 | 零输出精确等于原F，不能全零或再乘零gate | checks.py | verified | local_checks.json：逐位相等/state/RNG PASS | 所有后续物理路径保持 |
| A06 | 七 | 梯度穿过冻结层，无整学生no_grad | model.py/checks.py | verified | B首步梯度/A-K后续梯度 | 只参考父no_grad |
| A07 | 九 | FULL共同头规则，原W固定且不另训头 | model.py/optimizer.py | verified | 参数集合/逐项W不变 | 用户新附件的明确例外 |
| A08 | 八 | clean/As/keep三项，parent错时真值优先 | loss.py/train.py | verified | 解析公式/父detach/As全正CE | 用户选择L-only已确认 |
| A09 | 八 | 同clean输入，有限保持强度、无全特征强制相等 | loss.py/train.py | verified | 初始KL0/非零纠错梯度 | tau1、lambda1固定 |
| A10 | 七、九 | 同一父、三空间×NoAs/As、同E20+10预算 | design.py/experiment.json | verified | 六行显式配置/2220步 | NoAs第二槽raw，两CE总尺度匹配 |
| A11 | 九 | 完全相同训练观察与源诊断，L/U/V不越权 | data.py/bank.py | verified | 全ID序列与流消费/既有角色 | 不按模型重抽信道 |
| A12 | 九 | 区分训练实际As、同L新信道、V新观测 | diagnose.py | verified | 保存首次训练IQ/完整读出 | L1800，非69样本全L结论 |
| A13 | 九 | 新模块进入实际joint160和默认头预测 | model.py/checks.py/evaluate.py | verified | 零出→启用→抹除的实际输出变化 | teacher无部署依赖 |
| A14 | 九 | source决定后才读正式数值，不以目标调参 | report.py/dispatch.py | verified | predeclared source decision/scorer order | 已暴露benchmark如实标注 |
| A15 | 九 | 七环境、F1、最差RX、修复/新增错、成本 | evaluate.py/analysis/final_report | verified | 全预测/独立CM与配对重计 | n1标准差N/A |
| A16 | 九 | E20父成本+新增E10，源支持才可总E50 | design.py/report | verified | 实际epoch/update/optimizer/flow记录 | 原父optimizer/EMA不恢复 |
| A17 | 五、七 | 先比较LoRA/TEMP/FULL再考虑raw旁路 | 无本轮raw模块 | deferred | 本轮实际拟合/泛化结果后判断 | 附件明确条件式第二阶段 |
| A18 | 七 | 放开原W的独立消融 | 无本轮unfreeze行 | deferred | 首轮统一固定W后再判断 | 不是漏做当前矩阵 |
| A19 | 执行 | 单commit/聚焦负测/真实CP smoke/一次独审/唯一owner | checks.py/publish.py/dispatch.py | verified | 最小八项流程实际证据 | 不加SHA/审批链 |
| A20 | 本次用户回复 | 新增阶段仅L三项，U隐藏且不使用 | design.py/data.py/train.py | verified | 实际batch角色/无U分支 | 原健康任务U不修改 |

研究依据仅为 [LoRA](https://arxiv.org/abs/2106.09685)、[残差适配器](https://arxiv.org/abs/1705.08045)、[Learning without Forgetting](https://arxiv.org/abs/1606.09282) 的机制原则，已读取主源，不借用图像/LLM结果保证RFFI收益。当前数学结构计划严格对应附件提出的适配公式，新增预算、共同eval/autograd、L-only及NoAs槽位均在本报告显式固定。

本地结构验证 PASS：36 个合成 NN，0 checkpoint、0真实 IQ、0 query，8项负测。FULL实际可训练特征231233，LoRA2304、TEMP2384；零输出预测逐位相同、更新后joint160及默认预测改变、原W与适配器行固定原F状态逐位不变。FULL物理Sinc更新解析对照通过。详见local_checks.json。独立审查与真实CP/实际训练仍待执行，implemented不等于性能或远端验证完成。

一次独立 P0/P1 审查 PASS（review.json）：当前未发现会使下一次真实运行错误、越权、破坏、无法启动或不能生成合法预测的缺陷。19个Python文件语法和6行配置已核对。流检查另通过2220×128固定序列、45次完整L覆盖、源排序/失败及3项禁止角色负测，0NN。真实checkpoint smoke、共享实际观察生成、远端训练与七环境证据仍待返回。


技术恢复说明：r01首步记录拼接失败、0训练/0bank/0query，原产物保留；这里r02只修smoke参数统计中的重复style字段，科学方案完全相同。未更改parent、层、rank、kernel、目标、采样、epoch、seed、默认头或测试；不是重跑负科学结果。

一次独立 P0/P1 审查 PASS（review.json）：当前未发现会使下一次真实运行错误、越权、破坏、无法启动或不能生成合法预测的缺陷。19个Python文件语法和6行配置已核对。流检查另通过2220×128固定序列、45次完整L覆盖、源排序/失败及3项禁止角色负测，0NN。真实checkpoint smoke、共享实际观察生成、远端训练与七环境证据仍待返回。

同类记录拼接检查另发现completion中的default_head_frozen重复字段，一并删除显式重复值（counts仍保留True）。针对所有counts/info拼接的实际AST复现及修后扫描通过，0NN；不改变计算、状态、配方或预算。

本恢复实际代码/配置 commit 38316bc17e13db7acc0032dcda030afbcf414abf，远端分支 OID=本地 HEAD 已独立核实；先前r01版本仅为保留历史，不是本次执行身份。

启动证据见launch_evidence.json：单归档传输、远端compile、普通账号调度器PID/CWD/argv及日志增长VERIFIED；真实B0父8NN无query smoke PASS。共享观察逐步生成，没有据此声称六行已训练完成。

启动证据见launch_evidence.json：单归档传输、远端compile、普通账号调度器PID/CWD/argv及日志增长VERIFIED；真实B0父8NN无query smoke PASS。共享观察逐步生成，没有据此声称六行已训练完成。

源端完整回放与固定决定已保存，见source_closure.md。正式预测按固定流程继续。

完整六行结果见final_report.md、results_summary.json与validation.json；目标未据n1自动标完成。
