# 完整 practical E50：三种子正式结果与源域状态读出

**E50 进一步改善了 low／mid urban，但没有解决 clean 代价。** 固定三种子 2026092701／02／03 全部完成，训练后逐 row 自动生成 clean＋六种 practical 预测，再独立 truth-last 评分。相对各自 As30，As50 的 low urban 平均增加 **1.156746 ± 0.092682** 个百分点，mid urban 增加 **0.954167 ± 0.156568** 个百分点，三个种子方向一致。相对匹配 NoAs50，clean 平均变化 **−0.150198 ± 0.955701** 个百分点，超过允许下降 0.1 个绝对百分点的范围。尚未获得可重复的 clean／星地双提升证据；low urban 均值 51.926587%，仍低于 60%。

这是完整 practical 配方的预算复制，不是新增机制消融，也不是把三个种子挑成两个。03 的 clean 退步完整保留，不补选、剔除或选择性重跑。所有 ± 均为三个模型种子的配对差样本标准差（ddof=1），不是置信区间；信道 draw 不计为模型种子。

## 固定条件与执行来源

基座保持 F／SIG_TIME 三路、统一 Sinc 修正、joint160、跨 RX 同 TX 对比训练和现有内部门控，默认 cosine 分类头随网络联合训练。既有 CE、同 IQ dropout／RDrop、对比项及原半监督安排保持；本轮不能称为“只增加 paired-CE”。未新增部署结构或诊断头。L6300／U56700（TX 真值隐藏）／V27000、split392005、源 RX1／3／4／6／8 和源 day1／2／3 不变，没有留出一个训练 RX。

每行从自身合规 FINAL30 student＋own EMA 严格恢复，再追加 20 轮，重置追加段 AdamW／cosine20、段计数与缓存。累计 50 轮／11100 更新；轨迹为 20＋10＋20，不能冒充从零连续 cosine50。01 复用已完成原 E50 配对；02／03 由唯一原 owner 各完成 NoAs／As 两行。本对话没有重复启动同矩阵。

实际新增训练 release commit 为 `7bbb5536cb72c9b95c7c2045ba660e1e72e234e7`；原 owner 三种子分析 commit 为 `5f3a0fc9c128944994f40d4f4858a5f8e7730b40`。源域状态零前向分析 commit 为 `b217a2554ea89d148c4b21ab9ea556381599260f`。原 owner 为“审计并设计 CVS 联合增强”；读取其完成产物作此交叉读出，没有向该对话发送消息。

实际物理口径：fs=25 MHz、IQ256、native 归一化位置继承，practical residual／post_sync、EQ=false（G=1）。原训练 actual_channel.json 的 fc=2.462 GHz；固定正式六场景 received manifest 和源诊断 panel actual_engine.json 均为 fc=2.45 GHz。此处按原始元数据纠正引用文字中“训练／正式均为 2.462 GHz”的错误，没有改变 IQ、信道、SNR、困难样本或运行配置，不把这项差异解释为收益原因。

## 全部正式场景

每个模型、每个视图均为原固定 168000 个正式样本。As50 宏 F1 与最差 RX 栏为三种子各自指标的均值；最差 RX 不是所有种子中最小的那个值。

|环境|NoAs50 准确率 %|As50 准确率 %|匹配变化 pp（均值 ± SD）|As50 宏 F1 %|As50 最差 RX %|
|---|---:|---:|---:|---:|---:|
|clean|84.714087|84.563889|-0.150198 ± 0.955701|84.214124|77.687500|
|practical_high|47.103770|80.764286|+33.660516 ± 8.238954|80.492918|72.925000|
|practical_mid|45.710119|78.223016|+32.512897 ± 7.553454|77.979399|70.998611|
|practical_low_suburban|44.131746|75.784524|+31.652778 ± 6.656782|75.554855|68.918056|
|practical_high_urban|45.565675|77.272421|+31.706746 ± 7.664944|77.032625|70.173611|
|practical_mid_urban|38.417460|62.989484|+24.572024 ± 5.066220|62.754430|57.552778|
|practical_low_urban|32.764683|51.926587|+19.161905 ± 3.110887|51.537900|47.713889|

urban 三场景均值：NoAs50 38.915939%，As50 64.062831%，配对变化 +25.146892 ± 5.278997 pp。三个 suburban 与 As50 clean 均值的差距依次为 3.799603／6.340873／8.779365 pp，不能声称已接近 clean。

|种子|As50 clean %|clean 匹配变化 pp|As50 mid urban %|mid 匹配变化 pp|As50 low urban %|low 匹配变化 pp|As50 urban 均值 %|
|---|---:|---:|---:|---:|---:|---:|---:|
|2026092701|85.660714|+0.365476|63.901786|+29.864881|52.387500|+22.508929|65.010913|
|2026092702|84.394643|+0.436905|62.563690|+19.767857|51.659524|+16.358929|63.730556|
|2026092703|83.636310|-1.252976|62.502976|+24.083333|51.732738|+18.617857|63.447024|

01／02 的 clean 相对匹配 NoAs50 为正；03 为 −1.252976 pp。均值和全种子口径都未满足 0.1 pp 阶段目标，不能把单种子双提升写成稳定双提升。

## 多训 20 轮贡献了多少

预算交互 = (As50−NoAs50)−(As30−NoAs30)，同时包含 As 改善和 NoAs 退化。

|环境|As50−As30 pp|NoAs50−NoAs30 pp|预算交互 pp|
|---|---:|---:|---:|
|clean|-0.017659 ± 0.296162|-0.188492 ± 0.468416|+0.170833 ± 0.478334|
|practical_high|+0.437500 ± 0.463044|-3.768254 ± 1.438463|+4.205754 ± 0.984328|
|practical_mid|+0.587698 ± 0.372703|-2.836905 ± 1.375936|+3.424603 ± 1.007514|
|practical_low_suburban|+0.699603 ± 0.207028|-2.208135 ± 1.303563|+2.907738 ± 1.118605|
|practical_high_urban|+0.575992 ± 0.447469|-3.436706 ± 1.273573|+4.012698 ± 0.833477|
|practical_mid_urban|+0.954167 ± 0.156568|-1.673016 ± 0.867673|+2.627183 ± 0.718571|
|practical_low_urban|+1.156746 ± 0.092682|-0.720238 ± 0.538933|+1.876984 ± 0.491164|
|urban_mean|+0.895635 ± 0.219271|-1.943320 ± 0.890621|+2.838955 ± 0.671738|

As 自身 urban 均值增加 0.895635 pp，而 NoAs 自身下降 1.943320 pp。因此 As50−NoAs50 的 +25.146892 pp 不能全部解释为本轮追加预算带来的学习增益。03 的 practical_high 在自身延长后下降 0.071429 pp，保留该负值，不宣称所有环境和分层均改善。

|种子|clean 修复／新增错误／净修复|mid urban 修复／新增错误／净修复|low urban 修复／新增错误／净修复|
|---|---:|---:|---:|
|2026092701|4153／3539／614|56661／6488／50173|46642／8827／37815|
|2026092702|4984／4250／734|40362／7152／33210|36940／9457／27483|
|2026092703|3470／5575／−2105|46787／6327／40460|40064／8786／31278|

以上均为 As50 相对匹配 NoAs50 的逐包配对统计。全部逐 row／逐 seed／七环境、RX／day／TX、F1、默认头 logit／30 间隔和其他配对错误明细保存在原 owner 已完成汇总中，不另行预测或评分。

## 哪些 urban 状态获得改善

这里单独连接已完成的合法源 V1800 状态分析；该 CPU 任务只读已有源 logits 和物理元数据，未读正式文件。相对自身 As30，As50 在 mid urban 的 BLOCKED/degraded 组（每 seed 656 次重复观测）提升 +5.640244／+3.506098／+3.353659 pp，配对均值 +4.166667 ± 1.278429 pp；low urban 的 BLOCKED/unlocked 组（1565 次）提升 +1.533546／+2.044728／+3.130990 pp，均值 +2.236422 ± 0.815792 pp。二者三个模型种子均同向。

新增噪声比 3 至 8 dB 组的 mid／low 平均增量分别为 +2.785515／+3.232578 pp；8 至 15 dB 组为 +4.436860／+3.190319 pp。这里是 signal_to_added_noise_db，不是总 SNR。此类分组相关，不能把各组收益相加或宣称单独修复某种多径、CFO、噪声损伤。残余 CFO 1 至 10 kHz 的 mid 组仅 48 次，预算交互三种子均负；AGC、EQ 噪声着色和边界缺乏足够组间变化，不能宣称其机制得到修复。

源 panel EQ 关闭、噪声着色为平坦、reflect 边界，数值 FIR 延迟 24、核长 62、有效内部观测 195／256。未做新的受控移除损伤分类实验；因此“在这些状态中识别改善”成立，“改善由某个唯一物理因素造成”尚不成立。正式数据没有对应状态逐包标签，本轮没有正式状态准确率证明。完整分层及小／空组见 [源域最终分析](<E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01/final_analysis.md>)。

源 full V27000 与 panel 也分开报告：原冻结 full V clean 的 As50−NoAs50 三种子均值 −0.069136 ± 0.029937 pp；固定 V1800 panel 为 −0.259259 ± 0.084863 pp。前者来自原生冻结记录，不冒称全 V 逐包重算；后者的三个种子都为负。它们均不能替代正式跨 RX clean 的 −0.150198 pp。

## 实际预算、检查与产物

|成本组|模型 epoch|更新数|训练段及自身源／正式读出样本前向|
|---|---:|---:|---:|
|本次新增 02／03 四个 E50 追加段|80|17760|29842096|
|复用 01 两个 E50 追加段|40|8880|14921048|
|复用六个 E30 父模型|180|39960|54895716|

本次四行加共享 smoke 3074，合计 **29845170 次样本前向**。01 原追加段加其共享 smoke 1538 为 14922586；六个追加段共 120 model-epochs／26640 更新／44767756 次样本前向。连同六个 E30 父模型共 300 model-epochs／66600 更新／99663472 次样本前向；旧父模型共享 smoke 不任意分摊。本报告与连续 E30 报告共享部分父模型，不能将两份累计成本直接相加。

每新增行学生 train／eval 为 4543920／594002，own EMA 教师 train／eval 为 0／1134000；源读出 12600，正式预测 1176002。四行 native 训练段墙钟依次为 NoAs02 1626.249 s、As02 2195.350 s、NoAs03 1589.310 s、As03 2170.395 s；正式预测分别 47.767／46.756／46.929／46.388 s。它们是各模型段耗时，不是并行总墙钟或算法加速比。

独立后状态：原 dispatcher／四训练进程自然退出，failure 为空，source／formal／完整日志均完成。新增 17760 个 step、80 个 epoch 完整解析，匹配物理序列和普通增强 MSE mismatch 均为 0。每 row 七视图预测固定后才独立评分；后续汇总等待所有 12 逻辑条件×7 视图完成，核对 1512 个混淆矩阵及 252 次分区合并一致。正式分数没有回流训练、选模、挑种子或选择性重跑。预算结束不等于收敛。

本对话新增源域 wrapper 的六项定向 fixture、编译、一次独立 P0／P1 审查、N607 预检、单个发布包 SHA 比较和远端编译／六项检查均通过。源域 PID1197720 自然退出，96 个 source accuracy／F1 对齐、2178 个汇总分层保留；模型前向／IQ／checkpoint／更新／正式评分均为 0，CPU wall18.792 s。没有技术缺项；科学目标仍未完成。

公开结果：[原 owner 完整逐种子报告](<E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-As-budget-e50-s0203-m4-r01/final_analysis.md>)；[保存的正式汇总 JSON](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/owner_reference/formal_summary.json>)；[保存的成本 JSON](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/owner_reference/costs.json>)。

图表：[三种子正式配对结果](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_budget_E50_s0203_20261018/final_figures_r02/matched_e50_three_seed.png>)、[正式预算交互](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_budget_E50_s0203_20261018/final_figures_r02/matched_e30_to_e50_budget.png>)、[三种子源状态预算图](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/retrieved/result/three_seed_state_budget_deltas.png>)。[E31–50 完整训练健康曲线 PNG](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/report_figures/E31_E50_all3seeds_training_health.png>)／[PDF](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/report_figures/E31_E50_all3seeds_training_health.pdf>)；[CE／RDrop KL 分量曲线 PNG](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/report_figures/E31_E50_all3seeds_objective_components.png>)／[PDF](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/report_figures/E31_E50_all3seeds_objective_components.pdf>)。这些曲线使用原 owner 完整 training_logs.json 中三种子×As／NoAs 共 120 条实际 E31–50 epoch 汇总，没有补点或平滑；包含 native loss、日志中的源 V TX 准确率和骨干梯度。As／NoAs 输入难度不同，损失绝对值不能直接用来比较识别优劣，曲线也不证明收敛。绘图为 CPU 结果后处理，0 新前向／IQ／checkpoint／正式评分，独立于源状态程序的 18.792 s；[数据覆盖与渲染说明](<E:/type10-7/local_artifacts/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01/report_figures/rendering_note.txt>)。

源域可执行实现：[三种子 wrapper](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_budget50_source_states58_20261018.py>)、[六项定向检查](<E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/check_budget50_source_states58.py>)，实际命令及环境见 [执行记录](<E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01/experiment.json>)。

本轮结论：更多预算对固定完整 As 的 urban 适应有一致的小幅增益；clean 的种子波动仍是主要矛盾。原完整 As 保留为实际有效的星地参照，但不能晋级为满足 0.1 pp 或可重复双提升的最终方案。下一候选尚未启动；不从本轮正式分数挑 seed、改源阈值或扩到 E70。
