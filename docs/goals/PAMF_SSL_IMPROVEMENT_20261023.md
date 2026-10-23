# PAMF 半监督收益目标

用户于 2026-10-23 要求设计并验证适合 PAMF 的半监督方法，超过 PAMF 监督版与 PAMF 监督部分＋普通伪标签版，并再次明确这是**目标测试集性能**。提升单位固定为百分点。完成条件是同一方法、同一数据契约、相同默认分类头和 E50/11100 更新预算下，在固定未见接收机 R_t 目标测试集上，三个配对 F 模型种子的 raw clean 均值相对两个对照分别至少提高 1 个百分点，六种 practical 场景的种子内均值相对两个对照分别严格提高 1.5 个百分点。最终以未四舍五入的目标测试计数计算；用户表格对应近似门槛为 clean ≥85.98%、六景均值 >72.10%。两个指标必须由同一方法同时满足，不能拼接候选或种子。源 V 的准确率、损失和物理视图诊断只作健康及机制证据，不能用于宣布目标达成。

F0 每视图168000条、三个seed的准确率验收使用整数计数：clean总分母504000，六景总分母3024000。相对各对照，clean须满足`100*新增正确数 >= 504000`，六景须满足`200*新增正确数 > 3*3024000`。当前最强对照对应最低正确总数分别为433348、2180190。这保留原百分点评价，不将浮点舍入误差算成增益；每seed的差值与样本SD仍单独报告，不将包数当独立模型样本。

原数据为 ManySig F0：六 TX，源 RX1/3/4/6/8、源 day1/2/3，L6300/U56700/V27000，目标七 RX、四日期、168000 物理样本。raw clean 与六 practical 是主验收口径；DURA 读出不用于替代 raw 半监督收益。本研究不自动替换当前 PAMF-T1-E50 或 DURA v1.0 默认版本。

资源使用 N607 的 3 张显卡，模型种子 2026092701/02/03 各固定一张卡，每卡最多 2 个训练或预测 worker，总计最多 6 个；独立 CPU scorer 最多 1 个。首轮绑定 GPU0/1/2，启动前核实容量，不停止其他 owner 的健康进程。具体参数、命令、输出和技术停止规则见首轮技术替代 run：`20261023-phase1-PAMF-physical-softmatch-m6-r02`。原 r01 在正式派发前的 GPU smoke 因动态层设备放置失败，0 次正式训练、0 次目标访问；旧发布与失败证据保留，r02 不改变科学矩阵。

首轮路线为 PSM（物理视图软匹配）及 RXPSM（增加跨接收机标签原型参考），每条路线三个实际 F 种子。两者从零训练并保留原 20＋10＋20 三段日程、13L＋37LU、监督成对 CE/KL/CRIC 和默认头；不继承按目标成绩挑出的模型。已有正确输入绑定的 F20 监督对照及 F21 普通 PL 对照直接复用。

后续迭代只能基于合法源域训练、V 诊断和机制证据；参数及矩阵先于新目标访问冻结，不把目标差距变成调参反馈。科学负结果完整保留，不能作为技术故障终止健康任务。全部目标预测冻结后独立评分并报告每 seed、每场景、RX/TX/day、均值与样本 SD、配对差、无标签有效曝光和实测成本。未达到两项门槛时目标保持未完成，不将已实现、已启动或仅源域提升写成成功。

普通固定阈值 PL 是必须保留的对照。仅降低阈值无法单独检验物理鲁棒性与身份参考；实例记忆队列更大且可能强化同 RX 邻近关系。首轮因此选择连续权重＋同记录信道视图，再用其他源 RX 的 L_s 原型隔离验证跨 RX 身份证据。它们是研究假设，文献不能证明本项目的预期涨幅。

参考：

- Chen et al., SoftMatch: Addressing the Quantity-Quality Trade-off in Semi-supervised Learning, ICLR 2023, https://arxiv.org/abs/2301.10921 。本研究仅借鉴单边高斯置信加权，不声称完整复现或原创该加权函数。
- Zheng et al., SimMatch: Semi-Supervised Learning With Similarity Matching, CVPR 2022, https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_SimMatch_Semi-Supervised_Learning_With_Similarity_Matching_CVPR_2022_paper.html 。本研究使用标签来源受限、排除自身 RX 的类别原型，不声称其等于论文的实例记忆匹配。


## 首轮目标测试与继续研发

首轮 `20261023-phase1-PAMF-physical-softmatch-m6-r02` 的六行、42 个 raw 目标视图均完成独立 truth-last 评分。PSM 的 clean／六景三种子均值为 84.222619%／71.897288%，RXPSM 为 84.272421%／71.853571%，两者均未达到完整目标。完整逐种子、景、RX/TX/day 和实际训练成本见该 run 的 target_results.md 及配套汇总文件。该 run 为 ANALYZED，整体 goal 继续 ACTIVE，原 PAMF/DURA 默认版本不变。

后续 `20261023-phase1-PAMF-viewanchor-m3-r01` 已在首轮目标分数读取前固定 RXPSM E30 父来源、0.10 物理特征一致性和三个 seed，独立续训 20 轮，累计仍为 50。新调度器已独立核实，当前 QUEUED 等待 GPU 容量，尚未 smoke 或训练。不会因本轮目标分数改动这套配方、挑 seed 或选择性重跑。原目标仍为同一方法相对两个对照 clean 至少 +1 pp、六景平均严格超过 +1.5 pp，验收只看目标测试。


## 第二轮已实际启动

VIEWANCHOR r01 在任何 smoke／训练之前因原卡持续占用而被资源替代；记录和旧输出均保留，独立确认旧控制器退出。相同科学配方的新不可覆盖 run `20261023-phase1-PAMF-viewanchor-m3-r02` 已在 GPU5/6/7 各跑一个固定 seed，执行 commit 为 6226ad1e7cce97846702fe15ceba866d4acf64c9。真实 E30 no-query smoke PASS，三行 PID/CWD/命令／实际 GPU／日志增长均核实，首次 epoch 证据到累计 E32；部分 step 前缀确认 U 状态衔接和新增损失非零。当前仅有源训练证据，原目标保持 ACTIVE。

源端后续“干净／信道配对 U”假设已记录在 `20261023-phase1-PAMF-paired-U-m3-r01/report.md`，只固定同一 E30 分叉、完整 clean student U 分支及其损失分解；尚未实现或启动。当前健康 VIEWANCHOR 训练保持原不可变 release。


## 用户追加双 readout 测试（2026-10-23）

用户明确要求同时报告 PAMF 单模型与固定 DURA v1.0＋同一 PAMF。新增独立冻结补测 `20261023-phase1-PAMF-SSL-DURA-reference-m9-r01`，覆盖 PSM、RXPSM、VIEWANCHOR 各三个固定模型种子、七景168000物理样本；原单模型预测和匹配 F20/F21 的既有 DURA 结果复用。健康续训不热改，G 不重训或调参。两套结果分别与相同 readout 的监督及普通 PL 对比，并报告 DURA 的纠错/损伤。原单模型 clean≥+1pp、六景>+1.5pp 验收保持，DURA 参考不替代该目标。完整矩阵与资源见新 run 的 experiment.json/report.md。

后续每个新 PAMF 半监督候选均默认完成单模型 raw 与固定 DURA v1.0＋同一模型两套正式测试，分别与相同 readout 的监督版、普通 PL 版比较。各自覆盖原定全部模型种子、clean 与六种 practical；旧权重/已有预测合法则复用，新增组合单独登记，不热改健康任务。两套均报告逐景、逐 seed 和配对差，DURA 增加同模型 raw 的纠错/损伤计数；主 SSL 目标仍按单模型验收。


## 配对U方案已实现并发布排队

`20261023-phase1-PAMF-paired-U-m3-r01`：沿用此前源端固定假设，43项检查与一次限定P0/P1审查PASS，执行commit`de92f461949dbb0163745c0cde009a433e6fc854`。原RXPSM各seed E30学生/自身EMA/统计/标签原型独立追加20轮，full-U原IQ与同记录Practical输入concat，固定损失系数与实际曝光见该run的experiment.json/report.md。每seed在其九模型参考预测三行全部释放GPU后自动接续，不热改旧训练；最终E50同时测试raw与固定DURA v1.0。控制器已核实存活、三行排队，source smoke尚未开始；目标仍未证明达成。


## 当前已完成结果与配对训练（双读出）

VIEWANCHOR三seed raw已完整评分：clean84.379563%、六景71.946958%，对F20为-0.405754/+1.350761pp，对F21为-0.602183/+1.587599pp，原目标未达成。独立九模型DURA参考矩阵也已全齐：VIEWANCHOR组合84.448214%/77.521759%，监督组合84.857540%/77.854894%，普通PL组合85.029563%/78.074636%；三种新方法组合两个主指标均低于这两个对照。完整两套表与逐seed/景/RX/TX/day/净纠错、实际曝光和预测成本见`automation_reports/CV-SincNet/20261023-phase1-PAMF-SSL-DURA-reference-m9-r01/target_results.md`。

PAIREDU三seed已在各自GPU5/6/7实际续训，真实E30 no-query smoke和配对梯度PASS，累计E31；指标及父权重在上述新目标读取前已经固定，保持原计划。原目标仍ACTIVE，不能以局部六景收益、DURA相对raw收益或source运行健康替代达标证据。


## 无标签浅中层跨接收机扩展已实际运行

`20261023-phase1-PAMF-U-CRIC-m3-r01` 已固定同原RXPSM E30分叉、保持原U输入/软CE/原型损失，新增原生time_down/t2.pool U对比，系数0.10及温度0.20复用L CRIC；50项检查、一次P0/P1审查、真实E30 source-only梯度smoke全部PASS。执行commit`b249b75123e24acadd80d744b2b0f6e5d79f265d`，三seed实际训练，新增损失及浅中层梯度非零，未访问新候选目标。GPU5/6/7各有原PAIR＋新UCRIC两条本任务worker；含外部MATLAB总数最多3，仍保留>=10000MiB派发门槛，不热改或停旧任务。PAIR已累计E39，新UCRIC在E31；两路最终均自动完成raw与同一固定DURA的七景预测及独立评分。此前三种方法负结果完整保留，原目标继续ACTIVE。


## 源状态诊断与下一条预先固定假设

九模型源状态未显示类别塌缩或缺失L-only类/RX原型；固定720条source-V E30 CPU诊断未支持换监督父来源，继续原RXPSM E30。现有DURA记录披露历史正式暴露后的人工父选择，尚不能确认适合作新F训练祖先，故仅保留用户指定冻结组合参考。实际诊断和边界见九模型reference报告source_decisions.md。下一条`20261023-phase1-PAMF-U-MixStyle-m3-r01`固定同TX预测跨RX的U浅中层统计混合（MixStyle p0.5/alpha0.1，置信>=0.5，仅student U，原E30+20），三seed及双读出矩阵已登记，尚未实现或启动；等待同seedPAIR GPU工作结束后用第二名额，UCRIC健康任务保留。新目标未访问，原goal保持ACTIVE。


## 标签保护U统计混合已发布等待

`20261023-phase1-PAMF-U-MixStyle-m3-r01`严格实现13dcd1e6预先固定假设，61项聚焦检查、一次独立P0/P1审查PASS，执行commit`68ed7530a3c3caf96f7a54e1cfe4fc3c2a00d2f7`。CPU控制器已独立核实，三行排队等待同seedPAIR全部GPU预测完成，再与已派发的UCRIC共用最多两个本任务名额/card；无新GPU smoke/训练/目标读取。原E30+20及raw/DURA双七景自动收尾不变。当前两路健康训练不热改或停止，性能目标仍ACTIVE。
