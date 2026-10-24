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


## PAIREDU 完整双读出结果

`20261023-phase1-PAMF-paired-U-m3-r01` 已完成三 seed、每景 168000 条的 raw／同一 DURA 七景预测及独立 truth-last，42 项读出／场景完整，全部新增 13320 条闭合训练记录已核对。raw clean=84.543849%；DURA-LU clean=84.620437%；raw six_mean=71.978307%；DURA-LU six_mean=77.507606%；原数值目标仍未满足，维持 ACTIVE。完整负结果、两种匹配基线、逐 seed／场景／分层及成本均已保存，不从本结果修改已固定 UCRIC／UMIX 的方法、父权重或矩阵。


## 无标签特征统计混合实际运行

`20261023-phase1-PAMF-U-MixStyle-m3-r01` 已通过真实 E30 source-only smoke，三个 seed 在 GPU5／6／7 各接续已完成预测的 PAIR 名额，与 UCRIC 并行，每卡本任务两条、含外部最多三条。正式配方仍是原 E30＋20、原 U 输入与损失、time_down／t2.pool 同伪类跨源 RX 的停止统计梯度混合，每层独立 p=0.5、alpha=0.1；完整前缀计数与实际启用已核对，无新目标访问。一次 61 项检查／P0/P1 审查及真实 source-only 检查全部 PASS，未从刚完成 PAIR 的结果修改原假设、系数、父来源或测试矩阵。执行 commit `68ed7530a3c3caf96f7a54e1cfe4fc3c2a00d2f7`。


## 源域梯度冲突与下一条固定假设

原 E30 同 seed 的24个 CPU 源 L/U 受控批次中整体13个负梯度内积，中层与默认头各14个；无 optimizer／EMA／保存参数更新、U 真值或目标。具体输入、局限和数值见九模型reference源梯度附录。下一条 `20261023-phase1-PAMF-U-GradProtect-m3-r01` 固定原 RXPSM E30+20，现有教师/输入/损失保持，仅对 t2／默认头／其余参数分别单侧投影负内积 U 梯度，再限制 U 范数<=L；近零 L 组 U 置零。三 seed、固定最终E50、raw＋同一DURA七景及普通PL／监督对照登记完整，当前只是 PLANNED。等同 seed UCRIC GPU 完成后与 UMIX 各占每卡一个名额，CPU评分在UMIX之后。此假设在UCRIC／UMIX目标成绩读取前固定，不由刚完成PAIR或后续测试分数修改规则或父来源；原goal保持ACTIVE。


## UCRIC 完整双读出结果

`20261023-phase1-PAMF-U-CRIC-m3-r01` 已完成三 seed、每景 168000 条的 raw／同一 DURA 七景预测及独立 truth-last，42 项读出／场景完整，全部新增 13320 条闭合训练记录已核对。raw clean=84.290675%；DURA-LU clean=84.400198%；raw six_mean=71.866171%；DURA-LU six_mean=77.185979%；原数值目标仍未满足，维持 ACTIVE。完整负结果、两种匹配基线、逐 seed／场景／分层及成本均已保存，不从本结果修改已固定 UCRIC／UMIX 的方法、父权重或矩阵。


## 梯度保护三 seed 已实际启动

`20261023-phase1-PAMF-U-GradProtect-m3-r01` 已以 commit `0aba8a89c8b74e3058a1bec727fef86805d75f6c` 发布，80项检查／一次P0/P1审查／真实E30 source-only smoke全部PASS。真实反向 .grad 逐参数符合显式保护梯度，三个组的冲突均被处理，默认头／教师／单次EMA权限正确；正式三seed已重新载原E30续训，每卡UGP＋UMIX两个本任务worker，含外部最多三个。实际step前缀约束与曝光计数逐条核对，新目标未访问。后续固定E50＋raw／同一DURA七景及独立评分不变，原goal保持ACTIVE。


## 源接收机差异与预先固定的类别条件矩约束

原E30三seed sourceU960／V720冻结描述子中，浅／中层同类RX质心差对应约34%至42%类内散布，30个类×源RX单元齐；没有新头、参数／教师／原型更新、U真值或目标访问。完整比例和边界见九模型reference源接收机附录。独立下一条 `20261023-phase1-PAMF-U-CMR-m3-r01` 固定原RXPSM E30+20，原损失／输入／教师保持，新增一次raw U学生前向，对浅中层同伪类不同源RX的置信加权均值／标准差做0.1矩约束，权重>=0.5、ESS>=4，无矩EMA库或新推理参数。三seed、最终E50、raw＋同一DURA七景、双基线及成本完整登记，当前只是PLANNED。等待同seedUMIX GPU结束后与UGP各占每卡一槽，CPU评分在UGP后；UMIX／UGP健康训练保持，矩阵在它们的目标成绩读取前固定。原goal仍ACTIVE。


## UMIX 完整双读出结果

`20261023-phase1-PAMF-U-MixStyle-m3-r01` 已完成三 seed、每景 168000 条的 raw／同一 DURA 七景预测及独立 truth-last，42 项读出／场景完整，全部新增 13320 条闭合训练记录已核对。raw clean=84.362897%；DURA-LU clean=84.433135%；raw six_mean=72.000893%；DURA-LU six_mean=77.554134%；原数值目标仍未满足，维持 ACTIVE。完整负结果、两种匹配基线、逐 seed／场景／分层及成本均已保存，不从本结果修改已固定 UCRIC／UMIX 的方法、父权重或矩阵。


## 类别条件矩约束三seed已实际运行

`20261023-phase1-PAMF-U-CMR-m3-r01` 以执行commit `a3371f1339f8687ee6593afb71b8e0a79574a9cd` 发布，93项聚焦检查／一次P0/P1审查／真实E30 source-only smoke均PASS。新增raw U256、两层各56个同伪类跨RX对，真实浅中层梯度有限非零，默认头／教师／单次EMA及hook清理正确；正式三个seed重新加载原E30续训。GPU5／6／7每卡CMR＋UGP两个本任务worker，其他owner保留，全部完整step前缀计数和系数逐条核对，新CMR目标尚未访问。原E50＋raw／同一DURA七景／独立评分和原goal ACTIVE保持。


## 原始IQ身份梯度与双锚保护假设

原E30三seed固定24个源L/U批次的CPU诊断中，full-L保护后的U仍12次与原有raw L两前向真实CE负内积，最小余弦−0.291；full-L与raw CE本身24次均正。无参数／教师／原型持久更新、U真值或目标。L-only频谱RX估计仍有TX交互，当前暂缓。下一条`20261023-phase1-PAMF-U-DualAnchor-m3-r01`预先固定原RXPSM E30+20，只把原U梯度投影到full-L与raw CE双非负半空间交集，再限制范数≤full-L；不加网络前向／头／推理参数，原L梯度和全部scalar损失保持。三个seed与raw＋同一DURA七景矩阵已登记，当前PLANNED；在UGP GPU完成槽位接续，与CMR每卡各一条，CPU评分在CMR后。方法在读取UGP／CMR目标成绩前固定，原goal ACTIVE。


## UGP 完整双读出结果

`20261023-phase1-PAMF-U-GradProtect-m3-r01` 已完成三 seed、每景 168000 条的 raw／同一 DURA 七景预测及独立 truth-last，42 项读出／场景完整，全部新增 13320 条闭合训练记录已核对。raw clean=84.295437%；DURA-LU clean=84.366468%；raw six_mean=71.818254%；DURA-LU six_mean=77.420007%；原数值目标仍未满足，维持 ACTIVE。完整负结果、两种匹配基线、逐 seed／场景／分层及成本均已保存，不从本结果修改已固定 UCRIC／UMIX 的方法、父权重或矩阵。


## 双锚保护三seed已实际运行

`20261023-phase1-PAMF-U-DualAnchor-m3-r01`执行commit `51e4ccfc45a77d7e39c387e1157f43faf05365e8`；102项检查／一次P0/P1审查／真实原E30 source-only smoke均PASS。raw CE0.171661，t2双边界、默认头raw边界、其余full-L边界均实际生效，真实参数梯度与预定组合一致，L768／U256／单次EMA／无教师梯度正确；probe丢弃，正式三seed重新加载原E30。三个PID／CWD／命令／实际GPU和完整step前缀增长已独立核验，每卡双锚＋CMR共两个本任务worker，其他owner保持，新版尚未访问目标；固定E50、raw＋同一DURA七景与独立评分继续。原goal ACTIVE。


## 原E20视图梯度与下一条固定平衡方案

原F20 L-only E20与RXPSM E20、三seed各4批只读CPU源诊断，24次中16次两个U对raw真实CE作用异号；physical梯度约ordinary中层5倍／头8至9.5倍，合并后9次仍隐藏负分支。没有目标／U真值／V或持久状态更新；两起点均有此现象，不据此更换原E30父。下一条`20261023-phase1-PAMF-U-ViewBalance-m3-r01`固定同RXPSM E30+20，原输入与scalar损失保持，将两个U贡献分别双锚投影，每个非零方向归一到0.5full-L范数；可放大亦可缩小，总U≤L，无新增F前向／头／推理参数。当前PLANNED，待实现／检查／一次审查，接CMR释放GPU槽，与UDAP每卡各一条；CPU评分在UDAP后。方法在CMR／UDAP目标成绩读取前固定，原goal ACTIVE。


## UCMR 完整双读出结果

`20261023-phase1-PAMF-U-CMR-m3-r01` 已完成三 seed、每景 168000 条的 raw／同一 DURA 七景预测及独立 truth-last，42 项读出／场景完整，全部新增 13320 条闭合训练记录已核对。raw clean=84.336508%；DURA-LU clean=84.455159%；raw six_mean=71.944147%；DURA-LU six_mean=77.175198%；原数值目标仍未满足，维持 ACTIVE。完整负结果、两种匹配基线、逐 seed／场景／分层及成本均已保存，不从本结果修改已固定 UCRIC／UMIX 的方法、父权重或矩阵。


## 视图平衡三seed实际运行

`20261023-phase1-PAMF-U-ViewBalance-m3-r01`执行commit `81c4e8ba596412d850e478e45bef23c6d5fcbec8`；121项检查／一次P0/P1审查／真实原E30 source-only smoke均PASS。原raw CE、普通／物理各128和每视图双锚／半L预算实际生效，全部参数梯度、L768、单次EMA及无教师梯度正确；probe丢弃，正式三seed重载原E30。三个worker的PID／CWD／命令／实际GPU及完整step前缀增长已独立核验，每卡UVB＋UDAP共两条本任务worker；CMR对应GPU预测完成后交接，其他owner保持。原全局U系数不再控制归一化后范数，放大噪声风险披露；未宣称目标收益。新版尚无目标访问，固定E50、双读出七景和独立评分继续，原goal ACTIVE。


## 用户指定普通PL加权与监督机制用于U（2026-10-24）

实际原F21 E30在完整源V各27000条，.95接受后正确99.8868%/99.8366%/99.9027%，覆盖94.9037%/95.1815%/95.1296%；.98只增加少量纯度但减少正确样本覆盖，按预先源规则保留.95。U真值隐藏，不能称U实测精度。登记`20261024-phase1-PAMF-HardPL-T1U-m6-r01`，HPL2权重1→2与HPLT1U权重2＋U六视图监督/RDrop/AsKL0/LS.01/浅中层跨RX对比，CE剂量归一化1.25，均原F21同seed E30＋20到E50/11100；每seed各两候选。153项检查和一次P0/P1审查/一次P1定向修复复查PASS；待UDAP/UVB合法卡位，真实源smoke后自动训练及双七景独立评分。不以源V或本地检查宣布目标收益，goal ACTIVE。


## UDAP 完整双读出结果

`20261023-phase1-PAMF-U-DualAnchor-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增13320条闭合训练记录已核对。UDAP / raw clean=84.254762%；UDAP / DURA-LU clean=84.336310%；UDAP / raw six_mean=71.805324%；UDAP / DURA-LU six_mean=77.429729%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 高置信硬 PL 的 U 信道单因素对照（2026-10-24）

登记 `20261024-phase1-PAMF-HardPL-noUAs-m3-r01`：实际 F21 E30 同记录源诊断显示 accepted U 的普通噪声 hard CE 约 0.002、信道约 0.59；这仅说明难度，不能证明原标签失效或目标收益。在任何新 HPL/UVB 目标成绩读取前，固定 HPL_NO_UAS 三 seed 对照；原 F21 E30＋20、阈值 0.95、权重 2、六次 U 前向、LS/RDrop/As KL0/浅中层跨 RX 与 CE 归一化 1.25 均保持，只把 U mixed-half 信道输入换成 exact raw，原 L 信道保持。160 检查 PASS，一次 P0/P1 审查和唯一 P1 定向复查关闭；等待同 seed HPL2/UVB GPU 完成与 HPLT1U 已派发的合法卡位，真实源 smoke 后继续。双七景完整评分并列 F20/F21 及 HPLT1U，目标仍 ACTIVE。


## 普通硬 PL 的监督标签平滑单因素对照（2026-10-24）

`20261024-phase1-PAMF-HardPL-LS-m3-r01` 在原 HPL2 上仅加入 U CE smoothing0.01，保留门控0.95/权重2/一次raw U学生前向；原 F21 同seed E30＋20、L及默认头保持。完整 HPL2 新增日志三模型各4440条显示 U/总损失比约2.7%，但原E30源24批梯度比中位0.2985，故保留权重2；同logits标签平滑使源方向冲突13/24→8/24，19批余弦改善。这是探索性源依据，不是Adam或目标效果保证。166检查及唯一独立P0/P1审查PASS，在任何新HPL/UVB目标成绩读取前固定三seed双七景矩阵；接替noUAs GPU完成卡位，实际源smoke后执行，完整truth-last再判断。目标仍ACTIVE。


## UVB 完整双读出结果

`20261023-phase1-PAMF-U-ViewBalance-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增13320条闭合训练记录已核对。UVB / raw clean=84.611706%；UVB / DURA-LU clean=84.636905%；UVB / raw six_mean=71.675430%；UVB / DURA-LU six_mean=77.545337%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 完整 HPL2 方法先行汇总（成绩读取前登记）

HPL2 三seed双读出七景42cell已全部独立评分。为符合逐row自动测试及快速研发要求，在任何 HPL2／HPLT1U 目标成绩值读取前，解除本地collector额外的“等另一方法全部完成”报告等待（REJECTED_EXTRA_GATE）；先列完整 HPL2 方法，原六模型报告和后续全部健康实验保持。所有后续候选已在本次读分前冻结，不依据新成绩改权重／门控／父源／候选排序或选择性重跑。范围仅报告时序，非数据或模型协议变更，原raw双目标不缩减。


## HPL2 完整方法目标结果

原注册三seed E50默认头、raw／固定DURA七景42cell全部独立评分。HPL2 / raw clean=85.075595%；HPL2 / DURA-LU clean=85.103770%；HPL2 / raw six_mean=70.187533%；HPL2 / DURA-LU six_mean=78.078671%；raw相对F21 clean+0.093849pp、六景−0.171825pp；原双目标未达到，goal ACTIVE。报告时序调整先于HPL2读分登记；原六模型总报告与全部健康训练／后续已冻结noUAs和LS保持，不依据本结果改配方或父源。完整证据见该run的HPL2_target_results.md，不能把单方法完成称为整个六模型实验完成。


## HPL2 / HPLT1U 完整双读出结果

`20261024-phase1-PAMF-HardPL-T1U-m6-r01` 各方法三seed、共6模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部84cell完成；全部新增26640条闭合训练记录已核对。HPL2 / raw clean=85.075595%；HPLT1U / raw clean=84.867063%；HPL2 / DURA-LU clean=85.103770%；HPLT1U / DURA-LU clean=84.949405%；HPL2 / raw six_mean=70.187533%；HPLT1U / raw six_mean=71.974702%；HPL2 / DURA-LU six_mean=78.078671%；HPLT1U / DURA-LU six_mean=77.508929%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## HPL_NO_UAS 完整双读出结果

`20261024-phase1-PAMF-HardPL-noUAs-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增13320条闭合训练记录已核对。HPL_NO_UAS / raw clean=85.423016%；HPL_NO_UAS / DURA-LU clean=85.493849%；HPL_NO_UAS / raw six_mean=70.657242%；HPL_NO_UAS / DURA-LU six_mean=77.985648%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 高纯度普通 PL 的跨源接收机统计混合

`20261024-phase1-PAMF-HardPL-MixStyle-m3-r01` 固定原 F21 E30＋20，与 HPL_LS2 匹配，只在一次 U 学生前向的 time_down／t2.pool 加 accepted-only 同伪类跨源 RX 的统计混合，p0.5／Beta0.1／eps1e-6 沿原源研究设置。依据原 F21 完整 V 伪标签纯度及既有源接收机统计差异，保留门控0.95／权重2／LS0.01；不是依据目标成绩选择参数或父模型。52 检查和一次 P0/P1 PASS；真实 checkpoint smoke 及三个父模型 V720 冻结身份配对诊断通过后立即继续原三 seed／双七景矩阵。与健康 LS 共用第二卡位，goal ACTIVE。


## 特征统计混合的同配方技术恢复

原 r01 真实 F21 checkpoint smoke PASS，随后冻结源诊断的 NumPy→Torch 索引类型不兼容，正式三行仍无 PID、无训练／目标访问。保留原产物，`20261024-phase1-PAMF-HardPL-MixStyle-m3-r02` 仅换 Python int 列表索引；配方、父模型、资源和矩阵不变，8 项定向检查及唯一 P1 定向复查关闭。新真实 smoke／源身份配对诊断通过后自动继续。


## HPL_LS2 完整双读出结果

`20261024-phase1-PAMF-HardPL-LS-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增13320条闭合训练记录已核对。HPL_LS2 / raw clean=85.070040%；HPL_LS2 / DURA-LU clean=85.100397%；HPL_LS2 / raw six_mean=70.300331%；HPL_LS2 / DURA-LU six_mean=78.131581%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 监督 U 与统计混合的独立源因子组合

`20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01` 固定原 F21 E30＋20，与 HPL_NO_UAS 匹配，只增加 accepted-only 同伪类跨源RX的浅中层MixStyle；一批的六前向共享同一计划，门控0.95／权重2／RDrop／LS0.01／AsKL0／对比0.1／CE剂量1.25保持。复用同父／数据／STYLE的既有源身份配对诊断，通过真实六前向smoke后用每卡第二名额与MS并行；48检查与一次P0/P1通过。完整双七景自动truth-last，不依已暴露目标分数选参数、父模型或种子。goal ACTIVE。


## E20 后提前利用高纯度 U 的固定匹配矩阵

原F21 E20完整V27000/seed在.95门控后已达99.9002%／99.8687%／99.9164%，覆盖约93%，无状态更新／U真值／目标读取。`20261024-phase1-PAMF-HardPL-E20plus30-m6-r01` 固定同seed原E20父权重、自己native10＋20，比较HPL_E20_LS2与HPL_E20_NAMS2；原机制参数不变，只将新机制暴露从20轮增为30轮，累计仍E50／11100、13L37LU。原E30不作初始化。48检查与一次P0/P1PASS，真实E20两臂smoke／新的V720配对规则通过后自动六行三seed双七景。接替当前MS／NAMS释放卡位；当前两条方法目标分数尚未读取，不以其修改矩阵。goal ACTIVE。


## HPL_MS2 完整双读出结果

`20261024-phase1-PAMF-HardPL-MixStyle-m3-r02` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增13320条闭合训练记录已核对。HPL_MS2 / raw clean=84.996429%；HPL_MS2 / DURA-LU clean=85.043254%；HPL_MS2 / raw six_mean=70.328340%；HPL_MS2 / DURA-LU six_mean=78.075860%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 高纯度 PL 的更高权重与完整 L 梯度保护

`20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01` 固定原E20各续训native10+20到E50，.95门控／权重6，LS与NAMS各三seed。原E20完整V .95精度99.87至99.92%；新阈值诊断显示放宽至.8的新增样本仅约91%正确，故不放宽。24个原E20 source L/U 梯度探针weight2 U/L中位数.318/.304/.356，14个冲突，因而固定完整L anchor的单侧分组投影／U范数帽1；不声称Adam或目标保证。无新前向／EMA／Utruth／头或G训练，有额外autograd成本。49相关检查与一次P0P1PASS，原E20真实两臂smoke首卡通过即自动固定六行raw+DURA七景。接替旧E20两臂释放卡位、唯一CPUscorer串接旧E20终态；不变更健康任务。本矩阵在新NAMS/E20目标分数读取前固定；原raw双门槛不变，ACTIVE。


## HPL_NAMS2 完整双读出结果

`20261024-phase1-PAMF-HardPL-NoAsMixStyle-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增13320条闭合训练记录已核对。HPL_NAMS2 / raw clean=85.311111%；HPL_NAMS2 / DURA-LU clean=85.400000%；HPL_NAMS2 / raw six_mean=70.572983%；HPL_NAMS2 / DURA-LU six_mean=77.825000%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## HPL_E20_LS2 完整方法目标结果

原注册三seed默认头/raw及固定DURA七景42cell、源新19980步全部核实。raw clean=85.052778%；DURA-LU clean=85.083929%；raw six_mean=70.242956%；DURA-LU six_mean=78.089782%；raw数值门槛未通过，ACTIVE。原六行矩阵继续，不把单方法完成称为全实验完成，不改冻结配方/父源/种子/顺序。


## 高纯度 PL 的部署融合特征跨 RX 对比

`20261024-phase1-PAMF-HardPL-Joint160-m3-r01` 固定原E20续训native10+20到E50/11100，仅在既有GuardNAMS上加入joint160原生跨RX0.05/1.25。保持浅中层约束、.95／weight6、六U共享style、默认头和完整L梯度保护；复用前两ordinaryU forward，不增前向/头/G训练。原E20无更新源几何诊断显示style后pseudoCE仍约.003-.004，实际部署融合特征的源RX关联仍在；描述性关联变化部分由跨RX混合构造，不能当作已学会不变性或目标收益。48检查及一次P0P1PASS，原E20真实joint/sequence_pool梯度与guard smoke首卡通过即固定三seed，raw+固定DURA各七景。接替GuardLS释放卡位，排Guard唯一scorer后；健康旧任务保留。本矩阵在新E20NAMS/Guard目标分数读取前冻结，历史E20LS2暴露披露，raw双门槛和ACTIVE保持。


## HPL_E20_LS2 / HPL_E20_NAMS2 完整双读出结果

`20261024-phase1-PAMF-HardPL-E20plus30-m6-r01` 各方法三seed、共6模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部84cell完成；全部新增39960条闭合训练记录已核对。HPL_E20_LS2 / raw clean=85.052778%；HPL_E20_NAMS2 / raw clean=85.246230%；HPL_E20_LS2 / DURA-LU clean=85.083929%；HPL_E20_NAMS2 / DURA-LU clean=85.331944%；HPL_E20_LS2 / raw six_mean=70.242956%；HPL_E20_NAMS2 / raw six_mean=70.730225%；HPL_E20_LS2 / DURA-LU six_mean=78.089782%；HPL_E20_NAMS2 / DURA-LU six_mean=77.705919%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## HPL_E20_LS6G 完整方法目标结果

原注册三seed默认头/raw及固定DURA七景42cell、源新19980步全部核实。raw clean=85.101587%；DURA-LU clean=85.096627%；raw six_mean=69.936045%；DURA-LU six_mean=78.237996%；raw数值门槛未通过，ACTIVE。原六行矩阵继续，不把单方法完成称为全实验完成，不改冻结配方/父源/种子/顺序。


## 固定源证据后的剂量匹配互补监督

`20261024-phase1-PAMF-HardPL-Complement-m3-r01` 保留Joint6G三seed/原E20+30/E50/11100/.95/weight6/全部原RDrop及joint/浅中层/style/完整Lguard，只将ordinaryCE系数.5中的.25转给同pass joint减实际signature的CE，总分母1.25不变，部署完整签名和默认头不变。源C无签名视图CE.17-.22而完整仅.003-.004，暴露较弱路径；源诊断不是U真值或target收益。57检查与一次P0P1PASS，真实原E20检查首个释放匹配seed/card通过即重载继续，原三卡映射/2task上限保持，接替GuardNAMS并保留Joint。新增2既有head调用/步、无新骨干/teacher/头/推理参数；完整3seed raw+固定DURA七景42cell及原双raw目标保持。未读取GuardNAMS/Joint新target分数；goal ACTIVE。


## HPL_E20_LS6G / HPL_E20_NAMS6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01` 各方法三seed、共6模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部84cell完成；全部新增39960条闭合训练记录已核对。HPL_E20_LS6G / raw clean=85.101587%；HPL_E20_NAMS6G / raw clean=85.500794%；HPL_E20_LS6G / DURA-LU clean=85.096627%；HPL_E20_NAMS6G / DURA-LU clean=85.543452%；HPL_E20_LS6G / raw six_mean=69.936045%；HPL_E20_NAMS6G / raw six_mean=70.283433%；HPL_E20_LS6G / DURA-LU six_mean=78.237996%；HPL_E20_NAMS6G / DURA-LU six_mean=77.914649%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 20261024 Guard 全六模型闭合与 Complement 实际启动

Guard run `20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01` 全6模型／84项 raw 与固定DURA七景完成，全部39960条新增源步骤闭合核对；原先先行LS6G三seed结果逐项一致并保留。NAMS6G原始clean430924/504000＝85.500793651%，六景2125371/3024000＝70.283432540%；相对F21为＋0.519047619／−0.075925926pp。固定DURA为85.543452381／77.914649471%，相对相同DURA＋F21为＋0.513888889／−0.159986772pp。双主目标未达，goal ACTIVE。完整逐景／seed／TX／RX／day、净纠错、成本与匹配比较见该run报告。

固定权重2→6＋完整L梯度保护组合对同E20的NAMS2，raw clean＋0.254563492pp（三seed均正），raw六景−0.446792328pp（三seed均负）；DURA clean＋0.211507937pp、六景＋0.208730159pp。组合差异不解释为单独权重或保护的因果效应。冻结最终学生fullV27000×3门控0.95精度99.8954%至99.9613%、覆盖95.6222%至96.1778%；U实测准确率UNKNOWN，无目标反馈改变已冻结矩阵。

Complement run `20261024-phase1-PAMF-HardPL-Complement-m3-r01` 执行commit `7d58c705b65dfd0afa02ee8b7cb079e4e1b8468a` 在新GuardNAMS／Joint目标分数读取前固定并发布。真实原E20 source smoke PASS：首释放seed2026092702／GPU6；无签名CE实际signature梯度0，既有头0.829442375、主干2.382877837，完整joint签名辅助梯度0.370686237。普通CE0.5分配为完整融合0.25＋无签名0.25，总CE剂量1.25保持；两次既有头调用512条既有特征，U骨干1536与原L768保持。探针丢弃，正式行重新加载原E20。

三个正式seed已经分别在GPU5/6/7实际产生互补CE／joint／共享style／完整L guard原生步骤，独立PID/CWD/cmdline/GPU/loggrowth核实；当前前缀合计547步，非完整训练或目标收益。训练后raw＋固定DURA七景自动独立truth-last，健康Joint与前序任务未改变。部署T1E50/DURA默认版本保持。


## 主干互补监督固定矩阵

`20261024-phase1-PAMF-HardPL-PathComplement-m3-r01` 根据此前冻结源C的core-only CE.29-.37比去签名.17-.22更高，保持原E20+30/E50/11100/.95/weight6/完整Lguard/style/RDrop/浅中层与joint对比，仅将ordinaryCE.5分为full.25＋noSig.125＋core.125，总CE1.25不变，既有头与完整部署不变。每步4额外既有头调用，无额外骨干/teacher/头/推理参数。66检查及一次P0P1PASS；首匹配释放Joint卡位原E20smoke通过即重载，健康Complement保留、三卡2task上限、单CPUscorer接Complement。全3seed raw+固定DURA七景42cell及原双raw目标保持。历史GuardNAMS目标已知，尚未读取Joint/Complement目标；源V无更新，U实际正确率UNKNOWN。未启动/未达标，goal ACTIVE。


PathComplement发布后VERIFIED控制器存活/CWD/完整命令，执行commit `cf6794650506c4bca08425539efa3c3ae8d0b444`，3rowQUEUED、GPUworker0、原E20smokePENDING；等待Joint全部GPU预测释放对应卡位，健康Joint和Complement六worker日志继续增长。见该run queue_observation.json；不把排队当作训练或目标达成。


## HPL_E20_NAMS_JOINT6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-Joint160-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增19980条闭合训练记录已核对。HPL_E20_NAMS_JOINT6G / raw clean=85.641667%；HPL_E20_NAMS_JOINT6G / DURA-LU clean=85.685516%；HPL_E20_NAMS_JOINT6G / raw six_mean=70.171230%；HPL_E20_NAMS_JOINT6G / DURA-LU six_mean=78.080754%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## Joint 三seed完整闭合与 Path 实际三卡训练

Joint run `20261024-phase1-PAMF-HardPL-Joint160-m3-r01` 三模型原E20独立10＋20全部E50/11100，新增19980条源步骤闭合；每模型实际6660joint/guard、13320捕获批／3402000已有U特征，默认头、L/U联合及固定source权限均核实。全部raw＋固定DURA七景42cell独立truth-last完成。raw clean431634/504000＝85.641666667%，六景2121978/3024000＝70.171230159%；对F21为＋0.659920635／−0.188128307pp，对F20为＋0.856349206／−0.424966931pp。DURA为85.685515873／78.080753968%，对相同DURA＋F21为＋0.655952381／＋0.006117725pp。原raw双门槛未达，goal ACTIVE；不挑单seed或以DURA替代raw。完整逐景／seed／TX／RX／day、SD／MacroF1、修复损伤、成本及与GuardNAMS的固定匹配比较见该run结果。

Joint固定最终E50学生fullV27000×3门控0.95伪标签精度99.9384%至99.9574%、覆盖95.5111%至96.1741%；U实际正确率UNKNOWN，源健康不等目标收益。源view诊断在analysiscommit a7d299897387af9e335155c8b327a7e72a8873bb完整18行返回，39.837秒／CPU2、无V状态更新／Utruth／target／G训练。高纯raw标签经原训练信道后模型一致率约74%至77%、hardCE约0.61至0.66；只保留高置信同类视图会剩全部U约43%至46%，不是标签真值改变。

可靠As视图＋exactraw回退的下一独立原E20+30方案已于commit346b3fc2938e7b6fd52d803fca4db2e0267201a5、读取Joint／Complement／Path目标分数前固定，见Joint/reliable_As_followup_design.json；当前未实现/发布，不把设计当作结果。其完整源证据、旧目标暴露与额外半U teacher／模拟成本须如实保留。

Path run `20261024-phase1-PAMF-HardPL-PathComplement-m3-r01` 执行commit `cf6794650506c4bca08425539efa3c3ae8d0b444` 真实原E20 smoke PASS：首释放seed2702/GPU6，coreCE对exclusive签名／参考梯度均0，对既有头1.257945958／主干2.296314741；完整joint签名辅助梯度0.370686237。原L768／U骨干1536、4额外既有头调用1024packet、CE总1.25与完整L guard显式梯度一致，探针丢弃、正式重载原E20。三个seed已经在GPU5/6/7实际产生原生Path/joint/style/guard步骤，独立进程身份／GPU／日志增长核实；当前前缀共1041条，不等完整训练或目标收益。健康Complement保持，三卡每卡2任务；raw＋固定DURA七景自动执行，默认部署T1E50/DURA版本不变。


## 可靠As＋原始输入回退实施矩阵

`20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01` 按此前commit346b3fc固定源设计实施；保留Joint6G原E20+30/E50/11100/.95/weight6/全部原CE/KL/joint/style/完整Lguard/defaulthead，只在固定半U采用currenteval无梯度高置信同raw伪类As，否则exactraw回退，原acceptedU不删。新增As850500生成/teacher eval，累计teacher2948400、学生U10602900；实际used/fallback及成本另计。77检查、一次P0P1PASS，首匹配释放Complement卡位原E20smoke通过即重载；健康Path保留，三卡2task上限、CPU单scorer接Path终态。完整3seed raw+固定DURA七景42cell和原双raw目标保持。当前未启动、source smoke PENDING；历史Joint等目标暴露如实保留，Complement/Path未读target，不改固定方案。goal ACTIVE。


ReliableAs 发布后 VERIFIED：控制器存活及 CWD/完整命令匹配，执行 commit `0ffe0ba43a6b2ead84db96ff5b98494f29550594`；三 seed QUEUED、GPU worker 0、真实 E20 source smoke PENDING。等待对应 Complement GPU 预测完成并保留 Path 分发优先级，当前 6 个前序 worker 健康。见该 run 的 queue_observation.json，原 raw 双门槛尚未达到，目标继续 ACTIVE。


## Path＋ReliableAs 固定组合

`20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01` 从同seed原E20独立own10＋20/E50/11100，组合原Path CE(full.25/noSig.125/core.125，总1.25)和原ReliableAs(.95同类/原接受集合/否则exactraw)，其余权重6/六U/style/joint/完整Lguard/defaulthead保持。源D主干V子集90.51→97.59/98.54、full99.85；原生As视图诊断为另一源依据。Complement/Path/ReliableAs目标分数未读，配方和三个邻近比较预先固定，不加载候选权重。86检查/一次P0P1PASS，首释放匹配Path卡位真实E20smoke PENDING；保留活跃ReliableAs，每卡最多2任务，CPU单scorer接ReliableAs。累计teacher2948400、studentU10602900、额外既有头26640/6804000条，实际As使用/回退和成本另计；三seed raw+DURA七景42cell及原双raw目标保持，goal ACTIVE。


PathReliableAs 发布后 VERIFIED：控制器存活及 CWD/完整命令匹配，执行 commit `2ffbfcb8d04c2abd94ffc24f7f00fac0bebee529`；三 seed QUEUED、GPU worker 0、真实 E20 source smoke PENDING。等待对应 Path GPU 预测完成并保留 ReliableAs 分发优先级，当前 6 个前序 worker 健康。见该 run 的 queue_observation.json，原 raw 双门槛尚未达到，目标继续 ACTIVE。


## HPL_E20_JOINT_COMP6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-Complement-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增19980条闭合训练记录已核对。HPL_E20_JOINT_COMP6G / raw clean=84.958929%；HPL_E20_JOINT_COMP6G / DURA-LU clean=85.040079%；HPL_E20_JOINT_COMP6G / raw six_mean=70.252778%；HPL_E20_JOINT_COMP6G / DURA-LU six_mean=76.446362%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 完整特征接收机外推

`20261024-phase1-PAMF-HardPL-ReceiverExtrap-m3-r01` 固定源网格仅alpha.25通过，V准确率100/99.699248/99.849624%，U原伪类一致率100%但实际U准确率未知。原Joint6G配方从同seed原E20独立own10+20；前两已有joint160 U特征复用合法sharedstyle伙伴外推，普通CE0.5改full0.25+外推0.25，总1.25，无效伙伴差额0。保留完整特征/默认头/原接受集合/权重6/完整Lguard，额外2既有头调用，无额外骨干/teacher/As。110检查/一次P0P1PASS，真实E20smoke待释放RAS卡位；GPU5/6/7每卡最多2任务，CPUscorer接组合。源设计c0add33在Path/RAS/组合目标分数读取前固定，已知更早目标曝光如实保留。三seed42cell raw+DURA及原双控制raw目标不变，goal ACTIVE。


## HPL_E20_JOINT_PATH6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-PathComplement-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增19980条闭合训练记录已核对。HPL_E20_JOINT_PATH6G / raw clean=84.818651%；HPL_E20_JOINT_PATH6G / DURA-LU clean=84.902778%；HPL_E20_JOINT_PATH6G / raw six_mean=70.301984%；HPL_E20_JOINT_PATH6G / DURA-LU six_mean=76.296991%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 完整原生As监督对照

`20261024-phase1-PAMF-HardPL-NativeAs-m3-r01` 在源设计7d88684基础上实现，仅恢复Joint6G的U mixed半批原生As，原raw.95/权重6/接受集合/六U/损失/完整Lguard/defaulthead保持；无额外As教师、骨干、头或推理参数。原同seedE20独立own10+20到E50/11100，As生成850500/物理学生1701000，有效CE为2×rawacceptedhalf；累计teacher2097900/studentU10602900。119检查/一次P0P1PASS；真实E20smoke待匹配释放Combo卡位，GPU5/6/7每卡最多2任务，CPUscorer接RXE。固定Joint及RAS两个比较、三seed raw+DURA七景42cell和原双控制raw目标不变；RAS/组合/RXE目标分数尚未读取，当前健康任务保持，goal ACTIVE。


## 类别与接收机 CE 贡献均衡

`20261024-phase1-PAMF-HardPL-BalancedCE-m3-r01` 按 e7c9eb60 固定源设计实现，只在 NativeAs 完整配方中重分配当前已接受伪类/sourceRX 的 CE；接受集合、原 raw0.95/U权重6、六 logits/监督损失/完整Lguard/defaulthead保持。每批权重0.5至1.5、接受均值1；39960新增CE算术，无额外教师/骨干/头。原同seed E20独立own10+20/E50/11100，物理As预算与NativeAs相同。23新增及119受影响流程检查、一次P0P1均PASS；真实E20检查待匹配释放RXE卡位，唯一CPU scorer接NativeAs。三seed raw+DURA七景42cell、NativeAs匹配比较和原双控制raw目标固定，goal ACTIVE。


## HPL_E20_JOINT_RAS6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增19980条闭合训练记录已核对。HPL_E20_JOINT_RAS6G / raw clean=85.448810%；HPL_E20_JOINT_RAS6G / DURA-LU clean=85.518651%；HPL_E20_JOINT_RAS6G / raw six_mean=70.701488%；HPL_E20_JOINT_RAS6G / DURA-LU six_mean=77.864220%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## HPL_E20_PATH_RAS6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-PathReliableAs-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增19980条闭合训练记录已核对。HPL_E20_PATH_RAS6G / raw clean=84.821032%；HPL_E20_PATH_RAS6G / DURA-LU clean=84.932143%；HPL_E20_PATH_RAS6G / raw six_mean=70.758399%；HPL_E20_PATH_RAS6G / DURA-LU six_mean=76.386343%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## 原生 As 的 U 梯度范数帽对照

`20261024-phase1-PAMF-HardPL-NativeAsCap2-m3-r01` 按 28431a31 固定源设计实现，只将完整 NativeAs 配方三参数组 U/L 范数帽1改为2，保留负点积投影、tiny-L、一次完整 L 梯度桥接及原 raw0.95/U权重6/六学生/所有损失/defaulthead。原同seed E20独立own10+20/E50/11100，教师／学生／As预算不变，没有额外autograd或前向。源19980记录固定算术三seed均通过预定规则；它不等于新训练或目标收益。29新增及119受影响流程检查、一次P0P1均PASS；真实E20检查待匹配释放NativeAs卡位，唯一CPU scorer接BalancedCE。三seed raw+DURA七景42cell、NativeAs cap1匹配比较和原双控制raw目标固定，goal ACTIVE。


## HPL_E20_JOINT_RXE6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-ReceiverExtrap-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增19980条闭合训练记录已核对。HPL_E20_JOINT_RXE6G / raw clean=85.726786%；HPL_E20_JOINT_RXE6G / DURA-LU clean=85.790675%；HPL_E20_JOINT_RXE6G / raw six_mean=70.085516%；HPL_E20_JOINT_RXE6G / DURA-LU six_mean=77.897718%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## HPL_E20_JOINT_AS6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-NativeAs-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增19980条闭合训练记录已核对。HPL_E20_JOINT_AS6G / raw clean=84.831944%；HPL_E20_JOINT_AS6G / DURA-LU clean=84.944444%；HPL_E20_JOINT_AS6G / raw six_mean=72.264385%；HPL_E20_JOINT_AS6G / DURA-LU six_mean=77.428439%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。


## HPL_E20_AS_BCE6G 完整双读出结果

`20261024-phase1-PAMF-HardPL-BalancedCE-m3-r01` 各方法三seed、共3模型，每景168000条，raw／同一DURA七景预测与独立truth-last全部42cell完成；全部新增19980条闭合训练记录已核对。HPL_E20_AS_BCE6G / raw clean=84.878571%；HPL_E20_AS_BCE6G / DURA-LU clean=84.986111%；HPL_E20_AS_BCE6G / raw six_mean=72.323016%；HPL_E20_AS_BCE6G / DURA-LU six_mean=77.440608%；原数值目标仍未满足，维持ACTIVE。全部结果、匹配基线、逐seed／场景／分层及成本均保存，不从本结果热改其他已固定方法、父权重或矩阵。
