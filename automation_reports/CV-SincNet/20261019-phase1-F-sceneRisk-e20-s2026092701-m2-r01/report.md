# 纯 F E20：场景等权 CE 与有界动态场景风险

状态：ANALYZED / VERIFIED。完整终轮结果见 [final_analysis.md](final_analysis.md)；下文保留原预登记和运行快照。

唯一假设：既有普通完整 As 对不同传播场景平均计算 CE，可能没有充分改善持续困难的 mid/low urban。固定模型和 As 曝光，按传播场景训练风险调整监督份额，检验是否比场景等权 CE 更有效。源端困难并不证明增加权重必然有益；不可约噪声、过拟合与 clean 损伤都是可能反证。此路线不依赖外部教师，与另外聊天的 HQ/raw 额外监督、U practical 组织不同。

采用既有纯 F：SIG_TIME_X3＋CENTER_25K＋CENTER_UNITS，实际 joint160，默认 cosine6×160/scale30；骨干和默认头正常联合训练，900722 参数。0 CROSS_RX、0 RDrop、0 新部署分支或头。保留原普通增强、原生 U/ownEMA、默认 smoothing0.01 和 AdamW/cosine20 配方。两新行同 seed2026092701 从零 cumulative E20=13L+7U，每轮222步，共4440次更新，不加载任何父权重。L6300/U56700 隐TX/V27000、五源 RX1/3/4/6/8、day1/2/3、split392005 原角色不变。

逻辑矩阵及精确 row/config 以 experiment.json 为唯一登记。已完成的同 seed 纯F run `20261019-phase1-F-nontrue-CSIL-e20-s2026092701-m4-r01` 中 B0（ordinary NoAs）和 B1（ordinary＋普通完整 As）只读复用全部状态、源读出、原正式预测及分数，不重训、重测或重评。旧 B1 的 CSIL 仅 eval shadow，停止梯度、fork_rng 构建且无随机前向；它没有改变学生训练目标或 RNG，额外历史计算单列。本轮两个新行均无外部教师。

两个新行均每步原 L128 ordinary 前向＋同 L128 的完整 practical(raw) 前向，真实标签 CE/temperature1/smoothing0.01。ordinary 系数恒0.5、As 总系数恒0.5，原 U 目标不改。As 生成、硬件 session、随机流、采样器、全部真实标签样本均保持；不弱化信道，不按学生置信度筛选，不对 As 再叠普通增强。训练场景沿用 low suburban、mid urban、low urban 三个场景，测试仍是固定 clean＋全部六种 practical。

BALANCED：本批出现的每个场景先取 CE 平均，再等权平均。RISK：logq 初始三个0；每步在停止梯度的当前各场景 CE 上执行 logq +=0.01×risk、减 logsumexp；缺场景更新量0；q=softmax(logq) 在出现组重新归一化。实际场景权重为 0.5×出现组均匀权＋0.5×q。所有出现组和其中所有合法 As 都有正权重；三组齐全时每组权重不小于1/6、不大于2/3。以更新后的权重计算当前步 CE，不给风险权重反向传播。不用 V、目标、教师或置信度更新 q，不改变采样概率或信道强度。BALANCED 也记录 shadow q 供同计算复算，目标始终等权。RISK−BALANCED 隔离动态风险；BALANCED−B1 解释分组均值与逐包均值的差别。

依据是 [GroupDRO 论文](https://arxiv.org/abs/1911.08731)及[作者风险权重实现](https://github.com/kohpangwei/group_DRO/blob/master/loss.py)的指数群权更新。当前是传播场景分组及均匀底权的改编，不声称复现论文结果。历史源 RX GroupDRO 改善过训练最差 RX 风险，但目标泛化下降；该反证保留。RX 分组与传播场景分组不同，原先没有 As 场景风险消融，本轮不以改名重跑旧 RX 方案，也不因某个方法名称预判效果。刚结束的 CSIL signed true-row 输入导数源诊断已 FAIL 关闭，本轮不再使用该损失、教师或阈值。

固定源推进标准：终轮原 fullV27000 clean 及既有 V1800 两 draw 源场景读出；RISK 相对 BALANCED 和复用 B1 的 mid/low urban 两 draw 平均各≥0.5个百分点，且 RISK 相对 B0 fullVclean≥−0.2个百分点，才有累计 E50 资格。V 不用于拟合 q、阈值、信道或选择 epoch。决定先保存于 source_matrix_decision.json，之后新 scorer 才可读 truth。失败关闭固定配方，不扫 eta/混合系数、重分组、换 seed 或选择性重跑；所有健康已登记行继续 E20 和正式验证。PASS 仅支持考虑自身 E20＋30，追加前另固定 optimizer/EMA/q/U/采样器/信道计数衔接及匹配 E50 控制，绝不自动延长。

每行冻结 FINAL20 后自动源读出、原168000物理样本×7视图单包默认头预测；全部7份预测固定后，独立 exec scorer 才连接 truth。报告准确率、macroF1、最差RX、各 suburban 自身 clean 差距、修正/新增错误、相对B0/B1/BALANCED的配对差；单 seed 的样本SD不适用，不冒充可重复收益。正式目标数据与既有分数不回流权重、参数或扩展决定。该测试基准已暴露，不称新独立盲测。

命令 `CVS-RFFI/python -u experiments/cvs_scene_risk20/runner.py`；普通 N607 既有 FP32 环境、每worker CPU2；CWD `PROJECT/releases/cvs_F_sceneRisk_E20_20261019_r01`；不可覆盖根 `PROJECT/runs/20261019-phase1-F-sceneRisk-e20-s2026092701-m2-r01`，日志 `<run>/logs/<row>.log`，唯一owner `codex/root/F-sceneRisk20`。每 GPU 含其他 owner 最多2总实验进程，至少12000MB空闲，CUDA 初始化纳入共享 capacity 锁。健康旧任务不干预。

技术停止仅限实际数据/角色/来源/预算/非有限值/接口/输出错误，保留产物，不自动重启。每行预期568320 As、1136640 L训练前向包，原 U学生396900包/EMA4440次；最终源读出12600包、正式预测1176002包另计，运行时间和实际开销实测。首步原生 checkpoint 无query smoke PASS后立即训练。完整训练流、风险 CE/q/逐样本权重、真实 joint160、Sinc/签名/时频/PA/默认头更新均保留核验。

本地公式/正权/停止梯度/缺场景/置换/有界权/5协议负测 PASS；实际两个 F 同初态，原生 L hook 与全部关键骨干及默认头非零梯度、joint160 PASS。一次独立P0/P1审查结果待写入代码身份事件，发布前完成。Git commit 为代码/config身份，不增加设计SHA或审批。REJECTED_EXTRA_GATE：通用 brainstorming 的额外设计批准流程被当前用户自主研究授权及 exhaustive minimal workflow 覆盖；本报告与登记已包含唯一假设、矩阵、命令、预算和放弃规则。


发布状态：RUNNING / VERIFIED。实际发布 commit `9737b41af49a38516d85925b604c015b95ce5225`，代码/config `3f7062ff5a8be5074a280b410a45ec8be9213021`。一次独立P0/P1审查PASS；单归档传输比较、远端编译/负测PASS。BALANCED GPU2/PID2099403、RISK GPU3/PID2099480，实际PID/CWD/argv/CUDA和增长中的原生step日志已核实，参数均900722；含其他owner资源上限满足。当前两行正在原定训练，尚无源推进决定或新正式性能结果。


首段实际消费核验：BALANCED 656 个连续原生步骤、RISK 662 个连续原生步骤，L总损失与0.5ordinaryCE＋0.5场景CE的实际日志字段逐步一致，最大误差 1.1920928955078125e-07；已观察ownEMA更新序列全部匹配。此快照仍在L阶段，未以未发生的U阶段宣称已完成隐藏标签核验；不是全4440步结论、源晋级或性能结果。完整风险状态/逐包CE复算及正式配对分析代码已准备，待整行产物完成后执行。


U阶段实际边界已核实：BALANCED 已观察完整连续 2944 步，其中 U 阶段 58 步、RISK 已观察完整连续 2924 步，其中 U 阶段 38 步；全部已观察U TX真值可用计数0，ownEMA成功更新逐步匹配。两行真实生效配置另核实为从零纯F/13L+7U/ordinary启用、原U及ownEMA启用、native卫星拼接关闭、新As从raw生成/noEQ、无外部teacher、无CROSS_RX/RDrop。既有source角色句柄精确匹配，无原始数据重验。本次是已发生的执行前缀，不是完整4440步或正式性能结论。


完整源端结论：TRAINING_COMPLETE / VERIFIED，科学资格FAIL（不属于技术失败）。关闭固定 bounded scene-risk 配方，不追加 E50，不扫 eta/混合权/分组/seed。此记录在主 Agent 读取任何本轮新正式数值前固定；原七视图预测/独立评分队列继续。

RISK−BALANCED midlow=+0.222222个百分点，RISK−B1=-0.138889个百分点，均未达到各+0.5；fullVclean相对B0=-0.470370个百分点，超过−0.2容差。源资格已用保存的V1800×2draw预测与fullV27000读出独立重算。

|行|fullVclean %|源low suburban两draw均值 %|mid urban %|low urban %|
|---|---:|---:|---:|---:|
|B0|98.055556|62.694444|52.777778|42.750000|
|B1|97.762963|89.666667|75.638889|62.138889|
|BALANCED|97.725926|89.805556|75.305556|61.750000|
|RISK|97.585185|89.416667|75.666667|61.833333|

两新行全部4440机制步、4440原生步、4440物理单位步和20轮完整记录已复算；风险组均值/logq连续更新/实际权重/每包正CE/实际消费损失一致，最大误差低于4.3e−7，实际消费字段误差0。每行1554 U步/396900 U包真值可用计数全0，ownEMA更新1至4440逐步精确。四行初态及完整L/U序列相等，B1/BALANCED/RISK全部物理训练覆盖和session汇总相等。

RISK终轮均值权重为low suburban 0.16666667、mid urban 0.16666837、low urban 0.66666497，说明权重确实转移到了困难场景，但不能将此解释为识别收益。每新行参数900722、实际joint160，L前向1136640包，学生训练1533540包、训练期评估594004包，外部teacher0。完整正式指标、配对修正/新增错误及总目标仍待七视图收尾分析；不以本源结论代替正式结果。


最终状态：ANALYZED / VERIFIED。两新行均完成从零E20/4440更新、全部168000×7固定预测和独立truth-last评分，原进程自然退出。完整源轨迹/实际执行和同物理ID配对分析已完成，图已查看。RISK clean81.249405%、mid urban63.647619%、low urban52.819643%；相对普通As分别+0.038095/+0.175000/+0.183929个百分点，clean相对NoAs−2.914881个百分点。源关闭决定提交 `63fa12da51695d39d31f0dfa3ee2c093431c8a13` 早于主Agent读取本轮正式数字，且服务端源决定先于评分；固定配方保持关闭、不追加E50。单seed结果未达到low>60%和可重复clean净收益目标。详见 [完整分析](final_analysis.md)。总体研发目标仍在进行。

最终结果：两新行E20/4440步、全部七视图预测与独立评分完成。源FAIL在正式评分前固定，不追加E50。完整日志重放、四初态/L-U流/物理信道覆盖一致，科学负结果保留；目标尚未达到。
