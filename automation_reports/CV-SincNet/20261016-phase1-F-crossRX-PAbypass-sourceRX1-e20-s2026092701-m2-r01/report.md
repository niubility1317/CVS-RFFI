# F＋跨 RX 同 TX：同抽样普通 PA 旁路 E20 源对照

状态：PLANNED，尚未启动或完成训练。当前固定基线F＋CROSS_RX，E20从零，默认cosine头正常联合训练；不把源诊断或RDrop/SupCon当作IQ泛化完成。

## 唯一假设及证据边界

唯一待检验假设：只旁路普通有标签增强的PA输出，保留其余同次原生抽样和完整PA计算，能改善训练后未见RX clean，并保护practical。已有冻结fitL5040同抽样诊断直接请求20轮日程，实际变动3054观测修复640、新错27，支持存在可避免源身份损伤；不证明训练收益。实际native trainer把增强epoch截为min(training_epoch,label_epochs=13)。前探针E14至20直接函数调用属于更强诊断外推，已在原报告及日程更正文档明确，保留全部结果及原判定。前13轮描述性重算1475变动、254修复、7新错，净恢复16.745763pp；没有新增IQ或重跑。此次训练保持原生截断，PA概率最高0.035084047，未选择更强日程。

## 固定最小矩阵

| row | ordinary送入模型 | raw/S槽 | 预算 | 训练后自动读出 |
|---|---|---|---|---|
| B00-RX1-s2026092701 | 原native完整PA输出 | 两槽均raw | scratchE20/4440更新 | heldRX1 clean＋六practical |
| B10-RX1-s2026092701 | 同次native调用PA前prefix | 两槽均raw | scratchE20/4440更新 | 相同物理样本和信道抽样 |

两组真实原L5040/U45360/V21600契约相同，fitRX3/4/6/8×day1/2/3×全部6TX；原RX1 L1260只在fixedfinal20后读出。U真值隐藏，原13L＋7U oldgate/自有EMA流程保持，V只健康诊断不选checkpoint。不能冒充完整L6300或L5580/Ldev720；不调用target，source-only预登记结束于源证伪。复用现有carrier/角色过滤，不重复数据验证。

两组同模型seed2026092701、split392005，原训练增强全局随机流程、source practical信道/receiver/eval seeds均不变。只用不变backend0ae代码，正式训练不加载任何父模型。dispatcher真实checkpoint smoke仅使用自有匹配scratchE20/fixedfinal20/B00做2个合成包无query兼容检查，PASS直接继续；来源无祖先/目标选模，真实训练另从零开始。不继承E50/E200或诊断模型。

F三路SIG_TIME、统一完整Sinc修正、joint160、NoExplicitCFO与必要逐包补偿、默认cosine头和原联合训练保留。仍是三对共六L前向，.5raw＋.25ordinary＋.25raw，每对RDrop alpha1/LS.01，浅中层time_down/t2.pool CROSS_RX只取ordinary原两次前向，温度.2/最大权重.1/5轮warm＋10轮ramp不变；同TX同RX不作负对，U没有新对比或新增视图。

不改变任何概率或算子范围。native只调用一次ordinary augmentor，捕获post-general-channel的PA前prefix，原生PA仍完整计算。B00返回full，B10返回prefix；两组均记录全PA生成及捕获成本，保留原native Python/Torch随机消耗。DACp0、class_signature=false，PA选中子链仍按原抽样，未触发包full/prefix逐位相同。native n.configure按min(epoch,13)配置；记录真实training_epoch/configure_epoch/状态和每step有效PA概率，不能只凭默认值称启用。无新增RMS、参考窗移动、CFO身份输入、噪声/多径级联或As训练。新变体是旧ordinary配方消融，不直接冒称新增A_g已有效。

## 原定数值判断与成本

fixedfinal20后比较B10−B00：heldRX1 clean准确率≥+1pp，clean MacroF1≥0，六个practical各≥−0.5pp，seenV≥−0.5pp；两组完整20轮/4440更新/3409920LNN、3330非零SupCon调用、最终LCE≤.5、0As/头拟合/target。全部同时通过才支持下一独立源确认。任一失败关闭固定PA-bypass训练配方，不改seed/RX/强度/阈值救跑。源确认不等于正式target或多seed泛化，原联合目标及B00/B10/B01/B11矩阵仍未完成；不把clean降低造成gap变小当改善。

每组E20＝13L＋7U、4440更新；每组3409920L网络包前向，另实际U/ownEMA/V/2合成smoke/8820held；此前同配方参照每组4686678NN，当前按实际计数，不用预期冒充实际。每组568320个ordinary完整PA生成及prefix捕获，NN视图数不增加，PA实际触发/变动单列；0As训练生成/0诊断头拟合。两个模型40模型轮/8880更新；dispatcher真实CP smoke2另列。局部验证13056合成增强包、2312NN、2合成更新、0真实IQ，与正式预算分开。

## 发布与停止

唯一launch owner codex/root/F-CROSS_RX-PAbypass-E20-20261016；普通N607账号，CPUdispatcher2线程，GPU动态共享容量锁、每卡最多3个所有owner进程且启动前≥12000MiB空闲。健康旧任务不改/不停/不重启。CWD为PROJECT/releases/cvs_F_crossRX_PAbypass_sourceRX1_e20_20261016_r01，命令CVS-RFFI/bin/python -u experiments/cvs_pa_bypass20/runner.py，worker带固定--worker row_id。路径、六类seed、逐行完整config见experiment.json；Git实际commit及release_commit/resolved是代码身份。

一个不可覆盖run/output root、一个N607 preflight、一个归档单次SHA比对和远端compile；launcher第一步真实CP no-query smoke，PASS继续，不另建许可。启动后独立核实PID/CWD/argv/GPU/log增长。角色/初始化/预算/非有限/输出冲突/不可执行或不能合法完整预测为技术失败：保留产物、停pending、不干预健康worker、无自动重试。科学低分不停止、改配方或追加预算。

每row final20后自动完成clean及六practical源默认头读出；全部prediction固定后本地独立复算完整epoch/step/PA记录、实际日程和NN成本、物理L/U序列、accuracy/F1/CM/CE、配对修复/新错和TX/day分层及原判断。IQ/CP/逐包预测与完整日志私有，只代码/config/登记/汇总入Git。一次限定P0/P1审查及14个负例、原生输出/随机流精确匹配、同初始化/B00损失和头梯度精确对照、六前向同IQ/SupCon域、默认头联合更新和export160检查通过；这些不是识别收益。
