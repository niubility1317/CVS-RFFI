# 既有跨RX同TX对比项：实际视图覆盖与下一问题

2026-10-17。状态 RUNNING_SOURCE_E20／NOT_ANALYZED。频域相位全包源依据已失败关闭，不以本笔记改阈值、子集或算子救跑。两条源E20已启动；当前没有新Ag、默认头性能或目标结果。

## 已核实的调用路径

实际F＋跨RX训练底座见experiments/cvs_crossrx_satmix20/model.py和supcon.py。model.hook先在同一个已生成ordinary IQ上完成两次RDrop前向，从pending[first/second]的time_down与t2.pool计算原SupCon，随后清理捕获hook。raw和practical两个视图的后续前向没有进入该SupCon。scene_cycle20/model.py调用整个原hook，再对extra raw/As单独计算默认头RDrop；没有让extra As进入原跨RX同TX正对。

因此，现有As配方中的practical L确实全部参与默认头CE/RDrop，但原浅中层跨RX同TX项仍只看ordinary IQ。supcon.masks仅使用合法L TX与物理RX：正对为同TX不同RX，负对为不同已知TX；不把U当负样本。原层位为time_down和t2.pool，温度0.2，E6开始原schedule升至0.1；两次随机子网络前向复用同一IQ。这里是代码覆盖事实，不是导致urban错误的因果证明，也不是尚未执行的新方法有效性证据。

## 唯一待检验假设

在不增加loss项、系数、模型前向或部署模块的条件下，让一部分既有跨RX同TX正对跨ordinary／practical视图，能否比原ordinary-only及全practical对比控制保留更多urban身份，同时保护clean？这归类为As训练组织，不称为IQ泛化增强Ag。它不同于另一对话的As启停时序，也不要求整个joint/raw112/所有分支等于clean。

## 最小可判别实验

固定同一F＋跨RX、实际fitL5040/U45360/V21600和RX1原L1260 final-only、scratch seed2026092701/E20、原13L7U、默认头、原完整practical分布和receiver session。完整raw.75＋ordinary.25监督不减，匹配extra pair weight.25；均8次L前向／总pair weight1.25／4440更新。已有SupCon的层位、温度、schedule、正负对定义和总权重不变。

| 行 | extra IQ | 原既有SupCon的输入 |
|---|---|---|
| Q0 | raw | ordinary |
| Q_ord | 单次完整As(raw) | ordinary |
| Q_sat | 与Q_ord相同的单次完整As(raw) | practical |
| Q_mix | 与Q_ord相同的单次完整As(raw) | 每次L出现位置选ordinary或practical，单独随机流固定64/64，与TX标签无关 |

Q_mix在每次RDrop phase内仍是128个原物理包和原数量的SupCon项；两个layer的同一物理包必须使用同一视图选择，两个RDrop phase复用选择和已生成IQ。所有合法L的As仍无置信门控参与CE。只复用已有前向的实际maps，不增加projection/head，不平均IQ，不再次施加完整信道，不使用U真值／新增U规则。实际map捕获、额外中间量计算、显存／时间仍需计成本。

Q0/Q_ord有对应既有sourceCycle的B00/C_IID候选控制；可只复用原冻结源产物作比较，不加载其权重初始化新行。复用前必须核对完整数据角色／物理ID、初始化和全部预算／随机流／视图／loss条件；不能因名称接近就复用。若不匹配，按授权E20从零提供匹配控制；不得将节约的训练成本冒称所有行都新训练。固定四行逻辑矩阵，最少新增Q_sat/Q_mix两行E20；目前未登记或启动，不把这张逻辑矩阵作为已授权新预算之外的续训。

正式执行前在同一最小预登记固定逐行输入与判据、控制来源、实际成本和自动源七视图冻结评价。必须看Q_mix相对Q_ord和Q_sat的真实clean/urban/每suburban配对结果；对比loss下降、valid anchor数或梯度不能晋级。若不能同时优于两个控制的low和urban均值，或clean／suburban保护失败，就关闭此固定视图组织，不调比例、层位、系数、scene概率或seed救跑。数字阈值在实际登记前固定，不能在看到新行结果后补定。

本笔记不是额外审批或放行门槛。当前下一动作是核对控制实际契约并实现这一个可反证组织方案；未把“ordinary-only覆盖”当成成功，也未把As收益当Ag。匹配B00/B10/B01/B11、有效Ag、跨seed及正式clean＋六practical的low>60%／suburban接近自身clean仍是完整目标，保持ACTIVE。


## 固定执行矩阵登记

20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01已实现登记未启动。两旧控制Q0/Q_ord实际完整配置／LUV物理ID／seed／E20/4440/8L/1.25权重／默认头核对，并确认对应model/channel/source/design与原执行commit无diff；不加载控制权重。新增Q_sat/Q_mix两scratchE20，40新modelE/8880更新，40旧modelE另列。29反例／4108合成NN／2合成更新PASS，74组参数梯度与原ordinary-route接近、loss精确相同，As IQ／RNG／默认头／export160均通过；唯一一次P0/P1 PASS。primaryQ_mix与全部三参照的数字判据见[预登记](../automation_reports/CV-SincNet/20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01/preregistration.md)。当前0实际新训练／target。未把本路线当Ag或联合目标完成。


原SupCon视图组织 20261017-phase1-F-crossRX-supconViews-sourceRX1-e20-s2026092701-m4-r01 已实际RUNNING／VERIFIED，执行ed0ca91481f8c11a28cb8f4d4726a7c3d488720e；两旧控制复用、两新scratchE20实际GPU0/1 live，realCP2NN无querysmoke/归档传输/remotecompile/PID-CWD-argv-GPU-log增长均PASS。两行至少完整E1/222更新、227328LNN/28416fullAs/全部5040L覆盖，原E6启用对比项，当前warmup routing0，无新增NN/loss项。新训练/预测/正式评分未完成，无性能结论，不热改/重启。F+CROSS_RX/E20/默认头/原U保持，完整联合目标ACTIVE。
