# Phase3 源端星地编码器联合增量计划

前一轮仅改读出后，星地开发 AUROC 提高，但主判决的未知拒识仍只有38.8889%；中/低城市排序尚不足以同时实现95%接收与90%拒识。本候选检验源端星地输入匹配能否改善完整身份表征，主 arm 固定 practical_increment，clean_increment 为同预算控制。目标保持clean跨RX95%／星地跨RX90%，known接收尽量提高、FRR≤5%和原F同输入closed−open≤2pp；不依据开发结果改阈值、选arm或重跑。

run为`20261029-phase3-source-practical-encoder-s2026092701-r01`，单seed2026092701，科学设置见`configs/pamf_phase3_source_practical_encoder_20261029.json`。两arm均从同一个已验证uniform-OE E51开始，各固定222次更新；原11100＋E51增量222＋本次222＝11544步。重置相同AdamW CENTER_UNITS，lr1e-4常数、weight decay继承已验证parent args、物理全局梯度裁剪5；原默认六类head与完整骨干/Sinc联合训练，原Phase1/2权重及默认方法身份不变。

两arm保留原native L128六路监督、cross-RX0.1、LS0.01、ordinary epoch13和As机制，关闭U/EMA。每步在同一L128上加0.68倍附加CE：control使用同clean记录；主arm使用同记录的一份已缓存practical→冻结DURA输入。附加scene按独立固定随机流逐样本在三景中等概率选择，不能按TX/真值或类别比例选择。两组均对同源OE21的64条物理记录计算clean及对应第二view的均匀六类KL平均，整体权重0.1；control第二view是clean，主arm是相同抽样规则的practical/G。统一一次反传/裁剪/更新，不能给额外head单独训练或增加optimizer更新。

训练只使用原source五RX/三日L6300及已验证源OE21/6134。原L标签可见；OE仅known/unknown成员监督，真实TX隐藏；原U56700、V27000、source novel和所有target均不进优化。缓存只生成L/OE21的clean＋三practical/G，12434个原记录、49736计算view。固定source channel seed2026102931/receiver2026102912，同scene/opaqueID公式；输入保持现有一次unitRMS，G后不再RMS/EQ。G固定R69、FP32/TF32off、reuse/graph true/native false，参数不更新。缓存不是新的独立物理样本，不因更换模型重新验证原数据。

每arm实际曝光：L抽取28416条，native170496 packet加附加28416，共198912；OE抽取14208条，两个view共28416 packet。另记真实checkpoint合成smoke，不把它当训练更新。完整222步loss/梯度/额外view计数和scene histogram保留，终点固定，不用V早停。源known与OE随机流、view选择流、前向预算和优化器在两arm匹配。

训练冻结后每arm自动执行源评估与开发评估，不以source未知分数选择或取消某arm。源端L6300/OE4共4200/V27000/novel3共3150各clean＋三practical/G，共40650物理/162600计算view。使用该新encoder的clean L重新拟合自己的聚合几何，避免把E51坐标统计直接用在新特征。新linear101/MLP1633仅用L/OE4四view各固定200CPU步；V只作known-only阈值校准，整体/每RX/每view/RX×view共30组，FRR5%/原F同输入总损伤2pp，整组ties、每方法一个最高合法global阈值。源novel判决在阈值冻结后，全部四view预测固定后独立truth-last。全七方法/逐RX/TX/view及交叉均报告；源校准样本不是独立保护确认。

每arm开发C0/P1各Q864，RX24-1、2021-03-15、known576/unknown288，两个条件为同864原记录。原parent_E51状态精确复用上一run augmented_readout冻结预测，原F同输入baseline也复用；increment用自身冻结新F/几何/读出/阈值逐样本推理。P1只读原单份received并过同一冻结G，不读clean参考。真实类别构成只在builder/scorer，配置不暴露未知类数/配额；targetU0、formal RX24-18 U/Q0，暂无DA、发现或注册。source与开发跨RX但日期非严格未见，novel3与开发池已暴露，仅开发证据，不是正式新TX盲测。

最小执行为共享训练view cache→两arm各自train→source run→独立source score→C0predict/score→P1predict/score；资源可用时两个row独立前进，健康row不因另一row分数或技术失败热改。固定主arm实验证伪后才考虑扩seed，不因单项通过宣布完整Phase3成功。source参数、data角色和阈值在query保持冻结。

正式发布前在本run登记唯一launch owner、实际code/config commit、精确命令/Python/CWD/源与输出路径、GPU、stop rule和预期产物。N607每卡最多新增两个本任务GPU worker，至少每预约槽4096MB实际余量；保留其他owner，最多两row同时工作。实际checkpoint梯度与G合成smoke是各阶段首步，PASS立即继续。只执行八项最小流程：聚焦负测、一次独立P0/P1、一次preflight、单归档一次比较/compile、一次live PID/CWD/cmdline/GPU/log核对和prediction完成后的独立评分，不新增审批或哈希链。所有数据/模型输出使用新不可覆盖root。

技术停止限错误数据/继承身份、非法truth或Q拟合、非有限梯度/输出、输出碰撞或无法产出合法预测；保留partial，不因科学失败重启或调参。训练、source评分、开发评分分别记录实际状态；总目标仍须未来无标签适应、发现和匿名注册的独立观测证据。
