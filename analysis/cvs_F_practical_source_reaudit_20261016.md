# F/E20 的 practical 源域退化复核

状态：DESIGN_NOT_IMPLEMENTED_NOT_RUN。本文件只固定下一项独立科学诊断，不是新训练、WHITE30 晋级门槛或既有启动的额外审批。方案在 WHITE30 最终源成绩产生前固定；该训练健康继续，不据诊断改配方、阈值或延长预算。

当前固定基座已经从旧 RAW_HZ 单路切到用户指定 F（三路 SIG_TIME/CENTER_25K/CENTER_UNITS）。旧物理诊断证明了原模型的噪声遮蔽、状态与边界效应，但不足以给出新 F 默认头的分类保持率。新疑问是：**F 在真实 practical residual 下的哪些源域退化仍可恢复；独立轻度观测噪声训练是否改变了这种恢复性，而不只改变 clean 错误分布。** 不预设多径、相位、噪声或内部门控是主因。

固定两个模型，不按 WHITE30 得分增删：已完成 `20261016-phase1-F-raw-anchor-ordinary-RX1-e20-s2026092701-m3-r01/A0-RX1-s2026092701/source` 的 final20，以及 `20261016-phase1-F-white-observation-RX1-e20-s2026092701-m1-r01/W30-RX1-s2026092701/source` 的 final20（须先正常完成）。两者同初始化规则/四前向/F/defaulthead/原U，训练只含RX3/4/6/8；零新训练，零optimizer/EMA更新或拟合头。只从合法源缓存选择 heldRX1 的原L1260（6TX×3day×70），这些物理包未参与两模型训练；不使用V/U/target IQ。缓存可能物化其他源角色容器，必须如实披露其前向/使用量0。父训练成本各20轮/4440更新单列，无E50权重继承。

每模型在同一原L1260上记录一次clean，再使用原六practical、相同命名随机流/传播状态/RX-day-eq session、同一真实标签进行11条件对照：full；no_noise、flat_H、no_carrier、no_RX_IQ的固定状态与重算状态两种效应；actual_power_quality；edge_boundary。复用已验证 `cvs_urban_sigtime.controls.compare`，full必须与实际导入ResidualChannel逐样本一致；本次不重新发明物理渲染器。source随机流独立于训练/验证/测试，两个模型完全共享源波形；session不含TX标签。

实际配置必须打印模块`__file__`和完整Config：residual/post_sync、25MHz、IQ256、原reference_snr_db=30、reflect、EQ关闭G=1；记录质量适配、相位跟踪、IQ补偿、原归一化位置。不得通过改变正式测试IQ/SNR/信道获得收益。反事实仅用于源域科学解释，不能把去噪/flat_H准确率称作原practical性能。

接收状态效应明确区分：移除损伤后重算质量/锁定/补偿/AGC的总效应，与固定同一full接收状态、补偿参数及AGC的受控效应。分层保留LOS/PARTIAL/BLOCKED、locked/degraded/unlocked、白功率代理与实际波形功率差、signal_to_added_noise_db（不是总SNR）、残余CFO、残余核/数值延迟、噪声着色、AGC限幅、边界影响和有效观察区。EQ关闭不能假称已均衡；原引擎无逐样本硬剪裁时不伪造clipping mask。

沿实际F前向记录defaulthead分类间隔/30、固定各父clean正确集保持率、两模型共同clean正确集表现、修复/新错误、RX/day/TX/状态分层及macroF1。直接观测原time/frequency/PA、reference及实际三路SIG的激活贡献；验证记录路径确实构成最终joint160，不能由三个局部投影虚构三个部署分类器。冻结PA/reference/SIG的learned gain是模型参数，随输入不变；应报告加权激活变化，而非伪称冻结gate权重按场景自适应。

如记录梯度，只对既有参数或中间张量求无更新CE导数，复用既有前向、不积累`.grad`，不拟合probe、不把方向/范数当身份识别收益。模型始终eval，参数/buffer前后一致。新默认头分数只是source诊断，既不晋级已失败WHITE30，也不改其30dB或raw比例；物理归因需要分类修复证据支持。

预计每模型1260×(1+6×11)+2真实checkpoint smoke=84422网络前向，两模型168844；若实现为复用图求导，不增加网络前向，实际导数次数另记。首次实现需完成对应负例/合成图测试、一次直接P0/P1、独立登记与immutable root、发布后PID/CWD/argv/log增长核实。产物必须含逐模型/场景/条件与状态结果、真实模块配置、完整私有逐样本明细、分层图、实际成本及负结果。这里列的是预期产物，目前尚不存在新诊断图或成绩。
