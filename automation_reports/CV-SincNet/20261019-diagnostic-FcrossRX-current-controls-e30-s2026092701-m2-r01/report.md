# P79：当前P43基座的受控 practical 退化诊断

状态：ANALYZED_SOURCE_ONLY，预登记源条件已全部读出与分析。下述方案是运行前固定记录；最终结果见 [最终分析](final_analysis.md)。起点 f2f19130831ed2c6db220062deec913c6df66798。本轮明确仅源域冻结诊断，新增训练0轮/0更新，不是新的增强性能实验。

最近已完成P78两行匹配E30与all7评分，原As相比新增urban均值仅+.007540pp、clean−.075pp，固定候选已关闭。进一步去重表明EMA读出、名义质量风险、有界困难风险、U practical预算路由及公共传播TX单元均有既有源负结果，不按附件名称重复启动。

唯一诊断问题：当前P43原NOAS30/AS30在固定接收机状态下，新增噪声、多径选择性、载波残差、注入RX IQ残差和人工边界分别留下多大识别损失；去除损伤时重算接收状态的总效应与固定状态效应有多大差别。原十一条件诊断来自旧父，P74当前P43只覆盖mid/low full及固定状态去噪，不能当作当前模型已完整测过。结果用于确定下一项独立训练假设，不根据正式成绩挑方法。

两个模型固定P43 B00/B01各自ownE20+10 FINAL30 student与checkpoint内默认cos30头，骨干/头此前正常联合训练；不是EMA、P78或LRaw训练权重。严格核查ownE20 scratch ancestry、FINAL30/final_only、实际source role_ids、无target训练/选模、默认头6×160/scale30、900722参数/206状态。冻结后无更新、无新头、无拟合、无query或formal scorer。

复用现有VALIDATED_ONCE源面板：原L6300/U56700隐藏/V27000数据角色不变，只读固定平衡V1800，源RX1/3/4/6/8和day1/2/3无留出；两个既有信道draw、三个场景lowSuburban/midUrban/lowUrban、11条件。复用cache IQ及physics，不再生成信道、不改变formal received IQ/SNR/困难样本；加载cache可能反序列化未使用L，U真值不读。

11条件为full、no_noise/fixed,total、flat_H/fixed,total、no_carrier/fixed,total、no_RX_IQ/fixed,total、actual_power_quality、edge_boundary。固定状态条件保留full的锁定、补偿和AGC，total条件重算。分层始终使用同包full状态/锁定/新增噪声比，避免按不同干预重新分组。新增噪声比非总SNR；控制收益不可相加；flat_H保持白输入核能量但实际波形功率可变化；边界干预仅诊断合成边界，不虚构未来上下文。EQoff则G=1，实际缓存配置载频/采样率/补偿开关与__file__读取记录，不用代码默认冒充运行值。

每模型全源面板prediction120600样本前向，eval-mode smoothedCE局部梯度90×7=630（只用autograd.grad，无训练或.grad累积），真实CP无query smoke2，共121232；两模型总242464。前向复用读5个既有分支，记录静态坐标gain/默认头margin÷30，不编造逐包门控。只比较固定NoAs30和各自clean正确集合，非E20父集合。两个信道draw不是两个模型seed。既有两个父各累计E30/6660/native7960684前向成本不重复计入新增。

不设性能晋级门槛，不自动训练、正式测试、扩seed或E50。仅技术/权限/非有限/形状/来源/状态更新/输出碰撞故障停所属run，保留产物不自动重试。当前健康任务保持。结果较差仍完成所有预登记源条件；不把代理指标变成额外候选筛选。

环境：N607普通账号、既有CVS-RFFI Python3.10/Torch2.1/NumPy2.2/fullFP32/CPU2；CWD=PROJECT/releases/cvs_FcrossRX_current_controls_E30_20261019_r01。一个顺序GPU进程处理两模型，按全owner每GPU≤2/空闲至少4GB与原共享容量锁选择，不安装环境。不可覆盖run=PROJECT/runs/20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01；owner=codex/root/current-controls79。

命令：CVS-RFFI/python -u experiments/cvs_controlled_revisit79/runner.py --project PROJECT --device cuda:0。首步两个真实CP/defaulthead无query smoke合计4NN，PASS直接源读出；完成后源分析 analysis/cvs_current_controls79_analysis_20261019.py --project PROJECT --run-root PROJECT/runs/20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01 --out PROJECT/runs/20261019-diagnostic-FcrossRX-current-controls-e30-s2026092701-m2-r01/analysis_r01。无truth参数。实际代码提交发布前固定，由release_commit.txt记录。

产物：每模型predictions/branch_metrics/result；source角色/actualmodules/实际缓存引擎/config/CPsmoke/complete；逐draw×scene×condition/RX/day/TX/状态/噪声比accuracy、宏F1、CM、margin、纠错、固定正确集合保持率、固定与总效应、As−NoAs及控制效果交互；4PNG/4PDF和实际成本/缺项。只读完整现有聚合，不重新评分formal。

REJECTED_EXTRA_GATE：通用技能单独设计审批、报告hash/签名和重复审查不在用户八项最小流程内；不为这些停工。保留一次聚焦负例、一次独立P0/P1、一次preflight、单归档传输比较、远端compile与启动后所属进程读回。

实际发布：代码 308f4ca53bf87cb23c47cbcd43ec5b1da2ae8b56 已独立核实推送。单归档传输VERIFIED、13py远端compile PASS；实际所属进程PID/CWD/argv/CUDA/实际GPU0及378MiB已独立观察。两个真实CP/defaulthead/fivebranch smoke 4NN PASS，IQ打开前完成，4次ownE20/FINAL30 CP反序列化但不应用EMA；0训练/query。该段为启动时观察；随后两模型源读出和分析已完成。

最终：新增242464NN/0训练/0formal访问，源probe31.294900302秒，完整源分析105.893837690秒elapsed，4PNG/4PDF已核实。原记录真实上下文元数据补查为256点，候选关闭。无新增正式提升或跨seed双收益证据，目标ACTIVE。详细数字和负证据见final_analysis.md/results_summary.json。
