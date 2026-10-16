# Sinc 频带尺度位移＋原 RDrop：源 RX1 三条件

状态：VERIFIED_RUNNING。代码54ceb11f已推送并核对远端；三行均已通过own真实CP no-query smoke并完成E1/20。尚无本候选最终源域或目标性能结果。总目标仍ACTIVE：使Sinc优化与RDrop配合，在完整匹配三种子验证中提高默认头clean性能；不把源资格当成最终目标。

依据：已交付的 20261016-phase1-sinc-rdrop-gradient-source-s2026092701-m4-r02（结果提交a1c4c74a）拒绝KL删除/投影假设：Efinal50冲突9/50低于预定10，合计CE+KL未反向CE。现有E真实更新的只读复算发现初始带宽约416Hz至4.17MHz；E1保存13步×24通道中，9个观察的边缘移动超过当前带宽。只说明值得检验更新尺度，不证明这些移动有害。

唯一新假设：统一物理单位Adam仍以Fs为所有Sinc通道的位移尺度；按初始频带尺度调整自适应位移，可能让不同带宽的滤波器与原RDrop共同学习得更稳定。固定 w_i=max(B_i初始,Fs/K)/Fs，Fs=25MHz、K=79；Fs/K是既有K点频率格距，用作明确的尺度启发式，不宣称窄带滤波器没有信息。w范围由未训练初始模型直接决定，约0.012658至0.166913，不读目标、不扫倍率。

三条件仅区别Sinc的Adam自适应位移：FULL原统一方案；BAND逐通道乘w_i（low/band同增益）；NORM以正的全局倍率使其自身输入步的自适应L2等于BAND反事实步。NORM只匹配自身输入上的自适应范数，不匹配物理衰减在内的总位移或不同训练轨迹。FP32加法误差单独报告。

实现先保留旧CENTER_UNITS prehook按cosine映射lr_q=alphaFs/25000、eps_q=epsilon25000/Fs，按物理Hz全局clip5；物理衰减 q-=alpha lambda(q+p0/s) 已执行且原生qWD=0，然后原Adam自适应更新，最后只缩放自适应增量。物理衰减不缩放、不重做；Adam moment不变。实际q叶版本变化沿用旧滤波缓存失效；student/ownEMA/推理和feature-only导出沿用同一坐标。

SIG_TIME和joint160、默认6×160/scale30 cosine头、完整RDrop alpha1/LS0.01/普通增强、旧batch_neighbor门控与ownEMA/cos不变。两个随机L前向使用同次真实增强IQ。所有条件默认头随骨干正常联合训练；没有冻结、换头、单独头拟合或额外分类器。新机制只作用于实际f_theta的48个Sinc参数更新，部署结构和参数量不变。无额外模型前向、梯度调用、参数或optimizer步。

先执行单固定seed2026092701源资格：三个scratch20=13L+7U，每轮222步、每行4440步。RX1在创建学生/teacher前从全部训练、U、V、probe与calibration loader排除；有效L5040/U45360隐藏/V21600。1260个原L中的RX1物理样本只在fixedfinal20用默认头评估，144个fit-L anchors按既有fold读取但不用，成本单列。各条件初始化相同、不继承任何A/E或其他训练权重；引用AF09554c3a仅复用代码。NoExplicitCFO及内部逐包补偿保持，不用星地训练增强。

固定源资格：三行完整、finalLCE<=.5，实际q位移/物理clip/衰减和默认头/导出检查正确；BAND在同1260heldL上错误严格少于FULL与NORM，meanCE不高于两者。V只健康诊断，不选epoch。PASS后再固定另一个full-source50比较矩阵及源码、源决定上传后才访问任何新目标汇总；FAIL关闭固定公式，不改倍率、seed、parent、epoch或筛选结果救跑。本源试不自动访问clean目标、practical、Phase2。

成本：3新模型、60模型轮、13320次更新；每行额外L身份前向568320、原L身份前向568320，隐藏U观察396368（45360物理包按原222步预算循环）、1260heldL默认预测及144unusedanchors；每行ownCP smoke2合成包，feature边界40合成包。原初末CP、完整epoch/step、4440增量记录、e1/13/14/20各13个真实位移/物理衰减见证均保留私有产物，Git提交必要聚合及正负结果。

精确逐行seed/config/路径见experiment.json。命令CVS-RFFI/bin/python -u experiments/cvs_sinc_bandgain20/runner.py，CWD PROJECT/releases/cvs_sinc_bandgain_rdrop_source20_20261016_r01，root PROJECT/runs/20261016-phase1-sinc-bandgain-rdrop-sourcefold-s2026092701-m3-r01。CPU dispatcher的CUDA为空，workerCPU2/loader0；共享跨owner锁，每GPU含preCUDA最多3总进程且剩余>=12000MiB。保留所有健康旧任务。

技术停止：来源/角色错误、非有限、预算/合法产物缺失、输出碰撞或worker失败时保留产物、停止新pending、健康worker继续；不因性能低停止或重训。一个不可覆盖root、一次archive SHA/remotecompile、首先own真实CP no-query smoke PASS即训练；一个launch owner立即记PID后有限观察，随后核实CWD/argv/GPU/log增长。激活显式使用CVS-RFFI Python，避免已修复的系统Python问题。

本地CPU/CUDA已PASS：三条件物理origin衰减均未被缩放或双重施加；在完全相同的真实RDrop梯度下，其他特征参数与默认W首步逐位一致，仅Sinc位移不同。NORM局部范数差约4.07e-9/3.17e-9；3个非法输入负例通过。原生非确定性CUDA独立前后向梯度最大差9.10e-7已记录，因此不以独立轨迹的逐位相同作为断言。所有本地验证只用合成IQ，头单独拟合0。

本轮源资格只是开发证据，不是新的独立盲测、目标泛化、newTX或卫星部署收益。最终仍需匹配三seed默认头实际性能提升；目标保持ACTIVE。

发布核验补充：5个固定源契约负例与3个源资格布尔情形PASS；本候选唯一一次独立P0/P1审查PASS，无直接缺陷。

发布后核验：一次archive传输比较、remotecompile均VERIFIED；CPU dispatcher与三个worker的PID/CWD/argv/GPU及日志增长已独立核对。源有效L5040/U45360隐藏/V21600，fitRX3/4/6/8，heldRX1。每行E1的13个真实更新独立CPU复算，origin物理衰减及自适应位移最大误差均0Hz；NORM局部范数的FP32加法舍入差最高0.001177Hz。只复算既有见证，无额外IQ/模型前向。详见run_evidence.json、early_update_recount.json。
