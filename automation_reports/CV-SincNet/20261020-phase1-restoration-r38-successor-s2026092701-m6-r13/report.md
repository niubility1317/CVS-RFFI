# R38 79万参数基底的继续压缩与优化 R13

状态：RUNNING。用户最新明确在79万参数模型基础继续压缩优化。本轮唯一直接父权重为R38 dense_half、完整794529参数、固定source E20/4940、commit e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381；原R17为合法source祖先。用户在已暴露benchmark后指定此基底，如实披露，不称新独立盲测。R38训练/固定最终权重未使用TX/F/V/target优化或target选epoch；新训练不读取目标反馈。现有R11/R12健康六行继续完成。

|固定行|完整/可训练参数|context参数|相对794529减少|GPU|训练F知识|
|---|---:|---:|---:|---:|---|
|keep_iq|794529|548795|0.0000%|2|False|
|three_quarter_iq|442805|302471|44.2682%|3|False|
|half_iq|204561|140315|74.2538%|6|False|
|keep_task|794529|548795|0.0000%|7|True|
|three_quarter_task|442805|302471|44.2682%|4|True|
|half_task|204561|140315|74.2538%|5|True|

固定3宽度×2目标，原宽度是同新增预算对照。完整R38原宽度逐位copy；其余按原GN四组等量取通道，decoder拼接使用旧完整upstream宽度后的skip索引；保留谱分支首层投影拓扑，三步prior长卷积共享、每步bias/GN/1×1独立。0IQ/TX/F/V/target校准，coarse/action/prior全部联合训练，教师保留自身完整R38 context并冻结。三步×2CG保持。

所有行损失：最终paired source-clean IQNMSE +1×三个stage到冻结R38输出的IQNMSE均值，分母为detach clean能量/floor1e-4。task行额外T1KL(Fclean||Frestored)+0.5单位F160距离平方；IQ控制正式训练0F权重加载/前向，不能只把权重置零仍执行F。原F50/defaulthead0参数更新、0头拟合。task使用原监督F知识，不称严格无监督；全部0新TX/V/target训练标签。

每行只新增E10/2470、L63000/U567000/source630000，R38 IQ教师/studentcontext各630000；仅task行Fclean/Fstudent各630000。原cache63000=L6300/U56700、既定received01..10、15%clean/85%residual、model2026092701/data2026102001/FP32/noTF32。B256/micro64以真实globalbatch分母累积，尾24正确归一化，每globalbatch一次clip1/update。重置AdamW，LR1e-4/warm247/cos1e-5、WD1e-4；这是独立微调日程，不声称恢复原optimizer轨迹。原祖先coarse20/action20/R17prior20 +R38joint20 +newjoint10=90串行pass；各组件累计50/component sum150，原F50预训练50轮/11100与smoke/profile另计。

固定最终E10，无早停、源分数选行或target资格门槛。每行训练后自动source1800×7诊断/profile，然后正式同168000物理样本×七目标场景；每行七视图IQ/logit/features/IDs全部固定后，独立CPU才读truth/clean及原baseline汇总。逐场景同时报告相对用户79万R38和原R17差值，二者均以100×(参考正确数−新正确数)≤N作为1pp底线；另报真实提分、lowurban≥70%、其他场景距clean、paired-clean全帧/主体232/尾24/FFT/身份特征差及TX/RX/day、净纠错与完整10轮曲线。所有固定行及负结果保留。

推理只当前原生尺度float32 [B,2,256] IQ，输出同形恢复IQ并接原F50/defaulthead；0teacher/clean/TX/RX/scene/cache/跨帧状态更新。成本比较同时含原R17、794529父版与学生：固定16条source V/batch1/2warmup，包含完整context/CG，GPU payload/optimizer/F/teacher排除；不由减参预先宣称加速。

41项合成/协议检查PASS；唯一独立P0/P1发现预测入口原固定E20与新E10不匹配，已修复并经一次定向复审PASS。共享入口默认E20保持，R13显式声明E10/2470，3个实际调用路径回归检查通过。launcher首步实际R38父权重/F50/2Ls CPU noquery smoke，检查所有组件与F输入有限非零梯度、父/F/head不变、严格复载逐位输出、0TX/V/target/optimizer更新，PASS立即逐卡排队。GPU2–7每卡一own/max6；相应R11或R12旧row终态并且实际ownedPID/cmd句柄退出后才派发新行；不等待其他卡、不抢占健康任务。共享R06 score.lock/max1own CPU scorer/2threads；源训练空闲>=18GB且全owner每卡<2进程。

新不可覆盖run/release/output、一次普通N607预检、单archive两端SHA、远端compile/help、实际controller/worker PID/CWD/cmd/commit/GPU/日志读回。队列等待只称QUEUE_VERIFIED，不冒称GPU训练已开始；GPU启动后记录E1。技术错误停止受影响行并保留、健康行继续，不自动重试/回退。具体命令、环境、CWD、路径、停止规则和预期产物见experiment.json。

普通N607预检VERIFIED：R38实际父权重source E20/4940、0TX/F/V/target训练、各组件累计40及原R17source-only来源核实；新run/release不存在。R11/R12六个实际GPU句柄仍在运行，只提交CPU真实权重smoke与逐卡接续队列；0原任务停止/重启。

实际队列发布VERIFIED：方法commit `593bf5905f50fb7894e475b062bc191a2837a149` 已推送且远端OID独立比较一致，单归档SHA与远端compile/help通过。实际CPUcontroller PID/CWD/完整命令/commit及持续等待日志核实；真实R38最终权重/F50/2Ls noquery smoke PASS，六行全部组件及F输入有限非零梯度、父/F/head0更新、严格复载逐位输出一致。当前旧R11/R12六卡继续健康运行，新行按各卡真实句柄退出接续；排队状态不等于GPU训练或目标性能已验证。

GPU 实际训练 VERIFIED：keep_task (GPU7, E1), half_iq (GPU6, E1)。实际 PID/CWD/命令/commit/GPU 与完整首轮记录核实，0 新技术故障；其余行按原队列接续。目标结果尚未齐全，不宣称完成。

GPU 实际训练 VERIFIED：three_quarter_iq (GPU3, E1), keep_iq (GPU2, E1)。实际 PID/CWD/命令/commit/GPU 与完整首轮记录核实，0 新技术故障；其余行按原队列接续。目标结果尚未齐全，不宣称完成。

GPU 实际训练 VERIFIED：half_task (GPU5, E1), three_quarter_task (GPU4, E1)。实际 PID/CWD/命令/commit/GPU 与完整首轮记录核实，0 新技术故障；其余行按原队列接续。目标结果尚未齐全，不宣称完成。
