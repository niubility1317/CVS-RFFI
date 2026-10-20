# R14：保留原 R38 上下文，只压缩先验

状态：LOCAL_PREPARED。原 R38 dense_half 的固定源 E20/4940 权重为唯一直接父模型（e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381），不使用 R13 已测权重。其 coarse/action/prior 累计各40轮；本轮只给prior新增10轮。保留完整来源和同7/63/30数据契约，未重验未变数据；已暴露benchmark不称新独立盲测。

R13完整六行60轮记录与全部144行stdout无NaN/OOM/Traceback/提前终止。两窄IQ行末轮teacher-stage NMSE为原宽度15.56/39.58倍；half末轮72.87%更新裁剪，最后5轮总loss仍降18.74%。说明10轮后仍有源域函数拟合缺口，不能把目标下降直接断言为不可压缩容量下限。GN每组通道统计改变与四个网络同时切片是机制推断，尚未通过单因素实验区分。

R14以保留原物理上下文的匹配矩阵检验该推断：coarse/action共548795参数逐位复制、冻结且保持eval；三步prior保留三份bias/GN/1x1及长卷积共享，分别原宽、3/4宽、1/2宽。所有行只用source paired-clean IQNMSE加1倍三stage R38 IQ锚；与R13 IQ目标/预算/顺序/优化器一致。训练只计算一次冻结R38上下文，teacher/student共享该不可变上下文；部署仍自带自己的完整上下文。此复用只消除重复冻结前向，不以参数比例假定加速。

|固定行|完整参数|可训练prior|冻结上下文|较父版减少|GPU|
|---|---:|---:|---:|---:|---:|
|prior_keep|794529|245734|548795|0.0000%|2|
|prior_three_quarter|689129|140334|548795|13.2657%|3|
|prior_half|613041|64246|548795|22.8422%|6|

固定E10/2470、每行L63000/U567000、source630000；received01..10、seed/缓存/15%clean与85%残差不变。重置prior-only AdamW1e-4、warm247、cos1e-5、WD1e-4、clip1、B256/micro64、真实尾24归一化。coarse/action零更新、累计40；prior累计50；component-epoch130，祖先串行pass90，F50预训练另计。0TX标签、0F权重/输出训练、0V/target优化；教师与冻结上下文零更新。

每行无论源域结果如何，固定E10后自动source1800×7诊断/profile、target168000×七场景预测，全部七视图固定后才独立truth-last。固定原F50/defaulthead不重训头。报告原R17用户1pp容差、原R38保持/实际收益、lowurban70、其他五场景距rawclean5pp、clean损伤、波形主体/尾部/FFT、逐TX/RX/day/净纠错与完整训练曲线。没有源成绩资格门槛，不选择性重跑。

单样本float32[B,2,256]原生IQ输入/同形IQ输出，完整自身coarse/action/prior，3阶段×2CG；无teacher、clean、标签、scene、跨帧缓存或状态更新。成本包含完整上下文和物理求解，16sourceV/batch1/2warmup，MAC不含FFT/CG等并与实测延迟/显存分列。

普通N607预检已确认GPU2/3/6空闲；GPU0/1其他owner保持。每卡最多一本任务worker/max3，训练前共享容量锁要求free>=18GB且全owner进程<2，R13对应旧句柄终态与实际退出后才派发，R06锁max1own CPU scorer/2threads。一个新不可覆盖run/release、一个归档SHA两端比较和一次远端compile。launcher首步真实R38/F50/2Ls无query smoke：prior梯度有效，冻结context逐位保持、teacher/F/head零更新、严格重载输出一致；PASS立即启动。技术失败仅保留受影响行并停止该行，健康行完成；不自动重试/回退。

REJECTED_EXTRA_GATE：通用设计/规格/计划批准流程由项目八项最小流程覆盖，无额外用户审批。一次独立P0/P1仅查下一真实运行的合法性、可启动性、输出保护和合法预测，不增加报告完整性门槛。具体命令/环境/CWD/路径/seed/停止规则与artifact见experiment.json。
