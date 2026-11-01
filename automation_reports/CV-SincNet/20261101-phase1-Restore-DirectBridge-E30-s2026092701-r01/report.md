# 四步共享权重IQ恢复：从零单阶段30轮

**六景0/6，合并clean保持，clean准确率86.603571%。** cross_rx clean净变化+163，cross_both净变化-54。完整目标未达到，保持 ACTIVE 和默认 DURA。[完整性能与分组](analysis.md)。

- run_id：`20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01`
- group_id：`restoration-paired-direct-iteration-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（完整七景与独立复算已核实）

## 目的与对照

原源配对构造四个时间状态，单次estimate训练、固定四次共享权重部署；保留连续160指导和幅度约束，自动原Main七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

原 Main168000×七景完整，六景0/6、合并clean保持。84单元与完整分组/配对复算PASS，详见analysis.md。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 方法与源端依据

本版为独立四步直接迭代IQ恢复器，4677538参数全从零单阶段30轮/7410更新。不是多阶段训练或四个独立模型。每条训练记录只调用一次共享estimate；部署固定四次同一权重，始终以同一当前received IQ作条件，不能接收clean、TX/角色、额外观测、跨query统计或F0特征。输入编码器接收中间state2+原received2+时间1+坐标1，连续160回归与四个feature FiLM保持，新增四层time FiLM。

源训练state=(1-t)detachedclean+t detachedreceived，t=1概率1/2，t=.75/.5/.25各1/6，无额外噪声；clean IQ在训练中构造输入是明确的数据处理，不伪称只进入loss。连续clean F0特征与clean RMS仍只作损失。部署从z=received开始，每步z_next=z+(predictedclean-z)/remaining，remaining为4/3/2/1，最后直接predictedclean；每次预测clean端点并单位复功率投影，中间z不额外投影。更新形式借鉴[InDI Eq5](https://arxiv.org/html/2303.11435v1)，这里增加received条件、源F指导和IQ功率支持，不是原论文复现，不把图像感知收益当作身份保真证明。

原冻结源512×20面板中，低档城区162条Fclean不一致的131条来自模拟失锁；有限真作用/高斯先验并非上限，额外clean特征oracle有用也非可达部署收益。该证据支持检验逐步恢复先验，不证明跨RX问题已解决。本版是中间训练分布、结构条件和4步执行共同变化的系统比较，不按Main成绩挑父权重。固定步数，不扫描目标，不用Main错误、TX、V/OE或目标来选模。原PowerAnchor只作完整冻结对照，G没有任何旧G/coarse/DURA/神经IQ教师初始化或指导祖先。

损失保持IQ NMSE+.05 STFT+.1输出F0余弦/源方差标准化MSE+.1连续latent同尺度回归+.1投影前raw/clean RMS比误差。原Ls6300/Us56700、R07 20份draw+前10复用、order2026110102/model2026092701、B256tail24、AdamW/.001-warm247-cos2e-5/WD1e-4/globalclip1均保持。新增time RNG为globalCUDA2026110103并随技术恢复快照保存；逐轮time实际计数和各位置损失单列，不能将更接近clean的中间输入均值与旧received-only训练均值直接比较。源F0 clean63000目标与f57方差复用，在线F0特征1890000行，teacher不更新；方法F-guided，OE仅冻结诊断，V/novel/目标不训练/早停/选模。

17项相关检查（新7+FeatureCond7/PowerAnchor3）及一次独立P0P1通过。真实Ls2 smoke为启动首步，检查训练梯度、无状态更新、四步部署E0及严格重载，PASS立即完整30；E01..30只技术恢复、final30固定，源质量不取消正式测试。随后自动原Main126000+42000各七景，42单元全部预测固定才独立CPU truth-last/原共享锁；raw/DURA同IQ-ID精确复用，新G完整四步推理。逐景至少140345/168000正确=83.538690%，距同一raw clean86.538690%≤3pp，clean两个总体保持。报告TX/RX/day/RXday、相对DURA/对照/clean修复与损伤，保留单seed和已曝光开发基准限制；默认DURA不更换。

静态合计1043189760实数Conv/Linear/attention MAC/包（4×260797440）、0FFT/CG；训练每记录1core、部署4core且F0只在G之后读取1次。排除F0、归一化/激活/softmax/插值/FiLM逐点/单位投影/bridge更新/内存，不是实测时延。N607既有CVS-RFFI/Torch2.1+cu121/FP32-TF32off，GPU4/6 UUID准入free>=12000MiB/原compute<=2/只新增1条本任务NN/单卡总<=3，其他owner保持。不可覆盖root/命令/CWD/停止规则/产物见登记。


## 四步直接迭代恢复器已运行（2026-11-01）

`20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01`，实际代码`84a063c9ba1f9974cfe361e293a9e11fa93920b4`，4677538参数，从零单阶段30/7410、训练每记录1次estimate、部署4次共享权重，始终条件同一received IQ，无额外观测或部署clean/F0。源IQ配对构造t1/.75/.5/.25状态，原F0连续160与rawRMS参考loss-only；0TX/V/OE/目标训练或选模，OE仅冻结诊断。结构/训练状态/执行共同变化的系统比较，不冒称单因素。

VERIFIED / RUNNING：17重点相关检查/一次P0P1、真实源2训练梯度与完整四步E0/严格重载PASS。首轮247更新耗时59.003244秒，E01模型/AdamW/CPU-CUDA RNG独立CPU读回PASS，E0与固定seed factory逐张量一致；4个time实际计数和各项指标重组PASS，总loss确认原权重各加一次。GPU4准入free21865MiB/原2compute/仅新增1条，PID/CWD/命令/CUDA/GPU和日志核实，其他owner保持。固定final30后自动原Main168000×七景、全部预测后独立truth-last，当前性能PENDING；主验收仍六景各距raw clean≤3pp并保持clean，性能主线、效率支线和默认DURA不变。

## 源端方法限制补充（Main预测前）

[配对状态的代数捷径说明](paired_state_shortcut.md)记录同时输入state/received/time时可直接解出训练clean的恒等式及四步固定点。它不证明实际模型一定使用捷径；当前84a063c9已固定训练和完整Main矩阵保持。独立StateBridge只去除原received条件，不回流修改本版或取消测试。

## DirectBridge完整七景完成

VERIFIED / ANALYZED。全G从零单阶段30/7410，部署固定4次共享权重，两个Main总体各七景全部预测后独立truth-last；4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对复算PASS，误差1.25e-14、raw/DURA判决差0。完整60轮/stdout/源顺序draw/固定时间协议/从零E0/总loss与每t重组/30快照E30-final核实。完整目标未达到，保持 ACTIVE 和默认 DURA。
