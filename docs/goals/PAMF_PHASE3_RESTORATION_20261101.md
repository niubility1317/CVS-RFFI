# Phase3 独立星地恢复网络

## 最新优先级与主验收（2026-11-01 澄清）

用户明确“性能才是重点，训练效率是支线”；性能基座固定为地面训练的完整 Tri-SIG-TIME SSL，禁止把 Phase3 RX34 开放开发集或旧 F50 回归当作主结果。主测试采用 Main split 的原定全量物理总体，clean 加六种 practical，原跨 RX／跨 RX 与日期分组完整报告。固定同一 F、默认头与 raw clean 参照，六景逐个差距≤3个百分点，并保持 clean；效率、参数或少阶段不能抵消性能退化。此目标尚未达到。

新网络为 Phase3 准备，但本任务先解决上述恢复性能，不接管拒识／目标适配／发现／匿名注册。已训练 TF 纯 IQ 与 F0 指导两臂仍是合法原 L_s/U_s 从零联合30轮；其教师已是实际地面 Tri-SIG-TIME SSL 原 F0/3ff8/E50-11100。保留全部权重和既有诊断，补齐正确基座、正确全量七景的主测试，不根据辅助结果选择性丢弃模型或视图。OE继续只作冻结诊断，默认DURA保持。

## 用户指定与完整目标

2026-11-01 用户明确网络可以大幅改造，不要拘束于 DURA，新恢复网络为 Phase3 准备；同时选择保持原 L_s/U_s 训练范围，OE 只作冻结诊断。继续原完整目标：六景逐个距同一冻结识别器的 raw clean≤3个百分点，保持clean，减少多阶段训练对研发迭代的成本。新路线从零单阶段30 epoch，不以诊断或较小模型代替最终验收。

本文件为恢复器研发主线。旧DURA性能/源F指导目标与旧F50记录保留历史身份，不改成Phase3成绩；整体Phase3开放世界、无标签适应、发现和匿名注册目标由 `PAMF_PHASE3_OPEN_WORLD_20261028.md` 管理，本任务不接管其他owner。

## 方法自由度与独立路线

不固定DURA的coarse/action estimator/CG/prior/两阶段展开，不固定266951参数，不以压缩旧架构为先决条件。结构、容量、完整实测成本与识别质量一起比较。

第一条待验证路线暂名 `P3-Restore-TF`：直接残差式时域/频域联合网络，输入输出均FP32 `[N,2,256]`。不下采样的扩张时域分支处理局部与长程波形，逐记录频域混合处理整个256点包，融合门控只读本条隐藏特征。零输出初始化使初始G为恒等映射；不加载DURA权重或神经IQ教师，不把信道真值或身份作为部署输入。原物理逆算子不作为强制组件。

先实现完整架构并核实梯度、FFT端点、逐记录独立性、实际参数与GPU步骤成本，再固定正式从零30轮矩阵。正式两臂可比较同一新架构的纯源IQ目标与源F0特征指导；损失、种子、数据及预算在run预登记，不自动继承旧F50的特征目标/权重。代码草稿与零更新诊断不是性能结果，也不额外要求设计批准。

## 数据和教师边界

- G仍只训练原 L_s6300/U_s56700，使用合法IQ和源模拟配对，TX真值隐藏。复用已验证物理角色和数据句柄，不因换架构重验全量数据。
- Phase3主观测为既定G10：600 km、25 MHz、2.45 GHz、residual/post_sync/no额外EQ、reference SNR30 dB、reflect和原六分量概率。旧R07的2.462 GHz、20缓存和独立七视图身份保持，不能改标G10或重复归一化。新源配对生成路径、一次RMS和种子在run固定。
- 源clean配对仅用于已授权恢复器源监督；target clean仅冻结测试。源V/novel/OE不进G训练、选模或早停，目标U/Q也不进G训练。
- 指导模型为实际Phase3原Tri-SIG-TIME SSL F0/defaulthead，核实其合同和祖先后只读合法源特征。原F50仅历史对照。F0冻结，不使用TX损失或六类原型吸附，不另训分类头，F-guided知识与成本披露。
- source-only r03和目标适配PCD只用于冻结下游诊断，不作G教师或初始化。PCD已有target-U祖先，不能称整个系统纯源。

## 验证与交付（以下Phase3开放条款现为辅助诊断）

1. 每个实际训练新版冻结后完成原定自动正式测试，不因源分数较差取消目标测试，不选择性重跑。43577dc71发布的两个DURA全率分支继续原30轮与旧F50七景，不热改成新方案。
2. 新主线用实际Phase3 F0计算raw clean参照和新G输出，六景逐个≤3pp、clean保持。正确条数由该F0和同总体的冻结raw clean计算，不套用旧F50的142256/137216。原F50七景及R17容忍保留为注明身份的历史回归。
3. 新G与原DURA使用同记录/同received，报告身份准确率、双向纠错、逐TX/RX/day、IQ误差和成本；缺必需类别覆盖时保持N/A，不冒充全六景通过。
4. Phase3诊断同时含源域、跨RX clean和跨RX星地，接实际原F0与冻结开放路径，主报宏A_K/C_K/R_U/H-score。原C_K≥90%、clean R_U≥95%、星地R_U≥90%、同输入闭集到open损失≤2pp保持；另保留原DURA系统闭集基准，不能靠降低新G自己的闭集基线掩盖整体损伤。
5. 直接替换G、重用旧PCD/密度/阈值属于固定系统诊断，必须标明输入分布与祖先。需要匹配的新下游校准/适配时另行登记，按既有Phase3权限执行，不用旧DURA校准成绩冒充新G匹配部署；恢复器进步不自动完成发现/匿名注册。
6. 报告单阶段依赖链、全部祖先/F成本、步数/曝光、缓存制备/读取、单步/每轮/全训练墙钟、峰值显存、完整参数/MAC/FFT和实测G＋Phase3系统时延。完整质量配对后才讨论效率，少阶段或少参数不证明同质量提速。

只执行八项最小流程。默认N607，保持本聊天GPU4/6/7每卡新增1个、每卡总≤3、原compute≤2且剩余≥12GB、本任务每卡1个/整体≤2GPUworker、CPUtruth scorer≤1；不改其他owner或覆盖输出。旧pair健康完成后释放自身容量，再发布新主线。完整目标ACTIVE。


## 首个架构实现状态

独立包 `experiments/pamf_phase3_restoration/model.py` 已实现，不依赖DURA模型。宽64、六块膨胀1/2/4/8/16/32、全率时域门控与逐通道复频域滤波，输出为received加零初始化残差。FFT的DC/Nyquist不设置无效虚部参数。完整196898参数、实际Conv/Linear MAC19646464；每包六次rFFT和六次irFFT、每次64通道，共768个实通道变换，FFT/GN/激活成本不包含在MAC中。容量不是效果或速度证据。

9项聚焦检查PASS：恒等初值/随机流/严格state重载、零输出及激活后的所有时频/门控梯度、复频滤波与完整Hermitian DFT的前向/参数伴随、逐记录/置换/单条推理独立性、完整MAC/频域调用和输入形状/精度反例。随后r02已完成实际Phase3 F0加载、512源G10配对、真实源smoke和GPU成本诊断，详见下方完整结果。尚未完成63000训练配对缓存、任何新G优化或正式性能测试；下一步登记从零单阶段30轮匹配矩阵，不把成本PASS当质量完成。


## 首次真实源F0与GPU成本诊断

20261101-diagnostic-P3Restore-TF-source-cost-s2026092701-r01：原Tri-SIG-TIME SSL F0/3ff8/50-11100、L_s6300/U_s56700同392005合同。固定128Ls384Us的现成G10 MIX01 P0及原raw配对clean，0TX/V/OE/target/optimizer。新TF与固定R69只读图拷贝对照，两种损失/三块/每块2warm6timed/B256交替，共96次forward-backward-clip，0step。原R69不是新TF祖先或教师IQ；F0指导.1cos fused160、IQ输入梯度cap.5。首2Ls真实smoke后继续，参数状态不变；只证明成本与输入梯度接口，不冒充等质量提速或新性能。23聚焦检查和唯一P0P1 PASS，默认与完整目标保持。


首个源成本r01在真实smoke的DURA公共返回值适配处技术失败（engine reader=None仅返回IQ单项），旧root保留且PID已退出，无96步计时/优化/target。新20261101-diagnostic-P3Restore-TF-source-cost-s2026092701-r02仅修正取[0]并隔离root；来源、数据、96步与科学条件不改，一次直接问题针对性复审PASS。


## 实际Phase3 F0与成本诊断已完成

20261101-diagnostic-P3Restore-TF-source-cost-s2026092701-r02，执行b19c2ef6d9b75c5afecbafb4ecd3d44716e73c31。真实源2Ls smoke、512固定G10配对和96零更新步骤完成，独立96时序/512ID-RMS及计数复算最大差0。纯IQ步骤DURA128.642678→TF31.555337ms，减少75.470553%；实际F0指导172.115650→69.454961ms，减少59.646342%，两种情况3/3块均快，预定成本条件PASS。来源已绑定实际Tri-SIG-TIME SSL F0/3ff8/50-11100；0TX/V/OE/target/optimizer，模型状态不变。R02仅修复r01生产单输出适配，r01失败与成本保留。完整质量尚未验证，下一步单阶段从零30轮匹配纯IQ/F0指导及完整正式测试；原Ls/Us边界、OE仅冻结诊断、默认DURA和完整目标ACTIVE保持。


## 完整源配对与特征缓存完成

VERIFIED / ANALYZED。原 L_s6300/U_s56700 的完整 G10 MIX01 P0-clean 配对及独立 F0 clean fused160 缓存完成，执行 53410ce889adc94c53c1783741f970fbdaf51453。0 优化、0 TX 标签、0 OE/V/目标输入，未训练新版 G；OE 保持仅冻结诊断。

配对制备（含真实 2 条 L_s smoke）7.241917 秒，63000 条 F0 clean 缓存 2.309879 秒，完整 worker 10.528927 秒。F0 为实际原 Tri-SIG-TIME SSL / 3ff8 / E50-11100 / 232993 参数。缓存 IQ 两阵列 258048000 字节，F0 特征 40320000 字节；纯 IQ 优化 consumer 不打开指导阵列。共享缓存成本及 F0 原训练知识分别披露，不能称整个研究没有 F0 访问。

独立 0NN/0truth 核对：63000 条 ID/角色/RX/day 顺序一致、全部 IQ/特征有限；clean 复能量与 1 最大绝对差 2.07295787202e-08。与 r02 成本诊断重合的 512 条样本，其 clean、received、ID、角色、RX、day 六阵列逐位一致。原 raw ManySig 按 whole PKL 读取，包含其他角色字节与 TX 坐标命名，但仅索引原 L_s/U_s 包值；私有定位坐标不输入 G。63000 次原 raw RMS、2 次 canonical G10 审计、0 次信道后 RMS，首两条 P0 逐位复现。

真实源 smoke PASS，G 恒等初值、实际 F0 输入 VJP、0.5 梯度上限及模型状态保持。GPU 7 以原 1 个 compute 进程、22883 MiB 余量准入，新增本任务 1 个；worker 内 NVML 记录了自身 PID。首次独立读回时任务已自然退出，完整进度、smoke、manifest、结果及 controller 终态相互吻合；未为补遥测重跑。24 项聚焦检查及唯一独立 P0/P1 审查 PASS。

这只证明源训练数据已备好。尚无新 TF 的实际学习或正式性能结果，不能据此判断能恢复 clean 的几成、六景 3pp 或 Phase3 开放指标达标。下一步为独立纯 IQ / F0 指导、同初值和源顺序的从零单阶段 30 轮匹配组；自动冻结测试与独立评分已在实现核验中。默认 DURA 和完整目标保持。


## 独立TF单阶段30轮匹配组已运行

两臂 20261101-phase3-P3Restore-TF-IQ-E30-s2026092701-r01 与 20261101-phase3-P3Restore-TF-F0-E30-s2026092701-r01；执行 b33839dd1010dea36b68526043bbc4abe5eb1095。同初值/原63000源顺序/15%clean保持/30轮7410/B256尾24；IQNMSE+.05三尺度STFT，指导臂另加.1实际F0 fused160 cos/IQ梯度cap.5，全局clip1。AdamW.001/warm247/cos2e-5/WD1e-4。无TX/OE/V/target训练，固定final30，不选模早停。两臂实际真实源smoke、E0严格重载和teacher冻结通过，GPU7/4实际PID-CWD-argv-CUDA-NVML及日志核实。主Phase3四个角色与历史168000七景freshprocess全部预测后，独立CPU共同锁评分；固定PCD/密度/tau是输入切换诊断，非匹配部署。21聚焦检查+唯一P0P1两臂PASS，阶段资源修复一次定点PASS。完整目标与默认DURA保持，结果未定。


## 正确主基座全量七景已完成：性能未通过

20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03，执行d4c995c1139fc62779a2a254c269006324b0415f。同一地面Tri-SIG-TIME SSL原3ff8/E50-11100/232993/defaulthead；原Main split cross_rx126000与cross_both42000官方observations/positions/IDs，raw+DURAv1+两TF共56单元。1176000条raw决定与原正式记录差异0。source2实模smoke通过，1736评分/1302raw配对/1302额外配对独立0NN复算PASS，最大差1.24e-14。

合并raw clean86.538690%（145385/168000）；3pp阈值140345正确、83.538690%。默认DURA六景84.343452/83.016071/81.322024/83.049405/75.705952/68.801190，仅1/6，低城区差17.737500pp。两TF均0/6，各版clean均保持。F0指导较同结构纯IQ六景+1.404167/+1.863690/+2.373214/+2.470238/+6.370238/+8.314286pp，但指导版仍六景低于默认，不晋级。跨RX默认1/6、跨RX+日期3/6，两TF两组均0/6；不以合并均值掩盖差距。

性能主线与效率支线保持。后续先基于原L_s/U_s的源拟合与模拟覆盖诊断，设计更强的按包条件恢复和数据一致性结构；允许容量和计算增加，不以小网络或DURA结构作为前提。仍从零联合30轮、冻结原F0/defaulthead、不直接用TX/OE/V/目标优化；所有新版完整主测试，目标分数不回流调参/选模。Phase3开放和旧F50只辅助，其他owner与默认DURA保持。完整目标ACTIVE，六景逐个3pp尚未实现。


## 性能优先的新候选：MSAT 与源模拟配方匹配对照（2026-11-01）

源端已知信道诊断20261101-diagnostic-Restore-source-inverse-s2026092701-r01已VERIFIED/ANALYZED，10240实例/20draw/0更新-TX-V-OE-target，独立聚合复算差0。真实信道的各向同性Gaussian MAP弱于DURA，64到128CG所有实例指标变化0；只说明该固定求解器加迭代缺乏收益，不能当理论极限或主测试结果。

新run 20261101-phase1-Restore-MSAT-M20-F0-E30-s2026092701-r01，实际方法commit8720a8584803366d42d2879fb99502b8d220ac26。MSAT为4249922参数、四时间尺度/四个8头包内注意模块/全率跳连，不受DURA或小模型约束。TF_M20为196898参数原TF，同R07原Ls6300Us56700二十份模拟、同shuffle/损失/actualF0指导/AdamW，从零单阶段30轮/7410；21到30轮复用draw1到10。R07为2.462GHz六景均衡配方，不能改标G10，也不把旧TF到新TF的差异单独归因于覆盖。

两臂已在N607各通过真实源smoke并训练；每条final30冻结后自动完成正确地面Tri-SIG-TIME SSL/Main split cross_rx126000与cross_both42000的clean+六景，再独立truth-last。r03相同IQ/ID的raw与DURA预测精确复用；全部新G均实际推理，源差也不取消测试。性能主线/效率支线、六景逐个3pp+clean、OE仅冻结诊断和默认DURA保持；目前无新性能成绩，完整目标ACTIVE。


## MSAT 与匹配 TF 全量测试已完成（2026-11-01）

20261101-phase1-Restore-MSAT-M20-F0-E30-s2026092701-r01，实际代码8720a8584803366d42d2879fb99502b8d220ac26；两臂均从零单阶段30轮/7410步，正确地面Tri-SIG-TIME SSL/Main split两总体七景完整，独立truth-last后再NumPy全组复算。84单元/2604评分组/1736raw配对/1302额外配对，指标差≤1.20e-14，raw与DURA复用决定逐条差0，全部本任务进程已退出；登记ANALYZED。

原raw clean86.538690%，3pp门槛83.538690%。TF_M20六景0/6、clean86.642262%；MSAT_M20六景0/6、clean86.426786%（较raw下降0.111905pp）。MSAT六景为77.389286/75.841667/74.149405/76.324405/70.504167/65.055357%，全部低于DURA；低档城区相对匹配TF提高12.676190pp，仍低DURA3.745833pp。默认DURA保持，完整性能目标ACTIVE。

源端最终IQ NMSE TF0.045520→MSAT0.018515、F0cos距离0.027277→0.006411，不能代替跨RX正式性能。单纯扩大网络容量的本次检验未达目标；下一步需以原LsUs源证据区分物理反演、先验迁移和F细节约束，不能据目标误分类选择新超参或盲目扩种子。既有模型/失败结果保留，OE仍只作冻结诊断，无新训练已登记或启动。主报告 automation_reports/CV-SincNet/20261101-phase1-Restore-MSAT-M20-F0-E30-s2026092701-r01/analysis.md。


## 强先验与真实信道的源端反事实已完成（2026-11-01）

20261101-diagnostic-Restore-learned-prior-physics-s2026092701-r01，执行f57c54cad7491e3e8a36f7ea51a9caa13a3a0ab9。原LsUs128/384包×20draw=10240、额外63000源clean F0方差，DURA/MSAT及各自真实Action16/64CG共六路，0G更新/TX/V/OE/target，全部进程已退出。独立聚合差0、逐维差≤7.62e-9、DURA基线逐实例等于旧source_inverse。

MSAT源NMSE0.018428/F一致率96.133%，真实Action后0.017164/96.172%；低档城区NMSE略降，F一致率87.396%→86.838%（修复7、损伤15、净−8）。16→64CG判决逐条差0。源方差标准化误差揭示低/中方差方向较大相对误差，与Fclean判决偏离关联略好（MSAT总AUROC.986085 vs cosine.984633），不足以证明新训练或Main收益。不可把oracle或1−NMSE称理论恢复成数。

完整目标继续ACTIVE，性能主/效率次/正确Main TriSSL六景逐个3pp+clean/OE仅冻结诊断/默认DURA保持。候选设计必须来自原源证据，不因本诊断盲目扩大CG或继续扩容。暂无新G训练。


## 受控F特征尺度实验已启动（2026-11-01）

20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r01，方法3729a7135728500ab6013894ac060d85e3db8a67。MSAT结构/初值/原R07源数据与20+10draw/shuffle/优化器和30轮7410步均同旧MSAT。唯一方法变化是把.1cos源F约束改为.1(cos+逐维源方差归一化特征MSE)，保留统一.5 IQ梯度cap。方差是原63000源clean的F0-only moments，来自已闭合f57诊断，非G输出/身份原型/新分类头/神经IQ教师。新G仍随机从零联合单阶段30；源TX/V/OE/novel/目标均不学习或选模。

6聚焦检查/唯一P0P1 PASS，单archive SHA/remote compile/preflight及source2真实smoke/严格E0重载PASS；GPU7原0compute/24124MiB余量，仅新增1NNworker。全部原Main cross_rx126000/cross_both42000七景自动测试和独立truth-last保持，无源负结果取消测试。当前尚无新正式成绩，性能目标ACTIVE/效率次级/默认DURA保持。


## VarF实验因N607设备平台故障停止（2026-11-01）

r01在第30轮CUDA launch failure，GPU7随后独立查询不可用。29完整epoch合计1378.407422秒，未保存已训练状态，无final30/正式预测，不当科学负结果。独立r02实际代码88daaef5db065f88203da98d777f6711b999b97e，仅恢复资源准入/UUID/逐轮恢复快照，科学配方不变；source smoke前CUDA初始化仍失败，0epoch/0update/0预测。三个fresh-process微探针（4编号、4 UUID、6编号）均失败，故障在已授权GPU的CUDA执行层，具体驱动硬件原因UNKNOWN。两run登记FAILED/NO_PERFORMANCE_RESULT，无本任务活进程。

不再盲目第三次发布，不重置设备或改动其他owner。本机5070Ti/Torch2.10+cu128无模型CUDA探针PASS，但源缓存/原F0运行包未迁入、环境不同；已询问等待N607恢复还是迁移本机并重核基准，尚无选择，不能开始依赖该选择的正式训练。性能目标ACTIVE、六景逐个3pp+clean保持/原LsUs/OE仅冻结诊断/默认DURA不变。

## 2026-11-01 N607 CUDA 恢复：VarF r03 同配方继续

N607 GPU 7 的设备查询已恢复，GPU 4 在原 Torch 2.1.0+cu121/CUDA 12.1 中实际 CUDA 分配、同步和计算 PASS（驱动 580.173.02）。本任务未重置设备、改驱动或干预其他 owner。此前平台阻塞已解除，原 r01/r02 失败及成本记录保留，不视为性能负结果。

新独立 `20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r03`，实际代码 `9fdb1f5b164484f2bc438cb1e45d1ee805d2473a`。仅改 run ID，沿用已审查结构、原 L_s/U_s、原 F0 特征指导、初值/顺序/损失/学习率，重新从零单阶段 30 轮、7410 更新。9 项聚焦检查通过；真实源 smoke、严格 E0 重载、GPU 4 实际训练及 E01 模型/优化器/RNG 技术恢复文件独立读回 PASS。默认 DURA 不变，性能目标未完成。

每轮恢复文件不参与选模，固定 final30 后自动原 Main split 168000×七景完整预测和独立 truth-last；六景逐个距 raw clean ≤3 个百分点并保持 clean。训练效率仍为支线，失败重试及逐轮保存成本如实披露。详见该 run 的 report/launch_evidence；本次不迁移本机。

## VarF r03 原 Main split 七景测试完成（2026-11-01）

`20261101-phase1-Restore-MSAT-VarF-E30-s2026092701-r03`，实际代码 `9fdb1f5b164484f2bc438cb1e45d1ee805d2473a`，从零单阶段 30 轮/7410 更新，原地面完整 Tri-SIG-TIME SSL/default head。168000×七景预测固定后独立 truth-last，4 个执行阶段均退出；额外复算与既有原 MSAT 的 84 单元，2604 评分组、1736 raw 配对、1302 额外配对，最大指标差 1.24e-14，原 raw/DURA 判决差 0，30 份 PRIVATE 恢复文件及 final30 一致性通过。

raw clean 86.538690%，每景门槛 83.538690%。VarF clean 86.585119%（保持），六景依次 77.492857/75.969048/74.103571/76.473810/70.797619/65.251786%，达标 0/6；相对原 MSAT 依次 +0.103571/+0.127381/-0.045833/+0.149405/+0.293452/+0.196429 个百分点。完整性能目标仍未达成，保持 ACTIVE；本版不能替换默认 DURA。默认 DURA 和旧模型保持，源拟合或局部收益不替代六景逐个验收。

本次 CUDA 恢复后原环境继续，未迁移本机、重置设备或干预其他 owner。报告披露此前 r01 已知 1378.407422 秒完成轮次成本、未计入的部分 E30/启动时间，以及本次逐轮保存所含实测成本。单 seed、驱动恢复后轨迹不保证逐位一致；不以目标误分类重新选择超参或 checkpoint。详见该 run 的 analysis.md。

分总体补充：合并 clean 多对78条，cross_rx多对79条，cross_both少对1条；不宣称所有子总体clean均保持。已知失败加本次训练累计至少45.499分钟，部分失败轮次/启动等另计。详见analysis.md和results.json的scope/cost字段。


## 源 F 特征的 RX/日期均值检查（2026-11-01）

`20261101-diagnostic-Restore-Fgeometry-RXday-s2026092701-r01`，实际代码 `afd4af77ed637fd8a5a0908c264c1d8b0b90c72e`，复用原63000 L_s/U_s冻结F0特征，0 IQ/TX/V/OE/目标/模型加载或训练。完整160维、5RX×3日期，CPU统计及独立逐组复算完成，最大方差差1.50e-14、汇总比例差4.52e-16，进程退出、登记ANALYZED。

原尺度→VarF固定尺度后，RX均值相关变异份额0.890638%→0.612513%，日期0.085049%→0.052966%，联合1.129847%→0.780963%。不支持“标准化放大RX均值差异”的解释；不排除协方差、相位或非线性域偏移，也不证明因果干扰或允许删特征。没有新G版本或Main测试，性能目标ACTIVE、默认DURA不变。后续物理联合训练方案仍需独立实现/预登记/单阶段30与完整原Main七景，不能据源统计预先宣称性能改善。

## PHY-MSAT 联合物理网络已启动（2026-11-01）

`20261101-phase1-Restore-PHYMSAT-E30-s2026092701-r01`，实际代码`8198a2e70c655f24ebaf321f232a676cfb46d916`，8613449参数；新信道/噪声估计器加两次不共享context-MSAT先验，每次8CG，共16CG/81complexFFT-IFFT。全部从零单阶段30/7410，不继承旧G或分阶段预训练。原LsUs、R07二十份模拟加十份复用、原冻结地面Tri-SIG-TIME SSL F0特征指导；新增真实模拟H(clean)与噪声log源监督，非TX标签。部署只输入received IQ。原VarF仅作冻结对照，不能声称纯结构单因素消融。

13项聚焦检查和一次P0/P1审查PASS；实际源2条F0/新G/梯度/E0重载PASS。GPU4准入free22571MiB、已有2个计算进程，仅新增本任务1个。已完成首轮247更新（137.671秒），E01模型/优化器/RNG独立CPU检查PASS。仍需完整30轮及原Main两个总体七景固定预测后truth-last，不因源分数差取消。默认DURA不变，主性能目标ACTIVE/未达成。零target/TX/V/OE训练，未改其他owner。日志、权重与恢复文件保持PRIVATE，详见该run的report和launch_evidence。

## PHY-MSAT 原 Main 七景全量完成（2026-11-01）

`20261101-phase1-Restore-PHYMSAT-E30-s2026092701-r01`，实际代码`8198a2e70c655f24ebaf321f232a676cfb46d916`；8613449参数、两次不共享先验和16CG/81复数FFT-IFFT，额外源模拟监督，全部从零单阶段30/7410。原地面Tri-SIG-TIME SSL/default head，168000×七景完整预测后独立truth-last，4阶段均退出；与旧VarF84单元复算，2604评分组/1736raw配对/1302额外配对，最大误差1.24e-14，raw/DURA判决差0，全部30恢复文件和final30一致。

raw clean86.538690%，各景门槛83.538690%。PHY-MSAT六景依次79.983929/76.972619/74.065476/78.251190/69.658333/62.831548%，达标0/6；相对VarF依次+2.491071/+1.003571/-0.038095/+1.777381/-1.139286/-2.420238个百分点；合并clean 87.216071%（保持）。完整性能目标仍未达到，继续保持ACTIVE，默认DURA不变。结构、容量、物理求解与额外监督共同变化，不作纯结构归因；source误差或效率不代替正式指标。每个子总体clean净变化为{"cross_rx": {"count": 126000, "raw_correct": 111019, "PHY_correct": 112017, "net_correct": 998, "accuracy_change_pp": 0.792063492063491}, "cross_both": {"count": 42000, "raw_correct": 34366, "PHY_correct": 34506, "net_correct": 140, "accuracy_change_pp": 0.3333333333333286}, "combined": {"count": 168000, "raw_correct": 145385, "PHY_correct": 146523, "net_correct": 1138, "accuracy_change_pp": 0.6773809523809433}}。完整分组、配对、成本和限制见该run analysis.md。

## 联合 PHY-MSAT 源端四路替换完成（2026-11-01）

`20261101-diagnostic-Restore-PHYMSAT-factorial-source-s2026092701-r01`，执行ca5cf38aff632f4f037b15f056a320a5a0b88509。原Ls128/Us384固定512包×20 R07 draw=10240实例，冻结原F0与PHY-MSAT final30，0TX/V/OE/novel/目标/更新；blind/真实H/真实noise/二者同时替换全部完成。真实source2 smoke的blind逐点等于原G，原源面板/实际variance与20draw日志核对，独立汇总差0；全部进程退出，登记ANALYZED。

全体Fclean一致率97.509766%→真实H97.714844%（净21）；低档城区90.250696%→真实H91.643454%（修复38/损伤18/净20），仅真实noise净2，同时替换净14。真实H仍有可检验收益，但收益不是独立部件贡献之和：替换也改变学习先验的context与迭代输入。来源为训练源分布，Fclean一致率不是真TX准确率、Main泛化或理论恢复成数；20draw相关，不能当作10240独立包。默认DURA与性能主目标ACTIVE保持，OE仍仅冻结诊断。

## 未限幅源特征指导候选已运行（2026-11-01）

`20261101-phase1-Restore-MSAT-FullFGrad-E30-s2026092701-r01`，实际代码`940fa78ba450b686b1b03c7530c47af4360faafc`。相对固定已完成VarF r03，只移除每记录F特征输入梯度相对IQ的0.5限幅，原.1(cos+源方差标准化特征MSE)、IQ+.05STFT、全局参数clip1、MSAT4249922参数、同scratch初值/源R07/order/LR/30轮7410保持。非F-free，原F0源特征冻结，不读TX/logit/CE、不另训头；OE仍仅冻结诊断，V/OE/目标不训练或选模。依据为源端限制活跃及源反事实的剩余特征误差，并不预先证明扩大梯度有效。

3项针对性检查、一次P0/P1与真实源2条F0/G梯度/E0重载PASS；首轮247更新完成，E01模型/优化器/RNG独立CPU读回PASS。GPU4以原2个compute和22583MiB余量准入，仅新增本任务1个；其他owner保持。固定final30后自动原Main两总体168000×七景，所有预测完成后独立truth-last，不以源好坏取消测试；六景3pp+clean、性能主线/效率支线与默认DURA不变。当前正式性能PENDING，完整目标ACTIVE。

PHY同row冻结预测补充（无新增NN/预测/目标IQ/TX读取）：低档城区Fclean判决一致率源MSAT87.395543%→源PHY90.250696%，Main旧MSAT69.013690%→PHY66.810119%，默认DURA70.002381%。源512训练包×20draw和Main168000不同包×固定draw存在总体/采样/信道差异，不能拆分独立域偏移原因，也不能把一致率当TX准确率或理论上限。完整六景、clean和分总体见父PHY报告frozen_Fclean_agreement.json。这项补充统计前FullFGrad候选已固定并运行，未据此热改任何配置。

## 原固定源 V 冻结验证已排队（2026-11-01）

`20261101-diagnostic-Restore-fixed-sourceV-s2026092701-r01`，实际代码27197d5a45c720dabd624425eb30cf586263a171，4针对性检查及一次P0/P1通过。复用R07原V1800×七景，原5源RX×3日期每格120，角色V、唯一ID1800、与训练63000交集0及七景trace已核实；仅冻结验证，0训练/TX/OE/novel/目标/状态或统计拟合。固定四G final30、原F0/defaulthead、默认DURA/raw和PHY真实H读出；Fclean一致率不是TX准确率或理论上限，V历史曝光及训练/验证信道种子差异披露。

CPU控制器已独立核实PID/CWD/命令/CUDA空、WAIT_PARENT_COMPLETE，当前0GPUworker/模型加载/V神经前向，真实源2条smoke尚待执行。等待FullFGrad整个Main ARTIFACTS_COMPLETE后才按GPU4/6 UUID原准入运行，不按科学分数作条件，父健康任务不改。此项不能改变当前训练配方、checkpoint或主验收；完成后只作源验证与机制诊断。默认DURA和完整性能目标ACTIVE保持。

## 未限幅特征指导原 Main 七景完成（2026-11-01）

`20261101-phase1-Restore-MSAT-FullFGrad-E30-s2026092701-r01`，实际代码940fa78ba450b686b1b03c7530c47af4360faafc。MSAT4249922参数，与VarF同scratch初值/原LsUs/R07/sourceorder/优化器/30-7410；只移除逐记录F输入梯度cap，全局参数clip1保留。原地面Tri-SIG-TIME SSL/defaulthead/168000×七景全部固定预测后truth-last，4阶段exit0/liveempty。与既有VarF84单元、2604组/1736raw配对/1302额外配对复算PASS，最大差1.24e-14、raw/DURA判决差0；同E0/源顺序/draw/60日志/30恢复文件E30-final一致。

六景依次78.333333/76.964286/75.497619/77.198810/71.408929/65.849405%，达标0/6；相对VarF依次+0.840476/+0.995238/+1.394048/+0.725000/+0.611310/+0.597619个百分点。合并clean 86.484524%（未保持），各子总体损伤详见完整报告，不以合并均值掩盖。完整目标未达到，保持ACTIVE和默认DURA。本轮来源于源端特征误差与限幅证据，不据目标误分类调参/选模或取消测试；OE仅冻结诊断，V/OE/目标不训练，默认DURA保持。完整TX/RX/day/配对/成本见analysis.md。

## 原固定源 V 冻结验证完成（2026-11-01）

`20261101-diagnostic-Restore-fixed-sourceV-s2026092701-r01`，实际代码27197d5a45c720dabd624425eb30cf586263a171。全部父模型Main完成后，原V1800×七景/七读出完成，原Ls2条真实smoke与4项测试/一次P0P1通过，独立49单元/88200F0行/索引/RXday/日志及全部保存判决一致率复算PASS，汇总差0；0更新/TX/OE/novel/目标，所有状态保持、进程退出，登记ANALYZED。

低档城区Fclean判决一致率：received 60.611111%；DURA_v1 86.111111%；MSAT_M20 88.055556%；MSAT_VarF 86.388889%；PHY_MSat 91.166667%；MSAT_FullFGrad 89.555556%；PHY_true_H 91.833333%。这是原源V验证，不是TX准确率或Main。V历史曝光、原F0/DURA验证知识、1800包七次相关复用，以及训练/V物理包与receiverseed1911/1902差异均披露，不能单因称过拟合或理论恢复上限。默认DURA与六景逐个3pp+clean主目标不因源验证收益自动改变；完整方法/分场景/源面板对比见analysis.md。

## 源端成对变换稳定性完成（2026-11-01）

`20261101-diagnostic-Restore-source-style-equivariance-s2026092701-r01`，实际代码ac8015d9dcdcd9c5d1f05d7c09882a0591825d7d。原Ls128Us384固定512包×20draw×五T，冻结DURA/PHY/FullFGrad与原F0，比较G(Ty)与T G(y)，均对T clean评分。4项检查/一次P0P1/实际Ls2smoke通过；51200行/全部源面板/20日志/保存判决配对独立复算差0，identity差异0、phase/gain参考IQ不变性误差≤2.39e-7；0更新/TX/V/OE/目标，全状态保持、进程结束，ANALYZED。

低档城区对应1436源实例中，+90°使F0(clean)判决本身仅78.690808%保持；gain.7/1.4则将原clean单位功率变为.49/1.96。源等变差异不能直接解释Main下降或认定恢复器缺陷，结果不自动支持相位群平均/硬相位或增益等变结构。反射FIR也非native新信道，语义一致率与波形差异不单调。下一步改造需尊重原clean约定和冻结F0特征语义，仍仅原LsUs/单阶段30，所有新G完整Main七景；默认DURA和六景3pp+clean目标ACTIVE不变。

## 连续源特征条件恢复器已运行（2026-11-01）

`20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01`，实际代码`c8659b15bb3d316325867b154eede51f1a61e247`，4670178参数、260546560 Conv/Linear/attention MAC（不含FiLM逐点/归一化/F0等，非时延）、0FFT/CG。共享encoder预测连续160维F0源clean特征，四FiLM用预测latent调decoder，输出原clean单位复功率；teacher仅loss、部署只IQ。全模型scratch单阶段30/7410，原LsUs/R07/order/LR/seed保持，输出F指导加连续latent回归，无TX/logit/CE/六类原型/新分类头/旧G或IQ教师祖先；OE仅冻结诊断，V/目标不训练选模。结构/辅助损失/功率约束共同变化，非单因素。

7项检查/一次P0P1及实际Ls2F0/G梯度/E0重载PASS；首轮247更新完成，E01模型/优化器/RNG独立CPU读回PASS。GPU4准入free23484MiB/原1compute/新增1个，自身PID-CWD-CUDA-GPU与日志核实，未改其他owner。固定final30后自动原Main168000×七景、全部预测后独立truth-last，不因源质量取消。当前性能PENDING，六景3pp+clean主目标ACTIVE、效率次级、默认DURA不变。

## 连续特征条件恢复器原 Main 七景完成（2026-11-01）

`20261101-phase1-Restore-FeatureCond-E30-s2026092701-r01`，实际代码c8659b15bb3d316325867b154eede51f1a61e247。新G4670178参数，全部scratch单阶段30/7410；连续源F0特征预测+四FiLM+辅助回归+单位复功率输出，是系统比较。teacher仅loss、部署只IQ、原F0/defaulthead冻结；无旧G/神经IQ教师祖先、TX/logits/CE/新分类头。原LsUs/R07/源顺序/20+10draw/优化器/预算匹配已完成FullFGrad对照。V/OE/novel/目标不训练、选模或早停，OE仅冻结诊断。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景全部固定预测后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736 raw配对/1302额外配对复算PASS，最大误差1.24e-14、raw/DURA判决差0；60日志、同sourceorder/draw、新E0零输出/FiLM及30恢复文件E30-final一致性通过。

六景依次77.906548/76.511905/74.880357/76.644048/70.342857/64.372619%，达标0/6，相对FullFGrad依次-0.426786/-0.452381/-0.617262/-0.554762/-1.066071/-1.476786个百分点。合并clean 86.619048%（保持），分总体损伤和完整TX/RX/day/配对/成本见analysis.md。完整目标未达到，保持ACTIVE和默认DURA。性能为主线，训练效率为支线；源指标或训练完成不能替代逐景验收，不依据目标结果调参或重排候选，不把准确率保留比例解释为信息恢复上限。

## 冻结连续特征条件路径源诊断完成（2026-11-01）

`20261101-diagnostic-Restore-FeatureCond-source-paths-s2026092701-r01`，实际代码1b8b3625a3d7eb043129c034ad8639504b8576fa。原FeatureCond/c865/30-7410和原F0/defaulthead冻结；原R07 Ls128/Us384固定512包×20draw=10240相关案例。正常/zero-feature/无FiLM/sourceclean特征oracle/raw投影前五IQ路径及latent全部完成，复用旧source-style/ac801 identity DURA/Full源控制，不新增Main/V/OE/target推理或训练。3测试/一次P0P1/真实Ls2actual-latent-raw精确重放PASS，worker/controller结束0，模型state不变，独立预测/配对/20draw/源控制/均值复算PASS（差0）。

低档城区Fclean一致率正常88.718663%、zero特征29.874652%、无FiLM28.899721%、预测latent86.559889%、sourceclean特征oracle97.493036%（修复129/损伤3/净126）；IQ NMSE正常.071576/oracle.066893。条件确已使用；cleanoracle额外信息和off-support输入不代表可达上限或TX准确率。raw平均复功率75.827557，低档城区88.494702，实际输出单位功率；投影前尺度未被独立训练，不直接切换部署。下一项源端假设只加强连续latent回归权重.1→1，结构/源顺序/优化器/预算保持，另建scratch单阶段30对照并自动原Main七景。当前Main0/6负结果、默认DURA和完整性能目标ACTIVE保持；效率次级。

## 连续源特征回归权重1对照已运行（2026-11-01）

`20261101-phase1-Restore-FeatureStrong-E30-s2026092701-r01`，实际代码`3e5416df5b13dfcf9e16995988e988c5494f68b2`，结构/4670178参数/260546560实数MAC和原FeatureCond相同，唯一科学变化为连续latent回归权重.1→1；IQ/输出F指导/全局clip1/单位功率/源顺序/30-7410全部保持。实际E0与原FeatureCond逐张量一致，G从零单阶段无旧G祖先，源F0特征loss-only，部署只IQ，无TX/logits/CE/新身份头/V-OE-target训练或选模，默认DURA保持。

3针对性检查/一次P0P1及真实Ls2F0G梯度/E0严格重载PASS；首轮247更新完成，E01模型/优化器/RNG及同源初值独立CPU读回PASS。GPU4准入free23482MiB/原1compute/只新增1个，PID-CWD-CUDA-GPU与日志增长核实，其他owner不改。固定final30后自动原Main168000×七景/全部预测后独立truth-last；当前正式性能PENDING，六景3pp+clean目标ACTIVE，效率次级。

## 连续latent回归权重1原 Main 七景完成（2026-11-01）

`20261101-phase1-Restore-FeatureStrong-E30-s2026092701-r01`，实际代码3e5416df5b13dfcf9e16995988e988c5494f68b2。4670178参数/结构/源顺序/预算与FeatureCond相同，唯一科学变化为latent回归权重.1→1。实际E0逐张量一致，全模型scratch单阶段30/7410；IQ/输出F指导/单位功率/globalclip1保持。原F0/defaulthead冻结，teacher loss-only，0TX/V/OE/目标训练或选模，OE仅冻结诊断。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景全部固定预测后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736 raw配对/1302额外配对复算PASS，最大指标差1.24e-14、raw/DURA判决差0。完整60日志/同E0源顺序draw/仅latent权重配置变化/逐轮总loss权重/30恢复文件E30-final一致性核实。

六景依次75.979167/75.095833/74.039881/74.950595/69.618452/63.999405%，达标0/6，相对FeatureCond依次-1.927381/-1.416071/-0.840476/-1.693452/-0.724405/-0.373214个百分点。合并clean 86.822619%（保持），cross_rx clean净变化+515、cross_both净变化-38，不以合并掩盖损伤；完整TX/RX/day/配对/成本见analysis.md。完整目标未达到，保持ACTIVE和默认DURA。本项源端假设来自冻结source512×20路径诊断，不据Main误分类调参或选模；性能主线、效率支线和默认DURA保持。

## 冻结源端梯度几何完成（2026-11-01）

`20261101-diagnostic-Restore-source-gradient-geometry-s2026092701-r01`，实际代码96db4e5a0446c713cd2d36c2977703590c26d523。FeatureCond/c865与FeatureStrong/3e541固定final30/7410，原F0/3ff8冻结；原Ls128/Us384固定512×20draw/B256×2G=80批，仅autograd.grad，0TX/V/OE/target/更新或.grad积累，G最终导数标志恢复False/state逐值保持。3测试/一次P0P1/真实Ls2导数smokePASS，全部20draw/Gram分组/globalclip/点积cos/summary独立复算PASS，最大差2.78e-17，worker/controller结束0。

末轮encoder主/aux平均cos .547769/.599657，负cos批数0/40与1/40；所有80批在w.1/1均无globalclip，main·total/physical·total负值0。该末轮源几何不支持据此启动PCGrad/分组裁剪，不代表早期无冲突，也不预测AdamW真实更新。sourceworker耗时27.118665秒，在两次读取间完成；中途GPU快照未捕获，如实记缺项、不补跑，完整产物/控制器初始存活/准入UUID和退出状态已独立核实。下一源端假设针对既有raw功率75.83的未约束尺度方向，单独检验单位clean幅度约束；不是已验证改进。Main0/6和子总体clean损伤事实保持，完整目标ACTIVE、效率次级、默认DURA不改。

## 幅度约束对照已运行（2026-11-01）

`20261101-phase1-Restore-PowerAnchor-E30-s2026092701-r01`，实际代码`42ea957db57d7d90df67dadbeffe11fc014909a0`。VERIFIED / RUNNING：真实源样本检查与新增幅度项通过，同 FeatureCond 实际 E0 逐张量一致，配置仅幅度项和版本身份变化；E01 模型/AdamW/CPU-CUDA RNG 独立 CPU 读回通过，总损失复算确认幅度项仅加权一次。首轮 247 更新耗时 49.944650 秒。GPU 4 准入剩余 24110 MiB、原有 0 条计算进程，仅新增本任务 1 条；PID、目录、命令、GPU 与日志增长核实，其他 owner 保持。4670178 参数及部署前向不变，固定 scratch 单阶段 30/7410 后自动原 Main 七景，当前性能 PENDING；六景各距 raw clean≤3pp、clean 保持仍是主验收，效率为支线，默认 DURA 不变。

## 源幅度约束原 Main 七景完成（2026-11-01）

`20261101-phase1-Restore-PowerAnchor-E30-s2026092701-r01`，实际代码42ea957db57d7d90df67dadbeffe11fc014909a0。与固定 FeatureCond/c865 唯一变化为 .1×(rawRMS/同条cleanRMS−1)²，投影前尺度受监督；结构、输出/latent指导、单位功率/globalclip1、实际E0/源顺序/draw/LR/预算一致。4670178 参数全部从零单阶段30/7410，0旧G/IQ教师祖先，原源F0 loss-only，不读TX/V/OE/目标训练或选模，OE仅冻结诊断。

原地面完整 Tri-SIG-TIME SSL/defaulthead/Main168000×七景全部固定预测后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对复算PASS，误差1.25e-14、raw/DURA判决差0；完整60日志、同E0源顺序draw、仅新增幅度项/各轮加权loss/30恢复文件E30-final一致性核实。

六景依次78.990476/77.714881/76.166667/77.799405/71.533929/65.433929%，达标0/6，相对FeatureCond依次+1.083929/+1.202976/+1.286310/+1.155357/+1.191071/+1.061310个百分点。合并clean 86.644643%（保持），cross_rx净变化+176、cross_both净变化+2，不以合并掩盖子总体损伤。完整TX/RX/day/净纠错/训练幅度与成本见analysis.md。完整目标未达到，保持 ACTIVE 和默认 DURA。性能主线、效率支线保持，源尺度约束改善不替代真实TX识别，也不是信息恢复极限。

## 四步直接迭代恢复器已运行（2026-11-01）

`20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01`，实际代码`84a063c9ba1f9974cfe361e293a9e11fa93920b4`，4677538参数，从零单阶段30/7410、训练每记录1次estimate、部署4次共享权重，始终条件同一received IQ，无额外观测或部署clean/F0。源IQ配对构造t1/.75/.5/.25状态，原F0连续160与rawRMS参考loss-only；0TX/V/OE/目标训练或选模，OE仅冻结诊断。结构/训练状态/执行共同变化的系统比较，不冒称单因素。

VERIFIED / RUNNING：17重点相关检查/一次P0P1、真实源2训练梯度与完整四步E0/严格重载PASS。首轮247更新耗时59.003244秒，E01模型/AdamW/CPU-CUDA RNG独立CPU读回PASS，E0与固定seed factory逐张量一致；4个time实际计数和各项指标重组PASS，总loss确认原权重各加一次。GPU4准入free21865MiB/原2compute/仅新增1条，PID/CWD/命令/CUDA/GPU和日志核实，其他owner保持。固定final30后自动原Main168000×七景、全部预测后独立truth-last，当前性能PENDING；主验收仍六景各距raw clean≤3pp并保持clean，性能主线、效率支线和默认DURA不变。

## 仅状态时间四步恢复器已运行（2026-11-01）

`20261101-phase1-Restore-StateBridge-E30-s2026092701-r01`，实际代码`c50a9159b1fb891c6f1a6f8ad4bfa2d7eb66222e`，4676898参数，从零单阶段30/7410、训练每记录1次estimate、部署4次共享权重，received仅初始化state，估计器不读取原received条件，无额外观测或部署clean/F0。源IQ配对构造t1/.75/.5/.25状态，原F0连续160与rawRMS参考loss-only；0TX/V/OE/目标训练或选模，OE仅冻结诊断。结构/训练状态/执行共同变化的系统比较，不冒称单因素。

VERIFIED / RUNNING：12重点相关检查/一次P0P1、真实源2训练梯度与完整四步E0/严格重载PASS。首轮247更新耗时78.104723秒，E01模型/AdamW/CPU-CUDA RNG独立CPU读回PASS，E0与固定seed factory逐张量一致；4个time实际计数和各项指标重组PASS，总loss确认原权重各加一次。GPU6准入free20513MiB/原2compute/仅新增1条，PID/CWD/命令/CUDA/GPU和日志核实，其他owner保持。固定final30后自动原Main168000×七景、全部预测后独立truth-last，当前性能PENDING；主验收仍六景各距raw clean≤3pp并保持clean，性能主线、效率支线和默认DURA不变。

## DirectBridge原 Main 七景完成（2026-11-01）

`20261101-phase1-Restore-DirectBridge-E30-s2026092701-r01`，实际代码`84a063c9ba1f9974cfe361e293a9e11fa93920b4`。paired-state/原received-time条件/四步执行共同变化的系统比较，全部G从零单阶段30/7410，训练每记录1core、部署4core共享权重；sourceclean仅构造授权源中间训练state/参考，部署无clean/F0/额外观测，0TX/V/OE/目标训练或选模，OE仅冻结诊断。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景固定预测后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对复算PASS，最大指标差1.25e-14、raw/DURA判决差0；60完整日志/源顺序draw/从零E0/固定4步与time协议/各轮与各t总loss重组/30快照E30-final一致性核实。

六景依次77.663690/76.391071/74.903571/76.459524/70.286905/64.206548%，达标0/6，相对PowerAnchor依次-1.326786/-1.323810/-1.263095/-1.339881/-1.247024/-1.227381个百分点。合并clean 86.603571%（保持），cross_rx净变化+163、cross_both净变化-54，不掩盖子总体损伤。完整TX/RX/day/净纠错/相对clean/各time训练与成本见analysis.md。完整目标未达到，保持 ACTIVE 和默认 DURA。配对代数捷径是源端解析限制，不以其预判实际模型或取消测试；性能主线、效率支线保持。

## StateBridge原 Main 七景完成（2026-11-01）

`20261101-phase1-Restore-StateBridge-E30-s2026092701-r01`，实际代码`c50a9159b1fb891c6f1a6f8ad4bfa2d7eb66222e`。仅去除原received两stem通道，其他实际随机初值/时间抽样/损失/预算匹配，全部G从零单阶段30/7410，训练每记录1core、部署4core共享权重；sourceclean仅构造授权源中间训练state/参考，部署无clean/F0/额外观测，0TX/V/OE/目标训练或选模，OE仅冻结诊断。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景固定预测后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对复算PASS，最大指标差1.24e-14、raw/DURA判决差0；60完整日志/源顺序draw/从零E0/固定4步与time协议/各轮与各t总loss重组/30快照E30-final一致性核实。

六景依次71.752976/70.942262/69.885714/70.691667/65.321429/59.635119%，达标0/6，相对DirectBridge依次-5.910714/-5.448810/-5.017857/-5.767857/-4.965476/-4.571429个百分点。合并clean 86.107143%（未保持），cross_rx净变化-551、cross_both净变化-174，不掩盖子总体损伤。完整TX/RX/day/净纠错/相对clean/各time训练与成本见analysis.md。完整目标未达到，保持 ACTIVE 和默认 DURA。配对代数捷径是源端解析限制，不以其预判实际模型或取消测试；性能主线、效率支线保持。

## 冻结四步源轨迹完整分析（2026-11-01）

`20261101-diagnostic-Restore-bridge-trajectory-source-s2026092701-r01`，代码`128cc49df04f346f29c93d8173192cc56bd3bd77`。两版固定final30原source512×20，各4步共81920轨迹行；源2轨迹末步逐值等于实际G.forward、F0/defaulthead检查PASS，G/F0终态不变，0训练/新头/TX/V/OE/目标IQ-truth或选模。只等父Main技术完成、不读其分数后运行，不阻塞/热改两版。

DirectBridge低档城区Fclean一致率84.749304%→84.749304%，相对首步净变化+0；StateBridge低档城区Fclean一致率86.350975%→81.615599%，相对首步净变化-68。这些是源训练面板F(clean)一致，不是TX准确率、目标泛化或理论上限；原Main六景3pp+clean仍主线，默认DURA保持。独立全81920行/96组/配对复算PASS，误差0，旧源Fclean参考差异{'DirectBridge': 0, 'StateBridge': 0}单列非门槛；实时workerGPU快照捕获=True，完整产物与进程结束核实。图表/完整组/证据边界见analysis.md。

## 九个新恢复器的原 Main 同口径性能核对

只复用九版完整公开结果及固定raw/DURA，同一原地面Tri-SIG-TIME SSL/defaulthead、Main168000×七景，按研发顺序汇总，不排序选模或拼接最佳场景。正确数、六景3pp差距和两个clean总体分列；新神经前向/训练或目标原始数据/真值读取均0。准确率保留比例不作波形恢复率或理论上限，原默认DURA和完整目标ACTIVE保持。见[性能核对](../PAMF_RESTORATION_MAIN_PERFORMANCE_AUDIT_20261101.md)及同名JSON/CSV。

## 实际四步展开训练已运行（2026-11-01）

`20261101-phase1-Restore-RolloutBridge-E30-s2026092701-r01`，实际代码`7250d0a88160d6ab949dafbdc73edb7f302e290f`，4676898参数、与StateBridge/c50实际E0逐张量一致、全G从零单阶段30/7410。训练和部署均为当前received起点的四步state-only共享网络，clean只作最终loss参考，最终损失贯穿全部四步、不detach；0clean混合训练输入/时间抽样/旧G祖先，源F0连续特征loss-only，0TX/V/OE/目标更新或选模，OE仅冻结诊断。

VERIFIED/RUNNING：9重点检查/一次P0P1、真实Ls2四步训练部署精确一致/四步梯度/E0重载PASS。E01/247耗时203.122173秒，模型/优化器/CPU-CUDA RNG独立CPU读回及同State实际E0/原加权loss复算PASS。GPU4准入free20452MiB/原2compute，只新增1条；PID/CWD/CUDA/GPU和日志核实，其他owner保持。训练core数4倍、epoch/更新仍30/7410，不称同训练计算量。固定final30后自动原地面Tri-SIG-TIME SSL/defaulthead/Main168000×七景，全部预测后独立truth-last；性能PENDING，六景逐个3pp+两个总体clean主目标ACTIVE，效率支线，默认DURA不变。


## 射频细节保留的历史源证据补充

只整理原R31/c333e666历史R17源V1800公开聚合，不增加G/F0/IQ/真值读取。10个低档城区单元与原CSV逐值相同，1%微扰时四物理方向仅134至183/1800满足旧可观察条件，可观察子集PA幅度/相位/IQ镜像响应增益中位数约.74至.76，随机方向也衰减。方向是合成探针而非真实TX导数，条件包含已知A/有限MAP，不是Main恢复上限或当前新G保真证明；不能对全部弱方向强制放大。详见[射频细节分析](../PAMF_RESTORATION_RF_DETAIL_ANALYSIS_20261101.md)。当前RolloutBridge训练、原LsUs权限、Main七景与六景3pp+clean验收保持，不将旧F50/R17分析改标当前性能。完整目标ACTIVE。

## 可学习功率对照已排队（2026-11-01）

`20261101-phase1-Restore-SoftPower-E30-s2026092701-r01`，代码`1dd40c07970b664f18c57149a0a2ab02f000b114`。解析动机是单位范数clean的条件均值不必仍单位范数，合成双点反例中硬投影增加IQ风险；不据此预判真实F0性能。相对固定PowerAnchor/42ea只取消末级硬单位功率投影，保留同4670178参数/随机初值/原LsUs-R07/损失权重与.1软RMS/order/LR/scratch单阶段30-7410，vector_norm使同一RMS数学式在零点选取有限的次梯度。部署1core/rawIQ，无旧G祖先，源F0/3ff8冻结features-only，0TX/V/OE/目标更新或选模，OE只冻结诊断。

8针对性检查/一次P0P1通过；VERIFIED CPU等待进程PID-CWD-命令-CUDA空与日志增长，当前0GPU/模型加载/训练。源2及实际E0尚待首GPU步骤，不能把排队当训练完成。只等待当前Rollout整个Main技术ARTIFACTS_COMPLETE/terminal0，不读分数或预测，父任务与其他owner保持；此后按原4/6资源准入自动30及Main两总体七景/全部预测后truth-last。六景各3pp+两个clean总体仍主验收，性能主线/效率支线、默认DURA和完整目标ACTIVE保持。

## 固定30轮源训练完成，正式预测已接续（2026-11-02）

VERIFIED。30/7410完整源训练、189000L/1701000U、7560000 Gcore行、训练进程exit0并退出、final与E30文件存在已核实。全部训练状态/真实source2与实际E0的完整核对仍按既有证据及最终复算，当前不把源末轮指标当性能验收。实际Main预测进程已存活，跨接收机clean126000完成日志已观察；这不是准确率结果。全部两总体七景prediction完成后才独立CPU评分。SoftPower继续只在CPU等整个Main技术完成，原训练计划/默认DURA/源权限和完整性能目标ACTIVE保持。

## 可学习功率源训练已核实

VERIFIED / RUNNING。源2真实G/F0梯度/raw恒等E0与严格重载PASS，首轮247更新耗时45.930482秒。E01模型/优化器/CPU-CUDA RNG独立CPU读回，同PowerAnchor实际E0全部tensor/sourceorders/draws一致，配置仅投影/版本定义差异及加权loss复算PASS。GPU4准入free24110MiB/原0compute/新增1个，PID-CWD-CUDA-GPU与日志核实。原Rollout完整Main技术完成后才启动，0TX/V/OE/target更新或选模；固定final30自动原Main七景后truth-last，性能PENDING，默认DURA及完整目标ACTIVE保持。

## RolloutBridge原 Main 七景完成（2026-11-02）

`20261101-phase1-Restore-RolloutBridge-E30-s2026092701-r01`，实际代码`7250d0a88160d6ab949dafbdc73edb7f302e290f`。改为实际四步可微训练与最终损失，结构/E0/源顺序draw/LR/30轮/部署函数保持，训练core计算4倍的系统比较，全部G从零单阶段30/7410，训练和部署每记录均4core共享权重；sourceclean仅作最终loss参考，不构造clean混合输入state，部署无clean/F0/额外观测，0TX/V/OE/目标训练或选模，OE仅冻结诊断。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景固定预测后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对复算PASS，最大指标差1.2e-14、raw/DURA判决差0；60完整日志/源顺序draw/从零E0/实际4步训练与最终loss协议/各轮加权loss复算/30快照E30-final一致性核实。

六景依次75.960119/73.929762/71.875595/74.461905/67.470238/61.549405%，达标0/6，相对StateBridge依次+4.207143/+2.987500/+1.989881/+3.770238/+2.148810/+1.914286个百分点。合并clean 86.575000%（保持），cross_rx净变化+41、cross_both净变化+20，不掩盖子总体损伤。完整TX/RX/day/净纠错/相对clean/各time训练与成本见analysis.md。完整目标未达到，保持 ACTIVE 和默认 DURA。源端逐步退化是本项依据，不据Main错例热改模型或取消测试；性能主线、效率支线保持。

## 时频残差扩散源训练已核实

VERIFIED / RUNNING。源Ls2真实G/F0无更新smoke通过，固定15调用sampler有限、实际output-head梯度与严格E0重载PASS；不是恒等E0。E01模型/优化器/CPU-CUDA RNG独立CPU读回，全部E0 tensor等于scratch factory，源顺序draw同原FullFGrad；E01时/频输入与attention骨干及双输出head确已更新，固定latent/schedule/坐标不变，加权loss和63000条sigma histogram复算PASS。首轮247步19.601430秒。GPU4准入free24110MiB/原0compute/新增1个，PID/CWD/CUDA/GPU/日志核实。SoftPower完整Main技术完成后才启动，0TX/V/OE/target更新或选模；固定final30自动原Main七景后truth-last，性能PENDING，默认DURA不变，目标ACTIVE。

## 可学习功率原 Main 七景完成（2026-11-02）

`20261101-phase1-Restore-SoftPower-E30-s2026092701-r01`，实际代码1dd40c07970b664f18c57149a0a2ab02f000b114. 与固定PowerAnchor/42ea只改变硬单位功率投影，保留同结构/实际随机E0/原LsUs-R07/order/draw/LR/损失权重及.1软RMS；vector_norm对同一RMS数学式在零点选取有限的次梯度。4670178参数全从零单阶段30-7410，无旧G/IQ教师祖先，原F0/3ff8/defaulthead冻结features-only，0TX/V/OE/目标更新或选模，OE仅冻结诊断。

原Main168000×七景全部预测固定后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对及868相对raw clean配对复算PASS，误差1.25e-14、raw/DURA判决差0；60完整日志、同E0实际工厂/旧控制、sourceorder/draw/LR、仅投影配置差异/加权loss/30恢复文件E30-final一致性核实。

六景依次78.913690/77.574405/76.104167/77.705357/71.575595/65.526786%，达标0/6，相对PowerAnchor依次-0.076786/-0.140476/-0.062500/-0.094048/+0.041667/+0.092857个百分点。合并clean 86.535119%（未保持），cross_rx净变化+22、cross_both净变化-28，不以合并掩盖子总体损伤。完整TX/RX/day/净纠错/相对clean/功率与成本见analysis.md。完整目标未达到，保持 ACTIVE 和默认 DURA。解析MMSE例子不是F0收益或理论上限；性能主线、效率支线、默认DURA保持。

## 冻结残差扩散源采样轨迹完成（2026-11-02）

`20261102-diagnostic-Restore-diffusion-source-path-s2026092701-r01`，代码`1a7434baa2c579e78fc1119cb09c24a0ae6512b4`。原source固定128Ls+384Us、20draw、received+8内部clean读出，共92160行/108组；源2记录末步逐值等于真实部署G、F0/defaulthead有限检查PASS，G/F0终态不变，0更新/TX/V/OE/novel/目标IQ-预测-评分或选择。原fixedfinal30/5e5921c9，原Main完整8步/15调用保持。

整体Fclean一致率第4步90.869141%→末步87.294922%，低档城区69.428969%→62.465181%，对应同条一致判决净减少366/10240和100/1436；低档城区IQ NMSE .14207185→.25313855。第1步到末步的总变化另列，不能笼统声称每步改善或全面退化。训练单步noised-clean误差与实际sampler测量不同，不作同条件质量对比。源内轨迹不等于TX准确率、跨RX泛化或理论上限，不能借此热改正在测试的完整采样版本。

全92160行/108组/首步与received配对独立复算PASS，最大误差0，原source面板顺序/draw/scene-lock/既有Fclean参考差0；worker23.809560秒、GPU6 live快照已捕获，exit0/liveempty。图、完整分组和边界见analysis.md。后续可独立检验部署尺度训练覆盖与缩短采样，需新预登记及完整Main，不据目标成绩选step。原Main六景3pp+两个clean保持为主线，默认DURA与目标ACTIVE保持。

## 时频残差扩散原 Main 七景完成（2026-11-02）

`20261102-phase1-Restore-TFResidualDiffusion-E30-s2026092701-r01`，实际代码`5e5921c975678367a5d0a00ee72d4619e5f6927e`。全新联合时频Transformer/条件残差扩散，从零单阶段30/7410，1614608参数，训练1denoiser/记录、部署固定8步Heun共15调用，无DURA/旧G祖先。原Ls6300Us56700/R07/源顺序draw/LR保持，F0仅连续特征指导，0TX/V/OE/target更新或选模，OE仅冻结诊断。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景全部预测后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对/868相对clean配对复算PASS，最大差1.2e-14、raw/DURA判决差0，60完整日志及源顺序/LR/加权loss/噪声统计/从零E0/30快照与E30-final核实。

六景依次73.621429/71.008333/68.463095/70.593452/58.003571/48.688095%，达标0/6，相对DURA依次-10.722024/-12.007738/-12.858929/-12.455952/-17.702381/-20.113095个百分点；clean85.913690%，cross_rx净变化-821、cross_both净变化-229。完整分组与双向纠错见analysis.md。新训练曲线是加噪residual单步输出，含clean信息，与完整部署sampler不同，不能把低训练误差视作恢复性能。系统比较不冒充同E0/同计算量单因素，静态MAC与FFT不冒充实测时延。完整目标未达到，保持 ACTIVE 和默认 DURA。

## 源采样损伤与尺度覆盖的独立对照已运行（2026-11-02）

代码`3e5ecb4ffcc5658693bd9b6dc509711b4509a231`，根据已完成原source92160行/108组轨迹固定两个独立研究点，不依据Main错例/分组选择或组合方案。`20261102-evaluation-Restore-DiffusionEarly4-main-s2026092701-r01`：复用原5e5921c9/source-only scratch30-7410权重，0新训练，固定第4denoised读出/7core，原Main两总体七景自动评测，四路raw-DURA-full8-early4全预测后truth-last。`20261102-phase1-Restore-DiffusionSigmaMix-E30-s2026092701-r01`：全G同架构/实际E0从零单阶段30/7410，训练50%原lognormal+50%均匀8部署sigma，15core部署保持，固定GPU6。两项5聚焦检查/每候选唯一P0P1 PASS，实际source2/G-F0/early精确读出/CUDA mixture-RNG/E0重载和PID-CWD-CUDA-GPU/日志增长核实，GPU4/6各新增1/原0compute/free24110MiB，总ownNN2，其他owner保持。

SigmaMix E01模型/优化器/RNG独立CPU读回：全部E0 tensor同原full8实际E0与scratch factory、全30sourceorders/draws相同、配置仅sigma抽样/版本差异、E01全局CPU/CUDA RNG同原模型，9mixbins/5sigmabins及加权loss复算PASS。私有Generator seed2026110205+update不推进原Gaussian sigma/epsilon全局流，无额外技术恢复状态。两run当前均RUNNING、正式Main性能PENDING；不以更少调用或更低训练loss宣布达标。默认DURA与六景3pp+两个clean保持目标ACTIVE。

## 冻结扩散第4读出原Main完成（2026-11-02）

`20261102-evaluation-Restore-DiffusionEarly4-main-s2026092701-r01`，执行代码`3e5ecb4ffcc5658693bd9b6dc509711b4509a231`，G权重原`5e5921c975678367a5d0a00ee72d4619e5f6927e`/source-only scratch30-7410，0新训练。根据已完成合法source轨迹全局固定第4denoised读出，前三次Heun+第四次denoiser共7core，原网络/F0/数据不变，不按query或目标场景选择，非新从零网络。

原Main168000×七景/四路56单元全预测后独立CPU truth-last，3阶段exit0/liveempty，56单元/1736组/1302raw配对/868额外配对/868相对clean配对独立复算PASS，最大差1.24e-14、raw/DURA/full8复用判决差0。六景依次79.198214/77.956548/76.421429/77.031548/67.425595/58.480952%，0/6；相对同权重full8提高5.576786至9.792857个百分点，但六景仍低于DURA。clean86.876786%/合并净+568，cross_rx+635、cross_both−67，两个clean未同时保持。完整TX/RX/day/净纠错见analysis.md。

这是源端后段采样损伤与真实Main任务表现之间的受控推理对照证据，不证明所有扩散均应缩短、真实RF细节完整恢复或理论上限。1614608参数/1238802432静态实数MAC/21complexFFT，不冒充实测时延；祖先G30、F0 E50/源尺度成本保留。独立SigmaMix从零30及完整15调用Main自动继续，不按本项结果取消或组合。完整六景3pp+两个clean目标ACTIVE，默认DURA保持。

## 噪声尺度均衡扩散原 Main 七景完成（2026-11-02）

`20261102-phase1-Restore-DiffusionSigmaMix-E30-s2026092701-r01`，实际代码`3e5ecb4ffcc5658693bd9b6dc509711b4509a231`。同一时频Transformer/条件残差扩散架构，训练仅改为50%原lognormal+50%均匀8部署sigma，从零单阶段30/7410，1614608参数，训练1denoiser/记录、部署固定8步Heun共15调用，无DURA/旧G祖先。原Ls6300Us56700/R07/源顺序draw/LR保持，F0仅连续特征指导，0TX/V/OE/target更新或选模，OE仅冻结诊断。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景全部预测后独立CPU truth-last，4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对/868相对clean配对复算PASS，最大差1.24e-14、raw/DURA判决差0，60完整日志及源顺序/LR/加权loss/5sigma-9mixture统计/同原full8实际E0/E01-E30全局CPU-CUDA Gaussian RNG/30快照与E30-final核实。

六景依次73.260119/71.301190/69.273214/71.669048/63.918452/56.502976%，达标0/6，相对DURA依次-11.083333/-11.714881/-12.048810/-11.380357/-11.787500/-12.298214个百分点；clean86.718452%，cross_rx净变化+293、cross_both净变化+9。完整分组与双向纠错见analysis.md。新训练曲线是加噪residual单步输出，含clean信息，与完整部署sampler不同，不能把低训练误差视作恢复性能。sigma抽样改变期望训练风险，不做importance correction；损失系数/架构/E0/源顺序draw/LR/完整15调用部署保持，静态MAC与FFT不冒充实测时延。完整目标未达到，保持 ACTIVE 和默认 DURA。

## 源端完整 IQ 协方差与真实信道后验诊断完成（2026-11-02）

`20261102-diagnostic-Restore-source-covariance-inverse-s2026092701-r01`，实际代码`a3a90b55a73702954c453eec9c1ad17fec24d87f`。原63000LsUs clean拟合统一512维实IQ均值/C（固定.001同trace收缩），原128Ls384Us×20=10240相关病例比较同mu/tauI与完整C，均用真实模拟Action/噪声，FP64 Cholesky，无G训练或新增G前向。原地面完整Tri-SIG-TIME SSL/defaulthead冻结；0TX/V/OE/novel/目标/optimizer，源统计是实际数据拟合，不称零先验学习。

完整C整体Fclean一致96.601563%、低档城区87.743733%，相对DURA净+149/10240、+55/1436；相对各向同性净+871/10240、+341/1436。预固定三个支持条件全满足，支持另立学习未知信道/相关性先验的部署网络，但当前真实Action是oracle、先验拟合与诊断源包重叠，不是TX准确率、Main泛化、未知设备证明或理论上界。不会在Main使用真实A/噪声。

24组/所有病例配对独立复算差0，FP64协方差PSD/对称/谱/tau核实；同各向同性较旧FP32 FFT CG128的Fclean事件差0，完整C最大相对求解残差8.80e-14。实际source2/PID-CWD-CUDA-GPU6/loggrowth捕获，worker48.113852秒、exit0/liveempty，私有均值/C/逐病例不提交Git。源证据和完整分组见analysis.md。原Main六景3pp+两个clean保持仍未达到，目标ACTIVE、默认DURA保持。

## 新相关性先验物理恢复器已固定（2026-11-02）

`20261102-phase1-Restore-CovarianceMAP-E30-s2026092701-r01`，实际实现`bf29554a03e28a012eaaa134f5c7b3f7856f2268`。依已完成源fullC支持证据定义新结构：1230341参数时频Transformer只估计FIR64/主镜像调制/噪声，保存原LsUs统一mu/full512C，以FP64 Gaussian后验输出IQ，不含学习IQ解码器、DURA或旧G祖先。原源均值/C为数据拟合的额外知识与buffer，不冒充零先验知识。

从零单阶段30/7410，原Ls6300Us56700/R07/sourceorder-draw-LR，逻辑256/micro32。更新仅算完整Action矩阵相对误差(.1)+噪声logMSE(.001)，真实源模拟参数仅loss；G.estimate实际不执行后验或F0，0F0训练前向，不冒充F指导实验。部署只读单条received/state，不读真实Action或噪声/F0/clean。0TX/V/OE/novel/目标训练或选模，OE仅冻结诊断。5针对性检查、唯一独立P0/P1通过，源2实际smoke后立即完整30与原地面Tri-SIG-TIME SSL/defaulthead/Main2pop×7自动truth-last。

GPU4/6按实际free12GB/existingcompute≤2/新增1总≤3准入，一row一NN，不改其他owner。训练/部署实际耗时显存和每景性能待运行；静态176636032估计器MAC+269221888后验MAC、FP64 Cholesky约44739243FLOPs及2FFT不是实测延迟。矩阵、停止规则和产物见预登记report.md。主性能六景3pp+两个clean保持仍未达到，目标ACTIVE、默认DURA不变。

## 冻结相关性恢复器的源端参数归因已固定（2026-11-02）

`20261102-diagnostic-Restore-covariance-parameter-source-s2026092701-r01`，实际实现`5d7d9a2c4dd8e5d58b2843e47b1769d93d434c29`。固定父CovarianceMAP final30/bf29554a及原3ff8 F0/defaulthead，原源512×20、同一保存mu/C与FP64后验，对照盲/仅真实噪声/仅真实Action/二者真实四路。全48组/所有同条纠错与参数误差报告，0训练、新统计拟合、TX/V/OE/novel/目标IQ-预测-评分或选模。源oracle不作为Main部署版本，Fclean不是TX准确率或恢复上限。

父最后GPU阶段cross_both已从/proc核实liveon4/6后在另一卡准入，或该阶段exit0后选择4/6；父之后仅CPU评分，不会再占GPU。最多两条本任务NN且分卡，原compute≤2/新增1总≤3/free12GB，父健康训练/Main及其他owner不改。3项检查与唯一独立P0/P1通过，source2盲路径等于实际部署G/F0有限后立即全4路。当前父训练尚在运行，本项尚无新源结果。

## 冻结相关性恢复器的四路源端参数归因完成（2026-11-02）

`20261102-diagnostic-Restore-covariance-parameter-source-s2026092701-r01`，代码`5d7d9a2c4dd8e5d58b2843e47b1769d93d434c29`，父G固定bf29554a/final30/7410，F0原3ff8/defaulthead。原512源包×20draw，同mu/fullC/FP64求解，盲/仅真实噪声/仅真实Action/二者真实四路。整体Fclean一致依次95.029297/95.058594/96.523438/96.601562%，低档城区依次82.242340/82.660167/87.325905/87.743733%；完整配对修复/损伤与参数误差见analysis.md，不把Fclean一致当TX准确率、Main泛化或恢复上界。

10240实例/48组/所有配对独立复算PASS，最大差0，旧source-DURA/received标量与Fclean参考差0；source2盲==实际G.forward、全G/F0不变、20draw/exit0/liveempty核实，实际运行65.861080秒、worker live快照捕获True。0训练、新统计拟合、TX/V/OE/novel/目标IQ-预测-评分或选模；真实源物理参数不进入Main，父固定部署/正式矩阵及其他owner保持。主验收仍是原Main六景3pp与两个clean保持，目标ACTIVE、默认DURA不变。

## 相关性先验物理恢复器原Main完整七景完成（2026-11-02）

`20261102-phase1-Restore-CovarianceMAP-E30-s2026092701-r01`，实际代码`bf29554a03e28a012eaaa134f5c7b3f7856f2268`。1230341参数时频Transformer估计FIR64/主镜像/噪声，原63000LsUs拟合的统一mu/full512C作为额外1050624字节FP32 buffer，FP64 Gaussian后验输出IQ。无学习IQ解码器、DURA或G checkpoint祖先，全部学习参数从零单阶段30/7410；训练只算源全Action矩阵与噪声监督，G更新0F0前向/后验求解，源F0只作为方法诊断和smoke依据，不扩大成全流程无F知识主张。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景全部固定后独立CPU truth-last。六景依次83.500595/82.139286/80.539286/82.183333/75.047619/67.198214%，0/6，相对DURA依次-0.842857/-0.876786/-0.782738/-0.866071/-0.658333/-1.602976个百分点；clean86.572619%，cross_rx净+82、cross_both净-25。全部TX/RX/day/RX×day及真实TX修复/损伤、Fclean一致分别报告，不将源算子误差或Fclean参考当Main准确率。

42单元/1302组/868raw配对/434DURA配对/434clean配对复算PASS，最大差1.2e-14、raw/DURA判决差0；分组key覆盖、source30日志/顺序/LR/loss/E0-prior-E01-E30-final核实，4阶段exit0/liveempty。源审计按既有证据复用，0新增NN或训练。静态MAC/FP64Cholesky不是实测时延，详见analysis.md。本版未达到完整性能目标，目标ACTIVE、默认DURA保持。

## 实际后验 IQ 的源特征指导对照已固定（2026-11-02）

`20261102-phase1-Restore-CovarianceFeature-E30-s2026092701-r01`，代码 `5359a1b82b1fa5d640a394c3e3a77ce3b945fe1e`。沿新时频 Transformer/源完整协方差后验架构，与已完成 CovarianceMAP 物理估计对照保持 scratch 初值、原 L_s/U_s、源统计、顺序、draw、LR、单阶段 30/7410 和部署路径。新增原冻结 Tri-SIG-TIME SSL 的连续源特征约束，梯度通过实际恢复 IQ 后验回到信道与噪声估计，明确不是 F-free；不加载旧 G 权重，不使用 TX/logits/CE/原型/新分类头或 V/OE/novel/目标训练与选模。

全源物理监督保留；每逻辑 batch 首微批 32（末批 24）执行额外后验与 F0，固定 .01×(cos+源方差 MSE)，每轮 7896、累计 236880 条，另 63000 clean 源特征缓存。单阶段联合启用，采样项按所选记录均值计算，不额外缩小 1/8。3 项针对性检查和唯一独立 P0/P1 通过，实际 source2/full-posterior/F0/梯度/E0 重载通过后自动训练与原 Main 双总体七景，全部预测固定后独立 CPU truth-last。主性能目标仍未达成；训练速度是支线，默认 DURA 保持。


## 等价信道损失的冻结源端成本验证完成（2026-11-02）

`20261102-diagnostic-Restore-factorized-source-cost-s2026092701-r01`，执行代码`5ce71890e780ea3bfd8dc5c33389b30b70dacee4`，固定CovarianceFeature final30/5359、原3ff8 F0、保存E30首256源包/draw10。全矩阵与等价反射FIR因子路径保持同物理损失、首微批实际后验F0指导和clip，0新训练、optimizer、权重更新、TX/V/OE/novel/目标读取。实际loss相同，全部裁剪前梯度最大绝对差1.86264515e-09、相对L2差1.3145415e-07，G/F0全部state不变。

每路2预热/10计时/1梯度检查，共26逻辑反向。区域wall中位382.360001→353.696042ms，减少7.496589%；10配对中7快3慢，峰值allocated减少4.008942%。只含前向/反向/检查/clip，不含optimizer、全量数据IO和epoch日志，单batch短测不外推30轮同质量加速或识别收益。

实际source2/3新聚焦检查/一次独立P0P1通过，12项既有等价损失检查沿用；24记录与全部统计独立CPU复算差0。父最后GPU阶段在4时使用6，实际运行中快照与终态exit0已核实；辅助观察器外层回执失败和真实NN快照成功分开，没有重跑。本次仅测成本，当前5359原Main部署/矩阵保持，性能主线和默认DURA不变。详见analysis.md。


## 实际后验源特征指导的原 Main 完整七景完成（2026-11-02）

`20261102-phase1-Restore-CovarianceFeature-E30-s2026092701-r01`，实际代码 `5359a1b82b1fa5d640a394c3e3a77ce3b945fe1e`。同物理估计对照实际scratch E0/1230341参数、原源mu/fullC、顺序/draw/LR、30/7410和部署路径，新增冻结原3ff8 F0在实际恢复IQ上的连续特征指导，明确不是F-free。原LsUs全物理监督；每步首微批/每轮7896，累计236880后验/F0记录，另63000源clean参考；TX/logits/CE/类别原型/新头/V/OE/novel/目标训练和选模均无。

原地面完整Tri-SIG-TIME SSL/defaulthead/Main168000×七景全部固定后独立CPU truth-last。六景依次83.972024/82.464881/80.580952/82.608333/74.895238/66.962500%，1/6；相对DURA依次-0.371429/-0.551190/-0.741071/-0.441071/-0.810714/-1.838690个百分点，相对固定物理对照依次+0.471429/+0.325595/+0.041667/+0.425000/-0.152381/-0.235714个百分点。clean 86.742857%，cross_rx净 +377、cross_both净 -34。全部TX/RX/day/RX×day、真实TX修复/损伤与Fclean保持分别报告；源特征和波形指标不代替Main。

42单元/1302组/868raw配对/434DURA配对/434物理控制配对/434clean配对独立复算，最大差1.24e-14、baseline判决差0。完整30源日志/角色采样覆盖/归一化/顺序/draw/LR/prior/E30-final核实，复用E01实际匹配证据；4阶段exit0/liveempty。静态MAC不冒充实测延迟，训练与完整性能见analysis.md。完整主性能目标仍未达到；目标 ACTIVE，默认 DURA 保持。

## 2026-11-02 冻结源 V 物理／特征指导对照

源V low urban净增9/1800（+0.5个百分点），来自45次修复和36次损伤。配对双侧精确二项检验p=0.374174，单seed固定面板的微小正差不足以证明稳定改善。指导版仍比raw clean低13.888889个百分点，重景源端目标本身也远未满足，不能仅将Main差距归为域迁移。

此场景优化特征项均值降低1.118220%，而Action矩阵相对MSE增加16.747676%，noise logMSE增加56.248140%。七景的Action和noise logMSE均比纯物理版高；指导版没有独立IQ修正分支，特征梯度只能经物理估计器改变后验。这支持下一版分离物理估计与身份细节修正的结构假设，尚不能证明该改造有效，也不能用本验证选择权重或轮次。

low urban的特征改善组中42次TX修复、0次TX损伤；特征变差组中3次修复、36次损伤。因此该场景没有证据支持“更小特征误差主要造成TX损伤”的解释；更直接的现象是当前联合训练不能让全部样本的特征和物理误差同时改善。其他景的完整条件计数一并保留。

对照Main中同版low urban反而少396/168000条正确（-0.235714个百分点），只能说明这项小幅源V收益没有稳定转成Main收益。两者基线难度、记录分布和信道/接收机实例均有区别，不能分离单一域因素。

诊断run `20261102-diagnostic-Restore-covariance-guidance-sourceV-s2026092701-r01`；实际代码 `4178f020951d4799e912a941240bea36f45bedf1`；源V七景、315分组和105条件分组完成独立CPU评分与计数核对，0训练／target读取，默认DURA不变。完整结果见 `automation_reports/CV-SincNet/20261102-diagnostic-Restore-covariance-guidance-sourceV-s2026092701-r01/source_analysis.md`。

下一版候选方向：从零单阶段30轮，分别设置物理参数估计和IQ细节修正分支；连续F特征梯度只作用于后者，避免为了特征拟合直接改写Action／噪声估计。仅为源端证据支持的待检验假设，尚未实现或启动；不改训练角色、F0、默认头或原Main七景验收。性能主线继续，训练效率仍为支线。

## 2026-11-02 双分支恢复器已实际启动

run `20261102-phase1-Restore-CovarianceDetail-E30-s2026092701-r01`，实际代码 `1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6`。5481223参数（物理1230341、IQ细节4250882），全部scratch单阶段30/7410；物理只模拟参数监督，IQ／连续F特征梯度只到细节，独立梯度裁剪；1890000条完整源曝光，原Ls/Us范围和冻结Tri-SIG-TIME SSL不变。5项针对性检查和唯一P0/P1审查通过，真实source2完整网络/F0/梯度隔离/初始零细节/E0重载PASS，GPU4实际训练已核实。最终轮次后自动原Main双总体七景、独立truth-last；当前RUNNING，尚无新性能结论。默认DURA不变，原六景逐个3pp＋两总体clean保持的目标仍未达到。

## 2026-11-02 双分支首轮产物通过

实际E01/247步完成、63000条完整源曝光、1969次微批后验，核对PASS；物理E0与BF全部张量相同，先验不变，两分支均更新，源order/draw一致，损失复算误差4.00251432e-9。首轮230.688593秒。物理E01权重相对BF最大绝对差0.000418007141，具体数值来源未定位，不能声称轨迹逐值一致；该项为已预先声明的描述性比较，不改变训练或新增门槛。复合候选按原30轮继续，尚无新Main成绩。详见当前run `E01_analysis.md`。

## 2026-11-02 最终权重延迟测量已排队

独立run `20261102-diagnostic-Restore-CovarianceDetail-final-latency-s2026092701-r01`，代码 `33cb43c2d4d387ab18498b451ddc1444070e997c`，CPU controller 1065879 实际live／CUDA空／0新GPU worker已核实；必须等双分支父实验完整Main预测和独立评分完成后，才测固定final30的新G与默认DURA G、rawF以及同原3ff8 F0的两种G+F组合。B1/B128各3次预热，随后每路径100／20次计时，病例与路径均衡交错；只源512面板，不读取TX或target，不看父分数控制开始，不改父训练测试。4项检查和唯一P0/P1通过；实际最终checkpoint source2及性能测量仍待执行，当前无时延结果。

## 2026-11-02 最终源 V 波形与真实 TX 验证已排队

独立run `20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01`，代码 `f29b31417ce98a280eac25ea2fa78e0c49a5f0ff`；CPU controller 1086539实际PID/CWD/命令/CUDA空/0新GPU已核实。3新＋4原检查及唯一P0/P1通过，一次archive SHA/compile通过。等待Main与随后延迟任务技术完成后，只新算最终G的12600条源V输出与source2；旧4178三基线及1800clean参考按ID/F0身份复用，全部四方法七景输出后独立CPU才接V真值。0训练/选模/target/OE/novel，源V结果不改标Main，默认DURA不变。

为保持本任务最多一个活跃CPU真值评分器，人工Main独立产物复算与本源V的CPU评分错开，安排在其评分进程退出后；若源V在评分前技术失败，则确认无活跃评分进程后即可复算Main。不以源V分数或成功作为Main质量门槛；原Main自动正式评分始终先完成。

## 2026-11-02 双分支完整源训练已核实

run `20261102-phase1-Restore-CovarianceDetail-E30-s2026092701-r01`，源30/7410、1890000全源F0与后验曝光、30recovery、源先验不变及final等于E30全部核对PASS，weighted loss误差4.002514319978445e-09。训练6951.032337888995秒、worker6956.7112466519975秒、peak allocated 1518080512B。物理E30相对BF最大张量差0.44416457414627075，不声称同轨迹。原Main双总体七景由既有控制器自动接续；源结果不当性能验收，不据其取消测试。默认DURA、source权限与完整目标ACTIVE保持。

## 2026-11-02 独立物理与 IQ 细节双分支完整 Main

run `20261102-phase1-Restore-CovarianceDetail-E30-s2026092701-r01`，实际代码 `1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6`。独立物理估计与非线性细节网络，5481223 参数，scratch 单阶段 30/7410；完整 1890000 次源后验与冻结 F0 连续特征曝光，源 TX 隐藏、V/OE/novel/目标不参与训练或选模。不是 F-free，不使用旧 G 初始化或新身份头。物理与细节梯度隔离、物理 E0 与 BF 相同，实际轨迹有数值差异；结构、路由和曝光共同改变，不能作单组件因果归因。

原地面完整 Tri-SIG-TIME SSL/Main 七景全部固定后独立 truth-last。六景依次 80.140476/77.807738/75.495238/78.213690/69.447024/62.401190%，通过 0/6；相对 DURA 依次 -4.202976/-5.208333/-5.826786/-4.835714/-6.258929/-6.400000 个百分点。clean cross_rx 净 +151、cross_both 净 +12。合并六景逐个 3pp 与两个 clean 保持的完整主目标仍未达到。 两个总体各自分景与所有 TX/RX/day/RX×day、真实 TX 纠错损伤、Fclean 判决及完整混淆矩阵均报告；不以均值掩盖场景或 clean 损伤。

42 单元与全部评分组、raw/DURA/纯物理/原指导/clean 配对独立复算，最大差 1.24e-14，复用基线判决差 0。完整源审计单列保存，四阶段 exit0；图表经实际渲染检查。训练成本与静态计算量分别报告，最终实测延迟与源 V 验证在已登记独立 run 中补充；这些诊断不是主性能评分门槛。默认 DURA 保持，已有测试曝光如实注明，结果不回流本版训练/选模/重跑。

## 2026-11-02 最终双分支时延与源 V 诊断完成

最终 G 1fcb/final30 保持，同原 3ff8 完整 Tri-SIG-TIME SSL。实际延迟 run `20261102-diagnostic-Restore-CovarianceDetail-final-latency-s2026092701-r01`，B1 新 G+F 22.380381 ms、DURA G+F 12.734163 ms；B128 新 G+F 189.174147 ms。600 计时／30 预热／5 source2 路径全部独立复算，GPU 共享环境与计时/显存范围单列，不能解释成独占设备延迟保证或训练提速。

源 V run `20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01`，只新增 12600 条最终 G/F0 输出，原三基线和 1800 clean 参考逐值复用；28 单元/420 分组/105 条件组与独立真 TX 评分全部完成。低档城区源 V TX 91.000000%，相对纯物理版修复 165、损伤 43，全部场景/分组/物理误差与特征分析见源报告。类间/类内方差是固定 F 特征空间的描述性指标，不是硬件信息证明或未知设备泛化证据。V 未参加训练/选模，无新增身份头/原型。

这两个冻结诊断均在父 Main 全部评分完成之后启动，0 训练、0 目标输入/真值、模型状态不变；源 V 真 TX 只在预测后评分，不能描述成零标签读取。主性能验收仍以父 Main 合并六景逐个 3pp 与两个 clean 保持为准，不将源验证或时延当新的质量门槛，默认 DURA 不变。

## 2026-11-02 冻结源接收机均值残差反证


VERIFIED：同一冻结源V的52200条已保存特征行，87个汇总与667个组统计；0新NN/IQ/TX/目标读取，模型和原Main结果保持。新分析实际代码43f1d218fe54a2cffe707b953708baa6a9a10124，5项针对性检查与方差/配对误差恒等式通过（最大误差4.44e-16），全部28个总体特征MSE与既有评分一致。

低档城区新版RX均值误差0.000825112，物理版0.004257533，下降80.619941%；相对原clean均值模式的对齐增益0.952200，物理版0.713805。新版特征误差中RX均值项仅0.575860%，98.295941%在RX组内。这不支持“新版放大RX均值偏差导致Main下降”这一具体解释，也没有为删均值或接收机均值投影提供直接依据。源V的RX/day均在训练中出现，包分布未按TX作因果控制，不能排除未见RX的协方差或非线性域差异。

[完整RX/day分析](../../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01/receiver_geometry_analysis.md)保留全部场景、绝对量、占比、整体偏移与组均值模式；不是恢复性能优化或主目标达成证据。

下一项源端检验将区分细节修正与观测的物理一致性：仅在已冻结源V及已有模拟Action范围分析，不据Main错例选择网络或超参。当前尚未实现或执行该项检验，没有新G训练或Main预测。目标ACTIVE、六景逐个3pp与两个clean保持、原Ls/Us权限和默认DURA均保持。

## 2026-11-02 观测残差与去噪的源端区分


VERIFIED：实际代码62cf25c161afe1410ad6250249931389d9210a1c，复用冻结源V四方法IQ与clean参考、原R07缓存Action。840分组/168条件组完成，252个重叠分组IQ/Fclean复算差0；5项针对性检查覆盖既有稠密算子、反射边界、I/Q镜像与观测噪声反例。读取52200条保存IQ（29数组）、52200条保存判决和12600条Action trace；0原始IQ数据集/新TX标签/目标/新NN/训练。

低档城区新版有1731/1800条IQ更接近clean，其中1247条对观测的归一化残差增大。物理版/新版/clean参考的残差MSE分别0.101286/0.106892/0.109071；逐记录相对clean残差能量的绝对差距均值0.344762→0.232462，减少32.573363%。IQ改善且残差增大组包含138次Fclean修复、33次Fclean损伤；这是相对F(raw clean)的判决一致性，不是真TX，不能与原源V真TX纠错混同。

这不支持直接加强“贴近带噪观测”的损失。缓存Action只是包含实现后AGC的模拟代理，不是硬件Jacobian；缓存标量噪声也不是完整I/Q协方差，clean参考不可部署。这项源端检验不能解释Main的全部退化，也不提供恢复理论上界。原Main、模型、默认DURA、17候选及主目标保持。

[完整Action一致性报告](../../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01/measurement_consistency_analysis.md)包含全部七景与分组、均值/中位数、逐记录条件统计和证据边界。

下一项独立源验证将检验冻结恢复函数对I/Q坐标旋转的响应：旋转输入后把输出映回原坐标，再与同条原clean/原F0参考比较；不假设F0相位不变，不据Main选择角度或输出。当前尚未实现或启动，不是已证明有效的新恢复版本，也不构成额外发布门槛。

## IQ 坐标诊断已完成

480 条固定源 V 记录覆盖七景和四个 I/Q 坐标方向。恢复输出映回原坐标后交给原 Tri-SIG-TIME SSL。零度 3,360 条 G/F0 判决与 480 条 clean F0 参考复用原产物，新增 10,080 条 G/F0 输出，另有 8 条真实源 L_s 冒烟输出。独立复算覆盖 28 单元与 672 分组，指标差与原缓存变化均为 0。

旋转 180° 后，clean 的 IQ NMSE 从 0.000028 增至 0.188923，低档城区从 0.034477 增至 3.715660。低档城区的原 clean 判决一致率从 91.875% 降到 75.000%。这是同一源波形在可逆坐标变换下的实测差异。判决一致率以 F0 在原 clean 上的输出为参照，真实 TX 未读取。Main 差距的成因与理论信息上界仍待独立证据。

下一版将让物理算子和 IQ 细节路径满足四个直角旋转的等变关系。物理分支在同一坐标合并算子，求解一次高斯后验，再用共享细节分支修复 IQ。按原 L_s/U_s 从零单阶段训练 30 轮，并用同一冻结 Tri-SIG-TIME SSL 测 Main 七景。

[完整诊断](../../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-source-coordinate-s2026092701-r01/analysis.md)

## 2026-11-07 C4 等变恢复器已提交 N607

候选 `20261107-phase1-Restore-CoordinateMAPDetail-E30-s2026092701-r01`，实际发布代码 `bc6bcac3bf8463fb3b2770a0ec2a6115ab322ae1`。10 项本地检查及一次独立 P0/P1 审查通过。四个物理估计映回原坐标后合并，仅求解一次后验，再以共享细节路径修复 IQ。原 L_s/U_s、冻结 F0 连续特征、从零单阶段 30 轮，最终自动原 Main 双总体七景和独立 truth-last。已核实控制器在线，实际训练仍排队。 [实验登记与源先验分析](../../automation_reports/CV-SincNet/20261107-phase1-Restore-CoordinateMAPDetail-E30-s2026092701-r01/report.md)

## 2026-11-07 观测条件相位先验的冻结源对照

已完成 120 条源 V 记录、七景和四种先验的配对比较。低档城区的均匀 C4 高斯与观测条件混合，IQ NMSE 分别为 0.053467 与 0.052613，F(clean) 判决一致率分别为 91.67% 与 92.50%。同条比较有 1 次修复、0 次损伤。共享协方差下的固定均值对照为 0.052606 和 92.50%。

四路径共享模拟 Action 与噪声，F(clean) 只作冻结模型判决参照。无训练、目标或 TX 读取。 [完整分析](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-C4-phase-mixture-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 核对波形收益与额外纠错需求

低档城区 1,731/1,800 条 IQ 更接近 clean。其中 1,427 条原本和新版都判对，162 条从错变对，36 条从对变错，106 条仍共同判错。该组里原本判错的 268 条中修复 162 条，占 60.45%。IQ 改善组净增 126 条，其余 69 条净少 4 条，合计与原报告的净增 122 条一致。波形误差下降有实际识别收益，也同时包含误判和损伤，不能直接用波形改善比例推算识别准确率。

低档城区 DURA 判对 115,586 条。39 组配对中最大的额外修复量为 16,672 条，仍有 35,742 条两条路径均误判。即使保住全部 DURA 正确判决，最高也只有 78.725%，距 140,345 条门槛仍少 8,087 条。中档城区的现有固定配对也都达不到门槛。因此，强退化场景除了减少损伤，还需要产生能修复更多共同错误的新 IQ 输出。该数值仅约束已发布的固定两路判决，任意新恢复器和单次观测的信息上界仍待测定。 [源端逐条交叉核对](../../automation_reports/CV-SincNet/20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01/source_analysis.md) · [Main 纠错覆盖](../PAMF_RESTORATION_MAIN_PERFORMANCE_AUDIT_20261101.md)

## 2026-11-07 结构化噪声与源端条件波形误差

低档城区的盲估信道配对中，结构噪声让 IQ NMSE 从 0.058703 变为 0.058704，F(clean) 判决一致率从 81.67% 变为 82.50%，1 次修复、0 次损伤。已知模拟信道配对的 IQ NMSE 从 0.048568 变为 0.048563，一致率从 90.00% 变为 90.00%。该场景下，结构化代理的条件后验方差占源先验方差 17.417%。两条标量路径之间，已知模拟参数相对盲估有 12 次修复和 2 次损伤，净增 10 条。

后验方差以指定模拟器参数和高斯模型为条件。部署恢复性能仍由固定原始 Main 的真实 TX 评价。 [源端诊断](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-structured-IQ-noise-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 逐包相位和漂移的源端校正

低档城区的 120 条源记录中，原盲估、相位校正、相位加漂移和已知模拟路径的 F(clean) 判决一致率依次为 81.67%、86.67%、76.67% 和 90.00%。两条校正的配对修复／损伤分别为 7／1 和 4／10。

源端只作冻结验证，主验收仍为固定原 Main 七景和真实 TX。 [源端诊断](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-observed-phase-ramp-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 整体相位的 Main 冻结推理对照

VERIFIED / QUEUED。控制器 PID 3845271 在线，实际工作目录、命令、代码和 CUDA 空环境已核对。发布包 SHA 一致，远端编译通过。固定等待 C4 最后一个 GPU 阶段终止，再按原许可 GPU4/6 条件开始。快照时尚无本版 GPU 预测或真实源检查。完全复用原 E30 权重，只新增整体相位的冻结推理。 原 Main 双总体七景对照原 BF、DURA、raw；全部预测固定后由独立 CPU 评分。 [正式实验](../../automation_reports/CV-SincNet/20261107-evaluation-Restore-CovariancePhaseOnly-main-s2026092701-r01/report.md)

## 2026-11-07 整体相位源端真实 TX 配对

同一 120 条源记录的四路七景结果全部冻结并完成独立核对后，另做原源 V 真值评分。低档城区整体相位校正从 98/120 提高到 104/120，净增 6 条真实 TX 正确，修复 7 条、损伤 1 条。这七条修复与原来的 clean 判决一致性修复逐条相同。clean 仍为 118/120，整体相位版七景正确数均至少持平。

22 条原盲估误判中，整体相位和已知模拟参数共同修复 7 条，已知路径另修复 4 条。其余 11 条三路都错，其中 9 条在 raw clean 上可识别。已知路径使用缓存模拟 Action 和平均实噪声方差，仍沿用原高斯先验。因此共同错误还需检查观测损失、恢复先验和识别敏感细节，当前配对无法给三者分配因果份额。低档城区整体相位让 56 条 IQ 误差下降，7 条修复都在这组，另外 64 条 IQ 误差上升，含 1 条识别损伤。 [独立真值复算](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-observed-phase-ramp-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 源端算子误差归因

低档城区固定 480 条源记录。原盲估正确 401 条，替换 FIR 形状后 400 条，替换主路／镜像调制后 424 条，两者都已知时 430 条。最后额外换成真实噪声标量为 426 条，raw clean 为 474 条。只换 FIR 的修复／损伤为 7／8，只换调制为 31／8。

同一噪声下对齐后替换 FIR 与接收调制，源端全矩阵固定后独立真值评分。下一版优先改进接收调制估计，保留 FIR 与整算子约束，依据合法 L_s/U_s 从零单阶段训练。 [源端完整分析](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-action-factor-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 接收调制版从零训练已提交

VERIFIED / QUEUED。实际代码 3fc9201628e7602ea58c6a5fb13d6ec934513e99，控制器 PID 3869088 在线，实际工作目录、命令与 CUDA 空环境已核对。发布包 SHA 一致，远端编译通过。等待整体相位版最后一个 GPU 阶段终止，然后按原 GPU4/6 准入条件开始。当前真实源两条检查、训练和 Main 均待执行。 1229833 个学习参数，原 L_s/U_s 单阶段 30 轮、7410 步。包级复增益、独立斜率与局部相位，新增对齐后的分量监督。 完整算子使用精确因式计算，更新不调用 F0 或后验。冻结后自动原 Main 双总体七景和独立评分。 [实验登记与源系数摘要](../../automation_reports/CV-SincNet/20261107-phase1-Restore-ModulationCovariance-E30-s2026092701-r01/report.md)

## 2026-11-07 分类边界方向的源端条件误差

低档城区在已知模拟算子和标量噪声的同一高斯模型下，剩余 IQ 方差占源先验 17.42%。投影到原 F0 clean 第一与第二名的分数差，剩余方差比例为 20.49%，汇总所有五个竞争类别为 19.90%。

同已知模拟算子和原冻结 F0，固定源输入导数诊断，无训练或目标访问。 [结果和非线性核对](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-posterior-margin-uncertainty-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 矩匹配波形先验的源端对照

固定源 V120 的低档城区，盲估信道下真实 TX 正确数由 98 升至 102，4 条修复、0 条损伤。已知模拟信道和噪声时，两种先验均为 107/120。同批 raw clean 为 118/120。

低档城区有局部收益，高档和中档城区各多错 1 条。两种先验在已知模拟信道下的七景真实 TX 正确数一致。这项结果不更新正式 Main 的达标状态。 [源端分析](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-waveform-mixture-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 相位与波形先验联合恢复

同批源 V120 低档城区，原高斯、仅四组先验、仅相位修正、两者联合的真实 TX 正确数依次为 98、102、104、102，raw clean 为 118。联合比仅相位修复 0 条、损伤 2 条。 [七景配对和交互统计](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-joint-phase-waveform-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 接收调制分量归因

固定 480 条源 V 的低档城区，原盲估、主路相位、主路复增益、完整主路、完整镜像、双路都已知的 TX 正确数依次为 401／422／423／424／401／424。raw clean 为 474/480。 [七景配对、clean 逐条保留和分组统计](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-modulation-components-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 相干参考盲估主路相位

同批 480 条低档城区源样本，原盲估、整算子相位、主路等权、主路协方差加权、已知主路相位的 真实 TX 正确数依次为 401、414、378、414、422。raw clean 为 474。 [七景真实 TX 与主路角误差](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-coherent-main-phase-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 独立主路相位监督模型

VERIFIED / QUEUED。实际代码 36b13adad5b091bf1f8f4aa7934c403e3b50d8eb，控制器 PID 3933764 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待接收调制版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。 1241898 个参数，固定相干 FIR 相位坐标，显式主路角度、独立局部轨迹。 原 L_s/U_s 从零单阶段 30 轮、7410 步，使用模拟圆周相位监督。冻结后自动用原 Tri-SIG-TIME SSL 做 Main 两总体七景并独立评分。 [实验登记和排队核实](../../automation_reports/CV-SincNet/20261107-phase1-Restore-PhaseSupervisedCovariance-E30-s2026092701-r01/report.md)

## 2026-11-07 主路全协方差版开始正式 CPU 推理

VERIFIED / RUNNING。CPU 预测进程 3944144 已通过真实 L_s 两条检查，并已处理原 Main 首批。128 条源样本实测 19.20 条/秒，折算全部 117.6 万条模型调用约 17.01 小时。这是源端吞吐外推，实际进度由预测日志和各景产物确认。独立评分在两类总体全部七景固定后才开始。 复用原 BF E30 权重和原 Tri-SIG-TIME SSL，保持原 Main 双总体七景。其余 GPU 任务按既定计划。 [实验登记和启动证据](../../automation_reports/CV-SincNet/20261107-evaluation-Restore-CoherentMainPhase-CPU-main-s2026092701-r01/report.md)

## 2026-11-07 已知模拟信道后的身份信息损失

给定完整模拟信道和真实噪声，固定高斯恢复在低档城区保住 425/474 条 raw clean 正确记录。丢失 49 条，其中 45 条来自实测信噪比较低的一半（不超过 8.50 dB）。 丢失组与保留组 clean 分数差中位数接近。 [信噪比、频响及 clean 分数分层](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-action-factor-sourceV-s2026092701-r01/known_operator_error_breakdown.md)

## 2026-11-07 已知信道高斯恢复的收缩与噪声

固定源 V480 的低档城区，已知模拟信道和噪声时原高斯恢复识别对 426 条。只移除实际噪声对输出的贡献后对 452 条，只移除高斯收缩后对 458 条。raw clean 为 474 条。 低档城区原先丢失的 clean 正确中，两种单项去除各自共同修复 21 条。 [七景与信噪比分层](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-posterior-bias-noise-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 后验约束的条件 IQ 先验模型

VERIFIED / QUEUED。实际代码 8cf3669c4a5d98f6c4930117952203afd56f2f6e，控制器 PID 3965491 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待主路相位监督版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。 5481223 个参数，条件均值修正经同一后验投影后才输出。 原 L_s/U_s 从零单阶段 30 轮、7410 步，使用原冻结 F0 的源端连续特征和精确无噪收缩监督。冻结后自动用原 Tri-SIG-TIME SSL 做 Main 两总体七景并独立评分。 [实验登记和排队核实](../../automation_reports/CV-SincNet/20261107-phase1-Restore-ProjectedPriorDetail-E30-s2026092701-r01/report.md)

## 2026-11-07 原 Main 的 clean 保留与恢复独有纠错

低档城区，DURA v1.0 的总正确数是 raw clean 的 79.50%，而原 clean 正确样本的逐条保留率为 73.78%。另有 5,744 条 raw clean 和受损直接识别都错、经恢复后识别对。 [原冻结三路配对](../../automation_reports/CV-SincNet/20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03/clean_received_three_way.md)

## 2026-11-07 源 IQ 实际绑定与 Main 来源

原 F0 三个训练段均按底层存储物理 ID 绑定。将原 63,000 条 L_s/U_s 与 R07、G10 按同条物理 ID 对齐，原 F0 与 R07 的最大逐点差为 2.3842×10⁻⁷。确认正式 Main 两类总体全部复用原 168,000 条六景观测，实际原载频均 2.450 GHz。R07 源训练使用 15 个按 RX×day 的虚拟接收会话，原 Main 使用同一个虚拟会话。未测定会话与载频差异的性能效应。 [源 IQ 与历史 Main 溯源](../PAMF_RESTORATION_SOURCE_MAIN_CONFIG_20261102.md)

## 2026-11-07 重复段与整包 IQ 的源端识别归因

固定源 V480 的低档城区，原 BF 判对 401 条。只替换 80–159 采样点内的周期一致误差后判对 418 条，只替换窗口外误差后判对 473 条，raw clean 判对 474 条。窗口外两段合计占 IQ 误差能量的 84.84%。 下一步区分首段和后段，再决定独立整包架构的容量分配。 [七景反事实与真实 TX](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-preamble-component-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 前后段源端识别归因

固定源 V480 的低档城区，原 BF 判对 401 条。只替换包首 0–79 后判对 431 条，只替换后段 160–255 后判对 463 条，两段一起替换后判对 473 条，raw clean 判对 474 条。 低档郊区仅包首 474、仅后段 470，支持整包覆盖和独立分段能力。 [原冻结 F0 的四分量全组合](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-preamble-edges-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 整包时变复数逆滤波器核心

独立从零初始化的恢复网络已通过本地核心验证。输入和输出均为原 256 点 FP32 IQ。由采样点 80–159 的四个 20 点块估计周期相干参考，作为逐包条件。全率时域网络产生逐点 17 抽头复数主路、镜像路与 IQ 残差。包首、中段、后段有独立分支，边界 16 点线性过渡；后段有三块处理，包首两块，中段一块。总参数 490866，末层零初始化使 E0 恒等，依赖原 IQ 而无需加载旧恢复器。

四项针对性验证覆盖解析复数等价、相位参考及零输入、恒等与严格重载、激活后的完整梯度及逐包独立性。当前仅核心实现，本轮没有启动该模型训练或正式预测。接入原 L_s/U_s 单阶段 30 轮流程后，用原地面冻结 Tri-SIG-TIME SSL 测 Main 两总体七景。主指标为六景逐个距 raw clean ≤3 个百分点及两个 clean 总体各自保持。

## 2026-11-07 独立整包时变复数逆滤波恢复器

VERIFIED / QUEUED。实际代码 92d0dcbe0f25927d5a1cf4d98669ded45a410a5b，控制器 PID 4001257 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待后验约束条件先验版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。 490866 个参数，包首、中段、后段独立复数滤波，后段更深。 原 L_s/U_s 从零单阶段 30 轮、7410 步，使用批准的原冻结 F0 源连续特征约束，独立于 F-free 对照。冻结后自动用原 Tri-SIG-TIME SSL 做 Main 两总体七景并独立评分。 [实验登记和排队核实](../../automation_reports/CV-SincNet/20261107-phase1-Restore-PacketInverseFilter-E30-s2026092701-r01/report.md)

## 2026-11-07 原源模拟接收机配对归因

固定源 V480 的低档城区，原验证会话、训练会话、验证会话轮换的恢复正确数分别为 401、399、400。raw clean 为 474。 [固定传播和噪声的对照](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-receiver-session-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 源接收机先验的配额匹配对照

固定源 V480 的低档城区，盲估信道下全源、同 RX、其他 RX 先验的正确数为 401、412、341；已知模拟信道和噪声下为 426、434、379。raw clean 为 474。 [原源 RX 配对先验分析](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-receiver-prior-sourceV-s2026092701-r02/analysis.md)

## 2026-11-07 源接收机先验失配训练

VERIFIED / QUEUED。实际代码 14e7315d6da5d6fe242e7568f20c97ade064e8d9，控制器 PID 4033035 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待整包逆滤波版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。 5481223 个参数，原条件先验网络，训练时交替使用全源和排除当前 RX 的源先验。 原 L_s/U_s 从零单阶段 30 轮、7410 步，使用原源连续 F0 特征与无噪收缩约束，部署只输入 received IQ。冻结后自动用原 Tri-SIG-TIME SSL 做 Main 两总体七景并独立评分。 [实验登记和排队核实](../../automation_reports/CV-SincNet/20261107-phase1-Restore-MismatchedPrior-E30-s2026092701-r01/report.md)

## 2026-11-07 源先验均值与协方差分量

固定源 V480 的低档城区，盲估条件下同 RX、仅换均值、仅换协方差、两者都换的正确数为 412、408、364、341；已知模拟信道和噪声时为 434、425、394、379。 [同观测的交叉先验对照](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-prior-factor-sourceV-s2026092701-r01/analysis.md)

## 2026-11-07 观测条件化的先验方差网络

VERIFIED / QUEUED。实际代码 2f77f8cec389f7ff74a94a8200dd219c1b5e084f，控制器 PID 4049016 在线，实际工作目录、命令、CUDA 空环境和日志已核对。发布包 SHA 一致，远端编译通过。等待先验失配训练版最后一项 GPU 阶段终止，再按 GPU4/6 原准入条件启动。真实源检查、训练和 Main 均待执行。 1336263 个参数，当前观测生成非负逐点方差，完整后验可微。 原 L_s/U_s 从零单阶段 30 轮、7410 步，使用实际带噪 IQ/STFT 和批准的源连续 F0 特征约束，物理梯度隔离。冻结后自动用原 Tri-SIG-TIME SSL 做 Main 两总体七景并独立评分。 [实验登记和排队核实](../../automation_reports/CV-SincNet/20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r01/report.md)

## 2026-11-07 用户批准单卡最多五进程后实际训练

VERIFIED / RUNNING。控制器 4063502 与实际训练进程 4063584 的 PID、工作目录、命令、CUDA UUID、GPU6 计算 PID 和源日志已核实。启动前已有 4 个进程、剩余 17938 MiB，读回总数 5。原 r01 未开始任何 GPU 阶段，其唯一控制器按用户授权精确停止，其余七个旧控制器在线。真实源两条检查、物理/置信度梯度隔离及 E0 严格重载通过。当前首轮仍在运行，尚无新 Main 成绩。 [正式运行与源检查](../../automation_reports/CV-SincNet/20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r02/report.md)

## 2026-11-07 GPU6 首轮实际训练完成

VERIFIED：已完成 1/30 轮、247/7410 次更新，实际遍历全部 63000 条源记录（L_s6300、U_s56700）。独立读取 E0/E01 与优化器状态，确认物理分支和条件方差分支均有实际权重更新，优化器状态有限，源先验 mean/covariance 原样保持。训练器在轮末检查冻结 F0 未变化。首轮耗时 771.414 秒，81.668 条/秒；若保持此速度，30 轮训练约 6.43 小时。这是共享 GPU6 下的单轮测量和条件估算，未包含测试，也不能作为独占环境提速结论。实际训练进程仍在运行；单阶段30轮和自动 Main 两人群七景矩阵保持。暂无本版 Main 识别成绩，逐景距 clean≤3个百分点目标未达成。 [源训练执行摘要](../../automation_reports/CV-SincNet/20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r02/first_epoch_evidence.json)

## 2026-11-07 第二轮完成与分支活动跟踪

VERIFIED：2/30 轮、494/7410 次更新，累计126000条源记录曝光；E02优化器状态有限，固定源先验不变，E01→E02两个分支的权重均继续变化。第二轮769.632秒，两轮均值770.523秒。训练记录同时出现待跟踪现象：新增先验方差相对量均值由1.4987e-3降到1.6873e-8，置信度分支平均裁剪前梯度范数由5.2277e-2降到2.1048e-6。这些平均量显示分支作用明显减弱，尚不能证明分支永久失效或已达到最优。两轮的信道抽样和先验模式不同，IQ NMSE变化不作为单因素收益，也不代表Main准确率。检查只读取已保存训练状态和轮次摘要，新增IQ/神经前向/truth/target读取均为0。保持健康训练、单阶段30轮、final30唯一冻结点和完整Main七景测试；后续完成轮次继续跟踪分支活动。 [第二轮观察](../../automation_reports/CV-SincNet/20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r02/E02_signal_evidence.json)

## 2026-11-07 E02 条件方差分支的源端诊断

VERIFIED / ANALYZED。固定E02源训练70条（实际全部L_s，五RX×七scene×2），两种源先验×四种固定方差，共56单元完成且独立重算。当前分支与零附加方差的损失差仅约1e-8量级，当前softplus平均斜率约2.2e-7；统一增大方差在global55/70、other-RX44/70条上局部不利，但约43%/46%的坐标仍有负梯度方向。结果支持表达限制与梯度衰减并存，尚不证明永久失效或Main识别收益。源2检查、3项针对性测试、一次独立P0/P1和实际CPU执行通过，零模型更新、零TX/V/OE/target读取。GPU6当前已完成3/30轮，原final30和Main七景矩阵保持。 [源训练配对机制分析](../../automation_reports/CV-SincNet/20261107-diagnostic-Restore-conditional-variance-E02-sourceLU-s2026092701-r01/analysis.md)

## 2026-11-07 双向协方差谱适配版已实现并排队

VERIFIED / QUEUED。双向协方差版已发布N607；实际控制器PID/CWD/命令/CUDA空/等待日志已核实，子进程0、新增GPU训练进程0。等待当前ConditionedCovariance r02最后GPU阶段后再按GPU6容量准入。代码4项针对性测试、一次独立P0/P1、归档SHA一致与远端编译通过；真实源2检查仍待其训练阶段执行，不将排队或单元测试写成真实模型已通过。模型1340293参数，固定源top64方向，C+Bdiag(0.9softsign(raw))B^T允许增减且初始严格为原C；实际FP32模式因子的白化谱界保证正定余量。从零单阶段30轮/7410更新、原L_s/U_s与源F0连续特征权限保持，final30后自动同一原TriSIGTIME SSL基座Main两人群七景及独立评分。现有健康训练和CPU Main继续，新版尚无识别成绩。 [实现、矩阵与实际队列](../../automation_reports/CV-SincNet/20261107-phase1-Restore-SignedCovariance-E30-s2026092701-r01/report.md)

## 2026-11-07 前五轮训练与首个星地景预测已核实

VERIFIED：5/30轮、1235次更新、63000条唯一源记录完整曝光5次；累计315000次，其中L_s31500、U_s283500。E05与优化器有限、源先验不变、全部5轮源顺序为完整排列。物理估计误差下降，新增方差与梯度在E03至E05持续很小。实际平均739.993秒/轮；若均值保持，30轮训练约6.17小时，尚余约5.14小时，不含测试且共享负载可变。正式冻结仍为final30，尚无本版Main识别成绩。

VERIFIED：独立读取clean.npy与practical_high.npy，每份126000条int64类别预测且范围0至5，累计固定252000条，完成2/14个原Main“人群×场景”单元。实际进程已进入practical_mid，尚未提交truth评分；新IQ/模型/truth读取均为0。其余场景继续，预测固定不等于已取得识别成绩。 [训练分析](../../automation_reports/CV-SincNet/20261107-phase1-Restore-ConditionedCovariance-E30-s2026092701-r02/training_progress_E05.md)；[完整测试进度](../../automation_reports/CV-SincNet/20261107-evaluation-Restore-CoherentMainPhase-CPU-main-s2026092701-r01/report.md)。
