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
