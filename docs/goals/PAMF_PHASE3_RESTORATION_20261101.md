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
