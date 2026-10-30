# PAMF 身份与开放特征的联合密度：固定单候选

Run `20261031-phase3-joint-feature-s2026092701-r01`。唯一主joint_predicted_ratio；原pooled_ratio、原predicted_ratio、新joint_pooled_ratio为固定控制。当前无新性能，完整Phase3目标ACTIVE。

## 假设与方法

入选r03开放特征z160与原F0身份特征u160来自不同训练目标，可能包含互补的设备信息。本轮直接检验二者的联合分布；不重复已失败的局部协方差、四贡献640或星地P0/P1双视图。它是待证伪假设，不预设联合一定更好。

各自FP64单位化u、z，拼接后除sqrt(2)并转FP32得到320维。拟合和预测统一再FP64单位化。源L六类×五RX×三day90格各70条，用格内population残差的共享完整协方差、.9数据/.1各向同性收缩和Cholesky白化；known90各1/90质量，径向方差按70/(70+320)向pooled收缩。背景OE6134按15公共域各1/15权重、固定32容量、25Lloyd/seed2026103019、原effective-count方差规则。此320状态使用独立schema，旧160/640状态和默认维度不改。没有未知TX/K输入或新身份头。

新joint pooled分数为log pL−log pOE；主joint predicted只取原F0预测类的15个known模式、各1/15，扣相同背景。原两个ratio的160统计、六tau与source/Q控制逐值复用。原F0/default6、DURA v1.0和r03均冻结，旧身份及同DURA闭集基线保持。区别是完整联合密度及其源V校准，不能把所有变化只归因于拼接算子。

## 数据、祖先与新成本

父预测为class-density `20261031-phase3-class-density-s2026092701-r01` / code4af0668331682636ee212077b38cd912857a1b31；纯源r03 signal为code dc01a402582f93e452dbaa3ef59a9fbb44151818、headless232033＋MLP102688/原2220步。原Tri-SIG-TIME SSL F0 E50/11100/default6/232993、sourcecontract392005和DURA v1.0 R69/266951/E20/4940/reuseTrue/graphTrue/nativeFalse/FP32/TF32off祖先保持。helper复用local-cov中的父身份读取函数，不加载其局部协方差或任何U适配模型。

L6300旧六设备各1050、每TX×RX×day70，有旧TX真标签。OE6134只见outside-six成员和公共域，不读真实21设备身份或K。原F0 u160从已合规source feature缓存读取，ID/次序/metadata与同记录DURA P1 IQ核对；L/OE的z此前未保存，本轮新增12434条冻结signal前向，源F0/G真实新0。源RX1/3/4/6/8为1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。

源V27000（旧六每类4500）的u/z直接复用父NPZ，全部joint统计冻结后只校准两个新六tau，另两旧six-tau精确；hard12为overall＋5RX＋6scene，另外30交叉组完整诊断。源novel3150为19-15/19-16/7-1各1050，仅全部tau后预测，真值仅独立scorer。原Us56700本轮未读。L/OE/V/NV物理互斥，均已有历史源开发曝光。

跨RX clean/star Q均为RX24-1(index26)、2021-03-15的同864物理记录，旧六/未知三各96（576known/288unknown）。U384为同RX03-08、本轮未读；Q日期源已见，不称严格未见日期。父Q已保存z但缺u，因此本轮从原公共Q重新同DURA→原F0、两条件共G/F01728行，核同IDs/默认logits/closed与父一致；z逐值复用，Qsignal真实新0。clean也经过DURA，所有身份和闭集基线均保持父的同路径输出。重复处理不增加独立记录或新的盲测证据。

学习输入仍是既有G10 MIX01星地P1：600km/25MHz/2.45GHz/residual/post_sync/noEQ/refSNR30dB/reflect，郊区城区各.5及原六面积概率，receiver2027/scene2026103001/channel2026104001，一次输入unitRMS、输出不再RMS。clean仅冻结测试。U/Q不拟合、新目标TX/K/配额/正式RX24-18/未来query不读。

## 执行与验收

launcher首步实际F0/signal/G各2条synthetic no-query smoke并验证unit320，PASS立即继续；sourceL/OE配对导出与joint密度拟合→全部统计freeze→Vonly两新tau→四套tau全部freeze→sourceNV和两个Q完整预测→独立CUDAempty CPU scorer先核原控制/状态/四列raw与margin后各连接三truth一次。score不加载NN/重新fit/调tau/Utruth。即使sourceRU不足，也完成三类DURA预登记测试。

宏A_K/R_U/H为主，另报C_K、同输入原F0closed及损失、AUROC/OSCR、完整TX/RX/day/scene/42组及主对三个控制的配对gain/loss/net。gate为sourceknown12保护、cleanRU95/starRU90、CK90、原F0closed−AK≤2pp和所需主景覆盖；主starAK/RU均不得低于原predicted。high未知三类0/4/2缺19-15，保持N/A、不能认证六景，不补抽、改主或扫描超参。source同RX/历史NV和单开发RX/模型限制保持，不称正式晋级。无U适应/发现/REG，AN/HOW为N/A，完整目标不缩小。

新神经/optimizer更新0，新joint known90/OE32/normalizer1及源V阈值12标量实际拟合；源L/OE12434 signal和Q1728 F0/G真实前向，synthetic各2单列。原F0/r03/G训练、sourceF0缓存及父Qsignal成本继承；部署神经参数仍834665，新增联合聚合字节与旧state/tau分别实报，不部署逐样本source bank。时长报实际批处理，不假定B1延迟收益。

N607唯一owner/不可覆盖run root，每GPU新增本任务≤2且启动实际free≥4096MiB，不停止/修改/重启其他owner。一次实际代码commit、聚焦负测/真实CPsmoke/一次直接P0/P1审查、最小登记、单preflight/单archive SHA比较/remotecompile、一次PID-CWD-argv-GPU-log核实后自动score/report/Git收尾。精确环境/CWD/命令/路径/GPU/产物见experiment.json/report。非法来源/角色、非有限、输出碰撞或无法产生合法prediction时保留，不partial评分/自动重启；UNKNOWN先核原PID/root，科学负结果不停止健康工作。
