# PAMF Phase2 三池首批正式核心实验

2026-10-29。用户授权在三池数据制作完成并核验后直接启动，复用合规Phase1，每GPU最多2进程。当前本run只运行已固定PAMF/MILKR与两个机制对照；外部29链路/22版本设计完整保留，但不将未接入新episodic数据的外部方法写成已启动或用这些消融代替。

## 首批固定矩阵

|维度|实际设置|
|---|---|
|原始目标池|554,823条，P01/P02/P03＝184,936/184,944/184,943，池间互斥|
|池内重复|每池MIX01–08；LS/MU/LU分别01–05，共23组|
|目标RX|1-1、14-7、2-1、20-1、7-14、7-7、8-8，各自独立适应|
|K／N_new|K＝1/5/10/20；N＝0/2/5/10/20，原六旧类与原二十新类顺序|
|模型seed|2026092701/02/03/04，数据/信道不随模型seed重抽|
|输入|P0固定received；P1同一received经冻结DURA v1.0|
|算法|MILKR local_ridge为主；BranchRidge与InteractionRidge为预定机制对照|
|A|各K自身旧类query上的原默认头，无target support读取|
|B|旧6类当前K support闭式拟合，再对完整补集query全类判决|
|C|对当前6＋N类support独立重新拟合；不继承B核参数或声称参数协同|
|source|复用四个Tri-SIG-TIME SSL三视图E50/11100模型；L_s6300/U_s56700隐藏TX/V27000不变|
|DURA|复用固定R69 v1.0 Graph-RR、FP32/TF32off；不训练、不微调|

483个capsule×4seed＝1,932个配对P0/P1预测单元；每单元128行预测，合计247,296行。每单元有120次闭式support拟合，合计231,840次，均0优化器更新；源训练新增0。原始仿真观测12,760,929，重复seed/arm不增加物理记录。恢复结果由每capsule首seed建立后按相同合法行读取复用，不重复生成信道或恢复相同观测。

特征与恢复缓存按阶段延迟填充：A只计算本行query，B再开放实际support，C按N依次开放；未来类别不提前进入F/G，也不预先导出整个bank。缓存只保存固定逐样本函数值，不携带目标拟合/BN/原型状态，各K的B与各C重新拟合。完整received值和匿名ID与r10保持。

## 数据核验与运行

r10已独立回读DATA_COMPLETE：483bank、9660split全部制作完成，最小query176。此run在进入target F/G之前，对全部capsule核验物理池不重叠、跨episode池归属稳定、support标签/K/N嵌套、B/C旧support不变、完整query补集、每类query至少150、信道seed逐组不同、实际scene和仰角范围、residual/post_sync/no additional EQ/fs25MHz。MIX是已固定覆盖几何参考，不解释成实测卫星任务比例。

N607普通用户、既有CVS-RFFI Python，独占run/release。启动先执行一次真实source checkpoint无query smoke，再数据audit，随后P01/MIX01/RX1-1/seed2026092701的完整P0/P1/K/N单元。该固定复现单元预测闭合后直接展开其余预登记单元，不查看或以目标分数决定是否展开，不做候选选择或调参。新科学候选不因本复现授权自动获得全矩阵。

GPU0–7每卡最多2个可见GPU/待初始化进程，包含其他任务；启动前至少5000MiB可用显存。既有其他owner不停止、不改动。CPU最多2worker，其中独立scorer最多1。首次协议/数值/子进程或输出错误停止新派发，保留已有产物和健康在跑worker；不自动缩量/换权重/重试，不依分数停止。

每个capsule/seed的P0/P1、所有K/N/arm预测固定后，独立scorer才连接私有truth。主指标为class-macro accuracy/Macro-F1、旧新macro accuracy及H_macro，另报micro、class floor、class counts、day/scene和实际资源。三池先各自汇总池内随机组，再报告池间与模型seed变异，不把69组当物理独立数据集。

## 版本和证据

run：20261029-phase2-milkr-threepool-m4-r01。配置：configs/pamf_phase2_milkr_threepool_execution_20261029.json。精确命令、source/checkpoint路径、环境/CWD、六类seed、owner及输出见同run experiment.json/report.md。actual commit由full_publish从已推送Git HEAD写入runtime/submit，发布后回填记录；数据、checkpoint、原始日志及预测不入Git。

14项聚焦检查PASS；一次独立P0/P1审查PASS。直接N607 preflight与实际source/cache/checkpoint13个输入路径已核实，8卡当时均空闲，磁盘约2.1TB。旧source smoke已有PASS证据；本launcher仍以当前实际checkpoint无query smoke作为首步直接继续，不使用smoke许可文件。一次release归档SHA比较、远端编译后检查实际PID/CWD/argv、GPU和日志，提交成功不等于运行/实验完成。

完整外部方法的source物理契约审计已有36个匹配checkpoint可复用（8个主家族及旧Feature Separation辅助版，各4seed），不重复训练。它们的新episodic A/B/C适配与其他原生版本继续单独登记，不能将旧Feature Separation改名Hu2024，不能用PAMF-NCM充ProtoNet，不能用公共注册头掩盖原生CSIL/MoPC/ISSL机制。本核心run未启动这些外部行，后续接入不以核心目标分数挑选方法。
