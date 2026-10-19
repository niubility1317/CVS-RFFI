# 纯 F E20 同轮 As 重访：冻结源拟合诊断

状态：PLANNED。当前M正在健康训练；本诊断等待其固定源判定、FINAL20与source qualification完整后运行。完整矩阵、seed、命令、环境和路径以experiment.json为准。

目的：区分训练过程中重复命中时的在线正确率、冻结FINAL20对首次缓存观测的拟合、同一L新信道与源V新物理观测。0训练/optimizer/EMA/默认头拟合，0正式IQ、prediction或truth读取。此诊断不新增晋级门槛，也不救援已固定的源FAIL；是否追加E50仅用原事前两项源规则。

固定复用DR的180个L physicalID及原B1 E20实际step/slot，不换样本。完整关闭的mechanism/replay记录与训练序列用于恢复M实际首次缓存source_step/source_slot/namespace/scene，先核对L角色再取IQ，按原Config/session/seed重建。没有保存历史As IQ，不宣称历史字节一致。

M对raw180、原B1训练命名观测180、同L两新draw×三场景1080进行1440包前向；原seen与M缓存origin不同的d包才补M和B1各d。完全相同的origin直接复用M seen和旧DR B1 seen logits。另一次真实M checkpoint无query源smoke1，实际预算1441+2d≤1801；新V NN为0，复用source qualification及fullV27000完成指标。B0/B1其他既有读出均不重算，若d=0不加载B1模型。

分别报告标准默认logit CE与训练smoothing0.01 CE、正确数/样本数、同IQ模型配对修正/新增错误、raw180正确集合保持及V1800 clean正确集合保持。训练流与缓存流可能scene组成不同，仅在相同IQ/场景子集做配对；不把不同组成子集差写成机制净效应。180样本不冒充L6300，V1800不冒充fullV27000；两个信道draw不冒充两个模型seed。

保持固定纯F三路SIG_TIME/统一Sinc、实际joint160和默认cosine头，读取各自合法scratch FINAL20，仅eval/no_grad且参数冻结；前后权重/缓冲/梯度状态不变。M固定source decision先落盘，所有source completion和qualification准备好后才开始，未使用正式分数选择checkpoint或诊断面板。

普通N607，GPU6一个进程，包含其他owner每GPU最多2进程且启动前至少12000MB空闲，复用共享锁，CPU2、FP32/TF32off。CWD为原M不可变release，诊断代码在独立新release，输出为独占新run根。唯一launch owner codex/root/F-epochReplay-sourceFit20；任何角色/来源、非有限、缓存origin、模型状态或输出矛盾保留现场，禁止自动重试或干预健康进程。

代码完成后执行窄负测/编译、一次独立P0/P1直接缺陷审查、一个实际代码/config Git commit、一个归档一次传输比较、远端compile/预检。launcher首步真实checkpoint source smoke PASS即继续，不建立新签名/hash链或重复审查。未启动前不宣称后台任务或结果。

本地静态与窄测 PASS：py_compile/AST/whitespace，合成4440闭合step与180首次origin，4个角色/非法pid/缺步/错误origin负例，scene组成分别统计、配对修正/新错/CE和origin复用mask。0源IQ、0checkpoint、0NN、0更新；真实模型source smoke留作启动首步。

唯一一次独立P0/P1 delta审查 PASS，未见直接缺陷。审查覆盖L角色beforeIQ、完整origin恢复、同origin复用与d0不加载B1、1441+2d实际encoder计数、冻结state/grad、源判定前置、0目标/0新V/0更新、输出独占及GPU6共享容量。没有重审既有E20候选。
