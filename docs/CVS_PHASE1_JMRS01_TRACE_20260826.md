# PHASE1_JMRS01设计—实现—验证追踪表

## 1.当前结论

PA-M2.1停止继续单机制深挖，PA只保留在冻结Core90原有物理分支中。JMRS01第一轮只回答：接收机校正、无符号多尺度谱商、相位创新中，哪些机制能在跨source receiver条件下保留TX身份并提供Core90互补证据。

用户于2026-08-26进一步明确：删除线性/对数差分谱比值。原因不是工程实现困难，而是当前WiSig输入没有可验证的发射符号，也没有“同符号、跨时刻、不同信道”的合法配对。JMRS01不会把单包邻频幅值差分冒充该物理方法。

## 2.逐项处置

|指导项|处置|落地/验证|
|---|---|---|
|停止新增PA sidecar、48码challenge codebook|接受|JMRS01不得导入PA matcher/codebook；Core90冻结|
|Core90保留原始IQ旁路|接受|M0直接读取冻结Core90输出，新分支不改写主路径|
|RC-Feature32|接受|R1：有界低秩特征残差、身份保持、32维表示|
|RC-Smooth16|接受并收窄|R2：DCT低秩幅相校正，只服务R2分支；不作任意逐频逆滤波|
|DSQ-A多尺度循环移位谱商|接受并改称无符号MS-DSQ|D1：shift=1/2/4/8、双向商、分母floor、mask、clip、coverage|
|DSQ-B线性/对数差分谱比值|删除|当前数据无已知符号；候选注册与CLI都拒绝`D2`|
|DSQ-C局部时序谱商|延期|未知局部窗口是否携带相同内容，首轮不使用|
|PI-1与PI-1+PI-2|接受|P1/P2：低阶趋势去除、幅度门控、多尺度统计；单独检查噪声放大|
|WLIQ|延期|首轮矩阵不扩宽；未来仅在RX泄漏可控时复议|
|同容量sham|接受|S1与机制头预算对齐，作为容量增益对照|
|早期三分支拼接|否定|S0独立训练；未过门槛不得进入S1/S2|
|端到端解冻Core90|否定|整个JMRS01-S0冻结Core90|
|统一“完全校正信号”|否定|Core90、R2、D1、P1/P2使用职责不同的视图|
|目标receiver校准|否定|Phase1严格source-only；target/query不参与训练、选模、阈值和gate|
|嵌套leave-one-source-receiver|接受|外层7折，每折held receiver仅作audit；其余receiver使用既有L_s/V_select/V_cal角色|
|单机制统一预算|接受|32维表示、头结构同级、可训练参数不超过5万、相同epoch/batch/optimizer|
|联合可靠性gate|延期到P4|S0不训练gate；S1/S2必须由前序门槛触发|

## 3.冻结S0矩阵

|Row|机制|输入|首轮状态|
|---|---|---|---|
|M0|Core90|原始IQ|冻结基线|
|R1|RC-Feature32|冻结`z_id`|实施|
|R2|RC-Smooth16|分支专用幅相校正视图|实施|
|D1|MS-DSQ|未知符号IQ频谱|实施|
|P1|MS-PI-1|一阶相位创新|实施|
|P2|MS-PI-12|一阶+二阶相位创新|实施|
|S1|Sham32|同容量确定性控制|实施|

候选集合严格等于`M0,R1,R2,D1,P1,P2,S1`。不存在D2、D3、I1或PA新分支。

## 4.统一证据与入池阈值

所有阈值在真实实验前冻结：

1. LORO macro-accuracy相对S1的receiver分组bootstrap 95%CI下界大于0。
2. receiver probe归一化泄漏相对对应未校正输入下降至少20%。
3. between-TX margin至少保留对应输入的90%。
4. 固定安全残差诊断的source clean下降不超过0.30个百分点。该诊断只用V_cal确定固定幅度，不训练联合gate。
5. 7个held source receiver中至少4个不劣于S1。
6. 相对Core90的oracle gain至少0.30个百分点。
7. 有效coverage至少30%，并报告Accuracy–Coverage和Utility–Coverage曲线，禁止仅报告高质量5%。
8. 正收益至少覆盖2个receiver、2个day且不少于2个LEO场景，不能由单一切片贡献。

单机制必须全部满足才进入S1。若不足两个机制通过，S1/S2/P4均不启动；低性能是科学否证，不是系统技术失败。

## 5.协议与安全边界

- 只使用`L_s、V_select、V_cal`和外层held source receiver audit；`U_s`不用于有监督机制训练。
- target receiver、target day、query、truth、角色反馈全部不可访问。
- 每个完成训练的S0行必须产生clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`四项逐场景结果。
- prediction完成后才由独立scorer连接truth。
- 新run ID和输出根不可覆盖；旧PA-M2.1 A/B/C永久只读。
- 禁止因中途准确率低停止；仅协议泄漏、错误split/receiver/scene、输出碰撞、错误checkout、确定性系统异常或prediction无法闭合可停止。

## 6.状态追踪

|阶段|状态|证据|
|---|---|---|
|设计纠偏|已完成|D2删除；S0收敛为7行|
|基线回归|已完成|现有相关测试全部通过|
|TDD实现|已完成|23项聚焦测试通过；RED/GREEN已记录|
|P0/P1审查|已完成|修复probe和身份几何跨fold坐标混用；定点复测通过|
|既有回归|已完成|CCOI/PA相关93项通过|
|本地shell检查|受限但不阻塞|Git Bash被错误路由到WSL，停止；待N607远端编译|
|真实checkpoint smoke|待完成|不得访问query|
|N607正式实验|待发布|新run、不可覆盖|
|独立评分与报告|待完成|prediction闭合后执行|
