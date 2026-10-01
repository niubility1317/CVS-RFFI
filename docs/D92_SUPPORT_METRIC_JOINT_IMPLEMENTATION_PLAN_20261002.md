# LocalRidge 与 support 度量微调的下一步实现范围

日期：2026-10-02。状态：独立模块及新入口/分析器/控制已冻结，191项相关合成用例通过；固定support配置READY，尚未启动新实验。现行 ProtoFrame 和 GroupBarrier 的健康 query 任务保持冻结。此计划不把数学设计写成性能提升。

继续保留 LocalRidge 优先：解析 Ridge、自由截距、新类 head、barrier gate 与完整隐式导数是主体；少量目标 support 监督训练的 adapter 只改变合法特征几何。Phase1 与地面原型固定，源样本、源逐样本特征、query 拟合和评分反馈均禁止。

## 1. 一条结构改动

现行方向在原型差坐标中使用 `G+I5` 和欧氏球，会受列尺度影响。下一候选拟用物理原型 Gram 与合法训练 support 的类均衡预测 Fisher：

\[
F=\sum_i\frac1{C n_{y_i}}J_i^\top
\bigl[\operatorname{diag}(p_i)-p_ip_i^\top\bigr]J_i,
\qquad M=B+F,
\]

\[
\min_{d^\top Md\le0.25}
g^\top d+\frac12d^\top(G+M)d.
\]

`G` 保留 RMS 类交叉熵的 score GGN 与 RMS 外层曲率；Fisher 不替代它。真实目标仍是原 RMSCE，不添加实际 proximal loss。完整 JVP 必须包含训练核和交叉核两端、Ridge 系数、自由截距、新头及 gate 阈值。

B 在零 anchor、C 在本路径实际 B anchor 构造并冻结该阶段度量。合法 inner-held 是 support 监督；诊断 outer-held 和全部 query 不进入度量、拟合或 trial 选择。新候选先评估完整的物理受限二次方向，固定从 `eta=1` 开始、最多12次折半，用同一真实 RMSCE 与 Armijo 回读缩短。这是新算法定义；原 ProtoFrame 的 `eta=.125` 不变。它不产生比例网格，也不由成绩选取。二次方向、有限 trial 和真实目标仍须分别记录，不能用二次模型下降代替真实目标下降。

## 2. 先实现两个独立模块

`d92_support_metric_basis.py` 仅读取调用者给定的已存 binary64 字典。它按固定列顺序精确判秩，删精确零而保留微小非零方向。精确有理 Gram–Schmidt、缩放和整数平方根包络用于产生正交物理基的浮点近似及几何误差证据。整数资源预算是明确数值资源输入，不从准确率选择；所有实际工作与返回状态分别计费。

`d92_support_metric_step.py` 仅读取调用者给定的梯度、GGN、物理 Gram、完整 score JVP、概率和合法 support 标签。它构造中心化 Fisher，白化 SPD 度量，在至多五维空间求有界球方向，再回读原物理坐标中的 KKT、可行性和互补残差。无 jitter、伪逆或小奇异值截断。失败保留已耗工作与异常，不默默换回其他候选。

两个模块不导入数据、head 或实验入口，不加载权重、不读取真实产物、不构造实验。子 Agent 只拥有各自模块及合成测试；root 统一执行数值验证和后续集成。

## 3. 集成时的新状态必须明确

若采用正交物理基 `U`，新的训练坐标属于 `U`，不能当作原 `Q` 的五坐标。未来候选必须使用独立方法名、状态 schema 和归档声明，拒绝加载旧 adapter 状态。B 从固定 Phase1 的零 adapter 开始，C 只继承本次路径自己的实际 B；不会加载当前 ProtoFrame 或历史目标适应状态。

理想基满足 `U.T@U=I_r`；实际浮点实现必须使用实际 `U.T@U` 作为 `B`，不能遗漏基误差。有效秩 `r≤5` 与名义最大五坐标分开记录。若复用现有固定五方向 primitive，可显式补零列，再仅在前 `r` 个有效方向解度量球。秩为零和没有合法 OOF 时不更新 adapter，解析 final head 仍按原职责拟合。

本独立候选的新U坐标接口已完成源码与合成验证。现有 [数值稿](D92_PROTO_FRAME_SUPPORT_METRIC_NUMERICS_NOTE_20261002.md) 保留写作时讨论的原五坐标并lift方案；两种状态不能混用，合成验证不等于实绩或完整head区间认证。

## 4. 证据和有限验证

先验证两个模块的解析低维例子、精确零与近秩亏、非正交换坐标、实际 Gram 非单位矩阵、Fisher 类均衡、输入不变及工作计费。后续 head 集成还须验证完整隐式梯度、实际 B→C、物理样本配对和所有注册类竞争。这些是受影响行为的直接正确性检查，不给当前健康实验增加审批或额外门槛。

浮点残差验证不等于原解析目标的严格区间认证。完整 Ridge/gate JVP 误差包络尚未建立；模块必须保留 `FLOAT64_SUBPROBLEM_DIAGNOSTIC_NOT_COMPLETE_HEAD_CERTIFICATE` 等明确证据层级。不能因正定、低维或 OOF 目标下降声称 query 风险下降。

若模块与集成验证完成，才登记一条固定 support 诊断候选；不生成参数网格。独立实验使用不可覆盖的新 run_id，记录完整来源、实际资源预算、唯一 root launch owner 与失败保留规则。既有数据只按 VALIDATED_ONCE 复用，不重验；任何数据/schema 或实际角色变化仍按现行协议处理。

每次结果继续报告同一物理旧类的 A/B/C、C 新类、旧6、K×新增类、gain/drop/absGap/H及真实成本。理想目标是改进方向，不是每轮硬门槛。完整 query 预测固定后才由原独立 scorer 一次性接 truth，结果不得回流参数、候选选择或重跑。

数学出处与独立推导见 [坐标尺度稿](D92_PROTO_FRAME_GGN_COORDINATE_SCALE_NOTE_20261002.md)、[原组合函数及计算稿](D92_LOCAL_RIDGE_JOINT_GENERALIZATION_COST_NOTE_20261002.md)。本计划没有精度、泛化、时延或性能优势承诺。

当前验证见[实现验证记录](D92_SUPPORT_METRIC_SYNTHETIC_VALIDATION_20261002.md)。实现阶段的有限浮点证据不改变此前数学稿的严格认证未完成边界；原数学稿保留写作时状态。

新入口与完整矩阵分析验证见[补充实现验证](D92_SUPPORT_METRIC_ENTRY_ANALYSIS_VALIDATION_20261002.md)；正式启动及成绩以对应实验记录读回为准。
