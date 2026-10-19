# 实际运行状态

RUNNING／部分预测固定，尚无本轮最终性能结论。

|row|GPU／PID|训练|七视图预测|独立评分|
|---|---|---|---|---|
|NOAS_ADAPTER|原GPU6／2318182已正常退出|10/10，累计E30|固定完成|等待源决定|
|S1|原GPU7／2318268已正常退出|10/10，累计E30|固定完成|等待源决定|
|S2|原GPU4／2318352已正常退出|10/10，累计E30|固定完成|等待源决定|
|S3|GPU6／2333275真实存活|1/10，正在追加本段|未完成|未开始|

Dispatcher2318162与S3 PID/CWD/完整argv/CUDA/GPU均核实；S3真实父CP smoke PASS，source日志增长且已有第一轮。前三组的完成状态依据completion、prediction/complete与原进程终止核实，不因PID消失判为失败。无failure，无整组source_matrix_decision，未读取本轮新正式分数。

前三组实际记录：各2220新增optimizer更新、累计6660；4256新模块参数，原F206状态与默认头冻结，Sinc更新0、head_fit0。L前向NoAs284160／As568320，U567000，Tc/Ts各284160；NoAs学生As为0，其284160生成As仅teacher shadow。8新raw状态属于当前forward IQ，不叠旧局部adapter，不额外原F前向。训练raw/As在线正确数仅是动态权重／Dropout读出，不能冒充冻结拟合或目标测试。

实际代码/config commit：`28d14761575b080077fde038cc96baedd4d982cc`；归档运行commit：`6ea2f71f6654f4ceeadd460905a9d719b5dc1b89`。发布包SHA比较与remotecompile VERIFIED，先前push读回一致。后续状态元数据不修改运行代码。

整组S3终轮源资格完成后，先固定预登记3条件，再由独立CPU scorer连接truth；七视图prediction固定是必要前置。旧6row控制直接复用，任何中途source或测试结果都不热改参数、预算或样本。完整日志分析器与报告渲染器已准备好，等待四组完整产物，不生成伪完整报告。

最新只读证据：private progress010.json、first3_source_completion.json。运行root/命令/owner与固定矩阵见experiment.json。
