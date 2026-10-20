# R39：压缩恢复器的 F 独立源域 TX 关联诊断

固定比较原 source-selected R17、R38 dense_three_quarter 和 dense_half；不读正式测试成绩、候选排名或 target 完成状态，不改变任何训练、筛选或 R38 自动七场景正式测试。纯 CPU、2 线程、0 GPU、0 新神经网络预测/训练、0 checkpoint/F 权重或输出加载、0 target 访问。

复用已经完成并独立核实的 R34：原 R07 LU 前 8192 条的固定解析均值/标准差、源 V1800 的 raw 和原 R17 特征。没有新校准或 V 拟合。对两版全部七源视图的已冻结恢复 IQ 计算相同解析坐标：mirror64、spectral4、reference24、combined92。共 112 个单元，56 复用、56 新提取；两版新增解析变换输入 25200 条，不是模型前向或新增独立物理样本。

使用既有数学、固定 cap2 同 TX/RX/day 参考，每个待分析样本只与 RX 和 day 都不同的参考单元比较；六 TX 完整非零单元等权、其他五 TX 等权。独立 CPU score 在全部特征单元固定、有限且物理 ID 对齐后才打开源 V TX 标签。V 标签用于同 TX/异 TX 统计比较，不参与恢复器优化，也不拟合分类头。199 次 RX/day 内 query 标签置换固定种子 2026102034，参考库不变；这是条件描述性阴性对照，不是 p 值或多 seed 证据。

保留四种坐标、七视图、全部 TX/RX/day 和原 clean 的信息量，无结果导向删组。报告 1260 组关联、252 组条件阴性对照、84 组固定候选差；不产生晋级阈值或选中候选。mean margin、正 margin 百分比不是分类准确率。解析坐标也可能含 RX、载荷、采集日等成分；即使关联提高，也不单独证明每个生成细节属于输入 TX、未知 TX 泛化或在轨有效性。

新 root、固定 Git 代码，N607 普通账户。首步两条合法 L_s IQ 的参数自由无 query smoke；本方法不加载 checkpoint，实际 checkpoint/F50 smoke 由 R38 训练发布承担。等待两版 source_complete 和 E20/4940 元数据后运行；不读取源准确率和 target 分数。与 R38 共用 score.lock，extract/独立 score 顺序执行，避免增加本任务 CPU scorer 并发。父训练或输入技术失败时保留输出、不自动重试，不停止健康 R38 或其他 owner。

发布最小记录包括一次聚焦协议负例、真实源 IQ smoke、一次直接 P0/P1 审查、Git commit、矩阵/命令/CWD/路径/环境/停止规则、一次普通账户预检、归档两端单 SHA 比对、远端编译，以及 PID/CWD/argv/日志/产物读回。源角色和物理划分沿用原 READY/EXACT_MATCH，不新增数据重验或哈希链。执行状态和实际结果见同名实验登记；当前尚无 R38 独立解析关联结果。
