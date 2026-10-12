# 普通 cos 分类头：浅中层 TX 条件矩对齐预登记

用户再次限定每模型从头训练 30 轮，并要求常规 cos 分类头，便于 Phase2 重新拟合。本候选放弃 RX 专家分类头、分类头路由和专用集成读出。原已完成的多头实验及三成员集成保留为历史记录，不作为本候选结构或初始化；E30＋70 未获授权且不会执行。新研究只改变骨干的训练损失。

单 native 身份模型保留原完整时频、PA、参考响应、内部门控。判别为 30×cos(joint160,W6×160)，特征及类别权重均按 dim1/eps1e-4 归一化，无 margin、专家或路由，177025 可训练参数与纯 CE 相同。原融合属于 identity representation，分类权重可独立丢弃；本地实测同一 joint160 可接入新的 9 类 cos 权重，未使用 Phase2 数据。Phase1 CE 会训练普通 cos 权重，但 Phase2 不需要保留它。本次不改变 Phase2 权限或执行 Phase2。

训练限定对原 time_down（48×128）与 t2.pool（96×32）观察实际通道/时间图，计算每包各通道时间均值及总体标准差。描述向量用当前合法 L batch 的 detached 均值/总体标准差归一化，分母地板1e-4；无跨步统计状态。TX_CONDITIONAL 对每个 TX 的 source RX 描述均值作对齐，至少两个 RX 各有两个 batch occurrences 才形成一组；各 RX、各 TX、通道与两个层位等权。UNCONDITIONAL 控制按 RX 分组而不区分 TX，其余网络、初始化、采样、预算及系数完全相同。天数在合法训练源域内合并，不声称同一天配对；不声明统计量是纯 RX 或已获得接收机不变性。惩罚梯度实际只流向骨干，不流向 cos 类别权重。

前5轮只 CE，epochs6至15 的对齐系数线性从0.01升至0.1，16至30保持0.1。总损失 CE＋系数×两层平均对齐，raw/weighted/total、梯度范数及实际层位/groups/RX pairs 全步记录。warmup 不执行特征观察、对齐前向或辅助网络。推理只原 native forward，无标签/RX条件、batch归一化操作、附加分类器、持久 source statistic 或 query 更新。

原物理角色 L/U/V=7/63/30，训练池 TX 标注率10%；源 L6300/U56700/V27000。五个 source RX1/3/4/6/8 整机留出；每 row 四 RX fitL5040/heldRX原V5400。从零30×222=6660更新，每arm五row、总10模型66600更新。训练只 L clean、label_smoothing0、AdamW WD0.0001、coslr0.0002至0.000001、clip10、采样seedprimary+101。U、teacher、EMA、星地及其他增强关闭；held RX 连统计都不参与训练，原 V 无更新。固定final30后先保存预测，再sourceV评分。

资格固定：TX_CONDITIONAL 源均值至少比同 RX、同seed、同预算的合法 CE30 提高1个百分点，至少比 UNCONDITIONAL 提高0.5个百分点；分别至少4/5 RX非负，最差 RX 分别非劣1个百分点。匹配 CE30 五份已固定源产物仅用于对照，不用于初始化。全部10row完整后判断，不放宽、不按目标成绩调尺度或选epoch。通过则沿用本机制、普通cos单模型、全源从零30，三primaryseed2026092701/02/03，逐row固定后自动 clean＋六 practical 七视图各168000，prediction固定后独立truth-last；未通过不访问目标，保留全部负结果。84%为待证目标，既有benchmark已暴露，不称新盲测或真实卫星部署。

精确矩阵/commit/command/environment/CWD/path/GPU/owner/expectedartifacts见experiment.json。一次独立P0/P1、聚焦负测和真实合法checkpoint noquery smoke；原权重仅smoke、不初始化。一次实际Git提交，一次N607preflight、单archive SHA比对、remotecompile、postlaunch PID/CWD/cmdline/GPU/loggrowth。单不可覆盖run/release/output，总GPU实验进程≤3含其他owner/free≥12000MB；普通账号，不干预其他健康任务。仅预登记系统技术失败（权限、错误数据/输出碰撞、不可执行、不能产生合法预测、至少两row同指纹预测前技术失败）停止所属run，不因低分停止。旧额外设计/签名/hash审批为NONBLOCKING，不追加。

用户新上传优化报告后，主线转为跨RX元反馈；本候选未提交远端发布、未训练、未访问target，记为DEFERRED_NOT_LAUNCHED备选。
