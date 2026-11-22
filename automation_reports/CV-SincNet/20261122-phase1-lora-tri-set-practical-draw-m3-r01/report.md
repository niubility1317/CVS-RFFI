# LoRa Tri 候选集合的独立星地观测一致性

已冻结的原候选集合方法在三个种子的目标星地六景均为正。此次固定从相同 E140 监督权重与原硬伪标签策略出发，只改变无标签一致性与候选集合的学生波形。教师仍看原始 U，硬伪标签损失及跨 RX 供体分支仍用原始 U。同包的两个 practical 观测具有相同接收机、场景和对齐坐标，随机流独立。

一半 U 批位置尝试 practical 学生输入。两份观测均满足去啁啾相干度 0.40 时，按样本索引和批训练奇偶轮换。如果只有一份合格就使用这份，两份都弱时回到原始输入。学生使用原相位和噪声增强及交替 chirp 子集。原有权重、候选集合置信门槛、L/监督 practical CE、优化日程、每种子的 2464 更新及样本顺序不变。分别记录原 practical、第二份 practical、集合和硬标签用量。数据只用同合同源 IQ；U 的 TX 真值仅在冻结后由独立评分端读取。

三个种子预测相同 30000 个包、15 个未见接收机、clean 与六种 practical，新增 21 个单元。逐包比较原候选集合、原自适应 Tri、继续监督及 PL。比较 net corrections、场景及类别变化，以及候选集合正确率。发布不读取当前 DURA 新实验的目标成绩进行调参。

N607 普通账户、GPU0–7 按容量，一个本任务进程每卡，最多四个本任务 worker。CWD 为 `CV-SincNet/releases/20261122-phase1-lora-tri-set-practical-draw-m3-r01-release`，唯一输出 `CV-SincNet/runs/20261122-phase1-lora-tri-set-practical-draw-m3-r01`。沿用 `full_adaptive_publish`，传入原数据、冷监督、E140 对照、候选集合对照与第二份源 IQ 数据的 release receipt。真实 E140 权重的源烟测后启动。权重、物理顺序、预算、非有限或盲输出数量异常时保留产物并停止。待完整预测后用 `full_adaptive_score --controls ... --set-control ...` 生成独立目标和 U truth-last 结果。
