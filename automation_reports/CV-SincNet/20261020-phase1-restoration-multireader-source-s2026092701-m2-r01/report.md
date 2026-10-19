# 三套冻结原 F 约束的恢复实验

状态：ANALYZED / VERIFIED。用户要求 low-urban 超过 70%，其他五种 practical 接近 clean；“接近”暂按不超过原 F clean 2 个百分点解释，仍逐环境报告精确差值。评价分为恢复 IQ 与同条 clean 的差距、接入冻结原 F/default head 后的实际目标识别，两项不互相替代。

## 方法和源依据

前一源实验的 joint_posterior 在同一 1800 条源 V 上达到 low-urban 85.888889%、mid-urban 93.333333%、clean 98.444444%，但其约束来自单一 F1。新假设是三套原 F 的共同约束可减少单一识别器的盲点。此假设和候选在正在运行的前一目标测试得分被读取前写入私有源计划；不依据那次目标得分设置权重或筛选次序。前一目标正负结果均保留，不声称新的独立盲测。

teacher_mean、teacher_worst 均从原 R03 coarse20/correction20 独立初始化，联合优化 coarse 和 correction；原 F50 CROSS_RX 的三个 seed 2026092701/02/03、默认联合训练头全部冻结。每位 teacher 对每条恢复 IQ 分别计算 cosine feature distance + 0.1 * T² KL(T=2)，与其同条 clean 特征/后验比较。teacher_mean 平均三个标量 loss，teacher_worst 取逐样本最大 loss；绝不混合独立 embedding 坐标。共同波形项为 IQ NMSE +0.1 log power。

原 R02 前十轮合法源 L/U 缓存、路由 .15 clean/.60 residual/.25 full、物理顺序、paired observed→clean/clean→clean 等权、模型 seed2026092701/data seed2026101951 保持。未生成的 clean 槽不读取 damaged。新增 E10/2470、batch256、reset AdamW3e-5/WD1e-4、warmup247、cosine至3e-6、clip1、FP32。每臂曝光 L63000/U567000、IQ1260000；组件各至30轮、合计60组件 epoch、继承串行40+联合10=50 pass。复用已完成单 teacher E10 源结果作精确预算参考，不重复其训练；三 teacher 增加的实际时间/显存单独报告。原 F 保持50轮且不拟合新头；恢复训练不读取 L/U TX 真值，因此是识别器引导的自监督恢复。

## 冻结选择和自动目标收尾

源评分继续用相同固定 F1、原 1800 条 V、clean+六 practical，全部源预测固定后独立 CPU 评分。相对完成的单 teacher joint_posterior：low/mid-urban 各至少 +0.5 个百分点，其他四 practical 最多下降0.5个百分点；相对 raw clean 降幅≤0.1个百分点。六场景平均 IQ NMSE≤1.05×原 correction-only feature 参考0.028326367493718863，不能在 joint 参考上再复利放宽。按通过者的 low/mid 平均准确率最高选择，完全并列取较低NMSE，再取登记臂顺序。两臂均不通过时不访问新目标。

仅选中一臂自动预测原不可变168000条 clean+六 practical：high、mid、low_suburban、high_urban、mid_urban、low_urban。三套原 F 各自输出 argmax，不投票。全部七视图/所有恢复 IQ/物理 ID 固定后，独立无 CUDA CPU scorer 才连接目标 TX/RX/day 与评分 clean，报告accuracy/Macro-F1、净修复、逐F/TX/RX/day、距原clean差值，以及 full256/prefix232/tail24 未作 oracle 对齐的 NMSE 和 log FFT power 差距。practical 推断只见当前 received IQ，不输入 clean、目标真值、其他视图或参数更新。一个恢复器训练种子、三个既有 F 读出，不能当作三个恢复器 seed。

## 发布、资源和停止

26 项聚焦检查通过；一次独立 P0/P1 审查 PASS，无发现。代码/配置以一个 Git commit 固定，真实 R03+三个原 F 的两条 L_s no-query smoke 是 launcher 第一步，PASS 立即继续。核实两个恢复模块有限非零梯度、三个 teacher 全冻结、默认头梯度可回传到 IQ、新权重往返一致，0 query/0优化步。

N607 普通账号，GPU0/1 分别 teacher_mean/teacher_worst，每卡最多一条本 run worker，总 owner 数<2、空闲≥18GB才获取训练容量。CPU2线程、独立 scorer 最多1个；两源 worker 完成退出后在所选臂对应卡启动一个 target worker。保留其他 owner 和不可覆盖产物。单次预检、发布归档 SHA 比较、远端compile、实际 PID/CWD/cmdline/GPU/日志增长保存。协议、来源、commit、预算、输出冲突、OOM、非有限或预测失败时保留产物并停止对应技术失败，不自动重试，不修改健康任务。命令、路径、矩阵和预期产物见 experiment.json。


## 启动核验

方法commit：`286efa5bd71dcac5d108add346be40059e8ffa1c`。26项聚焦检查通过；一次独立P0/P1审查PASS，无发现。一次发布归档比较及远端compile/import通过。真实R03＋三套原F的两条L_s smoke证实coarse和correction均有非零有限梯度、三套F均无梯度、新权重往返一致，0目标访问、0更新。PASS后两臂直接继续训练。

独立后态已核实两条实际worker的PID/CWD/完整命令、物理GPU0/1归属和从0到首轮的日志增长。当前进度：

- teacher_mean：1/10轮、247/2470步。
- teacher_worst：1/10轮、247/2470步。

两个源参考特征缓存均为63000条L/U×三套F独立160维坐标，没有TX真值读取。当前没有源V评分或新目标性能结论；只有固定源规则选中的通过臂会自动进入目标七视图／三原F验证。


## 完整源结果与固定规则

两个新臂均完成E10/2470，L/U优化曝光63000/567000、IQ1260000。原R03的coarse/correction各由20至30轮，共60组件epoch；三套F及原默认头更新为0。以下均为同一1800条V、原F1的真实TX评分，不是目标准确率。

|源场景|原IQ|单teacher联合E10参考|三teacher平均E10|三teacher最差项E10|
|---|---:|---:|---:|---:|
|clean|98.500000%|98.444444%|98.500000%|98.500000%|
|high|54.277778%|98.388889%|98.222222%|98.277778%|
|mid|51.722222%|97.555556%|97.555556%|97.611111%|
|low-suburban|51.000000%|96.944444%|96.777778%|97.000000%|
|high-urban|51.000000%|97.500000%|97.611111%|97.777778%|
|mid-urban|43.388889%|93.333333%|93.277778%|93.388889%|
|low-urban|36.666667%|85.888889%|85.777778%|85.444444%|

|候选|low-urban改善（百分点）|mid-urban改善（百分点）|其他四环境最差改善（百分点）|clean降幅（百分点）|六环境平均NMSE|源资格|
|---|---:|---:|---:|---:|---:|---|
|teacher_mean|-0.111111|-0.055556|-0.166667|0.000000|0.02930071|未通过|
|teacher_worst|-0.444444|+0.055556|-0.111111|0.000000|0.02977491|未通过|

固定波形参考为原correction-only feature的六环境平均NMSE 0.02832637，上限为其1.05倍，不以新的联合参考复利放宽。资格要求两urban各改善≥0.5个百分点、其他四环境下降≤0.5个百分点、原clean下降≤0.1个百分点，同时满足波形上限；通过者取urban平均最高、NMSE较低、登记次序。

两候选均未通过既定源规则，未创建target目录，也没有新的目标IQ或truth访问。负结果完整保留；未测目标不能写成目标负收益，也不能由源指标推断目标改善。此前实际目标low-urban最好65.309127%，仍未达到用户要求。

## 成本和验证

- teacher_mean：训练1105.090 s，含准备/容量等待1122.647 s，可训练参数2140840，峰值显存5537224704 bytes。
- teacher_worst：训练1100.780 s，含准备/容量等待1119.026 s，可训练参数2140840，峰值显存5537224704 bytes。

复用完成的单teacher E10源参考，不重复训练；初始化、物理顺序、路由、组件预算及优化器与其匹配，三teacher增加的计算成本由实际耗时/显存体现。两个新臂之间仅teacher标量loss聚合方式不同。训练日志中的feature/posterior项均是三个teacher的均值；teacher_worst总loss使用逐样本最大组合项，不把均值项冒充总loss的相加分解。

完整10轮和全部stdout已解析，未见技术错误、NaN/Inf或OOM。26项聚焦检查、一次独立P0/P1审查PASS；真实R03+三套原F两条Ls no-query smoke、两个恢复模块梯度、全部F冻结、默认头到IQ的梯度、新权重往返、一次发布归档比较、远端compile、PID/CWD/完整命令/GPU/日志增长均有实际证据。未另训头、未读取L/U TX真值、未用目标训练或选模、未热改健康任务、未联系另一对话。原始IQ/预测/权重/日志保留在私有输出，不提交Git。

每个源row全部七视图预测后才接入V真值；若有选定目标row，全部七视图和三F决策固定后才独立CPU接入目标truth及评分clean。目标基准已暴露，不能声称新盲测、三恢复器种子、在轨性能或整个优化目标完成。
