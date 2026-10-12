# 2026-10-12 身份证据报告落实追踪

用户要求继续按照新报告优化。输入：[完整报告](../reference_materials/identity_evidence_report_20261012.md)。本记录在新方案代码编辑前建立。性能目标仍为 clean 三种子均值 ≥84%，当前未达成。

|ID|来源|验收要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|R01|9.1|当前 conditional9 按原配置完整收尾、保留全部结果|原 run 的 collect/analyze/final report|verified|9/9 clean评分、39960完整源step、final_analysis.md|健康训练不干预、不重复启动|
|R02|9.1|新核心假设设置同代码路径、同学生更新预算CE/SSL对照|cvs_nocfo_foundation／engineering|verified|CE3＋固定24行全完成；119880完整源step／4536000独立clean评分|历史P0不能替代匹配控制；SSL预训练教师和U成本单列|
|R03|9.1|新身份输入不显式拼接 CFO；保留其他参考响应与补偿算法，明确边界|cvs_source_identity_readout/input_policy.py；后续训练模型|verified|冻结读出3seed＋新CE三行训练/预测同mask；完整13320源step及504000独立预测核验|历史 run 保留；屏蔽显式坐标不等于原始 IQ 无频偏|
|R04|9.2|冻结浅／中／最终实际表示，L6300 全量分解 TX/RX/交互/within|cvs_source_identity_readout/core.py|verified|分解恒等式、PSD、全样本覆盖|不套用虚拟梯度的能量比例|
|R05|9.2|收缩 PSD 分母的广义特征值，身份均值秩≤5；不删除全部交互|同上|verified|人工矩阵／类置换／退化条件检查|固定 alpha=beta=1、shrink=.1、eps；属于诊断|
|R06|9.2|L 内物理 ID 交叉拟合，方向估计／原型与读出样本独立|同上|verified|fit/query 无交集；每 PID 每方法一次|跨 RX 只检验读出迁移，编码器已见五源 RX|
|R07|9.3|比较合并 TX 原型与分别保留 RX 原型的读出|同上|verified|全部类别统一竞争、固定 tau|判断共同坐标假设，不由目标混淆指定 TX 权重|
|R08|9.3|关系训练保留CE；关系任务、身份头、类margin、修复／破坏分开评估|cvs_crossrx_relation|implemented|固定15行发布VERIFIED；CUDA5／严格自身checkpoint5／22200合成源step／独立收尾15；实际9行配置核验，自动测试进行中|原型参考eval/no_grad；只query梯度；未证明目标增益|
|R09|9.4|作用网络先固定编码器、已知有限参数，验证留出 TX/day/质量迁移及线性／低秩基线|有限作用预测诊断|deferred|下一独立作用预测阶段；须先证明作用迁移再约束身份|拟合改进和真实 RX 残差覆盖分别量化，不称不同 RX 严格反事实|
|R10|8.2、9.5|U 新监督优先同包已知操作／重复前导；先测辨识性、周期复现和不确定性|作用/前导诊断|deferred|与 R09 同阶段|不读取 U TX；不能把周期重复当新独立激励|
|R11|9.6|工程优化单独归因；额外算力单列，同路径控制|cvs_nocfo_engineering／foundation|verified|实际24行完整106560源step和4032000评分、实测资源、全部负结果|旧engineering18保持未发布；新最高均值79.8986%，未达84%|
|R12|全文|禁止目标 TX 特殊权重／目标反馈；全 seed 与负结果保留、报告准确名词|所有 config/report/scorer|verified|独立 P0/P1、truth-last 与完整结果|既有公开 benchmark 仅探索，不声称新盲测|

R04–R07 三种子源诊断已完成，300前向／37800 L累计读出，未访问target。无CFO CE三行已完成，后24行同路径SSL/工程队列已发布。新的训练矩阵统一去除显式 CFO，并设置同路径 CE/SSL 对照，按固定 final20 自动 clean。R08–R10 的顺序来自报告本身，不是新增许可门槛。

三种子读出实际结果：noCFO浅层raw merged46.8624%／stable5 merged62.5344%；中层64.0106%／73.2540%；最终94.2487%／95.4339%；原身份头已见L98.1376%。不能由源读出断言未知RX提升，最终头并未显示落后于这些原型。R03的功能诊断已验证，新CE三行训练/预测及后24行实际源配置已核验输入一致。

无显式CFO新CE三行已训练及独立clean评分完成，均值79.4891%（81.0369/77.0994/80.3310%）。后续同输入24行SSL/replay/六工程配置在读取该成绩前登记；不据此改参或挑教师。R03已由实际新训练和严格自身权重预测验证，R02仍等待完整SSL对照矩阵。

完整27行：CE3已评分；同输入SSL/replay/六工程24行发布VERIFIED（commit 68d9221c07cbb77543cf2a4fe9b28e604863edbc），8GPU真实总CUDA进程均≤3，所有已生成resolved配置核验。R02/R11/R12尚需完整24行固定评分/成本收尾才能转verified；R08–R10下一独立阶段，不由目标排名筛选。

R08独立跨RX关系15行已固定登记，浅中最终与final、合并与分RX、L30 CE控制；每4身份更新30独立L，无额外头，支持stopgrad仅query梯度。登记先于读取24行目标评分。R09/R10仍为后续有限作用/结构证据阶段。

R02/R11/R12实际验证收尾：同输入CE3＋SSL/成本/工程24全部完成，119880完整源更新、4536000独立clean评分；24源日志自动全解析无缺项/异常。最高均值79.8986%（SSL_WARM_LR3），主组合78.8252%，未达到84%。关系15行在读取成绩前固定，不据目标评分改参/挑种子/重跑。

当前验收计数：verified9、implemented1（R08）、deferred2（R09/R10）、rejected0、blocked0。R08已发布并排队，实际矩阵未完整完成，不能升级verified。后续源端设计见[有限作用与U证据](finite_action_next_stage.md)，仅设计未启动；最高风险仍是新增辅助知识能否迁移并实际保留身份判别。
