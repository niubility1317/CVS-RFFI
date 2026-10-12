# 2026-10-12 身份证据报告落实追踪

用户要求继续按照新报告优化。输入：[完整报告](../reference_materials/identity_evidence_report_20261012.md)。本记录在新方案代码编辑前建立。性能目标仍为 clean 三种子均值 ≥84%，当前未达成。

|ID|来源|验收要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|R01|9.1|当前 conditional9 按原配置完整收尾、保留全部结果|原 run 的 collect/analyze/final report|verified|9/9 clean评分、39960完整源step、final_analysis.md|健康训练不干预、不重复启动|
|R02|9.1|新核心假设设置同代码路径、同学生更新预算 CE/SSL 对照|下一批无显式 CFO 训练矩阵|implemented|三seed无CFO CE本地CUDA/登记/独立审查；SSL尚未发布|历史 P0 另列，不能替代匹配控制|
|R03|9.1|新身份输入不显式拼接 CFO；保留其他参考响应与补偿算法，明确边界|cvs_source_identity_readout/input_policy.py；后续训练模型|implemented|冻结读出3seed及新CE CUDA slot严格零；等待运行配置|历史 run 保留；屏蔽显式坐标不等于原始 IQ 无频偏|
|R04|9.2|冻结浅／中／最终实际表示，L6300 全量分解 TX/RX/交互/within|cvs_source_identity_readout/core.py|verified|分解恒等式、PSD、全样本覆盖|不套用虚拟梯度的能量比例|
|R05|9.2|收缩 PSD 分母的广义特征值，身份均值秩≤5；不删除全部交互|同上|verified|人工矩阵／类置换／退化条件检查|固定 alpha=beta=1、shrink=.1、eps；属于诊断|
|R06|9.2|L 内物理 ID 交叉拟合，方向估计／原型与读出样本独立|同上|verified|fit/query 无交集；每 PID 每方法一次|跨 RX 只检验读出迁移，编码器已见五源 RX|
|R07|9.3|比较合并 TX 原型与分别保留 RX 原型的读出|同上|verified|全部类别统一竞争、固定 tau|判断共同坐标假设，不由目标混淆指定 TX 权重|
|R08|9.3|后续关系训练保留原 CE；关系任务与身份头、类 margin、修复／破坏分别评估|关系训练变体|deferred|待 R04–R07 结果|按报告先核验原型假设，再加入联合训练；不凭辅助损失下降宣称成功|
|R09|9.4|作用网络先固定编码器、已知有限参数，验证留出 TX/day/质量迁移及线性／低秩基线|有限作用预测诊断|deferred|待读出诊断落实后进入独立阶段|拟合改进和真实 RX 残差覆盖分别量化，不称不同 RX 严格反事实|
|R10|8.2、9.5|U 新监督优先同包已知操作／重复前导；先测辨识性、周期复现和不确定性|作用/前导诊断|deferred|与 R09 同阶段|不读取 U TX；不能把周期重复当新独立激励|
|R11|9.6|工程优化单独归因，实际额外算力另列；新矩阵统一输入并配同路径控制|cvs_scratch_engineering + 新无 CFO 矩阵|implemented|本地 CUDA/严格 checkpoint/日志分析 PASS|18 行旧输入版本尚未发布，保留实现和登记为未发布设计，避免按旧输入启动|
|R12|全文|禁止目标 TX 特殊权重／目标反馈；全 seed 与负结果保留、报告准确名词|所有 config/report/scorer|pending|独立 P0/P1、truth-last 与完整结果|既有公开 benchmark 仅探索，不声称新盲测|

R04–R07 三种子源诊断已完成，300前向／37800 L累计读出，未访问target。下一步发布无CFO CE三行，然后继续另外24行同路径SSL/工程对照。新的训练矩阵统一去除显式 CFO，并设置同路径 CE/SSL 对照，按固定 final20 自动 clean。R08–R10 的顺序来自报告本身，不是新增许可门槛。

三种子读出实际结果：noCFO浅层raw merged46.8624%／stable5 merged62.5344%；中层64.0106%／73.2540%；最终94.2487%／95.4339%；原身份头已见L98.1376%。不能由源读出断言未知RX提升，最终头并未显示落后于这些原型。R03的功能诊断已验证，下一批训练统一该输入仍需实际运行核验。
