# MILKR 宏 H 提升目标与当前证据

用户目标：固定 **Tri-SIG-TIME SSL** 基座，优化 MILKR，使宏 H 比匹配原版提高至少 5 个百分点；可用适当小矩阵，覆盖各 K/N。状态：**ACTIVE，未达成**。不得把实验技术完成写成科学目标完成。

## 固定范围

- F 为原完整三视图 SIG-TIME SSL（REPL全部机制、固定时序排列），原 seed2026092701，232993参数、源E50/11100；DURA v1.0保持原冻结身份。
- 每轮只用一个既有物理池的 MIX01，七个目标RX全部保留；K=1/5/10/20 × N=2/5/10/20，每单元全部注册类 query 补集。当前是有标签 K-shot support 的 Phase2 方法研发，不是零目标标签新类身份注册。
- 原版通过 r08 权威 jobs.json 连接 r04/r05 同 seed、同物理、同 split、同路径的冻结结果；不重新训练或重跑旧基线。
- 源L代理选择参数/规则。目标适应只消费当前 support；query始终逐样本测试，全部预登记预测固定后独立scorer才接真值。任何改进结论均按相同处理路径配对。
- 此基准已曝光，三池也不构成新盲测；不从目标成绩选择超参数、重排候选或选择性重跑。每张GPU最多两个新增训练/预测worker，其他owner保持。

## 三轮完成证据

三轮都完成了真实checkpoint无query smoke、针对性测试、每候选一次独立P0/P1审查、Git push后远端OID核实、一次release归档比较与远端compile、所属PID/CWD/argv/GPU/log核验、完整预测和独立truth-last评分。所有所属进程正常结束。新增F/G训练与更新均为0。

| run | 物理池 | 方法 | 宏H主S0Q0配对变化 | S1Q1变化 | 科学判定 |
|---|---|---|---:|---:|---|
| [semantic r01](../../automation_reports/CV-SincNet/20261031-phase2-milkr-semantic-kn-s2026092701-r01/report.md) | P01 | R1新核；R2默认头残差；R3继承；R4/R5双视图 | R1 +1.58pp；主R3 -2.24pp | R1 +1.87pp；R3 +0.45pp | 主目标失败，R3各16格均未+5pp |
| [support-CV r02](../../automation_reports/CV-SincNet/20261031-phase2-milkr-supportcv-kn-s2026092701-r02/report.md) | P02 | R6固定核网格＋支持条件LOO | +2.42pp | +1.68pp | 总均值未达+5pp，16格中1格达到 |
| [pooled-metric r03](../../automation_reports/CV-SincNet/20261031-phase2-milkr-pooledmetric-kn-s2026092701-r03/report.md) | P03 | R7核类内scatter度量＋精确scatter/centroid LOO移除 | +0.22pp | +0.97pp | 总均值未达+5pp，16格中1格达到 |

三个池不同，以上只是各轮对匹配原版的变化，不能把三个原始均值当成同row候选排名。严格S1Q0分别有明显改善（R1 +19.94pp、R6 +21.88pp、R7 +18.32pp），但该路线原来严重失配，不能拿它替代主S0Q0或同路径S1Q1的5pp目标。

每轮完整旧/新/H/F1/floor、四格及各K/N、逐RX值见原报告与同目录result_summary.json。共新增1960+560+560=3080行正式评分；多个行复用物理query，不是3080次独立物理试验。

## 实现与版本

- `experiments/pamf_phase2_practical/milkr/semantic_residual.py`：固定语义多尺度核、默认头残差和B函数继承。r01执行commit `01395832c4d3c1f116cf3f07f86354c9864d248a`。
- `milkr/support_calibrated.py`：24核×5ridge；K1源规则，K>1当前support条件LOO。r02执行commit `d6f505c227b98057a18960cad78f4d352f38ce6b`。
- `milkr/pooled_metric.py`：合并类内散度核Mahalanobis质心判决；LOO从scatter及centroid同时移除该物理样本。r03执行commit `d5d874cc73afd97f7a3d1dba7a5f4ade6524d251`。
- `milkr_screen.py` / `milkr_cv_screen.py`：源冻结→目标预测→独立评分。已释放快照不可覆盖、不可热改。

源码/测试/配置已在 `code/worktrees/standard_ce_baselines_20261010` 及发布承载面镜像；其他owner未提交更改保留。机器状态和完整私有产物不提交。根实验索引已登记三run，实际状态以各run证据为准。

## 下一步交接（未实现、未发布）

不要根据本轮query微调FFT、带宽、ridge或挑某个K/N。下一条待证伪机制是旧/新类分数的**支持标签校准**：旧新组注册类数不同，分类分数与宏H的组权重不同；可在支持LOO中选择一个全局旧类偏移，并在query前冻结。K1没有物理留出，应仅用源L伪旧/伪新组的多个类数比例选择固定规则，不能用本类唯一support的自拟合率当验证。

这只是机制假设，不是已实现方法，也不保证5pp。先在合法源L的既有十个代理episode检验偏移校准对held性能是否有稳定收益，再决定是否登记新小规模目标候选。目标端不读取真实query组别，不逐query路由；所有类别统一竞争。选择后端应依据源证据：现有源held平均H（K5/10/20）R6为0.84894/0.85690/0.85837，R7为0.83817/0.84858/0.84684；这些都是已见六源类代理，不能冒称真实新类结果。

续接时先读本文件、三个报告及对应配置；已完成run只读复用。未获5pp证据前保持goal active，不标complete，不将难度或低收益误记技术blocked。
