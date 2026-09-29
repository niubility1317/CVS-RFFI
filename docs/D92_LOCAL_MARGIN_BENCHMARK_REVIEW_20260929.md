# LocalMargin 重复基准入口独立 P0/P1 审查

2026-09-29。结论：本次限定代码范围内未发现未解决的 P0/P1 问题。结论只覆盖入口与共享集成的正确性，不表示 support 筛选通过、query 性能改善或已获启动许可。

## 范围与隔离

审查者为未编写本次预测入口、生成器、预检及共享集成的 `branch_local_core` 子任务。阅读范围为源代码和合成测试；未读取任何实际 query 分数、结果报告、索引、交接或预测产物。未运行测试、远程命令、配置生成器或实验。LocalMargin 数学求解器已有单独审查，本次不重复审查该求解器；本次审查者自己编写的三基线汇总工具不在此独立结论内，由主任务审查。

新增入口范围：

- [evaluate_d92_branch_local_margin.py](../tools/evaluate_d92_branch_local_margin.py)
- [prepare_d92_branch_local_margin_benchmark.py](../tools/prepare_d92_branch_local_margin_benchmark.py)
- [preflight_d92_branch_local_margin.py](../tools/preflight_d92_branch_local_margin.py)

共享集成范围为 [run_d92_confirmation.py](../tools/run_d92_confirmation.py)、[publish_d92_confirmation.py](../tools/publish_d92_confirmation.py)、[score_d92_confirmation.py](../tools/score_d92_confirmation.py)、[summarize_d92_confirmation.py](../tools/summarize_d92_confirmation.py) 和 [summarize_d92_repeated_benchmark.py](../tools/summarize_d92_repeated_benchmark.py) 的 LocalMargin 增量及其直接调用路径。

同时核对了缓存加载器 `export_d92_branch_features.load_features`、物理划分校验 `d92_orbit_feature_cache.validate_split` 和两组[预测入口合成测试](../tests/test_evaluate_d92_branch_local_margin.py)、[运行集成合成测试](../tests/test_d92_branch_local_margin_benchmark.py)。测试是否通过以主任务单独保存的执行证据为准，本次没有代跑或重复运行。

## 核实事项

| 事项 | 源代码观察与判断 |
|---|---|
| 当前物理 support 拟合 | 每个 split 独立调用固定 `fit_branch_local_margin`，只传入 support 索引下的五块特征、合法 support 标签、物理 ID 和全部注册类；不传入 query 特征、标签、预测或其他 split 的拟合状态。 |
| 单样本、全注册类预测 | 新状态固定后逐个 query 物理记录调用 `score`；输出列覆盖 `registered_classes`，按物理 class ID 处理平分。共享 scorer 同步注册相同 tie 规则。 |
| truth-last | 新入口不读取 truth。共享 runner 等全部四个模型行完成预测后才启动独立 scorer；scorer 先验证所有预测文件及物理 query ID/class/split 绑定，之后才打开 truth。 |
| Phase1 与继承来源 | 入口复用既有原始 received 五块缓存，不加载 checkpoint、不执行 encoder、不继承已拟合头。缓存加载器核对 SHA、model seed、capsule、类表、来源判定、scratch/epoch200、空继承链及冻结前无 target 访问，并把当前缓存绑定到原始行来源。 |
| 缓存输入边界 | 接受的 NPZ 成员集合固定，拒绝额外标签成员；capsule NPZ 只读取物理 ID，不读取 IQ。检查实际物理 ID 一致、float32 五块形状与有限值，返回不可写特征数组。 |
| 物理 K 与完整矩阵 | 划分校验拒绝 support/query 重叠、重复 ID、非法标签、query roles/truth/class-count 字段，并要求每注册类恰有 K 个物理 support。生成器继承完整 rx3/rx1、四模型、K1/5/10/20 和新增 0/2/5/10/20 矩阵；运行前核对预登记 split 数。 |
| 控制与缓存引用 | 原 D92 预测和 BranchRidge 原始特征生产者路径保持独立、不可重叠且逐模型绑定。生成器登记 LocalRidge 为直接比较基线、Interaction 为描述性参考；候选入口不消费它们的分数。后续汇总负责完成矩阵后的跨运行精确配对校验。 |
| 发布依赖与 CPU 执行 | allowlist 精确绑定方法名、目录、脚本和模式；发布闭包包含新预测脚本及冻结缓存加载依赖。LocalMargin 使用 CPU 缓存路径，未新增多视图导出、GPU、checkpoint 加载或基线重执行。 |
| 不覆盖与失败留痕 | 新输出目录及预测/日志文件使用拒绝覆盖语义。拟合失败保留核心异常审计，预测失败保留当前完整 fit 审计；旧 split 完整轨迹留在追加写磁盘文件，内存只保留标量摘要，避免跨 split 累积全部 sweep 数组。失败不写预测完成标记。 |
| 训练过程日志 | callback 逐 sweep 刷新 JSONL、文本及实测证书/核参数；结束或失败后生成 CSV。实际优化步数、sweep 数、耗时、常驻状态和 payload 分开记录，学习率与源验证缺失有明确理由。 |
| 汇总共享语义 | 新方法注册严格新类/H 改善；LocalRidge/Interaction 替代基线仅允许明确 `analysis_only`，保持 old-only 纳入全部单元旧类保护。没有增加全分层正向条件或独立样本声明。 |

## 结论的限制

预检只读已有 manifest、完成标记和缓存元数据，不替代运行时完整输入校验；它不证明将来的远程路径已发布或任务已启动。生成器只准备固定文档，不代表启动授权。现有健康 support 运行不受这些新增入口影响。

本次未发现需要阻断的源代码问题。实际合成测试、发布路径、support 完整分析及是否执行后续重复基准，由主任务按既定权限和唯一 launch owner 继续处理；不增加额外科学门槛。

## 主任务验证与补充审查

主任务统一运行数值环境测试：入口及共享兼容208项、三基线汇总58项、全拟合/全sweep审计58项，共324项通过。历史轨迹内存修正后，失败证据保留的2项相关测试再次通过。执行记录见当前support run的 `evidence/benchmark_tool_validation_20260929.json`；这些是合成正确性测试，不是性能改善证据。

主任务独立阅读三基线汇总器及成本审计器，未发现未解决的P0/P1问题。汇总先核对全部运行完成状态、完整矩阵与物理身份，再对同单元作差；主比较为LocalRidge。审计器不读取实际query分数或truth，逐fit/逐sweep核实终止证书、实测优化步数和JSONL/CSV/文本日志一致性，计算状态字节数与新增传输量分别报告。

代码已准备；尚未生成重复基准配置或启动该基准。当前support运行继续使用原冻结runtime，待完整support筛选后再决定下一步。
