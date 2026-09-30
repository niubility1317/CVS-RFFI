# 注册机制诊断接入审阅

2026-09-30。范围为本次 evaluator、prepare、runner、summary、preflight、publisher、analyzer 及相应合成测试的直接 P0/P1 正确性。保持 query-blind；未读取真实 query、ABC、历史结果索引、根交接或当前 LocalMargin 指标。未运行测试、Conda、远端命令或实验。Residual8 暂停草稿未修改。独立纯分解函数此前由根任务审阅并验证 39 项测试，本次不自审该函数。

当前结论：本次发现的两处交付问题均已修复并读回确认；当前实现未发现未解决的 P0/P1。最后的配置一致性与资源汇总增量也已审阅。根任务统一执行的相关合成测试已通过。

## 已核对的正确性

- 每条 OOF/proxy 路径先划分 train/held。B0 仅用旧类 train，C0 用全部注册类 train；二者调用未修改的冻结 LocalRidge。held 特征只用于固定头评分，held 标签仅在分数产生后用于离线分解。
- C0 始终输出全注册类分数。旧类列限制仅是独立评价反事实，不进入部署预测，不比较或拼接 B0/C0 的原始分数尺度。
- old-only 路径复用 B0 的分数作为 C0，类顺序一致。真实 K1 在任何拟合前返回，无持出准确率、OOF 或 proxy。
- 物理 ID 排序后按每类位置模 3 分折；OOF 拼回每个持出 ID 一次后计算，不等权平均不等大小折。proxy 遍历全部 K 个锚点，先在 parent 内均值，再让 parent 等权。
- 固定四个模型/cohort row、160 parent，其中 40 个 K1、120 个 OOF parent、1,400 个 proxy anchor、1,760 条路径。N0 每路径一个头，其他规模两个头，因此完整执行应产生 3,168 次 head fit，完成检查必须与实际计数一致；Cholesky 次数独立按核心实际 0/1 记录。
- 支持缓存加载核对模型 seed、checkpoint/capsule、完整物理 ID/标签映射及 source-only provenance；不加载 query IQ、truth、checkpoint 或 encoder，不训练 adapter。
- evaluator 在加载缓存前核对声明的 base algorithm 与核心 `FROZEN_CONFIG` 一致。summary 汇总实际 parent/stage 耗时，核对 stage 数与 head-fit 数一致；并发工作量之和与运行墙钟分开报告，无法取得的 RSS 保留 null。
- runner 为两个 CPU lane、每 lane 两个 BLAS 线程。新输出必须不存在；失败保留日志和其他 lane，不自动重试。summary 在读取任何分数 trace 前检查全部完成标记和来源绑定，再逐条复算分解、物理折、全部 proxy 及成本计数。
- 本 pilot 仅包含每个 cohort 确定性选择的两个 RX×场景组合，即选中 receiver 的 `practical_high` 与 `practical_low_urban`。原 capsule 同时提供 `practical_mid`，但本 pilot 未测该场景；不得表述为覆盖全部三场景。

## 已解决的问题

1. **P1：分析包用整个 `code` 目录检查未提交状态，导致保留的暂停草稿阻断分析。** analyzer 初版将整个 `code` 纳入 `PATHS`；必须保留的未跟踪 Residual8 草稿因此会触发 `Uncommitted analysis paths`。作者已将 `tools/analyze_d92_registration_diagnostic.py` 第 12 至 19 行改为冻结 LocalRidge 的五个明确核心依赖及其工具闭包。已读回确认，暂停草稿无需修改、提交或删除。
2. **发布完整性：包内 summary 缺直接依赖。** publisher 初版包含本次 summary，却遗漏其直接导入的 `summarize_d92_branch_support_probe.py`。作者已在 `tools/publish_d92_registration_diagnostic.py` 第 20 行补齐。已读回确认，隔离发行包测试覆盖 runner、evaluator、summary 的 `--help` 导入链。

## 验证边界

根任务确认纯分解工具 39 项、编排 34 项通过。最终受影响 entry 测试退出 0；审阅者只读核对 `E:/type10-7/.codex_tmp/pytest_utf8_1790702898230981800.stdout` 的 12 个通过点，并核对当前测试文件的 12 个 `test_` 方法，实际为 12 项，非先前口头估计的 13 项。相关测试合计 85 项通过，本审阅者未重复执行。初次测试中的临时目录/重复写文件夹具错误已由作者修复，失败记录保留；这些修复未改变拟合逻辑。

实现审阅不等于真实运行完成。当前没有启动本诊断，没有得出注册机制性能结论，也没有新增 10/1/3 pp、全分层正向或任何自动晋级门槛。逐阶段耗时保留在拟合记录中；CPU support 诊断不能代替星载时延、功耗或真实 query 泛化证据。
