# D92 E0连续session实验实施计划

> 用户已批准直接执行。实现按TDD推进，每个生产行为先取得真实RED，再做最小GREEN；不扩建通用发布平台。

## Task 1：连续session核心

**文件：**

- 新建`code/cvsrffi/stage2_d92_continuous_session.py`
- 新建`tests/test_stage2_d92_continuous_session.py`

**RED：**锁定S1 Ledoit-Wolf桥接、S2至S4前缀公式、S5原始E0逐字节等价、顺序/行置换等变、未来support和重复token拒绝、每session一次FULL/一次codec、query零访问。

**GREEN：**实现冻结DA锚点、不可变SessionLedger、累计support规范化、前缀D92统计量和D42状态发布；不修改既有D92 E0允许类数或state schema。

## Task 2：truth-free预测与truth-last评分

**文件：**

- 新建`code/cvsrffi/stage2_d92_continuous_session_prediction.py`
- 新建`code/cvsrffi/stage2_d92_continuous_session_analysis.py`
- 新建对应测试文件

**RED：**锁定未来support不打开、预测不读truth、每session不可覆盖artifact、未注册truth不计分、truth/预测/token绑定漂移拒绝、终态batch/session严格配对。

**GREEN：**复用既有封存包加载、backbone、D81/D42计算和不可变NPZ/JSON写入；新分析器输出session轨迹、终态等价、资源和八项指标。

## Task 3：矩阵、CLI与N607发布

**文件：**

- 新建连续session矩阵模块、runner/analyzer CLI、配置和聚焦测试
- 新建`automation_reports/CV-SincNet/<immutable-run-id>/report.md`

**RED/GREEN：**锁定5 outer×3 scene×4 schedules、K10/new5、seed713106、receiver集合、`[5]`/`[1x5]`正反序/`[2,2,1]`、2MiB/150ms资源门、DA1生命周期命名。

## Task 4：发布与分析

1. 本地真实checkpoint truth-free smoke。
2. 独立P0/P1复核，必须`P0=0/P1=0`。
3. Git提交、最小预登记报告、不可覆盖run ID。
4. 交唯一N607 runner执行；失败不重试同一run。
5. 完整取回后运行truth-last分析，更新同一报告，给出连续化是否可用、性能轨迹、累计注册代价和星上负载结论。
