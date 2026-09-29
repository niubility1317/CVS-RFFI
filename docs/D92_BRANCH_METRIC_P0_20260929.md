# D92-BranchMetric-v1 独立 P0/P1 审查

日期：2026-09-29。结论：**两项 P1 已定点修复并核对，当前审查范围内无未解决 P0/P1。** 此结论是代码、配置与合成验证的正确性审查，不是运行完成或性能结论，也不替代 launch owner 的发布后读回。

审查者：`/root/source_aux_feasibility`。本轮核心、入口、汇总和编排由其他 owner 实现；审查者仅新增本文。未读取任何历史/当前 query 成绩、正式结果或结果型 handoff，未运行真实数据实验、远端命令、checkpoint 加载、源数据访问、镜像或提交。设计文档引用的证据属于已授权 support-only 诊断，未将它解释为 query 保证。

## 范围与已修复发现

审查范围为 `code/cvsrffi/d92_branch_metric.py`、冻结算法及双 cohort 配置、`docs/D92_NEXT_SUPPORT_METHOD_20260929.md`、`tools/evaluate_d92_branch_metric_probe.py`、`tools/summarize_d92_branch_metric_probe.py`，以及本轮 run/prepare/preflight/publish/analyze 编排。沿调用链核对既有纯 support reader、交互核中心化和计数 helper；没有因审查读取真实 feature 数组。

| 级别 | 发现与影响 | 修复后核对 |
|---|---|---|
| P1，已关闭 | analysis release 最初未包含新 summary 导入的 evaluator 及其 support reader/exporter 依赖；隔离发布目录会在启动汇总时缺模块 | `tools/analyze_d92_branch_metric_probe.py:10` 的 PATHS 已补齐 evaluator、support reader、exporter 和 `cvs_native_artifacts.py`。新增测试仅复制发布集合，在独立临时目录用 `python -I` 执行 summary `--help`；owner 报告通过 |
| P1，已关闭 | 继承分析 transport 下载后读取 `coverage.episodes`，新汇总最初仅输出 `parent_episodes`，将完成下载后的核验变成 KeyError | `tools/summarize_d92_branch_metric_probe.py:287` 显式输出 `episodes=parent_episodes` 兼容别名，解释文字说明不重复计数；最终汇总测试期望值包含两键 |

上述修复只涉及发布依赖和输出兼容，不改变核函数、散度、正则、预测或支持筛选规则。已按 owner 修复后的最终文件定点核对，没有追加重复审查轮次。

## 数学与信息边界

- **固定表示和唯一公式一致。** 核为 `KB+KA+KB*KA`，B/A 的单观测归一化沿用原交互实现；无新增 view、核权重或参数网格。`S_W` 为物理 support 类内残差外积之和，正则固定 1，不将 sum 偷换成 mean。
- **dual 与最近中心判别一致。** `d92_branch_metric.py:44` 通过按训练类别减均值实现 R，计算 `Q=(I+RGcR)^(-1)RGcP`、`V=P-RQ`；`:81` 保留 `b=-0.5*diag(P^T Gc V)`。它对应 `(I+S_W)^(-1)` 下的共同度量最近中心，省略项仅为所有类相同的 query 二次项。类中心范数截距没有丢失。
- **K1 为精确 NCM 退化。** `:55` 在 K1 不分解或估计类内变异，candidate 与 `kernel_ncm` 走相同算术路径，V=P；两臂 score 差要求精确为 0。零/恒定特征有精确平局路径，物理 class ID 字典序打破平局。真实 K1 的 probe 在 `:241` 起直接返回数值诊断，OOF/proxy 均为 null，不把视图或其它 row 当验证样本。
- **表示拟合无 held 泄漏。** 逐样本固定归一化可以在拆分前执行，因其没有全样本统计。每个 OOF fold/proxy trial 都在 `:192` 使用 keep 子集重建 kernel、全局中心、类内残差、P、V/b；held 特征只进入该冻结状态的评分。old membership 仅用于报告指标，不参与拟合。
- **query 逐条且状态只读。** 公共 state 的 score 沿用逐样本核映射；每条样本只与固定 support 比较，不依赖其它 query。状态数组复制后只读，类顺序只影响输出列。诊断入口本轮根本不加载 query 或生成正式 query 预测。
- **无 ground/source 路径。** `source_inputs/summary_inputs/query_fit` 均为 false。当前协议 §5.3.1 禁止普通原型在线拟合协方差/持久头，§5.3.2 的冻结摘要权限也未在本候选中启用。当前 scatter 全部来自本 row 合法 target support。

## 物理隔离与完整计数

`tools/evaluate_d92_branch_metric_probe.py:38` 使用冻结算法配置，借用既有 reader 仅校验缓存契约。`tools/evaluate_d92_branch_support_probe.py:59` 校验 capsule manifest、SHA/model seed、source-only scratch final200 provenance、缓存单原始 view/frozen 状态，以及 support plan 的精确字段、物理 ID/index/label 对应和全矩阵覆盖。读取集合是既有 support-only cache 与 capsule manifest，没有重新读取 received IQ、query truth 或源样本。

核心每类按 physical ID 排序后，标准 OOF 使用位置模 `min(K,3)`；proxy 对 parent K 的每个排序位置恰选一次。每 trial 训练 K=1，held K=parent K−1；训练与 held 的并集恰为当前 parent support，不能借用全缓存中其它 row 的合法 support。所有类使用同一规则，未按成绩挑 anchor。

完整预登记矩阵为 8 model/cohort rows、4800 parent episodes，其中 1200 个真实 K1 仅数值诊断，3600 个标准 OOF parent。K5/10/20 分别穷尽 5/10/20 anchors，总计 **42000 个 proxy trials**；每臂 held occurrence 共 7879200，但这些复用物理记录，不能当独立样本数量。标准两项分解通常每 fold 各一次，加 proxy 的 ridge 一次，总上界 63600；恒定训练特征会减少 metric 分解，代码逐 stage 累计实际调用数，未把上界预填为实测。

`tools/summarize_d92_branch_metric_probe.py:81` 重新核对每类 fold 分配、每 trial 的准确 train/held 集合、parent/train/held K、全部 anchors 与计数。proxy 先在 parent 内平均全部 anchors，再对 parent 任务等权汇总；真实 K1 不进入任何 held 指标。注册类数与旧/新角色仅来自合法 support plan，不使用 query 类别集合。

## 汇总、筛选与资源口径

- `:27`、`:44` 从每臂 confusion、每类 NLL sum/count 重新计算 accuracy、macro、old/new/H/NLL；标准 OOF 还核对逐 physical 记录。proxy 保存紧凑计数，不重复数百万条 held 记录。
- 四格 paired count 与左右正确率、差值交叉核对。精确 NCM 等价为真时，两臂 confusion 与 classwise NLL sum 也核对相等。未校准 softmax NLL 仅作描述，不用于温度、方法选择或 query 保证。
- `:179` 按预登记分别检查 parent K5/10/20：标准 OOF 对 ridge 和 NCM 两个 control 均要求 new/H 正差且 old≥−0.01；proxy 对 ridge 使用同一护栏，并要求 candidate=NCM。new/H 只在 new-present 任务定义，old-only 分列。某一 K 的失败不能被其它 K 的平均掩盖，不自动晋级或替换正式方法。
- 顶层数值统计包含独立 `parent_numerical` 层，故真实 K1 的数值信息没有被丢弃或假冒 OOF。
- 实际 solve objective、梯度/normal-equation 残差、分解维度/次数、耗时和状态字节均来自执行路径。metric 二次求解目标可为负，日志明确它不是监督分类平方误差。学习率/epoch 为 null，optimizer 为 0。
- 当前 probe 不持久化部署 head。公共 fit 的数值状态为 `8*(N*(736+C+2)+C+2)` B，包含当前合法 support B/A、V、中心化向量、b 和两标量；不包含 Python/registry/audit 开销，不能误报为仅 736 维线性 W 的大小。

## 配置、发布和证据

预登记固定两个 cohort，四个 model seeds 2026092701 至 2026092704，沿用相同 raw support cache、capsule、SHA 和六类 seed 角色。配置记录 parent commit；publisher 记录实际已 push 的 release commit。`run_d92_branch_metric_probe.py:13` 检查八行、model/cohort 对、输出根目录、数据绑定、CPU4 lanes×2 BLAS、无 GPU；新输出独占，失败只保留所属 lane，没有自动重试。

发布沿用已 push commit 的 `git archive` 和逐路径未提交检查，CPU 路线去除 GPU 占用查询，不调用 checkpoint loader。run 与 analysis 的入口命令要求显式 spec，analysis 还要求显式 release 名。分析须等待完整八行终态；原日志与输出不会覆盖。包依赖闭包已按本轮发现补齐。

已读取只读预检证据：

```text
E:/type10-7/automation_reports/CV-SincNet/20260929-phase2-d92-branch-metric-support-m4-r01/evidence/metric_preflight_1790671460428903700.json
```

该证据为 VERIFIED：8 个既有 support cache 绑定通过，新 run/release/archive 路径均不存在；rx3 每模型 cache count=8273、rx1=2332。它证明当次只读预检状态，不证明后续启动或实验完成。本审查没有重复预检或远端访问。

实现 owner 提供并已核对对应测试源码的验证证据如下；按本次“不重复已通过测试”的要求，审查者未重跑相同套件：

| 套件 | 已报告结果 | 主要核验 |
|---|---:|---|
| `tests/test_d92_branch_metric.py` | 25 passed | 小维显式展开/dual、距离判别、K1、隔离、顺序/批次、退化、nbytes、错误输入 |
| 既有 BranchInteraction 核心测试 | 29 passed | 复用核/中心化/逐样本预测实现 |
| 新 evaluate + summarize 测试 | 44 passed，3 subtests passed | 合成 cache→真实核心→汇总；读取边界；proxy 物理映射/压缩计数损坏负测；每 K/control 护栏 |
| `tests/test_run_d92_branch_metric_probe.py` | 15 passed | 八行一次调度、失败保留、配置绑定、CPU 发布、独立目录分析依赖闭包 |

合计为 113 个 passed 测试及 3 个 passed subtests。新增文档经 UTF-8 与局部 diff 检查；未修改被审实现。剩余工作为 launch owner 的实际交付与运行后审计，不能用本次合成正确性证据预先宣称候选满足性能目标。
