# D92 E0当前256维模块消融纠正重跑

## 最小预登记

- `run_id`：`d92_e0_256_module_ablation_hard11_20260826_v2`
- 初始状态：`LOCAL_VERIFIED`
- 运行代码提交：`20bc5eb99ed57f99d4beaa7856d25ba1a3af78c6`
- 本地代码目录：`E:\\type10-7\\github_publish\\CVS-RFFI-repo`
- 本地运行环境：`C:\\Users\\lh594\\.conda\\envs\\ssr-gpu\\python.exe`
- N607运行环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

本run纠正`v1`的维度实现错误。`v1`的7个执行和独立评分已完整保留，但其`query_head_mac=3168=11×288`，不构成真实256维D92 E0性能结论。`v2`使用真实紧凑坐标：FULL、B0、S0、C3、D0、D2为`160+96=256`维；A0为160维身份坐标。不设置FP32状态对照臂。

## 冻结矩阵

所有臂共用：接收机`3-19`、`K10/new5`、method seed`7282101`、support seed`7282201`、query seed`7282301`、新类抽样seed`7282401`，以及`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`三个场景。query仅用于逐样本预测和独立truth-last评分，绝不进入拟合。

|逻辑臂|比较因素|相对`P2-256-FULL`的唯一配置差异|活动维数|
|---|---|---|---|
|`P2-256-FULL`|当前256维参考|无|256|
|`P2-256-A0`|联合特征|`feature_profile`|160|
|`P2-256-B0`|稳健中心|`center_profile`|256|
|`P2-256-S0`|自动收缩协方差|`covariance_profile`|256|
|`P2-256-C3`|旧/新任务均衡协方差|`covariance_profile`|256|
|`P2-256-D0`|双几何控制|`geometry_profile`|256|
|`P2-256-D2`|交叉拟合融合控制|`geometry_profile`|256|

## 已完成的本地核验

- 旧实现上的定向测试先失败：FULL与A0都错误编译为288维，且11类查询头错误报告为3168MAC。
- 修复后，完整7臂运行时核验通过：六个联合特征臂的已编译状态均为256维、块边界为`(0,160,256)`；A0为160维、块边界为`(0,160)`。
- 65项聚焦回归全部通过，包含feature投影、注册、row执行、量化闭合和计划目录测试。
- 独立P0/P1代码复核结果：`P0=0`、`P1=0`。新紧凑运行时文件已显式纳入运行代码提交。
- FULL的资源测试验证`query_head_mac=11×256=2816`；A0按其160维活动坐标单独计量。

## 固定输入、输出与停止规则

- 重用已验证输入，不重建也不重验received-IQ：`v1`的`VALIDATED_ONCE`cache binding和同一`capsule_id`、`split_id`。
- 本地报告根：`E:\\type10-7\\automation_reports\\CV-SincNet\\d92_e0_256_module_ablation_hard11_20260826_v2`
- N607输入、请求、运行和日志根均使用新的`v2`路径，不覆盖`v1`任何文件。
- N607GPU在启动前根据只读preflight分配；不得影响现有外部任务。
- 只在数据绑定或query边界错误、错误checkout、输出根已存在、同一确定性预预测异常在至少两个物理行复现、prediction closure缺失或独立scorer连接错误时停止。指标高低不触发停止。

## 发布与启动证据

- release代码提交为`20bc5eb99ed57f99d4beaa7856d25ba1a3af78c6`。
- release包本地/远端SHA256一致：`4b5b04c6a342c1e4fcb78faf6f79fbbec8763c9096cba07268a9ab1cd7dac700`。N607在`/home/szu2070436088/2510044040/CV-SincNet/releases/d92_e0_256_module_ablation_hard11_20260826_v2_20bc5eb9`检出该提交，远端编译通过。
- 封存计划位于`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/d92_e0_256_module_ablation_hard11_20260826_v2_20bc5eb9/sealed_plan.json`，包含7个物理执行和7个逻辑row；不写输出dry-run通过。
- 正式runner使用同release中的row predictor和独立scorer。7个物理row排入GPU0–GPU3的7个slot；GPU0已有一个外部任务，runner受每GPU最多2个进程限制而动态等待，不干预外部任务。GPU4、GPU5外部训练不参与本run。

## 预期交付与结论边界

每个逻辑臂应有一个独立预测闭合物和一个独立评分闭合物。只有7个row均完成prediction closure并由独立scorer连接truth后，才在同一行报告旧类准确率、已见新类准确率、`H`、`F`、`min-old`和`min-new`，并将结果写入各模块对应位置。该单seed困难切片仅用于筛选性消融解释，不替代多seed或完整矩阵确认。

## 启动健康证据

- 当前状态为`RUNNING`。N607主runner PID=`3677457`，其CWD和命令行均与冻结release和封存计划一致。
- 新run/log根、`runner_start.json`、7个物理row日志及launch记录均已出现，没有覆盖`v1`。
- 首次GPU读取确认6个预测子进程已在GPU0–GPU3运行。GPU0已有外部PID=`3660023`，故第7个row按每GPU最多2个进程规则等待；GPU4、GPU5的外部训练未受影响。
- 这些仅是启动健康证据，不能替代prediction closure、独立评分或模块效果结论。

## 完成、独立评分与最终状态

- 最终状态：`ANALYZED_SCREENING_SINGLE_SEED`。
- N607`runner_summary.json`证实：7/7物理执行完成、7/7逻辑评分完成、失败数0、无系统性停止。
- 全部评分row为`PASS`，使用`exact_scenario_query_token`连接truth；每行均具有`truth_opened_after_prediction_commit=true`和`scorer_output_must_not_feed_predictor=true`。
- 在本地对7份评分JSON重新运行汇总器后，评分schema、预测封存绑定、评分/行为/量化/资源/场景指标收据哈希全部通过；没有alias或failed row。
- `v1`因联合臂实际288维而永久排除出真实256维性能解释；其工件只作为纠错审计保留。

### 三场景等权同row结果

|臂|B-old|A-old|New|H|F|mean min-old|mean min-new|相对FULL\(\Delta H\)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|`P2-256-FULL`|73.056%|68.333%|62.333%|65.180%|4.722pp|38.333%|46.667%|+0.000pp|
|`P2-256-A0`|73.056%|37.222%|21.667%|27.110%|35.833pp|6.667%|3.333%|-38.070pp|
|`P2-256-B0`|73.056%|68.056%|63.667%|65.785%|5.000pp|36.667%|45.000%|+0.606pp|
|`P2-256-S0`|73.056%|57.778%|50.667%|53.645%|15.278pp|25.000%|28.333%|-11.535pp|
|`P2-256-C3`|73.056%|68.056%|63.000%|65.417%|5.000pp|38.333%|46.667%|+0.238pp|
|`P2-256-D0`|73.056%|67.222%|63.333%|65.218%|5.833pp|35.000%|46.667%|+0.038pp|
|`P2-256-D2`|73.056%|67.778%|63.333%|65.471%|5.278pp|35.000%|45.000%|+0.291pp|

指标先按三个LEO弱场景独立评分再等权平均。`B-old`/`A-old`分别为同一旧类query注册前/后准确率；New为注册后新类准确率；(H)为逐场景旧/新调和均值的平均；(F=\mathrm{B\!\text{-}\!old}-\mathrm{A\!\text{-}\!old})。

### 筛选解释

- FULL相对A0：ΔH=+38.070pp、A-old提高31.111pp、新类提高40.667pp，联合256维特征在本切片不可删。
- FULL相对S0：ΔH=+11.535pp、A-old提高10.556pp、新类提高11.667pp，支持高维少样本下自动收缩的稳定性价值。
- B0、C3、D0、D2相对FULL的(H)差异不超过0.606pp，但旧类保持、遗忘与新类识别方向互有取舍；单seed不能证明任何一个为普适胜者。
- 本轮没有FP32对照臂。量化收据的作用是储存/保真审计：FULL的预测一致率100%、argmax翻转率0，但`integer_kernel_used=false`，因此不能宣传INT8整数计算加速。

### 资源收据摘要

|臂|维数|查询头MAC|注册时间|候选头状态|部署状态|候选头batch-1均值|
|---|---:|---:|---:|---:|---:|---:|
|`FULL`|256|2816|378.166s|5742B|17216B|0.074005ms|
|`A0`|160|1760|37.681s|3586B|17216B|0.038359ms|
|`B0`|256|2816|387.114s|5742B|17216B|0.103397ms|
|`S0`|256|2816|198.502s|5742B|17216B|0.071898ms|
|`C3`|256|2816|383.854s|5742B|17216B|0.072647ms|
|`D0`|256|2816|12.694s|5742B|17216B|0.081900ms|
|`D2`|256|2816|25.858s|5742B|17216B|0.071438ms|

注册时间只表示support注册状态构建；batch-1只测临时解码、浮点点积与截距，不代表端到端延迟。完整逐场景评分和行级资源见主技术报告第12节及本地`automation_reports/CV-SincNet/d92_e0_256_module_ablation_hard11_20260826_v2/remote_results`。

该结果覆盖一个接收机、`K10/new5`、一个method/support/query/new-class draw与三个LEO弱场景；它完成当前真实256维模块筛选，不是多seed确认、全矩阵结论或实星硬件结论。
