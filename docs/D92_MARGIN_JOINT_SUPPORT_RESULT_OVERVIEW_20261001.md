# MarginJoint support 结果与数学方法状态

当前候选已实现并完成合法support的独立数学检查；它改善了注册后旧类准确率，但新类与综合H没有同步改善，不能称为全面提升或最新最强方法。完整重复query基准正在按原冻结方法运行，尚无query性能结论。

本报告整体口径为同一原source row/物理旧held配对的96个OOF可测parent，K5/10/20、新增2/5/10/20等parent平均。原声明160parent含真实K1和新增0；旧类6。每row两个预登记receiver/scenario组合、两个模型seed交叉两个capsule，support seed2026092711。

| 方法 | A旧（%） | B旧（%） | C旧（%） | C新（%） | H（%） | B−A（百分点） | B−C旧（百分点） | 平均绝对新旧差（百分点） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BranchLocalRidge / 当前R0 | N/A | 71.0417 | 63.7674 | 54.8464 | 58.4168 | N/A | 7.2743 | 10.7405 |
| MarginJoint | 62.6389 | 70.2083 | 66.5538 | 52.2734 | 57.8275 | 7.5694 | 3.6545 | 15.3220 |

R0的实际地面A尚未由该路径配对，不能借Margin的A或用R0/B0代替A。Margin的真实A与原固定B/C已独立补齐。H、下降与绝对差先按parent计算，不能从表中汇总准确率再算。

| Margin相对同次R0 | B旧 | C旧 | C新 | H | 注册下降 | 平均绝对差 |
|---|---:|---:|---:|---:|---:|---:|
| 变化（百分点） | -0.8333 | +2.7865 | -2.5729 | -0.5893 | -3.6198 | +4.5816 |

注册下降缩小包含B本身降低的影响；同时C旧确有提高，但C新下降且绝对差扩大。因此不把单项保旧收益解释为综合优化成功。理想方向仍是B−A≥10、B−C旧≤1、绝对差≤3个百分点，目前均不是每轮硬门槛。

完整K×新增类数、OOF/proxy及receiver/model分层见[实际三阶段报告](D92_MARGIN_GROUND_A_PAIRED_SUPPORT_RESULT_20261001.md)。真实K1无独立held，因此support诊断N/A；完整query基准中的K1正常评估。新增0时C直接复用实际B，新类准确率/H/差距N/A。Proxy不能当作K1独立准确率。

固定Phase1与practical residual/post_sync/noeq/25MHz。只用合法support学习adapter/解析头；无源样本、逐记录源特征、query拟合、truth/role/配额或全局重排。support独立分析与Ground A补充均query/source行数0；新query训练不继承旧probe适应状态，结果不反馈调参/选模/选择性重跑。

数学上，C使用实际B padding作为冻结先验，以全部旧训练点面对全部注册类的最小真实类分数差为下界。内层核ridge QP具有零残差可行解；完整KKT隐式导数包括自由截距和乘子响应，再更新小adapter。正原margin可保护旧train严格正确；ties/负margin和未见query不具备相同保证。详见[用户数学说明](D92_MARGIN_JOINT_USER_EXPLANATION_20261001.md)。论文工具和本项目推导在该说明中逐项区分，没有由论文推出本项目性能保证。

| 已测资源 | 数值与范围 |
|---|---|
| 原support实验程序墙钟 | 3135.413213s；四行、两CPU lane，不是单次部署适应耗时 |
| 原训练进程最大峰值RSS | 1319796736B；Linux CPU/float64、96逻辑CPU，每lane BLAS2 |
| 直接训练坐标数 | 最多736×8=5888；解析头和prior状态另计 |
| 实测QP最大factor buffer | 2212352B；两个QP峰值取MAX，工作计数取SUM |
| 实测最大小候选部署数值状态 | 5498976B；不是完整部署包 |
| 新增地面样本/统计payload | 0B；原模型头packet复用，其文件/数组字节与真实星地传输分开 |
| 实际星载时间/内存、完整部署包、传输与能耗 | N/A；尚未在星载硬件或真实星地链路测量 |

完整实测资源与独立数学计数见[原support摘要报告](D92_MARGIN_SUPPORT_DIAGNOSTIC_RESULT_20261001.md)。完整训练-only snapshot已捕获；其唯一extract仍运行，后续派生完整曲线与AI紧凑JSONL/CSV。该离线提取进程的RSS不是方法星载内存，训练目标RMSCE和算术平均CE必须分开。

完整重复基准run：`20261001-phase2-d92-margin-joint-repeat-m2-r01`；4row/2400parent，预先固定同两个最小model seed，覆盖四receiver、三scenario、五support seed以及K1/5/10/20×新增0/2/5/10/20。supervisor869328；全A/B/C固定前不评分，旧baseline仅比较相同seed子集，不能与旧四seed总平均混减。此基准沿用已评分库存，新增独立验证仍按用户要求暂缓。

来源：原训练e50de0ad4、独立分析ccb7f3469、Ground A和query runtime d86edc3235ec0ca0e1925223970e07373256fb9c。历史推导文档中的“尚未实现”描述属于当时状态，当前实现和真实证据由此报告及run记录说明。
