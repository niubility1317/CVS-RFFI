# Phase1 条件 IQ 流恢复原型

本原型探索不同于单次波形回归的恢复方式。对同一源记录，以接收 IQ、固定恢复锚点和物理上下文为条件，在锚点与该记录源 clean IQ 之间构造线性路径，训练共享网络预测路径速度；推理从锚点出发做四步确定性 Euler 更新。路径回归借鉴 [Flow Matching](https://arxiv.org/abs/2210.02747) 的向量场训练思想；这里采用每条观测确定的锚点，不声称实现了通用密度模型或后验采样。

DURA 结合路线使用固定 DURA 输出、物理残差和方差；源训练可以复用对应固定源上下文。独立路线可将接收 IQ 自身作为锚点，但两条路线必须分别配置、训练、登记和评估，不能把一个模型的成绩迁移到另一个模型。训练只用合法源 L_s/U_s 的 IQ 和模拟配对，不读取 F、TX 标签、V 或目标；推理仅接收一条观测及由该观测计算的上下文。DURA、现有权重及默认分类头不更新。

项目历史上已做过 [旧 R03 的四步确定性流修正](../automation_reports/CV-SincNet/20261020-phase1-restoration-residual-flow-source-s2026092701-m2-r01/report.md)：它使用三套旧识别器指导，low urban 三个读出均值为 64.8327%，高于其 R03 对照的 63.3925%，但未达到当时目标。本候选更换为 DURA 上下文、源 IQ 独立训练和 Tri-SIG-TIME SSL 主评测，不能继承该结果；确定性流思路本身不是项目首次提出。

零初始化时四步输出严格等于锚点。2026-11-07 本机 RTX 5070 Ti 使用已有固定 DURA 源上下文的 8 条 IQ 完成六次技术更新，数值有限，初始锚点严格一致；四项合成数据单元测试覆盖恒定速度积分、路径损失梯度、初始输出和非法时间边界。此检查只证明代码可运行，不是正式源训练、身份准确率或目标收益。独立 P0/P1 原型审查未发现阻止真实源 smoke 的问题。

下一步需接入真实 GPU 源训练、固定训练预算和唯一输出，并按项目协议在冻结后完成 Phase1 clean 和六景各 168000 条预测，再独立 truth-last、与相同 Tri-SIG-TIME SSL 的 DURA 做配对；不能按目标结果调整模型。网络需进行实际耗时评估，四步网络前向额外消耗不能按参数量推定。已跑模型和其他 owner 的健康进程不热改。

正式候选已登记：固定合法源 32768 个物理 ID、每条三个模拟视图、五轮、batch 32，使用均匀采样路径时间；参数 28162，DURA 保持冻结。N607 GPU 3 单一 owner 容量队列进程已启动，当前等待至少 4096 MiB 且计算进程少于三个；满足条件后依次执行真实 checkpoint 无 query smoke、源训练、源预测 smoke、七景冻结预测及独立评分。发布包传输和远端编译通过；排队不表示 GPU 训练已经开始。[源训练登记](../automation_reports/CV-SincNet/20261107-phase1-dura-iqflow-source-s2026092701-r01/experiment.json)、[七景登记](../automation_reports/CV-SincNet/20261107-phase1-dura-iqflow-ground-test-s2026092701-r01/experiment.json)。

另行登记独立 IQ 流：以单条 received 作为锚点，训练路径连接 received 和同记录源 clean，不把 DURA 输出或教师加入损失；推理仅需 received。为复用数据，训练流程加载包含 DURA 字段的源缓存和固定 DURA 用于 smoke／源对照；七景评测也计算 DURA 作为配对基准，不能称其全流程不读取 DURA。固定 32768 源 ID × 三视图、20 轮、batch 32，与 DURA 条件版五轮预算不同。七项相关单元／合同测试及本机 8 条源 IQ 的六次 GPU 技术更新通过，独立 P0/P1 审查未发现阻止运行的问题。N607 GPU 2 队列等待容量，尚未开始真实训练。[源训练登记](../automation_reports/CV-SincNet/20261107-phase1-direct-iqflow-source-s2026092701-r01/experiment.json)、[七景登记](../automation_reports/CV-SincNet/20261107-phase1-direct-iqflow-ground-test-s2026092701-r01/experiment.json)。

## 起点覆盖对照

对训练路径 `z_t = (1-t)anchor + t clean`，当 `t > 0` 时，训练目标速度可以从 `(z_t-anchor)/t` 代数重构；而部署积分从 `t=0` 开始。这个关系提示中间态训练可能存在依赖路径信息的捷径，但不证明现有网络实际使用了该捷径。为检验起点监督是否改善结果，独立登记同一 DURA 条件架构、相同源 ID、种子、优化器和五轮预算：每个训练样本以 0.5 概率使用 `t=0`，其余均匀采样；均匀版保持原配方。历史 R03 流也使用过起点覆盖，当前是不同父模型和无 F 指导的匹配对照，并非这一思路的首次提出。

九项相关测试及本机八条源 IQ 的六次 GPU 技术更新通过；独立 P0/P1 审查未发现阻止运行的问题。N607 GPU 5 队列等待容量，真实训练尚未开始。冻结后按相同七景和独立 truth-last 比较；训练采样的变化不证明目标性能收益。[源训练登记](../automation_reports/CV-SincNet/20261107-phase1-dura-iqflow-origin-source-s2026092701-r01/experiment.json)、[七景登记](../automation_reports/CV-SincNet/20261107-phase1-dura-iqflow-origin-ground-test-s2026092701-r01/experiment.json)。
