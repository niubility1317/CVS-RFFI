# PAMF-ChannelPL 三种子匹配续训与完整冻结测试

- run_id：`20261025-phase1-PAMF-ChannelPL-m33-r01`
- group_id：`phase1-pamf-channel-pl-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 固定方法与比较

本轮验证一个独立候选，暂称 PAMF-ChannelPL；不改写已选 PAMF-REPL 或默认 PAMF-T1-E50／DURA v1。完整 REPL 负结果的诊断与本次源域证据见 [源诊断报告](../20261025-phase1-PAMF-ChannelPL-source-m33-r01/report.md)。

保留普通 PL 的 raw 阈值 0.95、U 权重 1、原 U loader 和原默认分类头。原交替物理半批中，只有 raw 已接受、As 同类且置信度至少 0.95 的样本使用 0.8 raw CE＋0.2 As CE；其余精确回退原 raw PL。逐样本 U CE 总剂量保持 1，不叠加 REPL 的 U style、extrapolation、joint contrast、KL、guard 或 EMA。As 使用与监督部分相同的原 T1 三种源域物理仿真、独立实例，不读恢复 IQ 或 G。

11 个既有契约×三个 seed，共 33 条固定候选轨迹，全部已在目标访问前通过源诊断。源 V 转移伪标签正确率为 99.5291% 至 100%，转移比例为 24.9666% 至 65.0972%；源诊断没有更新模型或访问目标／U 真值。配方及 33 行矩阵不根据后续目标分数改动。

每条轨迹复用同 seed、同数据契约的普通 PL E20 前缀，再独立运行原生 E10＋E20 两段，累计 E50／11100 步，新增 E30／6660 步；日程为 13L＋37LU，前 7LU 属于原普通 PL，后 30LU 使用本候选。原监督部分的六次 L 前向、普通／raw RDrop、As 两次 CE 且 As KL=0、ORD128 跨 RX 对比、SIG_TIME_X3 和 CENTER_UNITS 均保持。默认头与骨干联合更新，不另训头。

同契约、同 seed 的监督、普通 PL 和 REPL 完整结果复用 m201 已冻结产物。G-LU 及其 IQ 缓存与各对照完全相同，不重训 G，不新建恢复缓存。G-LU 只是 F 冻结后的额外读出，不参与 ChannelPL 训练。

## 数据、seed与模型来源

全部数据契约、六类 seed 角色、逐行输出和期望产物以 [experiment.json](experiment.json) 为唯一矩阵定义。ManyRx 保持 10 类，其余保持 6 类。source-count 使用 m1/m2/m3/m4/m6、固定六个目标 RX；F0 不插入该曲线。source 的 L/U/V 与目标物理 ID 均复用已有 VALIDATED_ONCE 句柄。

真实 E20 的 complete、payload、seed、默认头、物理契约和完整 scratch 来源在加载前核对。旧 F0 的 own_EMA 声明字段不代表实际执行；F21 的真实 native U／EMA、teacher 和 inherited baseline flags 均须关闭。源 V 仅沿原流程诊断健康；checkpoint 固定最终 E50，不按 V 或 target 重选。

## 执行与存储

物理 GPU5/6/7 分别固定 seed2026092701/02/03，每卡最多两条本 run 训练／预测 worker；其他 owner 和待初始化 CUDA worker 计入总数，上限为 3，且启动前空闲显存至少 10000 MiB。CPU scorer 全机同时最多一条本实验评分，观察到其他 owner 评分时等待。首次真实权重 L/U 前向及反向 smoke 通过后立即进入训练。

每 row 完成 E50 后自动优先进入测试队列，不等待全部训练。预测 worker 不持有 truth；一次冻结该 row 的三类人群×七景×raw／G-LU 所有预测后，独立 CPU scorer 才连接 truth。scorer 不持有源模型／源 IQ／G 句柄。目标分数不参与其他 row 调度、参数、预算、seed 或 checkpoint 选择。

任一实际技术失败停止新启动，已健康的 worker 自然完成并保留输出；不因低准确率停止，不自动重试或覆盖失败 root。新 run 独立于全部历史产物，未授权其他进程停止或配置修改。

本地 55 项聚焦检查已通过，涵盖 CE 剂量、fallback 损失与梯度、U 标签隔离、6/10 类、错误 checkpoint／teacher／query 拒绝、固定训练预算、完整三人群／双读出预测约束、F0 168000 口径及样本 SD。发布前修复了 smoke 提前绑定 loader 导致 U128、以及绑定 hook 与显式 model 关键字冲突两处执行问题，均有修复前失败、修复后通过的聚焦检查。唯一独立 P0/P1 审查发现的 hook 参数冲突已完成一次定向复审并 PASS。执行 commit 与启动后证据在实际返回后记录。

## 结果与覆盖

尚无新的目标结果。测试完整报告 clean 与六种 practical 各景，并保留逐 RX/day/TX、confusion、G 相对 raw 的净纠错、三 seed 样本 SD 和 seed 内配对差。

原始主目标按 F0 全部 168000 个 target RX 样本检验，即 cross_rx126000＋cross_both42000；clean 比监督与普通 PL 中较强者至少高 1 pp、六景均值高于较强者 1.5 pp。raw 与同 G-LU 分开报告。论文的 cross_rx、cross_date、cross_both 三个人群结果另外各自完整列出，并报告全部 11 设置等权宏差及改善设置数。不会用更好的人群口径替换原目标，也不把缺 seed 的结果当三 seed 均值。

## 交接

该基准已因 REPL 的历史选择及用户本次诊断而暴露，本次属于自适应研发，不能称为新的盲测或据三 seed SD 声称显著。当前完成源诊断与配方固定，后续以实际训练／全预测／独立评分及同 row 分析推进；原目标保持 active，未达成前不宣布研发完成。
