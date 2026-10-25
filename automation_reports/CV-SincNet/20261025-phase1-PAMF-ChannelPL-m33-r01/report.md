# PAMF-ChannelPL 三种子匹配续训与完整冻结测试

- run_id：`20261025-phase1-PAMF-ChannelPL-m33-r01`
- group_id：`phase1-pamf-channel-pl-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING；六条实际训练进程已核实，新的目标结果尚未返回。

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

本地 55 项聚焦检查已通过，涵盖 CE 剂量、fallback 损失与梯度、U 标签隔离、6/10 类、错误 checkpoint／teacher／query 拒绝、固定训练预算、完整三人群／双读出预测约束、F0 168000 口径及样本 SD。发布前修复了 smoke 提前绑定 loader 导致 U128、以及绑定 hook 与显式 model 关键字冲突两处执行问题，均有修复前失败、修复后通过的聚焦检查。唯一独立 P0/P1 审查发现的 hook 参数冲突已完成一次定向复审并 PASS。

**VERIFIED：正式训练已启动。** 实际执行 commit 为 `1fa92a0b8b61fba7ddca532a12078a8ddae2bdf7`，不可变 release 为 `20261025-phase1-PAMF-ChannelPL-training-release-r01`。一次归档传输校验、远端编译与入口导入通过。独立 readback 核对了实际 PID/CWD/argv 与 NVIDIA compute PID 对应的物理 GPU：5/6/7 各两条 worker，F0 与 F1 各三个 seed 正在训练。

真实 E20 L/U 前向、反向 smoke 已 PASS：原 L 前向 768 包、L As64、默认六类头梯度有限且非零；原 U256 中接受 235、生成 As117、转移 51，U CE 剂量 1，query／U 真值／EMA 读取或更新为 0。独立读取各 worker 的首个完整新增 epoch，共 1332 条有序记录，确认每步 L768、As KL=0、U 权重1、实际 ChannelPL 损失等于原 raw PL＋重分配差值，均未发生非有限梯度跳步或 U truth 可用标记。此为首 epoch 快查，不能称为完整训练日志分析。

逐 worker 早期曝光见 [early_source_execution.csv](early_source_execution.csv)，精简启动事实见 [launch_evidence.json](launch_evidence.json)。原始运行细节和日志留在私有 local_artifacts 及 N607 输出，不提交 checkpoint、IQ 或原始日志。

首批 F0/F1 的六条轨迹已全部完成新增 10 轮，即累计 E30／6660 步。六段闭合日志共 13320 条记录已完整读取，确认 U 权重1、CE 剂量1、原 L768／As KL0、完整 2220 次更新及有限 loss／gradient；检查范围是已完成的 10 轮，不冒充完整 E50 分析。F0 三 seed 的原 U 接受率为 94.4067% 至 94.6305%，各增加 132636 至 136258 次可靠 As 学生视图；接受率仍不是正确率。

六条轨迹的后续 20 轮均已启动。读取实际原生初始化产物内的 config，核对父路径为各自的 segment30、step_offset6660、epoch_offset30、segment_epochs20、原 seed／source RX/day／默认头，真实源 IQ smoke 全部 PASS。未从头重置模型或改用其他父模型。阶段日志摘要见 [stage30_source_execution.csv](stage30_source_execution.csv)，实际初始化来源核对见 [stage50_resume.csv](stage50_resume.csv)。

## 结果与覆盖

首批 F0 与 F1 的六条轨迹均已完成 E50、全部3人群×7视图×2读出预测及独立 truth-last 评分。下表是原始 F0 全部168000个目标 RX 物理样本（cross_rx126000＋cross_both42000）的三个学生 seed 均值±样本SD，单位为%。差值是 ChannelPL−普通 PL，单位为百分点。

| 原始 F0 口径 | F监督 | 普通PL | ChannelPL | 较普通PL差值 |
| --- | ---: | ---: | ---: | ---: |
| raw clean | 84.7853±0.8236 | 84.9817±0.1893 | 84.8790±0.4067 | −0.1028 |
| raw 六景均值 | 70.5962±0.9519 | 70.3594±0.9576 | 70.5894±0.9856 | +0.2301 |
| 同 G-LU clean | 84.8575±0.8694 | 85.0296±0.2041 | 84.9514±0.4450 | −0.0782 |
| 同 G-LU 六景均值 | 77.8549±0.3960 | 78.0746±0.1843 | 77.9922±0.2382 | −0.0824 |

**首批未达标。** raw clean 低于普通 PL，raw 六景只比普通 PL 提高0.2301pp、与监督版基本持平；同 G-LU 两个端点均略低于普通 PL。与历史 REPL 比较，raw clean 下降0.8478pp、raw 六景提高0.5039pp，不能称为全面超过 REPL。F1 原始全部目标日期口径中，raw clean 较普通 PL 低0.4198pp、raw 六景高0.0281pp、G-LU 六景低0.2106pp，同样没有实现目标。

逐人群、逐景、对照均值／SD、seed内配对差／配对SD分别见 [F0三seed完整已评分汇总](interim_F0_three_seed.csv) 与 [F1三seed完整已评分汇总](interim_F1_rx_repartition_r01_three_seed.csv)。这里每张表均已完整三个 seed；整轮11设置矩阵仍未完成，其他健康训练与固定测试按原计划继续。源域99%以上伪标签正确率没有自动转化为目标提升，既有单纯As置信度筛选与固定小剂量CE不足以支撑性能目标。

汇总阶段发现两个历史 F0 监督 score 在原 m201 的 CPU 技术恢复目录，不能从统一 readouts 路径推断。读取其实际登记 job／config／completion 解析到了正确同 row、同 seed 的历史产物，没有重新测试、重选 checkpoint 或替换比较方法。当前正在修复仅供报告使用的引用路径；训练／预测／独立评分的科学执行 commit 仍为1fa92，不受此路径修复影响。

测试完整保留逐 RX/day/TX、confusion、G 相对 raw 的净纠错、三 seed 样本 SD 和 seed 内配对差。F监督＋G-LU 使用同一无标签恢复器，不称为全系统 L-only。

原始主目标按 F0 全部 168000 个 target RX 样本检验，即 cross_rx126000＋cross_both42000；clean 比监督与普通 PL 中较强者至少高 1 pp、六景均值高于较强者 1.5 pp。raw 与同 G-LU 分开报告。论文的 cross_rx、cross_date、cross_both 三个人群结果另外各自完整列出，并报告全部 11 设置等权宏差及改善设置数。不会用更好的人群口径替换原目标，也不把缺 seed 的结果当三 seed 均值。

## 交接

该基准已因 REPL 的历史选择及用户本次诊断而暴露，本次属于自适应研发，不能称为新的盲测或据三 seed SD 声称显著。当前完成源诊断与配方固定，后续以实际训练／全预测／独立评分及同 row 分析推进；原目标保持 active，未达成前不宣布研发完成。
