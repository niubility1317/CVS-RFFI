# PAMF 信道伪标签分配：33 个冻结前缀的源域质量证伪

- run_id：`20261025-phase1-PAMF-ChannelPL-source-m33-r01`
- group_id：`phase1-pamf-channel-pl-20261025`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；本源域诊断已完整完成，实际证据见下文及 events.jsonl。

## 完整负结果与研究判断

本轮从普通 PL 重新构建独立 SSL 候选，保留 PAMF-REPL 的名称、原权重及完整负结果。此前 REPL 根据已公开 F0 clean 成绩选定，本次研究继续使用已经暴露的基准，不能称为新的盲测。

已完整读取冻结论文汇总的 10512 行总体结果、13824 行配对效应、201 行训练曝光、111 行源特征诊断、1221 行源物理干预，并复核 99 行主公平性结果。三类测试人群分别统计：cross_rx 为目标 RX 的 D1–D3，cross_date 为源 RX 的 D0，cross_both 为目标 RX 的 D0。旧 F0 168000 条主测试合并 cross_rx 与 cross_both，与仅 cross_rx 的论文表口径不同。

下表为 11 个设置等权平均的 REPL−普通 PL，单位为百分点；不把不同数据集样本混成一个准确率。各项均先计算同 seed 差，再统计三个 F seed；固定 G-LU 由同契约两种 F 共用，不是三个 G seed。

| 测试人群 | 读出 | clean 差 | 六景差 | 六景提高设置数 |
| --- | --- | ---: | ---: | ---: |
| cross_rx | raw | +0.2147 | −0.7445 | 0/11 |
| cross_rx | 同契约 G-LU | +0.2203 | −0.1750 | 3/11 |
| cross_date | raw | −0.4441 | −1.1864 | 0/11 |
| cross_date | 同契约 G-LU | −0.4123 | −0.2268 | 6/11 |
| cross_both | raw | +1.3227 | −0.3485 | 3/11 |
| cross_both | 同契约 G-LU | +1.3475 | +0.1307 | 6/11 |

raw cross_date 六景的 33 个 seed 单元全部下降，说明问题不能只归因于某个接收机划分或某一个坏 seed。F0 去掉 style 后，raw 六景较 REPL 回升 0.3656 pp，较普通 PL 仅提高 0.0890 pp；G-LU 六景仍较普通 PL 低 0.0761 pp。去掉 extrapolation 后，G-LU 六景回升 0.2148 pp，但仍较普通 PL 低 0.0150 pp。去掉 joint 或 guard 并未同时改善这些端点，故不依据这次消融简单删除全部机制。

m1 的浅层、中层 U style 实际应用数均为 0，跨 RX 同类配对也没有可用第二个 RX；六景仍下降。这是“style 是唯一原因”的反例。

完整训练日志覆盖上述 66 条主轨迹与 12 条 F0 机制轨迹，共 189 个唯一训练段、666000 条有序逐步记录，均闭合，没有观察到非有限指标、实际跳过非有限 loss/gradient 或 U truth 可用标志。补查实际恢复控制器的日志索引后，189 个训练段的 stdout 均已定位：132 份唯一文件、136185 行全部解析，没有观察到 traceback、CUDA OOM、killed 或 warning 标记。日志结果支持“未见这些技术失败”，不能据此把性能下降归为确定的训练机制因果。

F0 两法的 U 曝光均为 6293700 次，普通 PL 接受率 94.8482%，REPL 94.7618%。REPL 没有增加原样本曝光或接受比例。普通 PL 实际 U 权重为 1，REPL 名义权重为 6，且后者包含多个 CE/KL/对比/特征机制。guard 会限制实际梯度，不能把名义权重解释为六倍实际梯度。接受率、置信度均不是伪标签正确率；U 真值继续隐藏。

当前可检验的假设是：先保留普通 PL 的稳定训练剂量，再让可靠的源信道视图承担一部分监督，以增加身份不变的物理覆盖。该解释尚不是因果结论，也没有新的目标收益证据。

## 固定候选与本次诊断

候选暂称 ChannelPL，机制详见 `experiment.json.matrix.recipe`。保留原 PL raw 阈值 0.95、U 权重 1、同一 U loader 和默认联合分类头。在原交替物理半批内，仅对已被 raw 接受的样本生成原 T1 As；额外要求 As 预测同类且置信度至少 0.95。通过时使用 0.8 raw CE＋0.2 As CE，相当于 As/raw 比例 0.25；未通过时精确退回原 raw PL。该比例来自监督部分的物理视图份额，不是根据新目标成绩搜索。关闭 REPL 的 U style、extrapolation、joint contrast、U KL 与 guard；不引入 EMA、新头或 G 训练。

本 run 只做源域诊断：11 个既有契约×三个 F seed，共 33 个冻结普通 PL E20 前缀。E20 本身已在 E14–E20 使用普通 PL，不称为纯监督前缀。先核对实际 checkpoint payload、源 RX/day/物理 ID、继承与默认头；旧 F0 声明字段 own_EMA=true 不能替代真实 flags，实际 native U/EMA、teacher 和 inherited baseline 均须关闭。每 worker 首先用两条合法 L_s IQ 完成真实权重无 query smoke，通过后直接遍历完整源 V。

固定证伪条件：转移到信道视图的伪标签源 V 正确率至少 99%；相对 raw 接受的物理半批，正确率下降不超过 0.1 pp；转移数至少占已生成信道视图的 10%，且非零。三个条件均按每个固定 row 报告，不按结果重排 row 或调整参数。科学失败仍是完整诊断，所有 33 行继续完成；技术失败停止新调度并让健康子进程完成。

该诊断的 V 标签只参与正确率计数，不参与模型、阈值、分类头、参数或 buffer 更新。U IQ 不在本诊断中打开，U 真值、target、G 权重/输出读取均为 0。源 V 正确率只能作为 U 伪标签质量的代理。

## 数据、seed与模型来源

实际数据契约、各 seed 角色、checkpoint 来源及逐行输出以 [experiment.json](experiment.json) 为准。ManyRx 是 10 类，其余为 6 类；m1/m2/m3/m4/m6 使用固定 6 个目标 RX，F0 的 5 个源 RX 设置不插入源 RX 数量曲线。数据按既有 VALIDATED_ONCE 句柄复用，不改 IQ、划分或标签比例。

## 执行与存储

采用 N607 既有环境、FP32/TF32off；物理 GPU5/6/7 分别固定 seed2026092701/02/03，每卡本 run 最多两条，总 CUDA／待初始化 GPU 进程少于 3 且空闲显存不少于 10000 MiB 才启动。只使用独立、不可覆盖的 release/run root，代码 commit 在实际 runtime 与启动证据记录。原实验、恢复缓存及其他 owner 进程均保留。

相关 31 项检查已通过，包括错误契约／继承／teacher／target 输入拒绝、6/10 类、U 标签隐藏、逐样本 CE 剂量守恒及无可靠视图时损失和梯度与原 PL 一致。唯一独立 P0/P1 审查结论 PASS，未发现会导致本次源诊断错误或越权的缺陷；实际 PID/CWD/GPU/日志证据在启动后记录。检查通过不等于目标性能改善。

## 结果与覆盖

**VERIFIED：33/33 项源诊断全部完成并通过固定条件。** 实际执行 commit 为 `f02436f53feeb0dad3a6c6feeae7b56afa775974`；源 V 共计 748269 次观测，包含跨设置及种子的重复观测，不表示同样数量的唯一物理样本。逐行结果见 [source_quality_results.csv](source_quality_results.csv)。

转移伪标签的源 V 正确率范围为 99.5291% 至 100%；转移比例为 24.9666% 至 65.0972%；相对 raw 接受物理半批的正确率，最大下降仅 0.0203 pp。F0 三 seed 正确率分别为 99.8999%、99.8570%、99.9140%；1% 标注设置为 99.5291%、99.6400%、99.5960%。这些是源 V 代理，不能称为 U 集正确率或目标集准确率。

全部 33 行完成真实 checkpoint smoke、完整源 V 遍历及 buffer 不变检查。模型／分类头更新为 0，target／query／U 真值／G 权重输出读取为 0。一次发布归档传输 SHA 比较、远端编译及入口导入通过。独立 post-state 验证了本 run 的 PID、CWD、完整 argv、GPU 映射、PASS smoke 和持续增长的源诊断 progress 记录；最终 33 个完整结果与控制器终态相互吻合。

本 run 无训练、无目标测试，不标作 SSL 研发完成。后续独立 run `20261025-phase1-PAMF-ChannelPL-m33-r01` 已固定同一个配方，执行匹配的 E20＋own10＋own20、raw／同契约 G-LU 的 clean＋六景及三类人群评测，保留匹配普通 PL 和监督参照。原 clean≥+1 pp、六景>+1.5 pp 目标仍未达成。

## 门控误差与类别集中诊断

进一步分解同一份既有计数，不新增 IQ、模型、U 真值或目标读取。以 raw 已接受的物理半批为候选集，信道门控接受／拒绝两组严格相加还原候选数及正确数；逐行和 480 个源 RX/day/真实类别分组分别保存于 [source_gate_excluded_quality.csv](source_gate_excluded_quality.csv) 与 [source_gate_strata_quality.csv](source_gate_strata_quality.csv)。

33 项中有 32 项的被拒伪标签正确率仍至少 99%；F0 三个 seed 的被拒组正确率分别为 99.9080%、99.8707%、99.8814%。因此，As 自身预测置信度／同类门控排除了不少 raw 标签正确的源观测。该证据提示门控可能同时减少较难物理视图的利用；当前计数没有保存全部 As 预测及逐视图可辨识性，不能据此证明 IQ 身份完全保真、所有被拒样本都适合训练，或宣布取消门控必有目标收益。

总体高正确率也掩盖了错误集中。1% 标注、seed2026092701 的真实类别3：候选 1654 条，raw 伪标签正确率 95.1028%；转移 623 条、正确率 96.3082%；被拒 1031 条、正确率 94.3744%。同 row 的整体转移正确率为 99.5291%，二者口径不同。这里的类别按源 V 真值分组，**不能据此给隐藏真值的 U 样本按真实类别设阈值**；要使用可观察的预测类别或其他可靠性证据，必须另做来源合法、独立固定的诊断。

下一步研究应区分“raw 伪标签错误”与“As 视图当前难识别”，同时检查低标注错误的集中性。当前固定候选仍按原矩阵完成，不因这些诊断热改权重／门控或取消测试。该后验源域分解不改变原 33/33 的全局科学条件，也不能替代目标集验证。

## 交接

源表：[完整冻结总体汇总](../../../exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/summary_overall.csv)、[完整配对效应](../../../exports/unified_figures/20261025-PAMF-REPL-complete-paper-r01/source_data/paired_effects.csv)、[逐设置主公平性](../20261024-phase1-PAMF-REPL-completion-m201-r01/main_fairness.csv)。逐步日志统计与原始诊断保留在私有 local_artifacts；公共数值摘要见本目录 CSV。
