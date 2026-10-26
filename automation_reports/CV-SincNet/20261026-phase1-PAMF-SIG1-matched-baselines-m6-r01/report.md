# 单视图完整 PAMF：监督与普通 PL 三种子匹配基线

按用户最新“补充单视图的监督，普通 PL 对比基线”要求，新建六模型，不重启已停止的两行源控制。现有三个 REPL 新学生保持运行。本次仅补缺失的同架构完整监督骨架对照，已完成的历史 X3/单视图其他配方全部保留。

完整 PAMF 保留时频、PA、参考响应和原 6×160 默认分类头，SIG_TIME 为一视图、18976 参数。两组原监督机制完全相同：ordinary/raw RDrop、As 两次真标签 CE 且 As KL=0、LS0.01、浅中层 cross-RX，默认头与骨干联合训练。监督版 U loader/所有伪标签/EMA 关闭；普通 PL 仅当前学生置信度≥0.95、单 U 学生视图、权重 1，无 EMA/REPL/新机制。PL 保留原第14轮启用U的日程。

模型 seed 2026092701/02/03。监督从零 20+10+20=E50/11100；PL 各自复用已核实且来源一致的 m201 单 SIG_TIME ordinaryPL E20/4440，再原生+10+20至E50/11100，累计13L+37LU。不得让监督继承U前缀或跨seed借权重。新训练合计240模型轮/53280步，继承三份E20/13320步和对应耗时另计。所有模型固定final，不按V或target选checkpoint。

沿用原F0 VALIDATED_ONCE L6300/U56700/V27000、源RX[1,3,4,6,8]及day[1,2,3]和原六TX。U真实TX始终隐藏，V仅原有遥测。训练只收到源context；目标句柄只进入独立预测/评分配置。正式测试为cross_rx126000+cross_both42000=168000包/seed，两个读出raw和同一原F0 DURA G-LU，clean及全部六practical。复用已完成固定G恢复IQ，不训练G，不重新生成目标IQ。G-LU用过源U的IQ，因此“监督F+G-LU”不称整个系统只使用L。

每行final后有资源立即自动测试。该行2任务×2读出×7视图预测全部固定，独立CPU scorer先核对完整ID/视图覆盖，之后才连接truth。保留逐TX/RX/day/scene准确率、Macro-F1、混淆矩阵和恢复净纠错；自动输出每seed、三seed均值±样本SD及配对差。168000口径按正确数合并，不能把126000跨RX和42000跨RX+跨日简单等权。单独跨日30000不加入本矩阵。

真实smoke为第一步，六个配置各执行实际L128联合backward；PL分别严格加载自己的真实单视图E20，U256隐藏标签；Sup U实际0。检验L768/AsKL0、单SIG_TIME和原head梯度非零有限，0optimizer；首个真实E20/defaulthead接同G还做sourceL2前向。PASS后立即训练。

唯一owner为codex-pamf-sig1-matched-baselines-20261026。代码/config由执行Git commit固定，N607既有Phase1 Python/Torch2.1、FP32/TF32off/CPU2，无安装。CWD PROJECT/releases/20261026-phase1-PAMF-SIG1-matched-baselines-release-r01，命令和不可覆盖独立输出见experiment.json。全部GPU0..7可用，每次派发全局CUDA+pending<2且free≥10000MiB，其他owner计入，不停止/改动其健康任务。评分单独CPU最多1。

一次archive SHA传输比较和远端compile；独立核实PID/CWD/argv/GPU及日志增长。首个技术/输入/来源/shape/非有限/smoke/输出失败停止新派发，健康在跑行完成，保留产物，不自动重复提交或覆盖。科学低分不当技术失败。新基线不热改现有REPL源选择阈值；原clean+1pp、raw星地>1.5pp及同G提升目标保持，源/训练完成不当目标成功。

当前：PLANNED；7项聚焦检查与18个协议子检查PASS，唯一独立P0/P1审查PASS；真实远端smoke现已PASS，见下方启动证据。


## 实际启动（VERIFIED）

执行commit `088f8f36a34a9cb750243414de91c084fe1f28d0` 已提交/推送并独立核实远端OID；一次archive传输比较VERIFIED、远端compile PASS。六个真实联合backward smoke全PASS：三个PL各加载自己实际E20并读取U256，三个Sup实际U0；所有L768/AsKL0、单SIG_TIME18976参数、原分类头/单视图梯度非零有限，0optimizer/0U真值/0target；同G+E20源L2推理PASS。

六行已全部运行，独立核实controller/六worker PID、CWD、完整命令、物理GPU与跨两次读取的step日志增长，当前可读1264条记录，仅为启动进度。三个PL从E20后续训、三个Sup从零；每行自动正式预测/评分已实际入队。当前没有完成E50或新的目标准确率；三seed均值、样本SD及seed内配对差待正式矩阵完成。原三个REPL候选继续原计划，不改其配置/门槛，不联系其他对话。[启动证据](launch_evidence.json)。


最新源日志检查：六行当前全部可读前缀L768/AsKL0和非有限跳步0；三个监督U_seen/accepted实际0，三个普通PL约92.4%至92.8%接受，仍是训练中。[九行可读前缀梯度诊断](../20261026-phase1-PAMF-SIG1-REPL-RXOffset-source-m3-r01/available_training_diagnostic.json)。未产生新正式目标结果。


## 完整三种子正式结果

ANALYZED/VERIFIED。六个模型、全部正式预测与独立truth-last评分完成；controller与全部worker均退出，无重复训练或预测。下表为F模型seed2026092701/02/03的均值±样本SD，原F0 cross_rx126000+cross_both42000按正确数合并为每seed168000包；六景为六个practical平均。G固定同一原F0 DURA G-LU，未另训或更换。不是三个G种子，SD也不是显著性检验。

| 单视图完整PAMF | raw clean | raw六景 | 同G clean | 同G六景 |
| --- | ---: | ---: | ---: | ---: |
| SIG1-Sup | 84.1169±0.8876% | 70.1681±1.0352% | 84.1927±0.9115% | 76.9945±1.3371% |
| SIG1-PL | 84.6938±0.6925% | 70.2332±1.0433% | 84.7746±0.6815% | 77.4329±1.1174% |

seed内配对PL−Sup：rawclean +0.576984±0.219579pp，raw六景+0.065179±0.037350pp，同Gclean+0.581944±0.255139pp，同G六景+0.438327±0.265597pp。普通PL在本单视图设置主要提高clean和恢复后识别，原始星地六景收益很小；不将这些数值当作新REPL方法达标或统计显著性结论。

这是新建严格同架构单SIG_TIME基线。历史Sup/PL的X3结果和单视图其他配方全部保留，不能将架构差异说成半监督效果。单视图基线均值低于相应历史X3结果，也不因此降低原目标：同一新方法仍须rawclean≥85.981746%，raw六景>72.096197%，同G六景>78.074636%。折算相对本单视图PL，分别仍需至少+1.287897pp、超过+1.862963pp和超过+.641766pp。严格单视图比较和历史绝对门槛取更高者，见goal_comparison.json。

独立核验完整53280条新增训练记录、六份实际单视图final/defaulthead与20+10+20祖先链。每行L768/AsKL0/原crossRX日程与finite正确，Sup所有阶段U_seen/accepted=0；PL新30轮实际U曝光/接受完整且U真值隐藏。Sup各50新增轮，PL各30新增轮+各自20继承轮，均累计E50/11100，原前缀耗时另计。

核验全部14112000个冻结类别决策的ID顺序/数目/类别范围和2任务×2读出×7视图完整性；从5208个保存混淆单元重算accuracy/macroF1及恢复repair/damage/net一致性，再独立重算逐seed、三seed均值/样本SD和配对差，与原汇总一致。核验没有新模型前向、IQ或truth读取；原评分仅在每行完整预测后独立CPU执行。该核验不是重新读取truth做第二次评分。

新三角色REPL配方已在读取本次目标分数前固定并发布，控制结果不回流改变候选、权重、epoch或seed。原目标未完成，当前改进学生尚无目标结果。

证据：[完整执行与核验](complete_evidence.json)、[逐seed](per_seed.csv)、[完整分群均值SD](summary.csv)、[seed内配对差](paired_summary.csv)、[实际训练成本](execution_summary.csv)、[目标比较](goal_comparison.json)。
