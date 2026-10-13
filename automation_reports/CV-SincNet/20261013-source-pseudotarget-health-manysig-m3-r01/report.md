# 冻结源域伪目标健康与浅中层 RX 探针

84%目标尚未达成，上一轮完成固定四组/三种子/30轮及全量评分，是实际进展。当前登记一个中间源域诊断，不缩减最终目标，也不把source健康指标当成84%完成。

## 研究问题与固定输入

当前U已实际展示并覆盖56700样本，累计约86%接受、末轮约96%；原UNIFORM_SSL独立U梯度非零。因此不按“U没执行/饿死”的假设改门控。已核实当前身份路径实际为residual_fusion/reference_response，67身份张量177025参数全部优化，无死支路/BatchNorm；外层域骨干冻结且不进入身份forward是现配方。只读核查不使用target错例。

全部三个MATCHED_BASE FINAL30固定模型只读加载；L_s6300、V27000、U_s56700，三个seed各90000源输入，总270000模型forward。数据角色/物理ID按未变契约复用，U TX标签始终隐藏。source_proof/own_model检查完整训练来源、固定30轮/6660更新、无祖先与角色匹配后才加载。无target index/IQ/truth/预测/score访问；零identity更新。

## 诊断矩阵

对L/V合法标签计算准确率、NLL、预固定confidence分箱及.95覆盖/高置信错误。U只计算confidence/entropy/伪TX预测分布/每RX/day分组，无准确率。三模型都报告自身弱eval目标和固定另两模型等权概率目标的同意/分歧；在L/V统计peer救回/新增错误，不能把这种最终状态扫描称为历史EMA或相邻pass轨迹。

提取冻结浅time_down、中t2.pool的时间mean/std和160joint。RX线性探针分别无条件、每TX条件；训练只能用L。每TX/RX/day按物理ID排序mod3，fold0/1训练、fold2诊断；V仅评估，不能训练探针、调超参、晋级或选模型。固定100全批更新、AdamW lr.01/wd.001、seedprimary+31337。Identity参数/缓冲/mode/RNG前后精确保持；探针不成为部署状态。

## 成本、范围与后续依据

本次只形成源机制证据。后续备选为自己的历史权重EMA目标或对称独立online peer，是否值得先检验由源健康证据固定；不存在已有历史EMA的FINAL替身。已有SELF_EMA来自不同heldRX/阈值/权重契约，不能冒充当前匹配对照。EMA理论参照：[Mean Teacher原论文](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html)。类条件域判别可作为另一源证据问题，相关方法原用于适应，本项目不读取目标训练数据：[CDAN原论文](https://proceedings.neurips.cc/paper/2018/hash/ab88b15733f543179858600245108dd8-Abstract.html)。文献不保证本项目收益。

sourceV只健康，U无真值，全部输入和阈值在诊断前固定。本扫描不能恢复未记录的训练时翻类，不能证明纯RX物理因果，不能取代未知RX或最终目标验证。一次合法源诊断不创建新的target测试或性能筛选。

一个owner/一个不可覆盖root，普通N607账号、单GPU总进程≤3且free≥12000MB、保护其他任务。代码/config一次提交push，聚焦负例/真实checkpoint无query首步、一次P0/P1、单archiveSHA/remotecompile、启动PID/CWD/argv/GPU/log增长；仅数据/来源/越界/非有限/输出冲突等技术失败停止。命令、环境/CWD、GPU策略、精确输入路径与预期输出见experiment.json。

|ID|验收|状态|证据边界|
|---|---|---|---|
|H1|固定3model合法同契约、仅源L/V/U|verified|代码/真实noquery首步待核实|
|H2|L/V质量与U truthblind分组|verified|组件/完整270000源扫描待返回|
|H3|另两seed固定peer信息，不伪造EMA/历史|verified|同row源统计待返回|
|H4|浅/中/joint RX probe仅L fold0/1训练|verified|组件fold/梯度与真实源probe待返回|
|H5|冻结identity状态/RNG，V无更新/无晋级|verified|实际诊断待返回|

CPU/CUDA组件检查与compileall PASS：U标签拒绝/合法概率与置信错误精算/固定peer覆盖/Lfold2排除/实际96、192、160特征接口/21probe每seed/identity状态与RNG保持/覆盖与登记差异拒绝。一次独立P0/P1发现CPU预检未显式传device，修复后定点复核PASS；实际REMOTE AST确认preflight CPU、正式run CUDA。未读取target IQ/truth/预测/评分。Lfold2仅对新线性RX探针独立，冻结identity历史用过全部L；本探针不被称为身份新盲测。peer_high是另两模型T1平均概率max≥.95，peer_agree是两argmax相同；未附加每个peer单独.95条件。63个probe固定训练参数无选择，identity不更新。

真实源诊断全矩阵已完成：270000 forward、63probe/189混淆、全量源概率/分组重算、完整stdout及冻结身份/RNG核验VERIFIED。H1至H5全部verified；目标84仍active。实际结果见final_analysis.md。
