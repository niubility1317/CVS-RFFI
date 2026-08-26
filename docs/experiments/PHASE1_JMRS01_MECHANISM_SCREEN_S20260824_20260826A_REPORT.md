# PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A完整实验报告（协议纠偏版）

## 一、结论先行

本实验已完成，最终状态为`ANALYZED`。唯一run产生428064条prediction，truth在prediction闭合后写入，独立scorer生成并成功解析8个JSON。7个source receiver的49个row/fold组合均完成clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`评估。全量stdout、owner日志和42条可训练曲线未发现Traceback、OOM、NaN、Inf、Killed或ERROR。

科学结论明确：本轮没有机制达到入池条件，`R1,R2,D1,P1,P2`均为`DO_NOT_PROMOTE`，不得启动后续`S1/S2`联合。R1是唯一保持Core90识别能力的候选，但没有降低receiver可预测性，TX margin仅保留62.20%；R2、D1、P1、P2的独立分类几何均显著坍缩。用户否定的D2线性/对数差分谱比值未被实现、未进入CLI、未进入矩阵、未产生实验结果。

本实验完成了“机制筛选”，没有完成“联合机制提升”。结果否定的是当前实现形式与训练目标，不是否定接收机校正、无符号谱结构或相位创新作为研究方向本身。

本次复核进一步纠正了结果的声明边界：JMRS01是“冻结Core90上的新增机制分支source-receiver LORO代理审计”，不是完整模型端到端LORO，也不是target receiver域泛化实验。M0的三LEO均值89.9233%来自`source V_select`、seen day0—1和source receiver0—6，只能作为机制筛选基线，不能写成ADV3B02的target DG、strict UDU或target-old三LEO成绩。该纠偏不改变任何prediction、评分数据或`STOP_JMRS01_S0_NO_POOL`结论。

## 二、研究问题与证据边界

本轮从PA单机制深挖转向接收机校正、无符号多尺度谱商、相位创新三个候选族，回答三个窄问题：候选是否在source-only、7折新增分支receiver LORO下携带跨source receiver的TX信息；是否降低receiver泄漏并保留Core90的TX margin；是否对Core90错误形成可被可靠门控的互补信息。

数据为WiSig/ManySig地面代理数据；LEO弱信道是物理启发压力代理，不是真实在轨数据。本实验是Phase1 source-only机制审计，不是Phase2 K-shot适配、Phase3多节点协同或真实卫星验证。target receiver、target day、query、truth反馈均未参与训练、选模、校准或gate，真实checkpoint smoke记录`target_or_query_access=false`。

### 2.1LORO究竟约束了什么

每一折都从新增机制分支的训练、`V_select`、`V_cal`、probe和几何统计中排除一个source receiver，再在该receiver的`V_select`样本上审计分支输出。这个隔离对R1、R2、D1、P1、P2和S1成立。

冻结Core90不满足端到端LORO：其历史checkpoint已经使用source receiver0—6和seen day0—1训练。held receiver只对新增分支未见，对M0及其`z_id/base_logits`并非未见。因此，本报告后续出现的“LORO”均指“新增机制分支LORO”，不得外推为backbone LORO或目标接收机DG。

### 2.2三LEO均值的定义

每个row/scenario使用同一组12600个source `V_select`物理样本；每个物理样本分别形成clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`视图。三LEO均值是三个弱信道场景Accuracy的简单算术平均，不是独立scorer定义的第四个场景，也不是三个互斥目标域数据集的总体Accuracy。Phase1允许这种配对压力测试，但它不能替代Phase2的单物理样本单LEO观测协议。

## 三、设计及逐项落地

|设计项|落地实现|实验行|状态|
|---|---|---|---|
|冻结Core90|严格加载195个state tensor，无missing、unexpected或shape跳过|M0|已落地|
|低秩接收机特征校正|IQ摘要+Core90身份表示生成有界32维校正、logit与可靠性；高置信Core90正确样本KL保持|R1|已落地|
|平滑接收机响应校正|16项DCT幅相响应、二阶平滑正则、分支专用校正视图|R2|已落地|
|无符号多尺度谱商|shift=1/2/4/8双向谱商；幅度、log幅度、相位通道；floor、mask、clip、coverage|D1|已落地|
|一阶相位创新|加权二次趋势去除、低幅度mask、4/8/16/32多尺度14项统计|P1|已落地|
|一阶+二阶相位创新|在P1基础上增加二阶创新|P2|已落地|
|同容量伪机制|确定性sham，参数量与D1同为6438|S1|已落地|
|线性/对数差分谱比值|未知发射符号且无合法同符号跨时刻配对，CLI显式拒绝|D2|已删除|
|新增机制分支source receiver LORO|held receiver从新增分支train、V_select、V_cal完全排除；fold内训练、选模、校准、probe和几何统计；冻结Core90不属于端到端LORO|R1/R2/D1/P1/P2/S1|已落地，声明边界已纠正|
|四场景闭合|每个held audit输出clean和三个`leo_*_weak`场景|全部|已落地|
|truth-blind评分|prediction先闭合，truth后写入，再由独立scorer连接|全部|已落地|
|后续联合|少于2个机制通过时禁止启动|S1/S2联合|未启动|

统一损失包含clean/satellite交叉熵、clean-satellite表示一致性、类条件receiver均值对齐、TX margin、R1身份保持、可靠性目标和R2平滑正则。safe fusion只在非held的V_cal上从`alpha={0,0.02,0.05,0.10}`选择，并受clean下降不超过0.30pp约束。

## 四、实验配置与闭合证据

- run ID：`PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A`
- 分支：`codex/phase1-jmrs01-20260826`
- 实验代码commit：`d17cf6fb8128b47f505fbd80e2fabfb7c8421284`
- release归档Git状态：`6be9748aae6e5ac109b3cc1ac9d1ee3f1112f3d2`
- 实验前Git头：`30529024cf368455c981589a13d403beab9a0495`
- release SHA-256：`d14d88067b11eb98d50623592c66f2483674f9ecda1a8bf1a5c7efae68221eaa`，本地与远端一致
- checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- seed：20260826；satellite seed：20260824
- epoch：200；batch size：128；evaluation batch size：256；learning rate：3e-4
- source角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- 样本数：`L_s=5880`、`U_s=52920`、`V_cal=12600`、`V_select=12600`
- 每折held audit 1800个clean样本；每个row/scenario共12600个样本
- 6个可训练row×7折×200epoch，共8400个训练epoch记录
- prediction：428064条；`truth_written_after_prediction_close=true`
- scorer artifact：8/8

M0没有训练，因此7个M0 history的`training_complete=false`和0 epoch表示冻结基线，不是失败。其余42个row/fold均为`training_complete=true`。

### 4.1实际数据切片

|维度|JMRS01实际值|是否进入本轮评分|
|---|---|---|
|TX|6个Phase1已知TX|是|
|source day|day0=`2021_03_01`、day1=`2021_03_08`|是|
|unseen day|day2=`2021_03_15`、day3=`2021_03_22`|否|
|source receiver|receiver0—6，共7个|是|
|target receiver|receiver7—11，共5个|否|
|评估角色|source `V_select`|是|
|正式named test|`test_unseen_day_seen_rx`、`test_seen_day_unseen_rx`、`test_unseen_day_unseen_rx`|否|

`target_or_query_access=false`证明本轮没有target/query泄漏，但它不证明本轮已经完成target测试。这里必须区分“没有使用目标域”和“在目标域上泛化良好”：前者成立，后者没有被JMRS01测量。

### 4.2与ADV3B02 Core90正式协议的同异

|项目|JMRS01|ADV3B02 Core90历史正式评估|判断|
|---|---|---|---|
|数据资产|ManySig|ManySig|同源|
|冻结checkpoint|`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|同一checkpoint lineage|相同|
|输入长度|256|256|相同|
|LEO场景族|三个`leo_*_weak`|三个`leo_*_weak`|名称和生成代码族相同|
|source角色比例|0.07/0.63/0.15/0.15|历史配置labeled=0.10、unlabeled=0.70、source val=0.20|不同|
|seed|20260826；satellite seed=20260824|checkpoint seed=392002；历史sat seed=2027；后续target-old重建使用713130/713912|不同|
|评估样本|source `V_select`，每场景12600个|正式named test或target-old池|不同|
|receiver|source receiver0—6|seen receiver0—6与unseen receiver7—11分场景报告|不同|
|day|seen day0—1|包含unseen day2—3|不同|
|聚合|7个held-source-receiver折|overall、strict UDU、receiver floor及目标接收机分场景指标|不同|

因此，JMRS01与ADV3B02共享数据资产、Core90 checkpoint、输入规格和LEO弱信道实现，但不共享正式测试样本、receiver/day切片、随机实现和指标语义。两者的绝对Accuracy不能按同row公平比较。

## 五、运行健康与完整日志诊断

唯一launcher PID为3196948，训练Python PID为3197723。运行期间连续核对PID父子关系、release CWD、完整cmdline、run根、GPU0、日志增长和artifact递增；结束后两者自然退出。GPU0上同时存在一个无关进程，未超过项目允许的每GPU两个训练进程，且未对其进行任何干预。

|检查项|结果|
|---|---:|
|stdout行数|51|
|owner日志行数|157|
|Traceback/OOM/NaN/Inf/Killed/ERROR|全部0|
|scikit-learn ConvergenceWarning|49|
|history非有限数|0|
|可训练fold|42/42|
|模型文件|42/42|
|评分JSON|8/8|

49条ConvergenceWarning来自receiver probe中的MLP达到300次迭代上限，每个row/fold一条。linear、MLP、KNN三类probe均已输出，warning没有中断scorer；但MLP数值只能解释为当前预算下的结果，不能宣称优化完全收敛。

### 5.1训练曲线

|row|epoch记录|平均首epoch loss|平均末epoch loss|首epoch V_select|末epoch V_select|最佳epoch范围|非零alpha折数|
|---|---:|---:|---:|---:|---:|---:|---:|
|R1|1400|1.5117|0.1082|98.35%|98.31%|1—90|0/7|
|R2|1400|1.8722|0.3764|21.82%|94.43%|180—200|2/7|
|D1|1400|1.8154|0.1728|41.67%|96.57%|190—200|0/7|
|P1|1400|1.9203|1.2763|18.42%|63.59%|170—200|0/7|
|P2|1400|1.8991|1.1578|23.42%|68.81%|160—200|0/7|
|S1|1400|2.0692|0.5184|31.06%|92.21%|150—200|1/7|

R1从首epoch起接近最终V_select，后续主要降低loss而没有选模增益。R2、D1、P1、P2和S1在source内层持续拟合，但没有转化为held-receiver能力。首个可定位分化是“V_select持续上升而LORO仍远低于Core90”，即泛化目标与训练目标错配；42条曲线均有限且总体下降，不是优化器爆炸或数据异常。

## 六、完整新增分支LORO代理结果

数值为source `V_select`、seen-day、held-source-receiver切片上的macro accuracy/macro F1/receiver floor，单位为%。M0没有新增分支，其行只作为同一切片上的冻结Core90参照。

|row|场景|Accuracy|Macro-F1|Floor|
|---|---|---:|---:|---:|
|M0|clean|98.3889|98.3886|95.8889|
|M0|leo_clear_weak|92.4921|92.4166|81.8333|
|M0|leo_low_elev_weak|88.9206|88.7882|76.6111|
|M0|leo_rain_weak|88.3571|88.1707|76.5556|
|R1|clean|98.2222|98.2220|95.2222|
|R1|leo_clear_weak|92.0556|91.9762|78.7222|
|R1|leo_low_elev_weak|88.3254|88.2008|74.2222|
|R1|leo_rain_weak|87.8968|87.7268|74.0000|
|R2|clean|59.0476|58.6684|33.7778|
|R2|leo_clear_weak|51.2222|50.8129|28.1667|
|R2|leo_low_elev_weak|48.4286|47.9101|25.8889|
|R2|leo_rain_weak|48.1429|47.6388|28.3333|
|D1|clean|55.9841|55.9833|35.7778|
|D1|leo_clear_weak|55.5714|55.8011|35.7222|
|D1|leo_low_elev_weak|53.2460|53.4813|33.7778|
|D1|leo_rain_weak|52.6825|53.0292|34.4444|
|P1|clean|37.1746|36.3410|21.3333|
|P1|leo_clear_weak|30.3730|29.3629|20.2778|
|P1|leo_low_elev_weak|28.7540|27.4208|20.5000|
|P1|leo_rain_weak|27.9206|26.5089|20.5556|
|P2|clean|40.5635|39.6874|19.2778|
|P2|leo_clear_weak|32.7063|31.2349|19.2778|
|P2|leo_low_elev_weak|30.5952|29.1844|17.7222|
|P2|leo_rain_weak|29.5476|28.0146|18.7222|
|S1|clean|49.3016|49.3788|22.7778|
|S1|leo_clear_weak|39.6429|39.6486|22.6667|
|S1|leo_low_elev_weak|38.0952|38.0173|20.7778|
|S1|leo_rain_weak|36.7778|36.9165|22.1667|

### 6.1相对同切片Core90

|row|clean|三LEO诊断均值|clean差值|LEO诊断均值差值|最低floor|
|---|---:|---:|---:|---:|---:|
|M0|98.3889|89.9233|0.0000|0.0000|76.5556|
|R1|98.2222|89.4259|-0.1667|-0.4974|74.0000|
|R2|59.0476|49.2646|-39.3413|-40.6587|25.8889|
|D1|55.9841|53.8333|-42.4048|-36.0899|33.7778|
|P1|37.1746|29.0159|-61.2143|-60.9074|20.2778|
|P2|40.5635|30.9497|-57.8254|-58.9735|17.7222|
|S1|49.3016|38.1720|-49.0873|-51.7513|20.7778|

R1是唯一接近M0的独立预测器，但三个LEO场景全部下降，最低floor下降2.56pp。D1相对自身clean的LEO退化较小，但绝对Accuracy只有52.68%—55.57%，不能把“相对稳”写成“识别有效”。

### 6.2为什么M0三LEO诊断均值较高

M0的三场景Accuracy为92.4921%、88.9206%和88.3571%，算术平均为89.9233%。高值来自较容易的审计切片，而不是新增机制带来的提升：Core90历史训练已经见过receiver0—6和day0—1；本轮又从source `V_select`取样，仅施加LEO弱信道压力。其最低receiver floor仍只有76.5556%，说明89.9233%的平均值掩盖了明显的receiver差异。

作为历史边界参照，ADV3B02 Core90正式闭集DG曾报告overall=89.18%、strict UDU=84.89%、receiver floor=75.55%、satellite strict floor=68.77%。后续严格target-old直接测试的三个LEO Accuracy为76.13%、70.83%和73.75%，均值73.57%。这些结果使用不同目标切片，不能与89.9233%做候选优劣检验；它们只证明“source诊断均值”不能被命名为target DG成绩。

## 七、几何、receiver泄漏与互补性

### 7.1身份几何

|row|D_TX|D_RX|D_day|D_sat|稳定性比值|margin保留|
|---|---:|---:|---:|---:|---:|---:|
|M0|11.5779|1.4513|1.1467|4.2677|1.6916|100.00%|
|R1|7.2018|0.3481|0.2691|1.1378|4.2689|62.20%|
|R2|3.5926|2.3424|1.5585|3.0323|0.5182|31.03%|
|D1|5.1043|2.7599|1.6555|2.0628|0.7884|44.09%|
|P1|2.3140|2.1434|1.3622|2.2514|0.4032|19.99%|
|P2|2.6233|2.2977|1.3961|2.3240|0.4381|22.66%|
|S1|3.9973|3.1290|1.9040|2.6803|0.5190|不适用|

R1压低D_RX、D_day和D_sat，但也把D_TX从11.5779压到7.2018，说明它同时压缩并重排了身份几何，不能只凭D_RX下降认定完成receiver去偏。

### 7.2receiver probe

|row|最佳balanced accuracy|归一化泄漏|相对M0泄漏下降|
|---|---:|---:|---:|
|M0|56.21%|0.4746|0.00%|
|R1|73.29%|0.6795|-43.18%|
|R2|86.57%|0.8388|-76.76%|
|D1|81.81%|0.7817|-64.72%|
|P1|70.96%|0.6516|-37.30%|
|P2|72.08%|0.6650|-40.12%|
|S1|87.31%|0.8478|-78.65%|

所有候选都使receiver更容易预测。R1的KNN达到73.29%，说明局部邻域仍保留receiver簇；均值距离缩小不等于receiver不可预测。R2、D1和S1接近完全可分，直接否定当前表征作为receiver-invariant身份分支的资格。

### 7.3互补性与可观测性

|row|Core90错时正确率|oracle gain|相对S1差值|95%CI|coverage|全覆盖utility|
|---|---:|---:|---:|---|---:|---:|
|R1|11.86%|0.9444pp|50.67pp|[46.15,55.25]pp|99.9960%|-0.4147pp|
|R2|29.96%|2.3849pp|10.76pp|[7.60,13.81]pp|100.0000%|-40.3294pp|
|D1|31.38%|2.4980pp|13.42pp|[10.60,16.49]pp|100.0000%|-37.6687pp|
|P1|16.20%|1.2897pp|-9.90pp|[-13.49,-6.62]pp|100.0000%|-60.9841pp|
|P2|16.15%|1.2857pp|-7.60pp|[-11.39,-4.36]pp|100.0000%|-58.6865pp|
|S1|25.02%|1.9921pp|0.00pp|[0,0]pp|100.0000%|-51.0853pp|

R2和D1确实包含少量Core90漏掉的信息，但当前可靠性无法从大量错误中提取它。R1在25%高可靠覆盖时准确率99.95%，utility仍为负；其可靠性更像识别容易样本，不是定位Core90可纠正错误。P1/P2相对sham的CI下界也小于0。

## 八、成本

|row|参数量|ms/sample|
|---|---:|---:|
|M0|0|0.00173|
|R1|10711|0.01858|
|R2|4454|0.00843|
|D1|6438|0.01627|
|P1|2086|0.03291|
|P2|3878|0.06173|
|S1|6438|0.00468|

P2运行时约为R1的3.32倍，却没有可用识别或互补收益。成本不是淘汰主因，科学门槛已经失败。

## 九、逐行gate判定

预登记要求同时满足：相对S1的LORO CI下界>0、receiver泄漏下降≥20%、margin保留≥90%、safe clean下降≤0.30pp、至少4/7 receiver不劣化、oracle gain≥0.30pp、coverage≥30%、正收益覆盖至少2个receiver、2个day、2个LEO场景。

|row|CI|泄漏|margin|safe clean|不劣化receiver|oracle|coverage/广度|结论|
|---|---|---|---|---|---|---|---|---|
|R1|通过|失败|失败|通过|7/7，通过|通过|通过|DO_NOT_PROMOTE|
|R2|通过|失败|失败|通过|7/7，通过|通过|通过|DO_NOT_PROMOTE|
|D1|通过|失败|失败|通过|6/7，通过|通过|通过|DO_NOT_PROMOTE|
|P1|失败|失败|失败|通过|0/7，失败|通过|通过|DO_NOT_PROMOTE|
|P2|失败|失败|失败|通过|1/7，失败|通过|通过|DO_NOT_PROMOTE|

入池数为0，低于启动联合所需的2个机制，因此不启动联合、不做多seed扩展、不做完整125矩阵、不恢复D2。

## 十、暴露的问题与根因

### 10.1safe fusion主要是拒绝候选

42个可训练fold中只有R2的2折和S1的1折选择`alpha=0.10`，其余39折为0；R1、D1、P1、P2全部7折为0。`safe_clean_drop_pp`通过主要因为校准器回退Core90，不是候选产生安全增益。今后必须分开报告“非零采用率上的净收益”和“允许回退的安全性”；alpha=0只能证明候选被拒绝。

### 10.2内层选模不代理held-receiver泛化

R2和D1末epoch V_select达到94.43%和96.57%，held clean却只有59.05%和55.98%。训练没有数值崩溃，当前CE与辅助正则允许分支学习receiver相关source捷径。后续选模必须直接约束fold内receiver可预测性和TX margin，不能只看V_select分类准确率。

### 10.3均值对齐没有消除局部receiver结构

R1的D_RX从1.4513降到0.3481，receiver probe却从56.21%升到73.29%。类条件均值对齐只压缩一阶中心差异，KNN仍能恢复receiver局部簇。应研究局部分布、条件协方差或邻域混合，而不是继续加大均值对齐权重。

### 10.4无符号不等于身份可辨识

D1、P1、P2的receiver probe为81.81%、70.96%、72.08%，margin只保留44.09%、19.99%、22.66%。这些统计在当前窗口和预处理下更容易编码接收链路/采集条件。该结果支持继续删除依赖未知符号的D2，也否定了“无需符号即自然稳健”的假设。

### 10.5receiver3是共同薄弱点

M0在三个LEO场景的最低floor均来自receiver3，R1进一步下降。后续可把它作为统一机制诊断重点，但不得按receiver ID设置专属权重、阈值或分支；修复必须对receiver标签置换保持同一形式。

### 10.6probe MLP未完全收敛

49个MLP probe触发300迭代上限warning。KNN或线性probe已给出同方向高泄漏，主结论不依赖MLP完全收敛。未来若精确比较小幅泄漏变化，可提高离线probe预算；这不是重跑本实验的理由。

### 10.7manifest是阶段封存而非最终状态

`run_manifest.json`保留`PREDICTIONS_COMPLETE_TRUTH_NOT_SCORED`，因为它在truth连接前封存。最终`ANALYZED`由owner日志终止标志和8个scorer JSON共同证明。后续可新增独立`score_summary.json`，但不得回写原manifest。

## 十一、逐机制结论

- **M0**：同一source诊断切片上的唯一可用基线，clean 98.39%，三LEO诊断均值89.92%，最低floor 76.56%；不是target DG结果。
- **R1**：clean仅下降0.17pp、三LEO诊断均值下降0.50pp；但泄漏恶化43.18%、margin仅62.20%、全覆盖utility为负，不入池。
- **R2**：source内层拟合充分，held clean下降39.34pp，泄漏恶化76.76%，失败。
- **D1**：oracle gain最高为2.50pp，但独立识别坍缩、泄漏恶化64.72%，仅可作离线诊断，不进入联合。
- **P1**：相对sham CI为负、0/7 receiver不劣化、margin仅19.99%，停止当前实现。
- **P2**：略好于P1但仍大幅失败且运行时最高，停止当前实现。
- **S1**：容量对照完成任务，证明增加分支容量不会自然产生稳定TX表征。
- **D2**：保持删除，不恢复。

## 十二、下一步路线

当前不发布新实验，也不启动联合。已有候选在更容易的source机制筛选中已失败，直接消耗目标域确认预算不能挽救候选，也不能改变入池判定。若后续重新进入该方向，应按两级证据设计推进：

### 12.1G0：新增机制分支source筛选

1. 将安全回退与候选有效拆成两个指标，要求非零alpha覆盖上具有正utility；
2. 把fold内局部receiver不可预测性纳入选模；
3. 对D1/P1/P2先验证TX条件信息是否超过receiver/day条件信息，再训练分类头；
4. 保持D2禁用，除非未来数据提供已知符号和合法同符号配对；
5. 仍从单seed小矩阵开始，达到预登记门槛后才扩展。

### 12.2G1：冻结候选的正式DG确认

只有新候选通过G0后，才能把source receiver0—6、day0—1上的训练和选择全部冻结，再一次性评估与Core90相同的三个named test：

1. `test_unseen_day_seen_rx`：day2—3、receiver0—6；
2. `test_seen_day_unseen_rx`：day0—1、receiver7—11；
3. `test_unseen_day_unseen_rx`：day2—3、receiver7—11，即strict UDU。

G1必须在候选与M0之间复用同一物理样本ID、LEO scenario、satellite seed、预处理和评分公式，分别报告clean及三个LEO场景的Accuracy、Macro-F1和receiver floor。三LEO算术均值只能作为附加摘要，不能替代场景级结果。候选必须同时满足目标接收机均值提升、strict UDU不下降和receiver floor不下降，才有资格讨论域泛化收益。

这不是立即启动授权。正式决策为`STOP_JMRS01_S0_NO_POOL`。

## 十三、优化修改追踪

|ID|修正要求|落地位置|状态|验证|
|---|---|---|---|---|
|C01|纠正“域泛化实验”命名|第一、二、六、十一节|verified|与实际`source V_select`取数路径一致|
|C02|区分新增分支LORO与Core90 lineage|2.1节、设计落地表|verified|held receiver仅从新增分支角色排除|
|C03|列明receiver/day/role实际切片|4.1节|verified|与`protocol_and_smoke.json`一致|
|C04|逐项比较JMRS01与ADV3B02协议|4.2节|verified|与历史resolved config及named test定义一致|
|C05|保留全部实验数据并纠正89.9233%语义|第六至十一节|verified|原8个scorer JSON数值未改写|
|C06|给出后续正式DG验证设计|第十二节|verified|G0/G1证据层级、切片和门槛均明确|
|C07|重复运行当前JMRS01|无|rejected|原run已`ANALYZED`，重复运行不会修复声明边界|

追踪统计：verified=6、deferred=0、rejected=1、blocked=0。本次是对已完成实验的严格证据纠偏，不是新方法实现；代码、prediction、truth和scorer artifact均未修改。

## 十四、证据位置

- 根报告：`E:\type10-7\automation_reports\CV-SincNet\PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A\report.md`
- Git镜像：`docs/experiments/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A_REPORT.md`
- 本地小型artifact：`E:\type10-7\local_artifacts\PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A`
- N607 run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_jmrs01_20260826/PHASE1_JMRS01_MECHANISM_SCREEN_S20260824_20260826A`
- N607日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_jmrs01_20260826`
- scorer JSON：identity stability、receiver probe、LORO metrics、clean-satellite consistency、complementarity、observability、cost、decision，共8项

## 十五、最终状态

`ANALYZED / STOP_JMRS01_S0_NO_POOL / NO_JOINT_LAUNCH`
