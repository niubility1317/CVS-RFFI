# 分离 Adam 状态 E30：技术失败，匹配性能无法判定

状态 FAILED；run 20261019-phase1-FcrossRX-splitAdam-e30-s2026092701-m3-r02；实际训练代码 3a984b391e60a3065165d4975f9bcb90fde6d1fe。

本轮固定完整 practical、ordinary/raw、同 IQ RDrop、ordinary128 CROSS_RX SupCon、原 U／自身 EMA 与默认 cosine 头；只比较共享 AdamW 和 rest/As 独立一二阶矩后 0.8/0.2 合成一次更新。没有新增网络、损失、部署参数、标签权限或测试信道改动。

|行|用途|成功新增更新|完整新增 epoch|固定最终模型|七环境评分|
|---|---|---:|---:|---|---|
|N01|NoAs／共享AdamW／两VJP|2220|10|已冻结，累计E30|完成|
|C01|完整As／共享AdamW／两VJP|1645|7|缺失|缺失|
|S01|完整As／分离Adam状态|2181|9|缺失|缺失|

C01第1646批、S01第2182批在原生physical clip后、optimizer.step前退出，均为 Component sum differs from actual clipped total gradient (parameter RMS)。dispatcher及三行进程自然退出，输出保留，未操作其他owner。失败的逐参数比较仅在内存，未落盘，故具体参数和误差幅度未知。R01首批重复VJP诊断只解释旧逐坐标nearzero误差，不能证明本次后期RMS失败也是误报。按预登记同类故障规则不重发R03，不改容差／种子／父状态／比例补救。

星地提升多少、clean代价多少、是否满足下降≤0.1个百分点，均**无法判定**，不是0，也不是方法被科学否定。源预定条件为S−C两draw mid/low均值≥+0.5pp、S−N完整V clean≥−0.1pp；C/S缺少最终源读出，两条件未判定，不具备E50或扩种子资格。source_decision.json先保存技术失败判定，再读已完成N01正式分数；目标分数未用于方案调整。单个模型seed2026092701，无可重复双提升证据，目标ACTIVE。

## 唯一完成行：N01 NoAs

以下不是候选表现。每视图168000固定物理样本，全部7视图prediction固定后独立truth-last评分，正式IQ／六种practical保持。

|视图|准确率 %|macro F1 %|最差 RX %|
|---|---:|---:|---:|
|clean|85.304762|85.035414|78.587500|
|practical_high|42.738095|39.283223|34.300000|
|practical_mid|41.032738|37.936164|33.491667|
|practical_low_suburban|39.466667|36.725380|32.016667|
|practical_high_urban|41.462500|38.439647|33.987500|
|practical_mid_urban|35.129167|33.468523|29.500000|
|practical_low_urban|30.129167|29.243347|25.779167|

urban均值 35.573611%；suburban与自身clean差距：practical_high 42.566667pp、practical_mid 44.272024pp、practical_low_suburban 45.838095pp。全部RX/day/TX分层与126个CM见technical_failure.json；从既有CM重算accuracy／F1一致，新增NN／truth读取0。C/S及所有配对修复、新增错误、交互项缺失，不能从N01推断。

## 完整日志与成本

完整解析step_metrics 2220／1645／2181条，training_sequence 4440／3292／4364条，epoch_metrics 10／7／9条和全部stdout。N01七份分量日志各2220条，6L、同IQ RDrop、原SupCon／U、两VJP、一次clip和默认头联合训练计数一致，U TX可见性始终0。无loss／grad非有限或OOM异常。C/S的分量缓冲因finish未执行未落盘，不能写成0。

|行|L前向|U学生前向|U教师前向|完整epoch验证＋初始检查|native总前向|额外VJP|成功更新|
|---|---:|---:|---:|---:|---:|---:|---:|
|N01|1704960|567000|567000|324002|3162962|4440|2220|
|C01|1264128|420452|420452|226802|2331834|3292|1645|
|S01|1675776|557404|557404|291602|3082186|4364|2181|

N01为实际counter；失败行按完整loader物理计数与固定调用路径重建：6L、1U学生＋教师，每完成epoch V27000＋geometry5400，初始2NN；最后失败批次NN／两VJP计入，optimizer／EMA更新0。公式与N01实际counter一致，不冒充失败进程已丢失的内存计数器。

R02 native8576982NN、源面板12600NN、正式1176002NN、共享CPU真实父模型synthetic smoke2358NN，合计9767942次样本前向、12102次额外VJP、6046次成功训练更新。计入R01失败26684NN／44VJP／16更新与零更新诊断1282NN／4VJP，本路线新增共9795908NN／12150额外VJP／6062成功训练更新。CPU玩具12次参数更新NN0，单列不算模型训练；两份原E20父模型复用不重复收费。

N01原生训练1226.167976秒，正式预测50.460308秒。失败行完整墙钟、源面板墙钟及并行总历时缺少可靠终态记录，保留null，不填0或猜测；同epoch／VJP数不等于同墙钟。

原发布的61core、96协议负例、actualTorch28检查、真实父模型无query smoke、一次P0/P1及唯一数值修复复查PASS；归档传输／远端编译／PID-CWD-argv-GPU-log增长VERIFIED。它们未覆盖这次后期RMS失败，不能用作训练或科学目标完成证据。不新增重复审查或发布门槛。

本轮证实技术未完成与成本，尚未检验完成分离状态的识别假设；此实现停止继续消耗训练预算。普通完整practical仍是已有收益的基础。本轮不启动第二机制；下一项须独立去重、依据源证据，不能用N01正式分数选方向。另一对话质量条件熵目标与ordinary噪声／多径旁路已进入实际实验，不换名重复。

缺项：C/S固定FINAL30、源面板、14个正式行视图、全部候选配对、跨模型seed复现及最终双提升。完整源日志摘要见source_log_summary.json，终态／成本／N01全部指标见technical_failure.json。
