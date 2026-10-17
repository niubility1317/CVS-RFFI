# 匹配 E30 No-As 参考完整结果

20261018-phase1-F-crossRX-matchedNoAs-e20plus10-s2026092701-m1-r01

状态 ANALYZED／VERIFIED。单一固定模型 seed，无新 IQ A_g。参考在读取当前 As 三行正式目标成绩前登记；没有按成绩晋级、重跑或修改配方。

|模式|clean %|high sub %|mid sub %|low sub %|high urban %|mid urban %|low urban %|
|---|---:|---:|---:|---:|---:|---:|---:|
|B00|85.245238|46.006548|44.314286|42.221429|44.751190|37.279762|31.712500|
|ORIGINAL|82.722619|81.605357|79.336905|76.969643|78.224405|64.376786|53.458333|
|MEASURE|82.691071|81.598214|79.344048|76.973214|78.211310|64.407738|53.443452|
|PROJECT|82.776786|81.572619|79.335119|76.983333|78.200000|64.379762|53.429762|

|As 模式相对 B00|clean pp|mid urban pp|low urban pp|
|---|---:|---:|---:|
|ORIGINAL|-2.522619|+27.097024|+21.745833|
|MEASURE|-2.554167|+27.127976|+21.730952|
|PROJECT|-2.468452|+27.100000|+21.717262|

|模式|high suburban距clean pp|mid suburban距clean pp|low suburban距clean pp|
|---|---:|---:|---:|
|B00|39.238690|40.930952|43.023810|
|ORIGINAL|1.117262|3.385714|5.752976|
|MEASURE|1.092857|3.347024|5.717857|
|PROJECT|1.204167|3.441667|5.793452|

数据与模型：全部五源 RX [1,3,4,6,8]、三日、L6300/U56700 隐藏 TX/V27000，split392005，无训练 RX 留出或 L_dev 划分；仅现有 256 点 IQ。F＋跨 RX 同 TX SupCon、SIG_TIME 三路/统一 Sinc/joint160/既有 cosine 头联合训练，NoExplicitCFO。

匹配边界：No-As 用原 fullsource REPEAT E20 student＋自身 EMA；As 三行用原 CONCAT E20 student＋自身 EMA。两父模型按同一原始 scratch seed、完整源契约、预算及初始化规则训练，因增强处理而权重不同。均追加 E10、重置 AdamW/cos10、固定 final30，不声称精确恢复原优化轨迹。

四次 L 前向、物理序列、隐藏 U 与教师规则保持；B00/MEASURE/PROJECT 每步额外两次 VJP，ORIGINAL 每步无额外 VJP，因此严格额外求导成本匹配以 MEASURE 为主要 As 控制。As 的传播处理成本属于处理差异。每个 RDrop 对内复用已生成 IQ；不声称不同模型全部浮点轨迹逐位一致。

L 损失质量明确：B00 为 .5ordinaryRD＋.5同一ordinaryIQ的额外RD＋.1SupCon；As 为 .5ordinaryRD＋.5完整practical(raw)RD＋.1SupCon。旧普通增强保留，所以 B00 不是完全无增强。As 视图直接从原 IQ 生成，没有在旧 ordinary 后重复加完整信道。此比较检验该固定 .5/.5 配方；不能把其 clean 损失解释为所有星地增强的必然代价。

完整源日志、370 个输入池、4440 条 L/U 序列和两父来源已只读核对；全部七视图完成后独立 truth-last。504 个混淆矩阵、84 个分区和21 个 As-vs-B00 配对比较复算，无原始 truth／IQ／checkpoint 读取或新增网络计算。全部 RX/TX/day 保留见 stratified_results.md。

No-As 新训练和预测 NN 3770644，另列真实 CP smoke2、开发合成 NN2048／更新4／VJP4；自身父 E20 NN 3661082。新增真实更新2220、额外训练VJP4440，含自身父累计6660更新/E30。原 As 三行及其共享父成本在原报告单列，不将父训练或开发成本计为0。

已关闭的固定梯度规则保持关闭。这里没有新 IQ A_g、B10/B11 新增强模型、Ag×As 交互或跨 seed 证明；完整联合目标未完成。

本轮判读：普通 As 相对 B00 clean −2.522619pp、mid urban +27.097024pp、low urban +21.745833pp。low urban 的 clean∩scene 共同正确率仍净增19.779167pp，因此不是仅将 clean 错误转移到 satellite；但 clean 净收益条件明确未满足。No-As 为 85.245238%／37.279762%／31.712500%，普通 As 为 82.722619%／64.376786%／53.458333%。这是单种子配方效应，不外推为跨种子证明或不可约准确率上限。

执行代码 00cc9bad869d98bbca408858a4adec55887cf6c7；分析工具 be259183d95f0c5a08b60372480f7b2a51ee4c85。图件见 figures/matched_view_results.png 与 figures/matched_training_curves.png。
