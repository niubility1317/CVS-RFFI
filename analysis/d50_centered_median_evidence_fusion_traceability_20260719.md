# D50全局锚定类级中位数证据融合追踪

|需求|设计位置|验证计划|状态|
|---|---|---|---|
|保留D45全局证据中心|`z0=C×mean(d)`、`delta`类均值为0|定向测试重算`mean(z)=z0`，verifier逐artifact复算|VERIFIED_PRE_RUN|
|类级稳健support证据|每类held-rank差的确定性median|合成outlier、mean/median闭合、rank置换测试|VERIFIED_PRE_RUN|
|类标签置换等变|所有类同式且只用类内证据＋类对称中心|随机class permutation测试|VERIFIED_PRE_RUN|
|K1/K2边界|K1 D45 bitwise fallback；K2等价head 1:1|D50定向＋D46继承链测试|VERIFIED_PRE_RUN|
|单affine int8部署|继承D46一次FP32类级融合与D42编译|105行int8/FP32 argmax变化0/0，margin翻转0|VERIFIED|
|无query/role/scene/scan|固定公式与D42 runner fail-close|105行query0，role/quota/count/global assignment均false|VERIFIED|
|资源闭包|复用D46，scalar上界继承D47保守账|36 LDA fit、1 state、总适配1,077,329,226|VERIFIED|
|每版详细性能|D50报告第10–20节|完整读取D50/D45/D46各105行并生成summary|VERIFIED|

设计边界：median仅是support fold证据的稳健位置，不是校准posterior或query泛化保证；`mean(z)=z0`只保证全局log-odds锚定，不保证平均概率或任何性能不退化，必须由outer evidence判定。

运行前验证：D50定向8项、D46＋D47＋D50联合45项、D42–D50全链152项均exit0；代码复核P0=0、P1=0。新增公式未改变数据、B20、head、query view、量化器或runner。

运行后：105/105行、exit0、query0；D50 int8与matched FP32一致。非平凡权重范围before`0.314–0.682`、final`0.322–0.717`，但相对D45为0/15 outer SHA和0/330 final argmax变化，所有性能精确回到D45。new/min-new为84.00%/70.00%，rain after/forget为76.67%/13.33pp，多个晋级门失败；最终`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不进入125。
