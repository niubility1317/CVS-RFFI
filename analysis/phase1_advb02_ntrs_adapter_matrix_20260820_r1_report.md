# ADVB02 NTRS Adapter-Only首轮实验矩阵详细报告

## 一、核心结论

本轮实验已经结束，并完成了训练后独立测试，不是仅完成训练。

- A0、A0-B、A1-R、A1、A2共5个有效实验行均训练至E200，`train_exit=0`、`eval_exit=0`。
- 每行均保存最终checkpoint，并完成clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`全量独立测试；checkpoint加载均为`missing_keys=0`、`unexpected_keys=0`。
- 最强可归因证据来自3个adapter行内部的冻结raw→always-on fused逐样本对照。A1的LEO均值下降0.061pp、Strict UDU均值下降0.113pp；A2的LEO均值下降0.134pp、Strict UDU均值下降0.286pp。
- A1虽然q梯度有效、raw参数零漂移，但其LEO总计`rescued=2322`、`harmed=2696`，净损失374个正确样本；A2为`rescued=4141`、`harmed=4964`，净损失823个正确样本。
- A1与随机冻结q的A1-R几乎相同：LEO均值仅高0.003pp、Strict UDU仅高0.019pp，说明可学习q已经更新，但没有形成可测的nuisance建模收益。
- A2增大修正上限并加入teacher KL与margin后，修正角度和raw/robust分歧明显增大，性能反而进一步下降。

run状态已由`ARTIFACTS_COMPLETE`推进为`ANALYZED`；科学判定为`ANALYZED_NEGATIVE_NO_GO_ADAPTER_V1`。A2未达到预登记晋级门槛，因此不发布A3/A4。当前应继续保留成熟D1的raw路径作为本矩阵最强有效基线：Clean88.234%、LEO均值71.768%、Strict UDU均值64.379%。

## 二、实验完成与证据闭合

原run ID`phase1_advb02_ntrs_adapter_matrix_20260820_r1`中的A0/A0-B曾因非v3训练参数统计路径缺少`nn`导入而启动即失败，未产生性能结果。该失败尝试完整保留，但不纳入下表。修复后两行使用不可覆盖run ID`phase1_advb02_ntrs_adapter_matrix_20260820_a0_fix1`重跑。

|profile|有效run ID|训练终态|测试终态|epoch记录|最终checkpoint SHA256|
|---|---|---|---|---:|---|
|A0|`phase1_advb02_ntrs_adapter_matrix_20260820_a0_fix1`|E200，exit 0|clean＋3个LEO_WEAK，exit 0|200/200|`8bdb7f99192b8962d04865172e00616bfb5f66ce4412b385a65c700c3dbc7dbd`|
|A0-B|同上|E200，exit 0|clean＋3个LEO_WEAK，exit 0|200/200|`16dd777977f4bd04750a8722771567a22a44a17e246bc6ba5c6d9906c7c845e9`|
|A1-R|`phase1_advb02_ntrs_adapter_matrix_20260820_r1`|E200，exit 0|clean＋3个LEO_WEAK，exit 0|200/200|`6b7ebd2d04904a6af33d67a2a4ba3e7a832b6342fccbaf49abe65e79cfa964b6`|
|A1|同上|E200，exit 0|clean＋3个LEO_WEAK，exit 0|200/200|`7cab69afc01279ddd83dd92a6e5273e6b2ede5551b4ee5416f15d23e78d43ce0`|
|A2|同上|E200，exit 0|clean＋3个LEO_WEAK，exit 0|200/200|`f863e8487a7d15329db2dcb9e5cb87d408349720404f79a33d8daefdb6bf9fd0`|

5个训练日志均有9019行、400个epoch起止marker；5份CSV和5份JSONL均为连续E1–E200，共200条结构化epoch记录。完整日志未发现Traceback、RuntimeError、ValueError或AssertionError。训练后N607上已无本矩阵所属进程。

## 三、协议与评测范围

- 阶段：Phase1 source-only，不访问target support、query或truth，不形成Phase2适配证据。
- 数据：`Dataset_WigSig/ManySig.pkl`。
- seed：`392034`。
- 训练角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 训练与测试信道：只使用`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；本轮没有使用`mixed_orbit`。
- checkpoint：预登记固定使用E200最终checkpoint，不以中间最佳epoch替代。
- 独立测试：clean204000条；每个LEO_WEAK场景204000条，其中Strict UDU为未见日＋未见接收机60000条。
- test物理隔离：训练接收机0–6、测试接收机7–11；训练日为2021-03-01/08，未见测试日为2021-03-15/23。
- LEO测试seed基值：`2027`；`eval_max_batches=0`、`sat_eval_max_batches=0`，即全量评测。

独立评测器报告中的`train_ratio=0.2`、`requested_val_ratio=0.8`是checkpoint重建时的评测侧train/val上下文，不是训练角色协议；正式训练记录仍为0.07/0.63/0.15/0.15。最终clean/LEO结论只使用隔离的test切片。

## 四、方法矩阵

|profile|初始化|训练参数|目标与用途|
|---|---|---|---|
|A0|从头|Core90全部参数|同release控制组|
|A0-B|从头|v2严格旁路配置|验证旁路运行行为|
|A1-R|成熟D1|rank-8 adapter3912参数；q冻结随机|随机q阴性对照|
|A1|成熟D1|q2368＋adapter3912，共6280参数|sat CE＋clean-zero＋relative correction|
|A2|成熟D1|q＋adapter，共6280参数|A1＋teacher KL＋margin；`alpha_max=0.05`|

adapter残差只读取q，不读取`z_anchor`；raw骨干、domain骨干和共享CosFace头冻结。原Core90几何、FISHR和伪标签路径继续使用raw输出，独立测试比较raw与always-on robust/fused。

## 五、最终性能矩阵

以下均为同一行E200最终checkpoint的独立测试准确率，单位为%。

|profile|Clean|LEO Clear|LEO Low-elev|LEO Rain|LEO均值|Strict UDU均值|最差LEO场景|
|---|---:|---:|---:|---:|---:|---:|---:|
|A0|87.342|72.647|69.629|69.581|70.619|62.966|69.581|
|A0-B|88.126|73.698|70.471|70.317|71.495|64.322|70.317|
|A1-R|88.224|73.863|70.749|70.500|71.704|64.247|70.500|
|A1|**88.233**|**73.865**|70.746|**70.510**|**71.707**|**64.266**|**70.510**|
|A2|88.205|73.791|70.660|70.450|71.634|64.094|70.450|

表面上A1是5个最终fused输出中最高，但这一差异不能直接归因于adapter：A1/A2从成熟D1初始化，而A0/A0-B从头训练。必须进一步看同一adapter行内保持不变的raw路径。

### 5.1 Clean三类域外切片

|profile|未见日＋已见接收机|已见日＋未见接收机|未见日＋未见接收机|
|---|---:|---:|---:|
|A0|92.108|86.280|81.730|
|A0-B|92.490|87.965|82.178|
|A1-R|92.608|88.048|82.262|
|A1|92.608|88.072|82.268|
|A2|92.606|88.022|82.227|

## 六、最重要的因果对照

### 6.1 A0-B−A0

|指标|差值|
|---|---:|
|Clean|+0.785pp|
|LEO均值|+0.876pp|
|Strict UDU均值|+1.356pp|

这不是adapter收益证据。A0-B中旁路后raw=fused且无任何rescue/harm，但A0-B额外构造了59008个NTRS相关参数，两个run从E1即走出不同训练轨迹。未保存E0 raw参数逐元素同一性和逐样本预测向量，不能排除未使用模块初始化消耗随机数或其他随机轨迹扰动。因此A0-B只说明本次从头训练重复存在约0.7–1.4pp的轨迹差异，不能把该差异宣传为v2结构增益。

### 6.2 A1−A1-R：可学习q几乎没有增量

|指标|差值|
|---|---:|
|Clean|+0.009pp|
|LEO均值|+0.003pp|
|Strict UDU均值|+0.019pp|

A1的q梯度最大范数为4.191，证明q确实参与优化；但相对随机冻结q的收益只有千分之几到百分之二pp。这说明当前目标可以更新q，却没有把q训练成对识别有用的卫星nuisance描述器。

### 6.3 A2−A1：更强修正与teacher/margin组合负收益

|指标|差值|
|---|---:|
|Clean|−0.028pp|
|LEO Clear|−0.074pp|
|LEO Low-elev|−0.086pp|
|LEO Rain|−0.060pp|
|LEO均值|−0.073pp|
|Strict UDU均值|−0.172pp|

A2没有表现出teacher KL和margin带来的保护，反而使三个LEO场景和Strict UDU一致下降。

## 七、同一checkpoint内raw→fused的直接证据

3个adapter行的raw路径都来自同一个成熟D1 checkpoint，raw参数漂移为0，且独立测试raw结果完全相同：Clean88.234%、LEO均值71.768%、Strict UDU均值64.379%。这比跨run比较更接近adapter净效应。

|profile|Raw Clean|Fused Clean|ΔClean|Raw LEO均值|Fused LEO均值|ΔLEO|Raw Strict|Fused Strict|ΔStrict|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A1-R|88.234|88.224|−0.010|71.768|71.704|−0.064|64.379|64.247|−0.132|
|A1|88.234|88.233|−0.001|71.768|71.707|−0.061|64.379|64.266|−0.113|
|A2|88.234|88.205|−0.029|71.768|71.634|−0.134|64.379|64.094|−0.286|

逐场景adapter净效应：

|profile|ΔClear|ΔLow-elev|ΔRain|ΔStrict Clear|ΔStrict Low-elev|ΔStrict Rain|
|---|---:|---:|---:|---:|---:|---:|
|A1-R|−0.048|−0.055|−0.090|−0.083|−0.122|−0.192|
|A1|−0.046|−0.058|−0.079|−0.097|−0.100|−0.143|
|A2|−0.120|−0.144|−0.140|−0.265|−0.307|−0.285|

三个场景、三个adapter候选全部为负；而且Strict UDU退化大于LEO总体退化，说明修正对最难的联合域外切片更不安全。

## 八、rescue/harm与修正幅度

|profile|LEO Rescued|LEO Harmed|净变化|LEO相对修正均值|相对修正p95|旋转角p95|raw/robust分歧率|
|---|---:|---:|---:|---:|---:|---:|---:|
|A1-R|2122|2514|−392|1.256%|2.000%|1.145°|1.371%|
|A1|2322|2696|−374|1.344%|2.000%|1.146°|1.429%|
|A2|4141|4964|−823|3.033%|5.000%|2.855°|2.559%|

A1提高了rescue数量，但harm同步增加且始终更多。A2把相对修正p95从2%放大到5%，raw/robust分歧率从1.43%升到2.56%，新增的决策翻转仍以伤害为主。换言之，当前问题不是“修正太弱所以没变化”，而是修正方向缺乏可靠的按样本正确性。

clean也不是严格零修正：A1的clean相对修正均值为0.813%、p95达到2%；A2均值为1.599%、p95达到5%。clean-zero只降低了平均能量，没有阻止尾部样本触及修正上限。A1/A2的gate与safe gate活跃率均为100%，这是预登记的always-on评测设置，不是A3支持门；其结果正好证明当前尚不具备进入A3的前提。

## 九、参数隔离与梯度证据

|profile|q可训练参数|adapter可训练参数|q梯度最大范数|adapter梯度最大范数|raw梯度最大范数|raw最大漂移|
|---|---:|---:|---:|---:|---:|---:|
|A1-R|0|3912|0|4.590|0|0|
|A1|2368|3912|4.191|4.067|0|0|
|A2|2368|3912|6.326|5.323|0|0|

成熟checkpoint共加载195项raw状态；新增q/adapter状态按设计重新初始化。A1/A2的q和adapter均有真实梯度，raw参数没有进入优化器、没有梯度且最大绝对漂移为0。因此本轮负结果不是“adapter没有训练”或“冻结失效”，而是adapter经过训练后没有产生正净识别效应。

## 十、训练过程与数值健康

- A0/A0-B从头训练时source val曲线正常上升，E200训练侧val分别为98.754%和98.730%。
- A1-R/A1/A2因冻结成熟raw骨干，训练侧val在200轮内维持约98.57%，没有发生主干遗忘或崩塌。
- A1-R/A1/A2的训练总损失在阶段目标激活后由约1.336升至约6.80，但raw val保持稳定、最终结构化指标有限且无异常终止，因此这属于新增损失尺度变化，不是数值发散证据。
- 每行共约9000个optimizer batch。非有限梯度保护分别跳过A0 9次、A0-B 10次、A1-R 5次、A1 5次、A2 4次，占比0.044%–0.111%；非有限loss跳过为0，E200均为0。该现象不改变终态合法性，但后续实现可继续定位这些稀疏梯度异常。
- 5行均无tail rollback，最终checkpoint固定为E200并被独立评测器成功重建。

## 十一、预登记晋级门复核

|门槛|A1|A2|
|---|---|---|
|raw保持冻结基线、raw漂移为0|通过；参数级证据完整，未单独导出逐样本预测向量|通过；同左|
|ΔLEO均值≥+1.0pp|失败：−0.061pp|失败：−0.134pp|
|ΔClean≥−0.5pp|通过：−0.001pp|通过：−0.029pp|
|每个LEO场景下降≤0.5pp|通过，但三个场景均为负|通过，但三个场景均为负|
|ΔStrict UDU≥0|失败：−0.113pp|失败：−0.286pp|
|rescued＞harmed|失败：2322＜2696|失败：4141＜4964|
|q梯度＞0|通过：4.191|通过：6.326|
|raw参数最大漂移=0|通过|通过|

结论不是“接近通过”，而是两个核心性能门和净救回门同时失败。按预登记顺序，A2不晋级，A3/A4保持未发布。

## 十二、资源观测

|profile|可训练参数|占总参数比例|峰值CUDA allocated|训练墙钟时间|
|---|---:|---:|---:|---:|
|A0|1049665|100%|9.493GiB|3.825h|
|A0-B|1108673|100%|9.494GiB|4.075h|
|A1-R|3912|0.370%|0.547GiB|2.796h|
|A1|6280|0.595%|0.547GiB|2.782h|
|A2|6280|0.595%|0.547GiB|2.788h|

adapter-only把训练显存从约9.49GiB降到0.55GiB，参数和显存效率非常高。但服务器上存在并发任务，这些墙钟时间只能作为吞吐调度观测，不能作为隔离延迟基准。资源优势不能补偿负性能结果。

## 十三、限制与不能声明的内容

1. 本轮只有单seed=`392034`。当前结果足以淘汰明显未达门槛的A1/A2，但不能给出跨seed方差或置信区间。
2. `LEO_WEAK`是物理启发代理信道，不是真实在轨同步采样；`leo_rain_weak`也不是完整大气降雨链路仿真。
3. A0与A0-B从头训练轨迹并非位级配对，不能把二者差值解释为旁路结构因果收益。
4. 本次卫星评测器未输出逐接收机LEO floor，相关字段为NaN；报告使用每场景overall和60000条Strict UDU，不虚构receiver floor。
5. 本轮不涉及unknown rejection、Phase2 K-shot适配、新类注册或真实多星协同，不能扩展为这些能力的证据。

## 十四、最终决策与后续方向

- 正式状态：`ANALYZED_NEGATIVE_NO_GO_ADAPTER_V1`。
- 保留：成熟D1 raw checkpoint及本轮全部日志、结构化epoch记录、最终测试和机制遥测。
- 不执行：A3支持门和A4极小core联合微调，因为A2没有取得正净救回，继续顺序晋级会违背预登记。
- 下一版若继续，应先重做adapter目标，使训练直接约束`rescued−harmed`方向和Strict UDU风险，而不是继续增大`alpha_max`；同时需要让clean尾部修正真正接近0，并在随机q阴性对照上产生明显增量后再讨论支持门。

本地完整证据位于`E:/type10-7/automation_reports/CV-SincNet/phase1_advb02_ntrs_adapter_matrix_20260820_r1/evidence/raw/`，共65个文件，包括5份完整训练日志、5份CSV、5份JSONL、5份最终测试JSON/TXT及机制/资源/终态记录。
