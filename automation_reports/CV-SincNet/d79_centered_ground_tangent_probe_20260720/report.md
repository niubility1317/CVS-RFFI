# D79中心化地面切向旋转实验报告

## 1.实验身份

|字段|值|
|---|---|
|实验ID|`d79_centered_ground_tangent_probe_20260720`|
|候选|`centered_ground_tangent_worstclass_top2_margin`|
|operator|Codex `/root`|
|状态|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|
|目标|消除D78未中心化地面切向产生的类先验漂移，同时保留旧类域适应收益与新类注册能力|
|数据单元|receiver`20-1`、new5、seed`713101`、K10(actual K8)、3场景×5fold|
|matched baseline|D62：`B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67%`|

## 2.假设、数学机制与唯一变化

D78的地面域切向使旧类`A+2.22pp`、遗忘`F−2.22pp`，但新类`N−2.67pp`。D79冻结D78的84-cell地面组件、rank13切向、smooth-worst top-2目标、20步优化、trust ball和D62基线，只把切向特征改为

`z=(x−mu_support)U`，

并把`Delta b=−DeltaW mu_support`编译进单仿射头，使部署残差严格等于`DeltaW(x−mu_support)`。`mu_support`来自全部注册类的等K support，对类标签置换不变；ground仍只提供共享域方向，不产生旧类专属分数。预期是保留D78对旧类的保护，同时减少new→old吸附。失败条件为任何`N/H/min-N`退化或old/new混淆交换。

## 3.协议、地面组件与资格边界

- 复用D18的`VALIDATED_ONCE/p2_min_v1`数据；没有重新验证数据，也没有改变received-IQ、physical ID、receiver/TX、场景、K或support/query划分。
- 单LEO_weak observation、support-only适配、每个query独立面对全部注册类；clean/source/query truth/role/quota/global reassignment访问均为0。
- 地面组件含84个有效域×类cell、14个有效域、逻辑状态25,428B；NPZ入口/出口SHA256均为`3c08c823d2e8a13c4233f0060ac67c332ecc8d6e8abec7352de975fead0267d7`，manifest入口/出口SHA256均为`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`，运行中只读。
- 当前地面组件仍为`formal_phase2_eligible=false`、`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，probe强制`formal_candidate=false`，因此本结果只能作为development diagnostic，不能形成正式Phase2声明。

## 4.实现、版本与窄验证

|文件|作用|
|---|---|
|`code/cvsrffi/stage2_d79_centered_ground_tangent.py`|包装锁定D78优化，中心化全support并解析生成bias补偿|
|`code/scripts/probe_d79_centered_ground_tangent.py`|在D62/D78单次compile点注入中心化bias，输出INT8/FP32和协议/资源闭包|
|`tests/test_stage2_d79_centered_ground_tangent.py`|中心零残差、单仿射等价、平移不变、确定性与K1回退|
|`tests/test_probe_d79_centered_ground_tangent.py`|公式锁、资源上限、ground只读和协议字段|
|`code/scripts/summarize_d79_performance.py`|完整解析D79/D78/D77/D66/D62各105行、stdout/stderr、逐类/场景/混淆/量化/资源|

- `E:\type10-7`根不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`，clean detached worktree为`E:\type10-7\code\snapshots\d79wt`。
- core SHA256=`1260de735e64aa32f11061be82ec2ca26e4c90e5590725d76d83299696848388`；probe SHA256=`aa2b0b4aee74eac0d551bf0387b2c87cfa2560df877924e579ca353b60499f68`。
- `ssr-gpu`下py_compile通过；D79专项6/6、D77-D79相邻专项24/24通过。
- 预注册Git提交=`b1a7f8df`；实现提交=`0f9b901e`。最终报告提交在本报告固化后记录。

## 5.运行与证据闭包

- 本地`cuda:0`运行，未连接或使用N607；2026-07-20T03:42:32+08:00启动PID`21308`，进程完整退出。
- 精确入口为`probe_d79_centered_ground_tangent.py --d79-arm centered_ground_tangent_worstclass_top2_margin`；除D79入口、arm、probe root和output root外，D18 before/after capsule、seal、authorization、D22 component、class binding、`--device auto --mode development_select_unverified_component --candidate-set d42_v1`与D78报告第7节锁定命令逐字一致。
- 输出目录=`E:\type10-7\automation_reports\CV-SincNet\d79_centered_ground_tangent_probe_20260720\centered_ground_tangent_worstclass_top2_margin`；stdout/stderr独立保存在报告根。
- 105/105条training row、30/30个target fit、1,080/1,080个D62 component fit全部解析；stderr为0B，`Traceback/RuntimeError/KeyError/OOM/Killed/NaN/Inf`marker均为0。
- `training_log.jsonl`为15,623,784B，SHA256=`55dbe245fade6160b71df0ee7e4788d1c5bd58dded426570b3554c30a4eb5e50`；RECEIPT SHA256=`347a82be0b269709f44d9bc5d1276718c598ae37787239dac5d4dd48dbcf7c9a`。
- `d79_full_performance_summary.json`完整解析D79/D78/D77/D66/D62各105行及完整stdout/stderr，SHA256=`75f47438b28bff0053805125d12d8cc6d7b2bd47ef13d6f5e03447d7c2ae9825`。

## 6.完整候选性能

所有百分比均来自同一候选15个outer row的联合统计；`B/A/N`为增量前旧类、增量后旧类、已见新类准确率，`H`为同row调和均值，`F=B−A`，`J`为同row联合floor。

|candidate|机制|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|判定|
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D79 INT8|中心化ground tangent residual|92.78|84.44|82.67|82.71|8.33|30.00|80.00/60.00/70.00|73.33/53.33/46.67|19/11/15|主候选，负结果|
|D79 FP32 matched|同一连续头|92.78|83.89|82.67|82.40|8.89|30.00|80.00/60.00/70.00|73.33/53.33/46.67|20/11/15|1个INT8/FP32翻转|
|D78 INT8|未中心化ground tangent|92.78|84.44|82.00|82.14|8.33|30.00|80.00/63.33/63.33|73.33/56.67/43.33|19/12/15|D79直接父版本|
|D77/D62 INT8|地面对角预条件/无效更新；当前最强合法开发基线|92.78|82.22|84.67|82.62|10.56|26.67|80.00/53.33/73.33|73.33/50.00/46.67|23/8/15|仍保持最强|
|D66 INT8|ground reliability residual|93.33|83.33|83.33|82.59|10.00|23.33|80.00/53.33/66.67|73.33/50.00/43.33|20/9/16|地面弱先验负结果|

### 同row差值

|比较|ΔB|ΔA|ΔN|ΔH|ΔF|ΔJ|Δmin-class B/A/N|Δold→new/new→old/new→wrong-new|outer hash变化|
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
|D79−D62|0.00|+2.22|−2.00|+0.09|−2.22|+3.33|0.00/+6.67/−3.33|−4/+3/0|5/15|
|D79−D78|0.00|0.00|+0.67|+0.57|0.00|0.00|0.00/−3.33/+6.67|0/−1/0|4/15|
|D79−D66|−0.56|+1.11|−0.67|+0.12|−1.67|+6.67|0.00/+6.67/+3.33|−1/+2/−1|7/15|

D79相对D78确实减少1次new→old并恢复`N+0.67pp`和min-N`+6.67pp`，说明中心化修复有效；但相对D62仍用3次new→old代价换取4次old→new减少，`N−2.00pp`且min-N`−3.33pp`，不满足联合无退化门。`H+0.09pp`只是错误交换后的微小均值收益，不能据此晋级。

## 7.逐场景性能

|场景|rows|B|A|N|H|F|J|min-class B/A/N|mean row floor B/A/N|old→new/new→old/new→wrong-new|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|`leo_clear_weak`|5|98.33|91.67|96.00|93.27|6.67|50.00|90.00/70.00/90.00|90.00/60.00/90.00|2/2/0|
|`leo_low_elev_weak`|5|91.67|80.00|76.00|76.92|11.67|20.00|80.00/60.00/50.00|70.00/60.00/20.00|7/5/7|
|`leo_rain_weak`|5|88.33|81.67|76.00|77.94|6.67|20.00|60.00/50.00/70.00|60.00/40.00/30.00|10/4/8|

相对D62，clear场景只损失`N−2pp`；low-elev获得`A+1.67pp`而新类不变；rain获得`A+5pp`、`F−5pp`，但`N−4pp`。保护旧类的信号仍主要来自困难场景，却没有形成对新类对称的共享域校正。

## 8.逐类性能与遗忘

|角色|类/真实TX|B|A或N|遗忘B−A|
|---|---|---:|---:|---:|
|old|`cls_75aa…`/14-10|96.67|93.33|3.33|
|old|`cls_8b02…`/14-7|80.00|60.00|20.00|
|old|`cls_1f33…`/20-15|96.67|90.00|6.67|
|old|`cls_f8df…`/20-19|93.33|93.33|0.00|
|old|`cls_a53c…`/6-15|93.33|76.67|16.67|
|old|`cls_33bb…`/8-20|96.67|93.33|3.33|
|new|`cls_09f8…`|—|70.00|—|
|new|`cls_1c2a…`|—|93.33|—|
|new|`cls_b8fb…`|—|70.00|—|
|new|`cls_d3af…`|—|90.00|—|
|new|`cls_f608…`|—|90.00|—|

D79把D78最差新类从63.33%恢复到70.00%，但最差旧类从63.33%回落到60.00%。地面切向对`14-7`和`6-15`仍未解决通用floor，而两个新类停在70%，远低于目标所需的稳健注册水平。

## 9.机制表现、缺陷与量化审计

|证据|结果|解释|
|---|---:|---|
|ground registry/effective domain/cell|26/14/84|真实只读组件完整参与切向构建|
|numerical/tangent rank|78/13|固定规则`min(domain−1,numerical rank)`|
|切向保留能量|77.7513%|没有扫描rank|
|有效更新/fallback|15/0|每个INT8 target row均执行更新|
|support中心误差最大值|`3.86e−17`|中心化数值成立|
|中心点残差logit|精确0|bias编译公式成立|
|bias残差范数|min/mean/max=`0.01137/0.02801/0.03753`|D78隐含先验项被显式抵消|
|smooth-worst目标变化均值|`−0.04334`|全部残差仍触及trust radius|
|OOF CE变化均值|`−0.004873`|连续代理明显改善|
|OOF非正margin数变化|0|误分类数未减少|
|support/OOF argmax变化|0|代理改善没有转化为held分类纠正|
|outer prediction变化|5/15|部署边界确实改变，但仍交换错误|

核心缺陷有三项：第一，ground tangent源自旧类聚合域残差，虽然不输出旧类分数，低秩边界更新仍会沿旧类辨识方向旋转；第二，support内连续margin下降但argmax不变，说明优化抓住的是边界内余量，不是能跨域泛化的误差方向；第三，所有更新都触及trust radius，继续调scale/rank/trust只是在同一路径上做结果驱动扫描，不能解决结构性old/new交换。

### INT8/FP32

|项目|结果|结论|
|---|---:|---|
|INT8相对FP32 outer argmax差异|1|出现新的量化近边界翻转|
|margin sign flip|1|该row决策对量化误差敏感|
|max score abs error mean/max|0.000853/0.001847|数值误差小，但边界余量更小|
|INT8−FP32 A/H/F|+0.56/+0.31/−0.56pp|INT8偶然更好，不能当作鲁棒收益|

D79不能把INT8偶然翻转带来的`A+0.56pp`解释为方法优势。正式路线需要量化前后同row稳定性，而不是依赖量化噪声越过边界。

## 10.资源与效率

|项目|结果|上限/结论|
|---|---:|---|
|trainable/peak parameters|2,159/2,159|≤80k|
|epoch/total optimizer steps|20/40|≤30/≤50|
|持久状态|34,011B|8,583B affine＋25,428B ground，≤256KB|
|D79额外适配MAC|351,232,584|含bias编译3,168 MAC；约为D62的1.41%|
|总适配MAC|25,242,456,554|资源闭包完整|
|query MAC|6,624|D79相对D62额外query MAC/state为0|
|CUDA峰值|22,886,912B|本地实测|
|dense query graph/query fit rows|0B/0|通过|

D79计算和状态都很轻，但性能门失败；资源达标不等于可晋级。

## 11.D77-D79三轮技术回顾

2026-07-20在启动第四轮前已重读活动目标和`项目.md`，刷新项目conversation index（1,005条）并搜索`D77 D78 D79 ground prototype tangent forgetting seen_new`，同时复核三轮完整报告、各105行日志、逐类/场景/混淆、量化和资源证据。

|轮次|地面信息用法|B/A/N/H/F/J|最差A/N|外层变化|结论|
|---|---|---|---|---:|---|
|D77|地面可靠性对角预条件＋全类共同下降|92.78/82.22/84.67/82.62/10.56/26.67|53.33/73.33|0/15|连续CE微降但决策完全不变，安全无效|
|D78|地面跨域残差SVD切向＋最差类top-2边界更新|92.78/84.44/82.00/82.14/8.33/30.00|63.33/63.33|3/15|旧类明显受益，新类等量受损|
|D79|D78切向中心化并编译bias补偿|92.78/84.44/82.67/82.71/8.33/30.00|60.00/70.00|5/15|部分修复先验漂移，仍未消除交换且出现量化翻转|

复盘结论：地面压缩原型不是“没有用”。D78/D79已证明它包含真实的跨接收机域方向，尤其能减少困难场景旧类遗忘；问题是把这些方向直接用于类logit残差，会把旧类Phase1判别几何带入`Y_old∪Y_new`共同边界，从而压制没有地面锚的新类。D77说明纯坐标预条件又过弱。Stage2-B和Stage2-C已用同一run的注册前后指标、`seen_new_acc`、`H`、逐类旧类遗忘和三类混淆同等审查；没有用均值掩盖floor。

协议复核：三轮均为LEO_weak-only、support-only、query独立评分；clean/source/query truth/role/quota/global assignment访问0，target-old/new使用同一类对称公式。当前ground组件未获正式资格，因此所有结论只用于研发方向选择。

明确关闭：不再扫描D78/D79的rank、中心化倍率、步数、温度、trust radius或类权重；不继续直接把ground中心/切向作为旧类anchor、类别bias或class-row残差。这些调整会扩大开发集过拟合，并不能消除结构性不对称。

## 12.下一轮决策与最终判定

D79最终状态为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`；不启动第二seed、125或N607。D62仍是当前最强合法开发版本。

下一轮D80转向“地面类内域扰动协方差先验”：从84个压缩cell的域内半径/残差构造只读共享噪声协方差，不产生类别中心、旧类分数或row残差；再用target全部注册类support估计目标域协方差，以预注册闭式经验Bayes收缩得到统一Mahalanobis度量，完全同等作用于target-old和target-new原型。主要差异是ground只决定哪些方向应降权，不再决定样本向哪个类别移动。若support-held代理、INT8量化稳定性或outer`A/N/H/min-A/min-N`任一退化，则关闭该协方差路线，不做参数扫描。
