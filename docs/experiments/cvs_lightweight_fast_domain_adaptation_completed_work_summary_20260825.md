# CVS轻型快速目标域适应已完成工作深度总结

日期：2026-08-25  
基线：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`  
协议：`p2_min_v1`、`VALIDATED_ONCE`  
分支：`codex/meta-adapter-tri-r4-v1-20260824`

## 一、结论先行

本轮已经完成一套可实际训练、可在Phase2对少量原编码器参数执行真实反向传播、仅消费合法目标域support且保持query只读的轻型快速域适应工程链，并完成10个单seed Target5候选的正式prediction与truth-last评分闭合。10个候选共形成150个Target5 row结果，全部满足训练参数低于总参数1%、正式更新3步、无D92式协方差／LDA／持久分类头、无source／clean运行时访问、无query真值／角色访问、query不更新状态。

科学结论与工程结论必须分开：

- 工程目标已经实现。Phase1可训练meta-adapter并生成严格可回读bundle；Phase2能够加载真实checkpoint，在原编码器少量非分类层上完成3步support梯度更新，再以同一冻结原型余弦规则分别产生`DA0_REG0`与`DA1_REG0`预测；prediction闭合后才由独立scorer连接truth。
- 当前没有达到晋级标准的正向目标域适应方法。10个已评分候选中，聚合旧类均值最高变化为`+0.1111pp`，但对应floor为`-5.0pp`；其余候选为0或负收益。没有候选同时达到预注册的旧类均值`≥+1.0pp`且floor`≥+0.5pp`。
- 最明确的正向机制信号出现在Phase1 source侧尾类：time+fusion rank4使三类LEO weak的floor分别提高`+6.7857pp`、`+6.3571pp`和`+5.8071pp`；class-floor scale8使三类LEO weak floor分别提高`+1.9071pp`、`+1.8643pp`和`+1.5214pp`。但这些都伴随LEO均值下降，而且属于source held-out评价，不能写成目标域DA收益。
- class-floor候选已完成实现、Phase1训练、Target5工厂和真实checkpoint无query smoke，但遵照用户“先别往下走”的要求，prediction未启动、truth未连接。因此它的最高状态是`LANDED / SMOKE_PASS / PREDICTION_NOT_LAUNCHED`，没有Target5性能结果。
- 没有启动任何Target25。原因不是工程链不完整，而是所有已评分Target5候选都未通过双门槛；class-floor Target5则在prediction前由用户主动停止。

## 二、目标、边界与实际方法

### 2.1 冻结目标

本轮围绕以下可证伪目标执行：在冻结`ADV3B02_CORE90_SOFT_E200`基线上，只对原编码器中的少量非分类层进行真实梯度更新；Phase2只读取`VALIDATED_ONCE`目标域LEO received IQ、合法old-class support标签、冻结checkpoint、冻结类原型和预登记配置；query只用于逐样本推理，不能更新模型、原型、归一化统计、阈值或任何其他状态。

资源与晋级规则固定为：

- 可训练参数不超过总参数1%；
- Phase2更新不超过40步，本轮正式候选均为3步；
- 最终判决保持冻结原型余弦argmax，不新增或训练D92式分类头；
- 只在prediction全部闭合后连接truth；
- `DA1_REG0-DA0_REG0`聚合旧类均值至少`+1.0pp`且旧类floor至少`+0.5pp`才进入Target25。

### 2.2 Phase1元学习任务设计

Phase1没有把query限定成单一形式，而是实现了五类source-only episode，用来模拟未来support适配后面对的不同分布变化：

|episode类型|权重|含义|
|---|---:|---|
|`Q_SAME_DOMAIN`|0.40|同域内support/query切分|
|`Q_RX_HOLDOUT`|0.20|接收机域留出|
|`Q_DAY_CHANNEL_HOLDOUT`|0.15|日期／信道域留出|
|`Q_CLEAN_TO_LEO`|0.15|clean到LEO弱信道迁移|
|`Q_LEO_CROSS`|0.10|不同LEO弱信道之间迁移|

正式meta训练为200个outer step、每步4个episode，共800个episode；每个episode内部3步更新。已完成的prototype scale16、prototype scale8和class-floor候选都保存了800／800 episode闭合、全部loss有限的日志证据。

### 2.3 少层适配位置

实际验证过的少层组合包括：

|profile|可训练参数|占总参数|张量范围|
|---|---:|---:|---|
|time+freq+fusion rank4|8670／1058341|0.8192%|双分支time、freq、fusion adapter，共30个非分类参数张量|
|fusion-only rank4|2890／1052557|0.274569%|双分支fusion adapter，共10个张量|
|fusion-only rank8|5458／1055125|0.517285%|双分支fusion adapter，共10个张量|
|time+fusion rank4|5780／1055449|0.547634%|双分支time与fusion adapter，共20个张量|
|time-only rank8|5458／1055125|0.517285%|双分支time adapter，共10个张量|

所有profile均不包含classification head、LDA、协方差估计或持久新头。Phase2真实smoke逐次审计了可训练参数集合、3次backward、严格checkpoint加载、`query_opened=false`、`source_opened=false`和`query_state_update_count=0`。

### 2.4 support更新目标的演进

本轮不是只改变rank，还逐步改变了梯度方向：

1. 原始P1～P4路线：比较随机adapter、source监督adapter、FOMAML固定LR和FOMAML+Meta-SGD，但仍同时更新time／freq／fusion。
2. 位置消融：依次验证fusion-only rank4、fusion-only rank8、time+fusion rank4和time-only rank8，区分“容量不足”与“更新位置不对”。
3. 判决对齐：在time-only rank8上引入`frozen_prototype_cosine_ce_v1`，分别用scale16和scale8，使support反向传播目标与最终冻结原型余弦判决处于同一空间。
4. 尾类约束：实现`frozen_prototype_class_floor_ce_v1`。损失由普通support样本均值CE与按类CE的归一化smooth-max各占50%组成，固定温度0.2；当各类CE相等时，归一化项严格退化为均值CE。该目标对类标签置换保持同一形式，只读取合法support IQ／标签和冻结原型。

## 三、已完成的工程工作

### 3.1 Phase1训练链

- 完成层级meta-episode采样、support／query-adapt／query-guard载体、FOMAML、可学习step size、source-only选模和正式deployment bundle输出。
- 完成time、freq、fusion三类adapter及其精确profile／rank白名单；可训练参数由真实bundle严格回读，而不是按配置估算。
- 完成冻结原型、objective、scale、adapter配置从Phase1 config到bundle再到Phase2 runner的传递。
- 完成clean和三类`LEO_WEAK`场景的最终checkpoint评价与artifact闭合。

### 3.2 Phase2 truth-blind执行链

- 完成Target5工厂：每个候选生成5个operating point×3个LEO weak场景，共15个truth-free row。
- 完成真实checkpoint无query smoke：只提供support，要求真实3步backward且不打开query。
- 完成`DA0_REG0`／`DA1_REG0`同row prediction、不可变row receipt和矩阵receipt。
- 完成独立truth-last scorer和矩阵汇总；REG0新类指标保持`N/A`。
- 修复N607现有NumPy2.2.5与Torch2.1.0之间的数组桥接故障，保留失败run并使用新不可覆盖run闭合修复。
- 修复CVS多场景truth sidecar连接逻辑；已有prediction不重跑，只在独立scorer层继续闭合。

### 3.3 验证状态

- 各新增机制或定点修复均执行与变更直接相关的RED→GREEN测试、邻近回归、生产入口编译和规定的一次P0/P1审查；未改变的factory／runner／scorer直接复用既有验证，不以重复审核阻塞实验。
- class-floor实现阶段通过175项聚焦测试、274项邻近回归和11个生产入口编译；真实Phase1训练闭合200／200 outer step、800／800 episode。
- 所有正式N607实验均使用不可覆盖run ID；需要新release的发布执行一次归档本地／远端SHA核对和远端编译；完成结果均回收到本地证据目录并写入对应Git报告。

## 四、Phase1已完成实验及source侧结果

下表每格均为“最终checkpoint相对同候选P0控制的旧类均值变化／旧类floor变化”，单位为百分点。这里的数据用于判断source侧训练机制，不是目标域DA结果。

|候选|clean|`leo_clear_weak`|`leo_low_elev_weak`|`leo_rain_weak`|
|---|---:|---:|---:|---:|
|P1随机adapter|-0.0167／-0.1571|-0.0214／-0.2429|+0.0143／-0.1571|-0.0226／-0.1857|
|P2 source监督|-0.6131／-3.7071|-0.0310／-3.7786|-0.0952／-4.5143|-0.2048／-4.8357|
|P3 FOMAML固定LR|-0.8881／-5.5714|-0.3429／-2.8214|-0.5298／-2.7786|-0.4655／-2.2500|
|P4 FOMAML+Meta-SGD|-0.8833／-5.5643|-0.3369／-2.8071|-0.5262／-2.7571|-0.4655／-2.2429|
|fusion-only rank4|+0.0560／+0.4786|-0.0452／+2.0929|-0.1381／+1.4143|-0.1893／+1.4286|
|fusion-only rank8|-0.2619／-1.5214|-0.6083／+3.3857|-0.6881／+2.8857|-0.5583／+2.9429|
|time+fusion rank4|-0.0738／-0.0071|-0.2155／+6.7857|-0.3524／+6.3571|-0.3881／+5.8071|
|time-only rank8|+0.3155／+0.0500|-0.0417／-0.2429|-0.0976／-0.3357|-0.1310／-0.5071|
|time-only rank8+prototype CE scale16|+0.0024／-0.1071|-0.2333／+1.1143|-0.3393／+0.8500|-0.3000／+0.6643|
|time-only rank8+prototype CE scale8|+0.1702／+0.5714|-0.2643／+1.9500|-0.3964／+1.6286|-0.3190／+1.5000|
|time-only rank8+class-floor scale8|+0.0964／+0.4071|-0.2821／+1.9071|-0.4095／+1.8643|-0.4107／+1.5214|

Phase1结果给出三个稳定事实：

- 直接同时训练time／freq／fusion的P2～P4会显著损伤source floor，说明“元学习完成”并不等于得到可部署的少步更新方向。
- fusion与time+fusion更容易抬高LEO尾类floor，但同时降低LEO均值；其中time+fusion rank4的floor信号最强，三场景均超过+5.8pp。
- prototype CE与class-floor能在time-only profile上恢复LEO floor，但仍牺牲均值。class-floor没有在Phase1解决“均值—floor”冲突，只是把尾类改善做得更稳定。

## 五、10个正式Target5结果

### 5.1 聚合结论

每个候选均为5个operating point×3个场景=15个row。表中的“决策变化”是15个row内`DA0_REG0`与`DA1_REG0`最终类别预测不同的query总数；非零score变化但决策变化为0，说明梯度更新真实发生，却没有越过冻结原型决策边界。

|候选|参数占比|决策变化|聚合均值变化|聚合floor变化|结论|
|---|---:|---:|---:|---:|---|
|P4 FOMAML+Meta-SGD|0.8192%|0|0pp|0pp|不晋级|
|P3 FOMAML固定LR|0.8192%|0|0pp|0pp|不晋级|
|P2 source监督|0.8192%|3|0pp|0pp|不晋级|
|P1随机adapter|0.8192%|0|0pp|0pp|不晋级|
|fusion-only rank4|0.274569%|0|0pp|0pp|不晋级|
|fusion-only rank8|0.517285%|6|-0.2778pp|0pp|不晋级|
|time+fusion rank4|0.547634%|0|0pp|0pp|不晋级|
|time-only rank8|0.517285%|0|0pp|0pp|不晋级|
|time-only rank8+prototype CE scale16|0.517285%|52|+0.1111pp|-5.0pp|不晋级|
|time-only rank8+prototype CE scale8|0.517285%|16|-0.1667pp|0pp|不晋级|

10个矩阵共150个row全部完成prediction和truth-last评分。没有候选达到`+1.0pp/+0.5pp`双门槛，因此Target25启动数为0。

### 5.2 分场景完整数据

每格依次给出“`DA0_REG0`均值→`DA1_REG0`均值；`DA0_REG0 floor`→`DA1_REG0 floor`”。场景值是在对应5个operating point上的汇总百分比。

|候选|`leo_clear_weak`|`leo_low_elev_weak`|`leo_rain_weak`|
|---|---|---|---|
|P4|70.0000→70.0000；30.00→30.00|65.0000→65.0000；35.00→35.00|72.5000→72.5000；40.00→40.00|
|P3|70.0000→70.0000；30.00→30.00|65.0000→65.0000；35.00→35.00|72.5000→72.5000；40.00→40.00|
|P2|68.3333→68.3333；25.00→25.00|62.5000→62.5000；25.00→25.00|69.1667→69.1667；45.00→45.00|
|P1|68.3333→68.3333；30.00→30.00|62.5000→62.5000；35.00→35.00|67.5000→67.5000；45.00→45.00|
|fusion-only rank4|67.5000→67.5000；30.00→30.00|59.1700→59.1700；30.00→30.00|65.8300→65.8300；45.00→45.00|
|fusion-only rank8|67.5000→67.5000；35.00→35.00|61.6700→60.8300；35.00→35.00|64.1700→64.1700；45.00→45.00|
|time+fusion rank4|63.3300→63.3300；40.00→40.00|60.8300→60.8300；30.00→30.00|63.3300→63.3300；40.00→40.00|
|time-only rank8|63.3300→63.3300；35.00→35.00|63.3300→63.3300；35.00→35.00|66.6700→66.6700；45.00→45.00|
|prototype CE scale16|63.3333→63.6667；35.00→30.00|65.0000→65.0000；45.00→45.00|65.0000→65.0000；45.00→45.00|
|prototype CE scale8|63.3333→63.5000；35.00→35.00|64.1667→63.5000；40.00→40.00|64.1667→64.1667；45.00→45.00|

### 5.3 score变化与判决变化

- P4和P3的每row最大余弦分数变化分别达到`0.052605152`和`0.054215699`，但15／15 row均无类别变化。这证明不是“没有梯度”，而是更新幅度／方向未跨越决策margin。
- fusion-only rank4的分数变化范围为`0.000483811～0.010616124`，无决策变化；rank8扩大到`0.000752628～0.054594249`并产生6个决策变化，但5个low-elev变化带来净负收益。
- time+fusion rank4的分数变化范围为`0.003379732～0.030114323`，time-only rank8为`0.000167698～0.015051037`，二者都没有最终类别变化。
- prototype CE scale16把变化范围放大到`0.0161863～0.6821303`并产生52个类别变化，clear均值提高`+0.3333pp`，但clear floor下降`5pp`。
- scale8把变化收缩到`0.002171695～0.198972940`和16个类别变化；clear有`+0.1667pp`，low-elev有`-0.6667pp`，最终聚合为`-0.1667pp/0pp`。

## 六、正向收益应该如何判断

### 6.1 严格回答

当前没有“可晋级的正向目标域适应方法”。按预注册定义，正向方法必须同时改善旧类均值和全类floor；已评分候选没有一个满足这一条件。

### 6.2 可以保留的正向信号

有三类信号值得保留，但不能升级表述：

1. **梯度确实能改变决策。** prototype CE scale16产生52个决策变化，证明少层、3步、0.517285%参数预算不是天然“动不了”。
2. **均值可局部提高。** scale16在clear weak上提高`+0.3333pp`，三场景聚合均值为`+0.1111pp`。但floor下降`5pp`，说明收益来自头部／中部类，代价由尾类承担。
3. **Phase1尾类可稳定改善。** time+fusion和class-floor在source LEO评测上持续抬高floor。这说明尾类约束方向具有机制价值，但尚未经过合法Target5 prediction验证。

因此，本轮最准确的表述是：已经找到“更新强度可控”和“尾类可被优化”的机制信号，但尚未找到能在目标域同row上同时提高均值与floor的更新方向。

## 七、失败机制的深度分析

### 7.1 单纯增加容量不能解决方向错误

fusion rank4没有决策变化；rank8开始改变决策，却在low-elev形成净负迁移。容量扩大解决了“更新太弱”，没有解决“更新往哪里走”。继续机械增加rank很可能只会放大错误方向。

### 7.2 source侧floor收益不能外推到target DA

time+fusion rank4在source LEO上出现+5.8～+6.8pp floor改善，但Target5仍为0pp／0pp。这直接说明source-held-out episode的尾类几何与真实目标接收机support/query判决边界不完全一致。Phase1结果只适合筛选和诊断，不能代替Target5。

### 7.3 判决对齐解决了“梯度太弱”，没有解决“类间公平”

把support目标改到冻结原型余弦CE空间后，score和类别变化显著增加；scale16过强导致尾类崩塌，scale8减轻过冲但产生跨场景方向不一致。固定全局scale只能控制总体幅度，不能保证每类风险都下降。

### 7.4 floor是当前路线的真正瓶颈

scale16是唯一获得正聚合均值的Target5候选，但floor下降5pp；fusion rank8产生更多变化后均值反而下降。已有数据说明，后续优化不能只追求更多prediction翻转或更大support loss下降，必须显式约束最差类风险。

### 7.5 class-floor候选的科学位置

class-floor objective正是针对上述瓶颈设计：对每个support类先独立计算CE，再用归一化smooth-max强调高损失类，同时保留普通均值CE。它在Phase1 source LEO上取得三场景floor`+1.52～+1.91pp`，但均值下降`0.28～0.41pp`。这说明它确实改变了类间权衡，却还没有目标域证据证明能通过双门槛。

## 八、技术失败与恢复记录

技术失败没有被伪装成科学结果：

- Phase1 tri r1／r2／r3分别暴露了release外部路径传播、运行环境和pre-artifact准备／原型写入问题，均封为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；r5使用新run ID完成P1～P4全矩阵。
- fusion-only rank4 Phase1 r1的bundle校验／episode覆盖失败被封存；r2完成9／9 artifact并进入Target5。
- P4 Target5 r1在真实无query smoke中触发NumPy2.2.5／Torch2.1.0桥接错误，prediction从未启动；r2用定点桥接修复完成15／15 prediction和评分。
- P4 r2评分阶段发现多场景truth sidecar join错误；已保留原prediction，只修复独立scorer并闭合score，没有重跑适配或使用truth反馈候选。

这些失败的主要工程收获是：发布归档必须同时校验launcher和下游实际消费者的绝对输入路径；进程／GPU存在不等于训练完成；technical smoke、prediction与scoring必须分层；评分错误不得反向污染或重跑已冻结prediction。

## 九、当前停止状态

用户已明确要求“先别往下走”。截至本报告：

- 不再启动新候选或Target25；
- class-floor Target5的15／15 truth-free工厂已完成；
- 正式class-floor Phase1 bundle无query smoke已通过，真实执行3次backward，参数占比0.517285%，query/source均未打开；
- class-floor prediction output未创建，prediction矩阵未启动，truth未连接，因此不存在其DA性能结果；
- 已完成的10个Target5结果保持不变，不因新分析重跑；
- 本报告只总结已有证据，不新增实验。

## 十、证据索引

### 10.1 关键正式报告

- 初始P1～P4 Phase1闭合：[phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5_report.md](phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260825_r5_report.md)
- P4 Target5闭合：[stage2_meta_adapter_target5_p4_s392002_20260825_r2_report.md](stage2_meta_adapter_target5_p4_s392002_20260825_r2_report.md)
- fusion rank4／rank8：[stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1_report.md](stage2_meta_adapter_target5_fusion_r4_p4_s392002_20260825_r1_report.md)、[stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1_report.md](stage2_meta_adapter_target5_fusion_r8_p4_s392002_20260825_r1_report.md)
- time+fusion／time-only：[stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1_report.md](stage2_meta_adapter_target5_time_fusion_r4_p4_s392002_20260825_r1_report.md)、[stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1_report.md](stage2_meta_adapter_target5_time_r8_p4_s392002_20260825_r1_report.md)
- prototype scale16／scale8：[stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1_report.md](stage2_meta_adapter_target5_time_r8_proto16_p4_s392002_20260825_r1_report.md)、[stage2_meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1_report.md](stage2_meta_adapter_target5_time_r8_proto8_p4_s392002_20260825_r1_report.md)
- class-floor Phase1与已停止Target5：[phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1_report.md](phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1_report.md)、[stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1_report.md](stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1_report.md)

P1、P2、P3及其他Phase1位置消融的报告均位于本目录，命名与run ID一一对应。

### 10.2 本地回读证据

N607完成产物已按候选回读到`E:\type10-7\local_artifacts\meta_adapter_recovery\`。其中prototype scale8 Target5完整证据为`target5_time_r8_proto8_p4_r1_complete_20260825`，class-floor Phase1完整证据为`phase1_time_r8_floor8_p4_r1_complete_20260825`，class-floor Target5仅有工厂证据`target5_time_r8_floor8_p4_r1_factory_20260825`。这些运行产物不纳入Git，只在Git报告中记录路径与结果。

## 十一、最终判断

本轮完成的是一个严格source-free、query只读、真实梯度更新、参数占比小于1%、3步适配、无D92式持久分类头的轻型快速域适应研究平台，并用10个Target5矩阵证明了多个直觉路线为何失败。科学上尚未完成“得到正向方法”这一目标；已获得的最好target均值信号为`+0.1111pp`，但floor`-5pp`，不能晋级。class-floor是当前最有针对性的未评分候选，但在用户停止指令下保持`PREDICTION_NOT_LAUNCHED`，不得提前推断其目标域收益。

如果未来恢复实验，最小下一步只有一个：在不改变既有Target5 row的前提下运行已经通过真实smoke的class-floor prediction，并按同一truth-last scorer判断双门槛。除此之外，不需要重新训练、扩大矩阵、启动Target25或增加新审核。
