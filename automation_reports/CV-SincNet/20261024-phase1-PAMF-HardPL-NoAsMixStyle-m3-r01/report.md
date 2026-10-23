# 监督 U 机制与同伪类跨接收机统计混合

状态：LOCAL_VERIFIED_NOT_LAUNCHED。48 项相关检查、一次独立 P0/P1 PASS；原 F21 E30 真实六前向联合梯度检查为首个 GPU 步骤。

用户要求高纯度普通 PL 加权后叠加监督机制。本版为独立源因子组合，以原 F21 同 seed E30 为父权重，与 HPL_NO_UAS 配方严格匹配；不继承任何新候选。门控0.95、权重2、U标签平滑0.01、ordinary/raw RDrop KL1、As KL0、浅中层跨RX对比0.1与CE剂量归一化1.25均不变，只在已有六次U学生前向增加同伪类跨源RX统计混合。源依据是原F21完整V高纯度、源接收机统计差异及既有源视图难度诊断；不由目标分数选择参数、父模型或种子。

每批U只生成一个配对／lambda／两层开关计划，全部六前向共享，RDrop成对前向不额外随机改变统计混合。双方均raw置信≥0.95、硬伪类相同、源RX不同，才混合停止梯度的均值／标准差；拒绝样本不供给也不接收。time_down／t2.pool每层独立p0.5，Beta(0.1,0.1)，eps1e-6沿固定源设置；先变换再捕获CRIC特征。池化后时间长度1时变成特征均值插值，不声称非退化时间方差。教师、L和推理没有临时混合；原L信道64保持，U As输入仍exact raw，U不运行信道生成。无EMA、原型、U真值、G训练、新推理参数或另训头。

复用 MS r02 在相同原F21父模型、相同数据handles与STYLE上冻结的V720/seed配对身份诊断。raw接受后精度100%／99.5595%／99.8525%，同真实TX配对100%／99.2658%／99.7050%，均满足预先99%规则；V未更新任何状态。完整V27000原门控纯度另列，不把720当全集，不称U或目标正确率。启动时精确核对三个父路径、source_contract/source_cache/STYLE并保留原结果；不重复V模型更新或读取目标。真实checkpoint首步另外核实六次style、一次共享计划、实际浅中层／头有限梯度、RDrop／AsKL0、CE剂量、原L768与U256／1536前向、Uphysical0／rawAs128及hook移除。PASS立即丢弃probe、正式三seed重载原父状态。

固定三seed2026092701/02/03，GPU5/6/7一seed一卡，与健康MS各占一名额。LS同seed全部GPU预测完成且MS已派发后才启动；MS启动未注册CUDA的PID同样保留。每卡最多两条本任务、含外部最多三条、空闲≥10000MiB。唯一CPU scorer排MS终态之后。原E30+20至E50/11100、13L＋37LU、固定最终默认头学生；每模型新增4440步／U1134000／学生U6804000／style调用26640／共享计划4440；累计U2097900／学生U7767900／rawAs567000／Uphysical0。共享GPU墙钟时间不视为独占小时。

环境/CWD/命令/不可覆盖路径见experiment.json：N607普通账户、固定CVS-RFFI环境、FP32/TF32off。一代码Git commit、单archive本地/远端SHA比较和远端compile；两个相同技术失败停止新增派发，保留健康进程与产物。无目标成绩停止、调参、重排或选择性重跑。冻结后每row自动raw与固定DURA v1.0各clean＋六practical、168000条/景、三个seed共42cell，全部预测固定后独立truth-last。完整F20/F21及NoAs匹配差、seed/景/TX/RX/day、修复损伤、实际接受／混合与训练预测成本均保留。DURA只作冻结参考、不用于F训练；G＋F20仍含G的U源权限。

原raw目标继续要求clean同时比F20/F21至少高1pp、六景均值严格高1.5pp，当前ACTIVE。历史测试暴露如实披露，不声称新的独立盲测。预期保留smoke、复用源身份检查、全部step与hard_pl_state、E50源checkpoint、默认头源诊断、graph/native源检查、双七景predictions/scores及完整报告。源配对通过、排队或训练启动均不替代目标收益。


## 实际源诊断与三卡训练

VERIFIED_NOAS_MIXSTYLE_RUNNING：执行 commit `810019e38bd7f0fe4f9bdd8fffb0c28c6016743c`，单次 archive 对照及远端 compile PASS。真实 F21 E30 检查通过，两层实际 U 混合／有限梯度、权重2／门控0.95／LS0.01、六次U学生前向／一次共享计划／原RDrop与AsKL0／实际混合后跨RX捕获／rawAs128、原L768／信道64及临时hook移除均核实。已精确核对原父路径／数据handles／STYLE，复用固定源 V720 三seed接受后伪标签精度100%／99.5595%／99.8525%，同真实 TX 配对100%／99.2658%／99.7050%，满足原先≥99%源规则；不替代完整 V27000 的99.84至99.90%纯度，也不代表U或目标正确率。源证据不能保证目标收益。探针丢弃，正式重载原父权重，三GPU实际PID／CWD／命令／GPU与每条已刷盘step增长均独立确认，源训练继续。当前MS健康peer及每卡两本任务／含外部最多三进程均独立核对。详细源检查见 source_style_identity.json、source_activation.json。


## 三模型源训练完整，正式评测继续

VERIFIED_SOURCE_FINAL：HPL_NAMS2 三seed均新增20轮／4440步至E50／11100，全部13320条新增闭合训练记录逐条核对。门控0.95、权重2、标签平滑0.01、原L及联合默认头保持；无EMA／原型／U真值／G训练／另训头／U物理生成。每模型累计U2097900，学生U7767900；接受后同伪类跨源RX的实际style调用、可混合样本、逐层应用、4440个六前向共享计划、RDrop／AsKL0／对比0.1／剂量1.25及rawAs567000全部按预定执行。完整状态和父／增量／累计经过时间见source_training_complete.json，共享GPU墙钟不视为独占小时。

冻结模型自动接续raw与同一DURA各七景，全部决策固定后独立truth-last，不读取部分目标成绩或以源证据宣称达标。
