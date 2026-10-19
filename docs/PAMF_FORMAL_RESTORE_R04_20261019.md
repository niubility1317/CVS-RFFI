# 冻结复时频恢复器的正式固定观测确认

状态：RUNNING / VERIFIED，待发布。5项新增与23项回归、compileall、正式入口help通过；未启动或取得本轮分数。R02复时频候选在原source-only规则A/B共同选中并通过源资格后，已完成1200条配对面板；本轮按既有源资格进入正式固定输入确认，不按目标成绩选择网络或参数。R03独立三行继续其原配置；本轮结果在R03源决策冻结前不回收为研发反馈。

固定候选为R02 `tf_unet_mixed`，父方法commit `7efdba92932a86bf4b0492a565d35377564a26da`，源最终E20/4940、全L/U且0TX训练。只评估原冻结cleanF及AsF/default heads；训练/适配/分类头拟合均为0。源角色、继承核对及已有候选P0/P1结论直接复用，不追加runner review或数据重验证。

正式输入复用 `20261002-phase1-cvs-residual-sixscene-manysig-m8-r01/received_views`，其metadata为VALIDATED_ONCE、168000个原物理ID、source_target_disjoint、六practical、residual/post_sync/noEQ。clean来自原final-evaluation capsule。receiver_seed=2027、namespace=phase1_full_six_fixed_20261002_v1、fc=2450MHz；与恢复训练/1200面板的注入会话和随机流不同，仍是同一原生引擎，不称独立物理族或真实在轨。保留全部原输入字节，不重新生成信道，不拆分或重抽样。

原predict_row复用，只增加文件格式适配：七IQ文件是原资产的只读链接，opaque物理ID从index的ids字段导出；不读取index标签或truth。rx/day占位仅满足旧scorer入口，truth-side独立评分时由已有真值字段替换，不参与恢复或默认头预测。原checkpoint严格加载，先用2个真实源样本检查，再顺序固定七视图全部预测，独立CPU scorer随后连接原truth。

准确率、macroF1、TX/RX/day分层、修复/损坏/净修复、原坐标IQ/FFT/PA等波形指标沿用。唯一辅助成本调整为paired-clean feature margin对固定512个均匀位置参考记录比较，分块计算；不得与完整最近其他TX检索胜率混报，不进入推理/选模。原小面板scorer默认分支保持。当前仅单R模型seed、固定一个AsF，不冒充多seed系统证明。

预计独立GPU4一条预测worker，计入其他owner的cap2/free12GB；CPU scorer2线程。R03的0/1/5和其他健康任务保持。输出/release不可覆盖；非有限、输入/权重不符、已有输出或预测缺项才触发技术停止，保留产物，不以低分重跑。精确命令和路径随experiment.json固定。

|ID|要求|文件|状态|证据/边界|
|---|---|---|---|---|
|R04-01|复用正式固定输入与原物理ID，不读取标签|formal/reference|verified|元数据/原文件存在已读回；窄测试验证原字节绑定与仅ids访问|
|R04-02|固定源选中权重、F/default heads、0更新|formal/runner|implemented|来源/源资格负测通过，原候选审查复用；实际源smoke待启动|
|R04-03|全七视图完成后独立评分|原predict_row+formal/runner|implemented|默认预测/scorer复用；CLI池参数接线及负向检查通过|
|R04-04|有界且显式标注参考池的特征诊断|feature_diagnostic/score_row|verified|完整池数学对照/标签置换/无其他TX/非法cap测试通过，默认分支保留|
|R04-05|资源/输出/提交与运行状态可核实|experiment/发布|implemented|单次资源/路径预检通过；GPU4空闲约24GB，待发布|
|R04-06|多seed和独立物理族最终证明|后续冻结扩展|deferred|本轮只有原单seed；R03结果不依据本轮分数调参|
