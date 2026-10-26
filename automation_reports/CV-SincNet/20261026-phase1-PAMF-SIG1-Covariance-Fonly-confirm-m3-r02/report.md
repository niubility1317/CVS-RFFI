# 协方差三组 F 直接识别正式确认

状态：FAILED_BEFORE_TARGET。按用户最新撤下DURA配合路线，固定CovSel-U、CovRandom-U、CovSel-LU三份已完成E50/11100单视图完整PAMF，直接预测rawclean和六practical。新训练0，不读取G权重/恢复IQ，不新建恢复缓存。全部矩阵/命令/环境/CWD/输出/GPU/停止规则见experiment.json。

三候选按独立F-only源必要条件入选，依次rawclean少6/5/6条，raw三景总正确数多58/38/103条；fullmask三景少31条未入选。全部使用冻结final，非按目标排名；不把小源收益称泛化成功。旧含G判定保留，不追认通过。此次范围变更来自用户明确要求，发生在新covariance target访问前。

各行固定cross_rx126000和cross_both42000，共168000物理包×7raw视图；按正确数合并，单独cross_date30000不加入本次与严格SIG1对照匹配的正式矩阵。原监督/普通PL同SIG1三seed模型及已完成预测复用，seed01配对本轮；无对照重训/重复预测。原三seed和更完整目标范围仍未完成，不能仅凭此pilot宣布成功。

加载实际来源一致的E20+10+20/defaulthead后，先sourceL2无query smoke，PASS即连续预测。预测配置无truth/G句柄；每row两任务全部7视图固定后独立CPUscorer连接truth，评分配置不含source/model/G。保留逐TX/RX/day及receiverday的accuracy/macroF1/混淆。目标不回流调mask、权重、epoch、候选顺序或选择性重跑。

候选已完成一次独立P0/P1 source审查PASS；此处复用同一冻结加载和原truth-last评分，新增raw-only配置/权限边界5项测试PASS，不额外添加runner审查或数据hash gate。源码/config由执行Git commit固定，单releasearchive一次传输比较、远端compile和实际PID/CWD/argv/GPU/loggrowth核对。只在全局空闲卡启动，每卡一条ownworker；首个技术失败停止新启动、健康在跑自然完成，其他owner不操作。

最终至少检查rawclean≥85.981746%、raw六景>72.096197%，并按严格同SIG1PL/Sup的逐景和分群配对结果报告；原三seed等目标保持。DURA指标因用户最新范围撤下，不再用于本路线验收。

## r01 技术替代的准确边界

r01在模型smoke/target访问前因确认context的source_cache路径不匹配而停止，全部进程已退出，原root完整保留。r02直接从三个实际冻结训练配置取source_cache/source_contract，三行必须一致；实际源记录离线检查3/3PASS，旧错误context仍FAIL。模型/数据契约/seed/epoch/超参数和正式raw矩阵均不变，不重训。实际no-query E50/sourceL2smoke仍作为本次预测第一步执行，PASS后立即继续。

## 独立加载环境的技术失败

执行commit `d5ec09e206c0de58468fced18a62e5e84b77372d`。单archive比较VERIFIED/远端compilePASS；source实际句柄检查已通过，随后torch反序列化祖先checkpoint缺少`baseline_origin_sat_view.SatViewStage`模块而退出。controller及三worker均自然停止，未构建完成F、未执行正式sourceL2smoke或进入target预测/评分，输出progress/smoke为空。完整三traceback和终态已读回，没有重启此root。

该模块是既有可信checkpoint训练日程的metadata dataclass，不是G网络或恢复权重。联合bootstrap曾把它与G历史code路径一起提供，独立Fbootstrap漏掉了当前release/code路径。普通账号独立CPU进程在同一实际checkpoint、四份F依赖和release/code路径下完成严格加载+sourceL2读出，shape2x6、单SIG_TIME、有限输出PASS，optimizer0、G权重0、target0。未修改checkpoint或复用目标结果。

替代r03仅加入现有可信metadata模块路径；原r01/r02的代码、输出与失败判定完整保留。探测中一次/proc/environ权限不足仅影响观察，之后通过终态和完整日志核实失败，不因探测报错重复启动或使用管理员。
