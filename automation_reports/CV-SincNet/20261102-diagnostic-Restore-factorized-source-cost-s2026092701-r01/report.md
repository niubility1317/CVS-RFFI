# 冻结最终恢复器：完整矩阵与等价因子损失的源端反向成本

- run_id：`20261102-diagnostic-Restore-factorized-source-cost-s2026092701-r01`
- group_id：`restoration-exact-action-factor-source-backward-cost`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（源计时、状态不变和独立复算已核实）

## 目的与对照

固定5359 CovarianceFeature final30和原3ff8 F0，E30保存首源batch256/draw10，比较同物理+实际后验特征目标的前向/反向/clip成本；0optimizer、新训练、目标读取。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

源端两路同损失/梯度与计时完整，详见analysis.md；不是新模型训练或正式识别成绩。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 零更新源端前向与反向成本

固定原CovarianceFeature final30/5359/1230341参数和原3ff8地面完整Tri-SIG-TIME SSL。保存E30第一逻辑batch256、原R07 draw10，不按TX、ID或质量选样。比较全实矩阵与等价反射FIR因子损失，两者均为.1×全Action矩阵相对MSE+.001×noise logMSE，首微批32追加实际FP64后验IQ的.01×原冻结F0(cos+源方差MSE)。原mu/C与F0源方差知识保留，不是F-free。

source2实际最终G/F0后立即核对两次完整逻辑反向：损失rtol2e-6/atol2e-7、全部G裁剪前梯度rtol2e-4/atol2e-6；随后每路2次预热、10次计时，交错次序各5次先测，共26次逻辑反向。6656次G、832条后验/F0、256条clean参考，额外smoke2次G/后验/F0/defaulthead；重复256条不是新独立样本或训练曝光。没有optimizer对象/step，训练轮次和权重更新均0，全G/F0最终状态逐值不变。

计时含前向/反向/有限性与求解检查/clip，边界同步，分别报告wall/CUDA event、中位数/p90和allocated/增量显存。每次区域前清梯度，数据预驻留；不含optimizer、数据IO、epoch指标日志，显存也不含全63000缓存。所有路径和负结果均报告，不将孤立区域速度外推成完整30轮同质量提速。数学等价不保证浮点训练轨迹逐位相同。

性能主线仍为同原F0/Main双总体七景，健康父实验保持原计划。控制器只在父最后GPU阶段cross_both已经实际live在4/6时用另一卡，或该阶段exit0后使用4/6；父随后只有CPU评分。每卡新增1、总compute≤3、free≥12GB，本任务NN最多2且分卡，其他owner保持。任何源/父/非有限/梯度差/状态改变/技术故障保留，不自动重试或迁移复位。

3新针对性检查和唯一P0/P1通过；1031等价损失组件的12项检查与独立审查沿用，不重复不变代码审查。source2实际checkpoint smoke是worker第一步，PASS继续全部矩阵；单release archive SHA/远端compile/实际PID-CWD-CUDA-GPU-日志增长核实。完整命令、CWD、输入/输出、GPU和停止规则见登记。此项0TX/V/OE/novel/目标IQ-预测-评分、0新统计拟合/分类头；不是恢复性能或理论极限实验，默认DURA和目标ACTIVE保持。


源随机性角色说明：登记的model2026092701来自冻结父G；data2026101901/augmentation2026101912及旧evaluation字段2026101911只追溯R07的builder/augmentation/receiver实例化。实际本次不新生成观测、不运行目标evaluation，batch来自父已保存order2026110102的E30首256条、draw10，不重置或抽取新的训练随机数。


## 固定源端计时已完成并独立复算

VERIFIED / ANALYZED。source2真实最终G/F0、两路全部裁剪前梯度及冻结状态检查PASS；26次逻辑反向访问、24预热/计时记录、20timed/10配对、分位数和显存独立复算，最大误差0。区域中位wall减少7.496589%，10对中7对更快、3对更慢，区域峰值allocated减少4.008942%；不含optimizer/全量数据IO/epoch指标，不代表完整训练或识别收益。实际GPU6 source worker运行中PID-CWD-CUDA-GPU/日志快照已保存，最终exit0/controller-worker自然结束，0权重更新/TX/目标。辅助观察器的外层存活回执FAILED与实际NN快照VERIFIED单独记录，未重复启动。当前父Main按原矩阵继续，默认DURA保持。
