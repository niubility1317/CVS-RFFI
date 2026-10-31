# MILKR 三视图投影前特征 K/N 验证

预登记，未启动。唯一矩阵/命令/owner/seed/路径/stop rule在experiment.json。基座是原完整Tri-SIG-TIME SSL（232993参数），DURA v1.0冻结，源网络新增训练与F/G更新0。

## 机制

实际CompositePool先连接三个112维signature视图，形成336维描述符，再经336→160线性投影、LayerNorm与gain输出。R9用forward_pre_hook读取实际投影输入，保留原前向及默认头，不用重建的假descriptor。每112维块独立单位化后连接/√3；与原joint160或历史多分支核等权混合，三种几何，nearest/all旧类带宽、global0/0.5、single/multi尺度，共24核×5ridge。查询仍每条一次F，无额外训练。特征总宽度1072，头部存储/计算成本相应增加。

K1用五源RX×两固定源episode，1support/32互斥held，在三伪旧/三伪新四路径上固定一套全局核规则。K>1只用当前support条件LOO选择核/ridge，全部注册类竞争；kernel几何条件于完整support，不称无偏nestedCV。源六类已被F见过，不能替代真新类验证。

## 验证

P02/MIX01、seed2026092701、全部七RX，K1/5/10/20×N2/5/10/20，全补集query、四严格S/Q，560行指标。原MILKR同数据预测复用。主目标S0Q0平均宏H至少+5pp；全16格均+5pp才声称各K/N均达标。P02已有开发曝光，不声称盲测，不依目标结果选择特征或参数。

全部七RX预测固定后独立scorer接truth。真实checkpoint无query smoke会核对实际336维捕获与原z/logits逐位一致、F权重状态不变。新特征核PSD/单位自核、query分批独立、schema错误拒绝及旧后端/R8回归共四项测试PASS；本候选唯一独立P0/P1审查PASS。

GPU0/2/4/6每卡最多2新增worker，启动前检查剩余显存，不触碰其他owner。不可覆盖run/output root，系统技术失败保留产物不原地重启；低收益不记技术故障。不把R8源端负结果重标为本轮结果。
