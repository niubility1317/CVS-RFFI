# 四条特征任务 CE 源试验

RUNNING，独立启动/实际训练核实VERIFIED。真实E30/defaulthead/G L2配对检查、完整H缓存和12批无更新梯度校准已完成，HG原L128/U256联合backward smoke PASS。四条学生训练分别在GPU0/2/3/4运行，PID/CWD/完整命令/环境/GPU与训练日志增长均独立核实。实际执行代码`1f34b3ac51a355fb5da73591ae43f9c28f73130d`；7项聚焦测试、local/remote compile及唯一一次P0/P1检查PASS，一次archive传输比较VERIFIED。

## 源配对与冻结系数

H缓存[3,56700,2,256]已完整生成并独立逐块核实有限值，使用与旧G缓存相同的物理U与namespace。实际G小批配对检查通过，G与教师均未更新。校准总耗时260.1165秒，含H缓存构建；12批L128/U256，无optimizer/V/目标，学生全部参数和registered buffers恢复不变。分类阶段weight在GFeature/HFeature的附加梯度中严格为0，GJoint非零；特征融合norm/gain继续训练的定义保持。

原主梯度中位数0.978037417。按预登记公式得到λG=0.059415326575376119、λH=0.11323142806482039；GJoint/GFeature共用λG，HG各半，不是按V/目标分数选择。全参数norm已按CENTER_UNITS换算；这是初始E30梯度代理，不保证后续AdamW更新或收益。

| 额外分支 | 梯度norm中位数 | 加权norm/主梯度中位数 | 与主梯度cos中位数 |
| --- | ---: | ---: | ---: |
| GJointHard | 1.646102905 | 10.000000% | -0.249252 |
| GFeatureHard | 1.610788822 | 9.785469% | -0.224113 |
| HFeatureHard | 0.863750845 | 10.000000% | -0.145018 |

同输入G两组的12批CE值完全相同。联合梯度平方与“特征梯度平方＋分类weight梯度平方”的最大相对误差为1.095e-07，支持实际checkpoint上的路由隔离与定义一致。分类weight约占额外梯度norm的中位数21.9064%；屏蔽后主/辅cos中位数从−0.249252变为−0.224113，仍为负值，H分支为−0.145018。这不能证明分类weight更新是退步根因，也不保证骨干冲突的因果效应；固定四条健康训练继续原预算，不能据此提前挑选或改系数。

## 实际训练前缀

截至独立审计时间`2026-10-25T22:19:18.407740+00:00`，共8772条完整训练记录被逐条核对。只有一个学生seed2026092701，四个候选，不是三学生种子结果。四个实际initial checkpoint的来源、配置和原六类默认头匹配；原L768包、AsKL0、ORD.1、rawPL与U曝光/系数/升温/HG计数符合配置。六组方法的额外接受共同前缀完全一致，无非有限跳步、U真值、G/教师更新。训练仍在运行，以下是已核实前缀，不能视为完整E50或性能结果。

| 方法 | 物理GPU | 完整新增步 | 累计额外接受 | 额外输入 |
| --- | ---: | ---: | ---: | --- |
| GJointHard | 0 | 2229 / 4440 | 421063 | G+G |
| GFeatureHard | 2 | 2174 / 4440 | 410802 | G+G |
| HFeatureHard | 3 | 2175 / 4440 | 410987 | H+H |
| HGFeatureHard | 4 | 2194 / 4440 | 414597 | H+G |

每条计划从own原E30追加20轮/4440步到E50/11100，自动完成完整V27000八视图。仍按held额外RX1 G三景≥+.3pp且整体V rawclean≥−.1pp筛选，全部四条完成后按固定规则选择；无通过者不扩种子/目标。RX1只是额外分支留出，父模型/L/rawPL/G已见它，不是未见baseRX的独立泛化结论。

目前没有新的目标性能或方法晋级，原同候选三个学生种子的raw clean/raw六景/同G六景联合目标不变。完整矩阵与停止规则见[预登记](preregistration.md)，[源校准记录](source_gradient_calibration.csv)、[源校准核验](source_gradient_calibration.json)、[全部可用训练前缀](available_source_execution.csv)、[训练执行核实](training_evidence.json)均保留。
