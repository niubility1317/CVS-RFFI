# 非对称H/G任务与关系

RUNNING，实际执行独立核实VERIFIED。HJoint/HAnchor/GJoint/BothJoint均同一个学生seed2026092701，各自own原E30+20至E50/11100；尚无完整源结果或新目标性能。代码`f064f495716de7434be99027ae383b2a49c88666`，archive传输/remote compile、真实E30/defaulthead/G和原L128/U256联合backward smoke通过。7项聚焦检查及唯一一次P0/P1检查PASS；四条实际进程/GPU/step增长已独立核实，见[启动证据](launch_evidence.json)。

## 真实源校准

12批、72次完整参数梯度调用，加60次CE特征梯度与48次H/G路由核对，无optimizer/V/目标/Utruth及模型/buffer更新。primary中位数0.978037417；四route附加中位数：HJoint=1.820321858；HAnchor=1.448445201；GJoint=2.317050219；BothJoint=1.823718369。按预定共同公式固定λ=0.042210454011369962，加权初始范数比：HJoint=7.856204%；HAnchor=6.251246%；GJoint=10.000000%；BothJoint=7.870863%；与主任务cos中位数：HJoint=-0.090867；HAnchor=-0.073268；GJoint=-0.153737；BothJoint=-0.142235。这是初始CENTER_UNITS代理，不是后续AdamW或性能保证。

逐批实数记录重算确认：HJoint/HAnchor的CE/CL/总loss相同，HAnchor恢复视图F特征梯度为0，其他route的H/G特征梯度均非零；全部附加分类weight梯度为0。所有F/G参数与registered buffers未变，temporary buffers恢复，原L768/AsKL0保持。CE1/CL1相对系数在校准前固定，不能从当前源分数解释为最优。

## 实际训练前缀

当前850条完整step记录及四份真实初始默认头/配置已核实。四路同H/G输入、冻结q/接受集合、共同剂量和原主项，六组共有前缀的额外受理/held/有效anchor/正负pair计数匹配；CE_H/CE_G实际曝光符合各自路由，Both两CE先平均。附加总loss与CE+CL值相符，未见非有限跳步。下表为当前前缀，不是最终成本；anchor不是独立物理包。

| 方法 | 完整新增step | 附加受理包 | CE_H受理 | CE_G受理 | validanchor |
| --- | ---: | ---: | ---: | ---: | ---: |
| HJoint | 210 | 39768 | 39768 | 0 | 79536 |
| HAnchor | 214 | 40513 | 40513 | 0 | 81026 |
| GJoint | 210 | 39768 | 0 | 39768 | 79536 |
| BothJoint | 216 | 40888 | 40888 | 40888 | 81776 |

每row计划U1134000/held226800/extra838780、H/G学生FW各1134000。原+.3pp/−.1pp源门槛及全四条完成后固定选择保持；RX1仅从额外分支留出，不是全网络未见RX。源通过后才另行冻结三学生种子raw/同G七视图正式矩阵，原联合目标仍未证明，不替换默认方法。

见[预登记](preregistration.md)、[源校准](source_gradient_calibration.json)、[实际执行](available_source_execution.csv)、[前缀证据](available_source_evidence.json)。
