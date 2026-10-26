# 恢复前后跨RX身份关系

RUNNING，实际执行独立核实VERIFIED。四路为同一个学生seed2026092701，分别own原E30+20到E50/11100；尚无完整源结果或新目标结果。代码`c0fc6a9a98e1a28669c2c41b59e26d2887311303`，远端OID/发布archive/compile和真实E30/defaulthead/G及集成L128/U256 backward smoke通过；8项聚焦检查及唯一一次P0/P1检查PASS。四条实际进程/GPU/step日志增长均核实，见[启动证据](launch_evidence.json)。

## 真实源校准

12批、72次全参数物理梯度计算，无optimizer/V/目标/Utruth及模型/buffer更新。primary中位数0.978037417；四route附加中位数：SeparateRG=1.689651430；BridgeRG=1.753729701；BridgeHG=1.315275609；BridgeGG=2.090845704。按预定共同公式固定λ=0.046777120618086768，加权初始范数比：SeparateRG=8.081187%；BridgeRG=8.387657%；BridgeHG=6.290639%；BridgeGG=10.000000%；与主任务cos中位数：SeparateRG=+0.012153；BridgeRG=-0.005712；BridgeHG=+0.043640；BridgeGG=+0.028931。所有附加分类weight梯度0、原L768/AsKL0、两视图配对数量匹配。该代理不是后续AdamW上界或目标收益保证。

## 完整日志前缀核算

当前895条完整step记录及四份真实初始默认头/配置已逐条核实；各row进度不同，下表不是最终成本。六组方法共有前缀的额外受理、held、有效anchor及正负pair数量完全一致，实际输入路由/系数/升温/原主项/隐藏U符合预登记，未见非有限跳步。validanchor为两个视图上的实际数量，不是物理包数。

| 方法 | 已核实新增step | U曝光 | 附加受理包 | validanchor |
| --- | ---: | ---: | ---: | ---: |
| SeparateRG | 227 | 57980 | 42883 | 85766 |
| BridgeRG | 224 | 57212 | 42316 | 84632 |
| BridgeHG | 226 | 57724 | 42700 | 85400 |
| BridgeGG | 218 | 55808 | 41271 | 82542 |

训练和逐条源V27000八视图仍在运行。原heldRX1 G3≥+.3pp、overallrawclean≥−.1pp及全四条完成后选择规则不变；无通过者不扩种子/目标。RX1只在额外分支留出，不能称全网络未见RX。原同候选三个学生种子的raw clean/raw六景/同G六景联合目标尚未证明，不替换默认方法。

见[预登记](preregistration.md)、[源梯度](source_gradient_calibration.json)、[逐row执行](available_source_execution.csv)、[前缀证据](available_source_evidence.json)。
