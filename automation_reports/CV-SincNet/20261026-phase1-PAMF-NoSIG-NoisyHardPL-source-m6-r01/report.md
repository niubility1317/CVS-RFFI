# NoSIG 有噪声困难伪标签：六行完整源结果

ANALYZED。**六行均未通过源条件，不扩三seed、不以部分指标选择成功方法。** 原三学生seed的实际clean／星地／同DURA联合目标尚未达到。

执行代码`b298bbc5afbb734e91c19c4c88ef39622a3ff6e8`。单学生seed2026092701，六行均同NoSIG普通PL E30＋20至E50/11100。完整L、As KL0、原默认6×160头joint及原优化预算保持。原U56700真值隐藏；固定RawLabeler产生2858个noisy candidates，源V代理precision93.829947%，实际U正确率未知。旧可信标签图／恢复专家／raw锚点失败结论不变。

## 固定源判断

全部V27000八视图完成。下表单位%，差值为百分点；hard net是原E30困难V的raw净正确，相对NoSIG-PL E50。这里的G是源RX1三景，不是原目标六practical。

| 方法 | 源raw | ∆源raw | ∆RX1 G三景 | 困难raw净正确 | 固定规则 |
| --- | ---: | ---: | ---: | ---: | --- |
| NoisyCE | 97.896296 | +0.333333 | -0.290123 | +89 | FAIL |
| NoisyScaledCE | 97.888889 | +0.325926 | -0.234568 | +88 | FAIL |
| NoisyGCE | 97.900000 | +0.337037 | -0.216049 | +91 | FAIL |
| NoisyGCE2Raw | 97.862963 | +0.300000 | -0.209877 | +81 | FAIL |
| NoisyGCEG | 97.862963 | +0.300000 | -0.222222 | +81 | FAIL |
| NoisyGCEHG | 97.859259 | +0.296296 | -0.253086 | +80 | FAIL |

条件保持：∆源raw≥−.1pp、∆RX1 G三景≥+.3pp、困难raw净正确>0。源NoSIG-PL基准raw97.562963%、RX1 G三景92.567901%。RX1原基座和G已经见过，只从新增U分支排除，不能称真正未见接收机验证。该条件只能筛选研发候选，不能证明目标提升。

## 匹配解释与全量核验

CE、统一缩权CE、GCE区分标签与梯度剂量；Raw2、G、交替HG区分额外前向与物理输入。六组完整配对差见[匹配差值](source_matched_differences.csv)。不同loss数值不能直接比较正确率，单seed的小差也不能当显著性。

完整解析26640条新增记录、120个epoch、4794行stdout和六个实际final checkpoint，六行共同4440步的U/noisy曝光序列逐项相同。每行U曝光1134000、2858个候选共曝光57160，第二视图行额外57160包，其余行0。实际q／rho／统一缩权、L768／As KL0、原跨RX日程、完整budget、原NoSIG头与parent、nativeU/EMA关闭、零非有限和无U真值／target均核实。教师与G没有更新，无新头。

逐包从全部冻结logits独立重算816个RX/day/TX/overall/留出分组评估单元与固定source选择，和原结果一致。所有训练／控制器进程已结束。复核新增模型前向／IQ读取／U真值／target均0。已保存全部完整曲线、source endpoint和实际objective-state／view曝光。

另两个seed的原NoSIG E20真实张量／来源已核实，但没有因本行完成就自动声称三seed。后续如扩展，仍用各seed自己的E30／RawLabeler、同固定配方、原目标F0 168000包/seed，raw和同G clean＋六practical，全部预测后独立truth-last。原绝对门槛和新NoSIG三seed匹配控制要求均不降低。

[原计划](../../../docs/goals/PAMF_NOISY_HARD_PL_PLAN_20261026.md)、[预登记](preregistration.md)、[完成证据](complete_evidence.json)、[120轮曲线](source_epoch_curves.csv)、[全部源单元](source_cells.csv)、[科学判断](source_decisions.csv)。
