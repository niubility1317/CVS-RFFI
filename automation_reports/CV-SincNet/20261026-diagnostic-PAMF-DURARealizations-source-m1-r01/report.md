# 同包多信道随机实现经DURA：冻结源L诊断

ANALYZED，独立重算VERIFIED。原F0可见源L6300，单个原seed2026092701 E30默认头及同一固定G-LU；原始L defaulthead accuracy为99.428571%。这不是三种子、U准确率或目标测试，没有新增训练、V/目标访问或模型替换。

## 直接结论

**相同H重复恢复：18900/18900包逐点完全一致，最大绝对误差0。改变同场景的信道随机实现：恢复输出仍有明显变化，且会改变冻结F的分类判断。** 固定receiver/day hardware、场景参数与G执行路径，只改变预登记的三个随机namespace；所有实现保留，不按源L成绩选择种子。

| 源场景 | 三fresh的G相对L2中位数范围 | 四实现间G预测有变化的包 | fixed G准确率 | 三fresh G准确率范围 |
| --- | ---: | ---: | ---: | ---: |
| practical_low_suburban | 9.41% 至 9.49% | 10.84% | 96.0159% | 96.1111% 至 96.5397% |
| practical_mid_urban | 13.64% 至 13.86% | 24.94% | 91.0794% | 90.6984% 至 91.6349% |
| practical_low_urban | 19.36% 至 19.62% | 38.73% | 85.0159% | 84.5079% 至 84.9365% |

相对L2为`norm(Gfresh−Gfixed)/norm(Gfixed)`，百分数是波形距离比例，不是准确率百分点。预测变化比例是四个固定信道实现中预测类别不完全一致的物理包比例，不是四个模型种子的差异。原始L准确率高于这些恢复视图，说明这些视图对该冻结F仍有难度；不能据此声称恢复后的TX细节已被物理证明保真。

三fresh并未一致提高G准确率：midUrban有升有降，lowUrban三条均略低于fixed；也不能据此认定拿来训练一定有害。该诊断只支持“不同真实信道随机实现能产生非重复的恢复训练视图”。数值未剔除全局复增益/相位，不能将波形差异自动解释为新增身份信息。所有正确次数分布是描述统计，不提供依赖真值的预测择优、oracle或部署ensemble。

## 完整执行与复核

实际代码`189e94efaf0fa5c52a8980032c0e27453182d5d7`。3项聚焦检查/local compile、唯一一次独立P0/P1、archive传输及remote compile通过；实际GPU5单进程身份/CWD/argv/环境/日志增长与L进度独立核实，真实E30/defaulthead/G L2 smoke PASS。正常完成后所属进程退出，F/G参数及全部registered buffers不变，optimizer/U/V/target/另训头均0。当前AsymmetricHG四路按原配方继续，不因本诊断改变。

完整L覆盖6300、25个F视图，逻辑F评估157500包、G94500包，另有smoke各4包。实际总walltime131.169633秒，包含加载/CPU信道生成/推断，不是单包延迟benchmark。375个viewcells、180个H→G cells、45个repeatcells由保存预测/逐包误差重算，810个几何分布由保存的逐包几何量重算，物理ID/y/views与原L对齐；复核没有新增F/G前向或optimizer。

固定namespace历史名含U，但本run输入严格L_s；原source_channel保留normalize_input及residual/post_sync/noEQ/reflect/SNR30、25e6/2462e6和固定hardwareSeed20261016423，fresh只改channel random namespace，不对G另做归一化。私有NPZ保留预测/逐包量，无新增IQ缓存或权重。

## 后续用途与目标边界

优先等当前AsymmetricHG四路完整源结果，沿原+.3pp/−.1pp门槛决定其扩展。若仍无候选通过，可单独登记“固定源U H/G缓存”与“多信道实现H/G缓存”的匹配训练对照，保留同raw PL/伪标签/默认头/预算；同时记录额外G缓存成本。当前不自动构建U缓存、挑最佳随机实现、改变健康训练或访问新目标。

最终仍要求同一方法三个学生种子的raw clean≥85.98174603174603%、raw六景>72.0961970899471%、同G-LU六景>78.07463624338624%。本诊断不能证明上述联合目标，目标保持未达成。

证据：[25视图完整单元](source_view_cells.csv)、[H到G配对](source_restoration_cells.csv)、[重复控制](source_repeat_cells.csv)、[几何分布](source_geometry_cells.csv)、[全部fresh总体](source_overall_realizations.csv)、[实现间一致性](source_scene_consistency.csv)、[复核证据](source_result_evidence.json)、[启动证据](launch_evidence.json)、[预登记](preregistration.md)。
