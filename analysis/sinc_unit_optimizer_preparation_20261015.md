# 固定 SIG_TIME 的 Sinc 优化单位：实现准备

状态：LOCAL_COMPONENT_VERIFIED／NOT_LAUNCHED。当前只有优化器组件和本地机械验证，尚无新实际训练、目标结果或完整发布入口；不称实验完成、远端验证或泛化提升。SIG_TIME结构、原Hz参数张量、原始频带初始化、输入处理和默认cosine分类头保持。分类头按原流程正常联合训练。

## 已有真实源证据

前序9个真实final50模型中，Sinc的24个低截止参数有23个不变、24个带宽参数全部不变，实际滤波核相对L2至多2.4071e-8；48参数都有11100步真实优化器状态，47个一阶矩非零，末轮重建数据步长全低于对应float32半ULP。源证据已保存在主台账的sinc_actual_checkpoint_measurement／sinc_saved_optimizer_EMA，不重复运行诊断。末轮单步重建不代表完整尝试步长历史。

当前固定SIG_TIME首种子的三个真实新模型也独立确认：低截止只变1/24、最大变化SIG7.2479e-5Hz／随机8.3923e-5Hz／RSC3.8147e-5Hz；带宽均0/24变化。来源：20261015-phase1-temporal-rsc-oldgate-cos-e50-clean-manysig-s01-m4-r02/source_feature_mechanism.json中真实initial→final逐张量记录。源损失和其余fθ在学习，不能把该现象归为冻结骨干或遗漏optimizer。它是数值和优化单位问题的证据，不证明Sinc是泛化瓶颈。

## 唯一训练假设和实现

原Sinc采样率Fs=25000000Hz，以Hz保存low_hz_／band_hz_；全模型原AdamW峰值LR为0.0002。检验把这两张量的优化更新按f/Fs单位表达，是否使原滤波器获得可辨识的学习并改善实际特征；不增加卷积、分支、滤波器、损失或投影，不修改Sinc前向和参数化，不以目标成绩选择父权重或扫描倍率。Fs来自既有网络，不是搜索的超参数。

NORMALIZED_UNITS只为原48个Sinc参数建立优化器组：lr_sinc=Fs×lr_native、eps_sinc=eps_native/Fs、wd_sinc=wd_native/Fs。其余参数（含默认W）保持原AdamW选项；lr×wd的收缩因子保留。按[远端Torch2.1的AdamW实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.1.0/torch/optim/adamw.py)，这给定同一组已裁剪Hz梯度时与f/Fs参数坐标的自适应更新等价。原全局clip5仍在Hz梯度上执行，不声称整个优化器的坐标不变性。

原cosine日程每epoch会覆盖所有组的LR，因此在真实step前从group0恢复该原轮次LR，再施加固定Fs策略；不改cos日程或轮数。峰值对应Sinc数据更新量级上界约5000Hz，而不是给全网络设置5000学习率。既有ownEMA路径继续更新相同原参数，未增加第二教师或EMA配方。

## 本地证据和边界

7个无效/遗漏/重复参数或选项负例通过；直接使用实际原生L前向、原生参数收集、同logits和clip后梯度。STANDARD的模型更新、AdamW矩与原生逐元素一致；NORMALIZED_UNITS第一次相同梯度更新中所有非Sinc参数（含默认W）与原生逐元素一致。合成输入下24个截止／23个带宽得到非零更新，实际joint160最大差0.0146172，原EMA能跟踪相同Sinc变化。

float64给定梯度的4步坐标等价误差最大1.862645e-9Hz，原cosine覆盖问题已在该检查中复现并处理。完整机器结果见 [local_checks](research_data/sinc_unit_optimizer_local_checks_20261015.json)。这些是合成输入机械检查，不代替真实训练权重结论，不证明精度、源健康、DG、新TX、Phase2或84%。原数据、checkpoint和目标均未读取。

## 下一步最小实验范围

保持原旧batch_neighbor＋ownEMA＋cos、33label＋17pseudo、默认头正常联合训练、L6300/U56700隐藏TX/V27000及NoExplicitCFO。只准备首seed两个新scratch50臂：同路径STANDARD SIG_TIME与NORMALIZED_UNITS，另复用旧门控+cos历史结果；不加载本轮目标最高权重续训。完整训练接线、实际每步LR／Sinc前后值／EMA和feature-only witness、自动默认头clean truth-last、源资格与逐row预算将固定在发布预登记后再运行。每新模型50轮／11100次更新，机制不增加模型前向。仍需完整launcher检查、唯一P0/P1、真实checkpoint no-query smoke和既定最小发布步骤。

当前RSC02/03六模型继续冻结方案，与本优化单位准备独立；不得按它们的后续目标数值决定倍率、重排或选择性重跑。本方向依据上述真实源参数/优化器证据，在读取其剩余目标数值前记录。


完整训练入口已固定于 00132f7d54aff45b6e52fc41f89d7705a043bc71，唯一独立 P0/P1 缺失release依赖已修复并确认PASS；两行原生入口/合成16step验证和3行完整canonical恢复均PASS。预登记 [Sinc单位实验](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r01/report.md)；当前未启动真实训练，源资格与目标性能均未知。网络和默认头保持，只有原48个Sinc参数的优化器单位设置变化。


实际发布更新：R01因缺失间接依赖在训练前失败（0更新），产物保留。修复7221e39e0301e0de68231c3526e6c755295270d8仅补依赖与新ID。R02执行9fc348bc4be71f498a7684484fd86de81e3e4870已VERIFIED启动；两模型E2、首轮真实13批Sinc更新证据已回收。源资格与目标收益均未知。当前报告为 [R02](../automation_reports/CV-SincNet/20261015-phase1-sinc-units-oldgate-cos-e50-clean-manysig-s01-m3-r02/report.md)。
