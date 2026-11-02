# 最终双分支恢复器：源V波形与真实识别验证

- run_id：`20261102-diagnostic-Restore-CovarianceDetail-sourceV-s2026092701-r01`
- group_id：`restoration-final-detail-sourceV-IQ-feature-TX-validation`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定1fcb最终30/5481223 G，原1800V七景只新增detail输出；旧received/BF/guided控制与clean特征全部复用。完整预测后独立CPU评分TX，分开报告IQ/特征/Fclean与真实准确率。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


本项只作最终冻结后的原源V验证，排在父实验完整Main和随后延迟测量全部技术完成之后，不与它们争GPU，不读取其成绩控制开始。固定1fcb最终30/5481223 G与同原3ff8完整Tri-SIG-TIME SSL，原R07的1800源V×七景（5RX×3day×120）。V只验证，0训练/选模/校准/目标访问。

只新增12600条detailG/F0输出，另真实Ls2完整G/F0检查；旧4178源V的received/BF/5359控制IQ、特征、logits和1800cleanF0参考按相同ID与F0身份复用，0新基线神经前向。先固定四方法全七景输出，再独立CUDA空CPU进程才按原V_metadata映射TX评分。分别报告真实准确率、修复/损伤、IQNMSE、连续特征误差、Fclean参考margin以及物理参数误差，特征改善/持平/变差容差固定1e-7，条件比较为detail相对physical。

源V已有研发曝光，同属五源RX，验证接收机实例seed2026101902、训练实例2026101911；不能单独证明Main的域原因、未知设备保真或理论上限。Main未缓存全量restoredIQ，所以本源V波形误差不能改标Main波形恢复率，IQNMSE也不是“发射机信息恢复几成”。这项诊断补充波形与识别收益的对应关系，不替代原Main六景3pp＋两个clean保持。

3新检查与4原检查、唯一P0/P1通过；实际source2为worker第一NN步骤，PASS继续全部矩阵。严格final30/config/state、冻结状态保持、缓存基线身份/ID绑定、一次archiveSHA/compile与独立子进程PID/CWD/CUDA/GPU/log增长核实。原source数据与权限验证沿用，不新增签名或重验流程。失败保留，不自动重试；默认DURA及原训练测试保持。

## 实际 CPU 等待队列

VERIFIED_CPU_QUEUE_ONLY：controller 1086539 的实际PID/CWD/命令与CUDA空环境已核实，未新增GPU worker。一次archive SHA一致、远端compile通过；实际代码 `f29b31417ce98a280eac25ea2fa78e0c49a5f0ff`。按Main完整预测评分→最终延迟→本源V验证串行执行，真实source2与结果仍pending。原模型训练保持，无新训练、目标访问或基线NN重算。

## 冻结诊断完成并独立复算

VERIFIED / ANALYZED。28 单元／420 分组／105 条件分组独立复算；缓存基线判决变化 0，最大指标差 1.42e-14。全部预测固定后独立源 V 真值评分；本次独立检查使用 1800 条源 V 标签及保存预测/参考 IQ，0 新 NN，0 原始 IQ 数据集读取，0 目标真值。特征类间/类内分析仅描述固定面板，没有新分类器或类别质心部署。 [完整分析](source_analysis.md)。0 优化／模型更新／目标输入，未改变父 G、默认 DURA 或主验收条件。

## 冻结源 RX/day 均值残差分析


VERIFIED：同一冻结源V的52200条已保存特征行，87个汇总与667个组统计；0新NN/IQ/TX/目标读取，模型和原Main结果保持。新分析实际代码43f1d218fe54a2cffe707b953708baa6a9a10124，5项针对性检查与方差/配对误差恒等式通过（最大误差4.44e-16），全部28个总体特征MSE与既有评分一致。

低档城区新版RX均值误差0.000825112，物理版0.004257533，下降80.619941%；相对原clean均值模式的对齐增益0.952200，物理版0.713805。新版特征误差中RX均值项仅0.575860%，98.295941%在RX组内。这不支持“新版放大RX均值偏差导致Main下降”这一具体解释，也没有为删均值或接收机均值投影提供直接依据。源V的RX/day均在训练中出现，包分布未按TX作因果控制，不能排除未见RX的协方差或非线性域差异。

[完整RX/day分析](receiver_geometry_analysis.md)保留全部场景、绝对量、占比、整体偏移与组均值模式；不是恢复性能优化或主目标达成证据。
