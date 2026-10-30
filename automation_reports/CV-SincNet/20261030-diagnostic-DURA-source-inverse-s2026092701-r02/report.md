# DURA v1源端求解误差与算子失配诊断

- run_id：`20261030-diagnostic-DURA-source-inverse-s2026092701-r02`
- group_id：`phase1-dura-performance-source-inverse`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Frozen DURA; fixed128L+384U positions x cachedepochs1/20; native2CG/fixedlastprior16CG/sourceoracle action and noise counterfactuals; no F/TX/V/target

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 预登记与本地验证

固定128 L_s＋384 U_s位置，共512个独立源训练记录；读取R07原epoch01和20的received及对应模拟器kernel/coefficient/variance，不读取TX/F/V/target。冻结默认DURA v1.0，以原生末级2CG精确重建作运行自校验，比较同一末级prior的16CG、source oracle算子/方差，以及clean-prior一致性偏差。按L/U、scene、state、lock报告均值、中位数、P90/P95和逐例数组。Oracle均为源反事实，不是部署prediction或严格准确率上界。

2026-10-30本地4项聚焦测试通过：源角色拒绝及位置采样、恒等真实算子MAP与解析解一致、零能量/尾部指标有限性。一次独立P0/P1审查PASS，无需改动。旧模型依赖使用既有不可变release `pamf_stage2_inference_20261021_r01`；本次只发布新诊断overlay，原模型代码和权重保持。

运行使用CPU2/0GPU，不等待或干预其他owner。输出碰撞、输入/来源/角色不符、非有限数、原生末级重建不一致或冻结状态变化则保留partial并报技术失败，不自动重试。launcher首步两条真实Ls checkpoint smoke后立即继续，无额外审批artifact。

尚无新性能结果。完整优化目标、旧证据和后续候选边界见[目标](../../../docs/goals/PAMF_DURA_PERFORMANCE_20261030.md)。本诊断不是新网络，无目标测试/训练；完成后仍不能宣称星地接近clean。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 技术修复范围

r01首步smoke在Torch2.1/NumPy2 C桥失败，原产物保留。r02仅切换源数组传入/传出为direct-buffer/tolist；科学矩阵、输入及预算不变。新增禁用C桥回归通过，原故障一次定点复审PASS。实际远端恢复以本run真实checkpoint smoke和诊断产物核实。

## 独立读回与完成结果

VERIFIED / ANALYZED_SOURCE_ONLY。实际远端首步2Ls checkpoint smoke通过，epoch01/20各512条完成，进程自然退出。CPU2/0GPU，实际诊断墙钟10.155秒。独立取回两份逐例NPZ，按all/L/U/scene/state/lock逐组重新计算全部mean/median/P90/P95，与results.json逐项一致；512个位置唯一且L128/U384，冻结全部模型状态不变，0TX/F/V/target及训练更新。

以下为完整源位置均值IQNMSE（越小越好，非识别率）：

|缓存epoch|coarse|stage1|最终prior|原生stage2|固定prior CG16|真实算子CG16|真实方差CG16|真实两者CG16|clean prior/学习算子|clean prior/真实算子|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0.4932697|0.0283037|0.1749918|0.0202128|0.0220397|0.0226658|0.0272530|0.0251412|0.0086788|0.0017555|
|20|0.4873983|0.0262959|0.1747256|0.0194397|0.0213655|0.0203554|0.0255737|0.0228765|0.0077909|0.0016345|

更多CG使两批均值误差分别增加约9.04%和9.91%，不能据此预设加迭代会提升。最终prior误差约0.175，而输出约0.020；clean子集的coarse/prior误差约1.00/0.49，但最终输出误差约0.00003，说明当前链路对一致性修正有较大依赖。真实算子/方差直接替换不稳定改善，符合先验与学习算子协同补偿的可能性；这属于解释，未证明因果或物理不可辨识。

low urban子集在两批分别66/70条：原生最终NMSE0.0667389/0.0683482，固定prior16CG为0.0678780/0.0694083；采用clean prior时学习算子仍引入0.0193653/0.0181164误差，真实算子仅0.0022784/0.0022605。Oracle控制只说明该源诊断的差异，不是部署上界，也未证明新候选必然改善TX识别。

下一独立候选保持现有两阶段266951参数及CG预算，检验对两个内部prior增加source pairedclean辅助监督，配套同父同预算常规续训。候选将在目标访问前固定源目标、轮数和七景矩阵；每个训练版都进行正式测试，目标分数不回流。当前未产生新恢复权重或新正式性能，本goal仍ACTIVE。

执行代码commit `11c5b4a6eea17c4e6b1e72b5a7ab1edcc2007402`；远端branch OID已独立匹配。科学输入和全部原始产物保留；本地逐例与原始汇总置于local_artifacts/pamf_restoration/<run_id>/，不提交逐例数组。

同份epoch20源记录还显示长尾集中：BLOCKED 62条均值NMSE0.100885，LOS 330条0.007779；degraded/unlocked各38条均值约0.08895/0.08891，locked355条0.008993。BLOCKED的mean log(估计方差/模拟器方差)为−1.9398，提示在该源子集可能偏于强一致性；模拟器标量方差不代表完整有色噪声协方差，不能把比值直接当成唯一病因。全体最终tail24/prefix232均值误差0.0515872/0.0161141，按各自长度平均后约3.20倍；最弱/最强FFT功率四分位的相对误差0.5211085/0.0140802，但分母不同，不能把该比值解释为身份信息损失比例。这些细分均来自已独立复算的同一诊断，不是新增实验或目标性能证据。
