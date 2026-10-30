# DURA v1源端求解误差与算子失配诊断

- run_id：`20261030-diagnostic-DURA-source-inverse-s2026092701-r01`
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

2026-10-30本地3项聚焦测试通过：源角色拒绝及位置采样、恒等真实算子MAP与解析解一致、零能量/尾部指标有限性。一次独立P0/P1审查PASS，无需改动。旧模型依赖使用既有不可变release `pamf_stage2_inference_20261021_r01`；本次只发布新诊断overlay，原模型代码和权重保持。

运行使用CPU2/0GPU，不等待或干预其他owner。输出碰撞、输入/来源/角色不符、非有限数、原生末级重建不一致或冻结状态变化则保留partial并报技术失败，不自动重试。launcher首步两条真实Ls checkpoint smoke后立即继续，无额外审批artifact。

尚无新性能结果。完整优化目标、旧证据和后续候选边界见[目标](../../../docs/goals/PAMF_DURA_PERFORMANCE_20261030.md)。本诊断不是新网络，无目标测试/训练；完成后仍不能宣称星地接近clean。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 技术失败，原产物保留

2026-10-30远端状态VERIFIED/FAILED。提交`6a9916cd57670de052aa7424de379fdf3b49f881`的归档传输SHA一致、编译通过；实际checkpoint严格加载成功。首次两条Ls smoke在`torch.from_numpy`处发生`TypeError: expected np.ndarray (got numpy.ndarray)`，原Torch2.1与NumPy2 C桥不兼容。进程自然退出，仅有launch.json，未运行诊断或产生prediction，未更新模型。原日志保留`logs/<run_id>/stdout.log`，不原地重启。

技术修复复用项目已验证的direct-buffer传入，并移除Tensor.numpy传出，新增禁用两种C桥接口的针对性回归检查；本地4项检查通过。替代run `20261030-diagnostic-DURA-source-inverse-s2026092701-r02`，科学矩阵/位置/预算/权限均不变。
