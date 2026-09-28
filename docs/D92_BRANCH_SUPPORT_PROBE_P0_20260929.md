# D92 分支 support 探查 P0/P1 检查

日期：2026-09-29。结论：在本次实际实现和已生成配置范围内，未发现会越权读取、错误执行、覆盖已有输出或阻止启动的 P0/P1 问题。本记录只检查直接正确性，不构成额外审批，也不提供预测性能保证。

## 范围与证据

检查对象为 [冻结设计](D92_BRANCH_SUPPORT_PROBE_DESIGN_20260929.md)、[纯核心](../code/cvsrffi/d92_branch_support_probe.py)、[support 导出器](../tools/export_d92_branch_support_features.py)、[探查入口](../tools/evaluate_d92_branch_support_probe.py)，以及本轮新增的 [准备脚本](../tools/prepare_d92_branch_support_probe.py)、[串行导出与并行探查编排](../tools/run_d92_branch_support_probe.py)、[发布脚本](../tools/publish_d92_branch_support_probe.py)。未重复审查其他旧方法。

直接执行的入口测试是 [7 项合成测试](../tests/test_evaluate_d92_branch_support_probe.py)：覆盖真实核心 K1/K2/K5、每次调用只含当前任务 support、完整矩阵、物理索引/ID/类标签绑定、来源和缓存负测、拒绝额外 query 成员、失败不写完成标记、拒绝覆盖，以及实际 exporter 到真实核心的合成联调。测试命令为 `python -X utf8 -m pytest tests/test_evaluate_d92_branch_support_probe.py -q`，结果 7 项通过。导出器 fixture 中 query IQ 置为 NaN，联调仍只提取并使用合法 support。

核心负责人已报告 21 项合成测试通过；导出器负责人已报告 35 项通过；编排负责人已报告 8 项通过。本检查阅读相关实现和测试，不将负责人报告写成独立重跑结果。另直接核对实际 rx3/rx1 配置的算法等于核心 `FROZEN_CONFIG`，Cartesian 矩阵分别为 900/300，主规格通过 `validate_spec`，发布清单 8 个路径全部存在。`git diff --check` 无空白错误。

## 通过的直接正确性项

1. **输入权限和冻结来源。** 导出器沿用 `verify_source` 检查源域角色、scratch 继承和目标接触记录，再核对实际 checkpoint SHA、epoch=200、seed、初始化参数和原生 exact loader。仅加载 checkpoint 和来源元数据，不调用源样本 loader。实际前向处于 eval/inference 模式；参数和 buffer 逐值比较，避免只依赖 tensor version。时间、频率、PA 出口与原 identity 出口来自同一次 aux forward；代理检查原 identity selector 不增加 encoder forward。
2. **support 专用读取。** `SupportIQ` 只读取 IDs 和 NPY/ZIP 头，按 ZIP_STORED offset 建立只读 mmap；只有已登记 support 索引可 gather。原始 split 的 query indices 仅作为排除元数据，不读取其 IQ、标签或分数。support union 与 query union 不交，物理类标签跨 split 一致。支持专用缓存不包含 query 元数据；物理全局索引、ID、类标签及固定出口契约同时保存。
3. **逐任务隔离。** 探查入口仅读 capsule manifest、支持专用缓存及导出元数据，不读原 received NPZ 或原 split 文件。缓存 ID 恰好等于所有登记 support 的并集；每个任务各类实际 K、索引与 ID、label 与注册类精确对应。只有当前任务数组切片进入核心，无历史适应状态，也不返回部署分类器。
4. **固定数学与 OOF。** 六臂固定为两种背景各自的 baseline、duplicate 和 aux；不根据 OOF 选臂。目标是物理样本平方误差之和加 ridge=1，截距不惩罚；primal/dual 使用同一中心化系统，联合 RHS 的分类列与重构列可分。所有中心和映射在 trainfold 内重新估计；physical ID 排序分折，同一记录的全部出口共同留出。旧/新 membership 仅用于诊断。K1 不拟合、不造 held 指标；K≥2 只产生 support OOF，不作全 support 部署头。
5. **输出和预算。** 入口拒绝已有输出目录，完成标记只在完整 Cartesian 矩阵执行后写入。full trace 保留物理 OOF、重构和配对记录；compact JSONL/CSV 去除大数组但保留分量汇总和 K1 缺失原因；每折每臂全部解析阶段 scalar 单独保存。每折 6 次 Cholesky 分解、数值诊断另计 5 次谱分解；optimizer steps=0，持久部署状态=0，同时报告临时头字节。新增 source payload=0；完整训练 checkpoint 文件字节不称作最小推理包，部署状态和增量模型传输未知。
6. **发布和运行。** 实际两份矩阵配置与 CLI 对齐。发布包包含 core 目录、导出器、探查入口、来源元数据工具、runner 和配置；入口导入导出器常量不会加载 Torch 或模型。发布前检查已推送 commit、相关路径未修改以及新 archive/release/run 路径；远端发布拒绝已有输出。GPU0 串行导出，最多 4 个 CPU probe lane，各 2 个 BLAS 线程；单 lane 技术失败保留产物且不自动重试，不干预其他健康 lane。父编排根据产物绑定和任务数量确认完成，不仅依赖 exit code。

## 边界

本检查没有启动远端实验、重新加载真实 checkpoint、读取真实 IQ/feature 缓存或任何 query 成绩，也没有验证真实运行完成。真实 checkpoint 的运行兼容性与最终远端产物状态仍由发布执行证据确认。全 8 lane、4800 episodes 的分层 support 诊断由独立只读汇总工具处理；OOF 的重复 support draw 不等于独立泛化样本，任何分支增量均不自动支持 query 改善或方法晋级。
