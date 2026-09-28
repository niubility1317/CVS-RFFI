# D92-BNNA-v1 独立 P0/P1 审查

日期：2026-09-29。结论：**在本次检查范围内未发现 P0/P1 阻断。** 这是一轮独立正确性审查，不是性能结论，也不增设审批或真实实验前置门槛。

## 检查范围与依据

审查对象为最终的 [BNNA 核心](../code/cvsrffi/stage2_d92_bnna.py)、[冻结配置](../configs/d92_bnna_frozen_20260929.json)、[冻结特征导出入口](../tools/export_d92_bnna_features.py)、[support 拟合与预测入口](../tools/predict_d92_bnna.py)，依据 [BNNA 设计](D92_BNNA_DESIGN_20260929.md)及当前项目输入权限。核心作者完成 `_leave_one_ce` 等价聚合优化后才收束这次最终审查；没有对未变化的历史候选、runner 或数据胶囊重复审查。

本子任务另实现 [完整训练审计工具](../tools/collect_d92_bnna_audit.py)和[对应测试](../tests/test_collect_d92_bnna_audit.py)。它只读取完整拟合 trace、紧凑日志、逐步日志、状态及资源元数据；对 cache 仅查询文件大小，不读取 scores、query truth、predictions、checkpoint 内容或 feature 数值。正式配置要求四个模型 seed 与全部矩阵：rx3 每模型 900 次、rx1 每模型 300 次，两个完整批次合计 4,800 次拟合。当前尚未产生本候选的真实 4,800 次拟合审计结论。

## 直接正确性检查

- **输入与冻结模型。** 导出入口先按现有 `verify_source` 核对训练角色元数据，再校验实际 checkpoint SHA、最终 epoch、seed、scratch 与继承参数，使用指定 native code 的精确模型加载器，并检查导入模块来源。模型保持 eval、无梯度；参数版本与 buffer 前后保持不变。Torch/NumPy 交界复用已修复的 Python 数值列表中转。代码不调用 source loader、source feature bank、地面原型或摘要拟合。
- **物理观察与 query。** 每个 received IQ 只派生四个固定全长相位视图；全部视图仍属于同一物理 ID。缓存逐观察生成，不用跨 query 统计。predictor 将当前 split 的 support 子集传入 `fit_bnna`；query 只进入冻结状态的逐样本打分，始终面对全部注册类，没有按真实角色、配额或 query 类别数路由。不可写状态与物理类 ID 稳定并列规则同 scorer 一致。
- **完整外层分折。** 每类按物理 ID 排序，按位置模 `min(K,3)` 留出，同一物理 ID 的视图不分离。每次 `_fit_train` 的协方差、B、a 与原型均只接收该 fold 的训练 support。heldout 只评分；没有全 support 先学表示再仅对头留一。选择 trained 后，完整 support 重新从零估计并训练；选择 identity 后不保留 fold 的 B/a。旧、新类身份只进入 support 风险比较。
- **K1、目标与预算。** K1 没有候选风险或 OOF 调用，只按固定规则训练；其跨视图项是训练约束，不能称独立验证。非零 rank 使用 64 次完整批更新及固定投影 Adam 参数，采用最后一步系数而非最优 loss 步。rank=0 按设计为精确退化、零更新；不是静默失败回退。loss 权重使用各次训练实际 `n`。非有限输入、损失、梯度或状态走明确技术错误。
- **解析梯度与等价优化。** 负类共享 prototype 的 GEMM 计算与正类 leave-one 覆盖仍保留直接表征梯度和 prototype 两侧梯度。正类 prototype 梯度按类聚合后扣除自身项；没有建立 `N×C×D` 张量。近零 common prototype 使用原贡献顺序，避免 epsilon 分支放大聚合舍入误差。该优化未改变固定公式、选臂规则或训练预算。
- **输出与启动接口。** 新导出目录与预测目录拒绝已存在路径；预测、完整 trace、紧凑 JSONL/CSV、逐步 JSONL/CSV 均独占创建。新增候选/feature allowlist、CLI 参数、marker/schema、包内共享桥接依赖及 scorer tie 扩展相互匹配。评分沿用全部模型预测固定后才连接 truth 的既有流程。
- **日志与字节。** 审计核对每步 loss 分量及权重、LR、梯度范数、gate 边界与连续性、step 64 最终状态、rank/协方差/正交/残差界、每折训练和留出 ID、两臂 OOF 按物理样本权重聚合、最终选择与完整矩阵。完整 trace 与紧凑/逐步两套 JSONL/CSV 逐项一致。状态字节按实际 prototype、B、a 计算，训练工作数组与进程 RSS 分列。模型部署状态及新增模型传输字节为 null；训练 checkpoint 文件包不称最小推理包。新增 source payload 为 0，不表示已有模型传输为 0。

## 验证证据与限制

本子任务在最终核心上运行 `tests/test_collect_d92_bnna_audit.py`，**38 项通过**，包括真实合成 exporter→predictor→auditor 联调、完整物理分折与训练元数据、rank=0/K1/identity-selected、old-only、损坏证据拒绝、CSV 一致性、同一远端脚本的本地合成执行及已有输出拒覆盖。远端脚本测试未连接 N607。

核心作者提供最终 **14 项合成测试通过**，涵盖原朴素实现 oracle、解析梯度与中心有限差分、n=1/n≥2、近零及相消 prototype、heldout 扰动不改变对应 fold 的 B/a/prototype、K1、rank=0、26 类、状态只读和 query 分批不变。入口作者提供 **25 项合成测试通过**。主 Agent 已验证 runner/scorer/联合汇总的相关检查，本审查没有再次运行其未变化部分。

本次审查**没有加载真实 BNNA checkpoint、执行真实 target/source 数据、启动实验或读取本候选真实目标成绩**。代码中的 runtime provenance 检查与纯合成 smoke 路径已经检查，但不能据此宣称真实 N607 启动或完整矩阵已经成功。审计工具验证的是记录的完整性和一致性，不通过元数据重新证明全部数值计算。方法能否改善各 K 的旧类、新类和 H，仍只能由完整固定预测后的独立评分回答；本审查不提出成绩驱动的超参数、公式修改或性能保证。
