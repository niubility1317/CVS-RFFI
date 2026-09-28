# D92-MVRidge-v1 独立 P0/P1 审查

日期：2026-09-29。结论：本次本地代码和合成证据范围内，未发现越权拟合、错误预测、无法启动或覆盖历史输出的 P0/P1 阻断问题。仅审查新增 MVRidge 核心、入口、审计工具及必要发布接入；不重复旧方法、既有数据或不变控制面的审查，不增加审批。

## 范围与输入边界

依据当前本机[项目协议](E:/type10-7/项目.md)和[最终设计第 10 节](D92_SUPPORT_INFORMATION_DESIGN_20260929.md)，检查：

- [MVRidge 核心](../code/cvsrffi/stage2_d92_multiview_ridge.py)及[固定配置](../configs/d92_multiview_ridge_frozen_20260929.json)。
- [预测入口](../tools/predict_d92_multiview_ridge.py)和复用的[纯缓存边界](../tools/d92_orbit_feature_cache.py)。
- [日志审计工具](../tools/collect_d92_multiview_ridge_audit.py)与[合成测试](../tests/test_collect_d92_multiview_ridge_audit.py)。
- 两份正式 spec 与 collector 对接；runner/publisher/scorer 的 MVRidge 名称、CLI、依赖和并列处理定点检查。

审查者知道此前任务的结果，本轮未读取任何新的 target 成绩、已有成绩文件或评分汇总，也未据历史结果提出公式、尺度、预算或参数建议。所用测试全部为合成输入。

## 直接正确性检查

**特征与物理权重。** 核心采用 `unit(concat(unit(identity160),4*unit(FFT96)))`，各归一化下界为 1e-12。四个固定相位来自同一物理观测，FFT 共用原缓存描述子；输入没有增加物理 K。目标按所有物理样本求和，每个物理样本的四视图平方误差取平均，类别输出维求和；没有额外除以 N、C 或 V，也没有旧新组权重。每折的实际样本总数为 N=Cn，中心化 one-hot 的总平方范数为 `N*(1−1/C)`。

**目标分解与闭式解。** 实现的均值误差、view scatter 惩罚和 ridge 项分别为：

`0.5*Σ_i ||m_i W+b−t_i||²`；

`0.5*Σ_i mean_v ||(x_iv−m_i)W||²`；

`0.5*||W||²`。

三项之和等于固定目标；view 惩罚与均值误差之和等于显式逐视图数据损失。ridge 系数为 1，截距不正则化。`G=I+M_centeredᵀM_centered+Σ_i V_i`、`H=M_centeredᵀT`，经 Cholesky 解 `GW=H`，再取 `b=−mbarᵀW`。等 K 保证目标均值为零，因此该截距符合固定目标。零特征对应零 W/b 和可核对的常数损失；不切换其他方法。G 的解析最小特征值下界为 1，条件数上界为 N+1。

**真实残差日志。** 核心从展开的实际残差计算 W/b 梯度，另记录 `||GW−H||F` 和截距梯度范数，不把无优化器写成无梯度数值。入口 compact 的 `gradient_norm` 是最终解析解的已测量驻点残差；startup 尚未拟合时为 null 并说明原因。学习率和 epoch 为 null，优化器更新为 0；不作迭代收敛声明。有限精度下展开 W 梯度与中心化 normal-equation 残差可相差 `mbar⊗grad_b`；collector 使用 `||mbar||≤1` 的界核对，不把两种残差当成数值上必须完全相同。

**fold 隔离与 K1。** 所有 fold 的中心、view scatter、G、H、W、b 都在 `_fit_train` 内从训练子集重新计算。held 物理样本及其全部视图一起排除；不等长 fold 使用实际 n 和 N。点式归一化不依赖其他样本。最终全 support 重新拟合；不复用其他任务的头或适应状态。K1 只执行同一闭式目标，没有 OOF；K≥2 固定物理 fold 的 OOF 仅为诊断，不改变公式、候选、尺度或 ridge。old membership 只用于 OOF 分组。

**query、类表与持久状态。** query 逐样本求四视图特征均值并乘冻结 W/b，不使用 query 真值、全局类别数、配额或跨 query 统计。核心保持外部注册类表顺序；预测及 scorer 对精确并列采用物理类 ID 升序。持久数值参数只有 float64 W[256,C] 和 b[C]，共 `8*C*257=2056*C` 字节，C=26 时为 53456 字节；不保存 support bank。内存数组由不可写字节缓冲构造，评分不追加缓存或改变状态。

**缓存科学绑定与启动。** 复用的缓存 helper 已包含真实 native provenance 缺少 SHA 字段的技术修复：旧 D92 startup 仍必须匹配 SHA；旧 origin 如果显式携带 SHA，则必须匹配；BNNA 生产 marker、provenance、startup 和 NPZ 标量均继续精确绑定。source-only scratch、final200、空继承链、未接触 target、类表、seed、capsule、固定四相位/FFT schema 和物理 ID 校验保持。复用此前已核实的 8 份缓存绑定证据，不重验数据或重新载入真实模型。入口合成测试采用真实旧 origin 无 SHA 的正例。

**发布与输出。** 两份 MVRidge spec 各 4 个模型，rx3 每模型 900、rx1 每模型 300 个 split，与 collector 对接通过。publisher 明确包括新 predictor、完整 code、共享缓存 helper、其通用导入依赖和冻结 BNNA 生产配置；新增依赖路径存在。MVRidge 进入 CPU 缓存复用分支，CLI/folder/mode 一致；无新增 encoder/GPU 导出。新预测目录及文件 exclusive 创建，已有输出拒绝；原缓存只读。新增 source payload 为 0，完整训练 checkpoint 文件大小、缓存容器、适应状态和 RSS 分列，模型部署/新增模型传输未知字段为 null。

## 验证与限制

审查者执行 `tests/test_collect_d92_multiview_ridge_audit.py`，57 项测试通过，包含实际 core 的 K1/K2/K5、零特征/仅旧类、物理 loss mass、N×C 目标尺度、损失恒等式、梯度残差、完整物理 fold/OOF、2056C 字节、阶段日志及 compact/stage JSONL/CSV 一致性。还对实际 predictor 的合成输出做端到端 collector 核验，覆盖来源权限、只读缓存、非终态提前拒绝、已有输出在 SSH 前拒绝、生成的远端核验脚本一致性。

入口 owner 报告 13 项合成测试通过，含独立重算展开目标和 W/b 梯度、单 query/重排一致、无 SHA 旧 schema、技术失败无 completion marker、拒绝覆盖。核心 owner 确认最终稳定且 9 项测试通过，包含独立 augmented least-squares 对照、目标/解析梯度与有限差分、held 扰动后 trainfold 全状态不变、K1 无验证、退化输入及状态字节；本审查未重复执行 owner 已负责的整套测试。最终 unit 规则在退化 identity view 下也按真实范数执行，不假定所有 view 的归一化后 FFT 块必然完全相同。

collector 读取完整元数据和本次 fit 日志，核对完整矩阵、每次 fold/final stage 以及日志间一致性；不读取 scores、truth、predictions、checkpoint 或特征数组。损失因 N、C 和拟合子集而变，日志统计仅用于一致性与成本描述，不能据不同尺度的累计损失直接比较方法优劣。OOF softmax NLL 不代表判别分数已校准。

本轮没有连接 N607、运行真实 checkpoint、读取真实输入数组、启动实验或提交。审查结论不证明远端执行完成、目标性能改善、显著性或独立泛化。最终结果仍需既有完整执行与独立评分证据，不另增门槛。
