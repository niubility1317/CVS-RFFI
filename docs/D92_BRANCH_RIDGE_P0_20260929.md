# D92-BranchRidge-v1 直接正确性审查

日期：2026-09-29。结论：本次唯一 P0/P1 审查未发现阻断问题。检查范围是会让下一次实际运行越权、覆盖、无法启动或产生非法预测的直接正确性；不重新评价已完成 support 探查的科学效果，不增设审批或性能保证。

## 范围与实际核对

阅读 [冻结设计](D92_BRANCH_AUGMENT_DESIGN_20260929.md)、[核心](../code/cvsrffi/d92_branch_ridge.py)、[新导出器](../tools/export_d92_branch_features.py)、[预测入口](../tools/evaluate_d92_branch_ridge.py)，以及 [runner](../tools/run_d92_confirmation.py)、[publisher](../tools/publish_d92_confirmation.py)、[scorer](../tools/score_d92_confirmation.py) 的 BranchRidge 接入和 [prepare](../tools/prepare_d92_branch_ridge_benchmark.py)。只检查相关新增路径，不重复无变化控制面。

直接用本机 Python 导入并核对两份实际 rx3/rx1 配置：candidate 等于核心冻结配置；方法、folder、predictor、mode、feature exporter/flag 配套；四个模型 seed 和 SHA 对应同一原始 source-only final200 来源；两 capsule 分别为既有 900/300 split；复用 baseline rows 为真，复用旧多视图 cache 为假。launch command 的 `--spec`、`--release` 和配置内 release 路径一致。动态发布工具清单全部存在，包含 exporter 的 support-reader/原生来源工具依赖和 predictor 的 split-validator 依赖；code 目录包含新核心及其纯数学依赖。

## 检查通过项

1. **固定公式和当前任务拟合。** 核心使用 736 维固定 zfft_aux 拼接：归一化 identity/FFT 背景，加等权归一化时间、频率和 PA 三分支；无最后整体归一化或数据标准差缩放。目标是物理样本误差之和加 ridge=1，截距不惩罚。只用当前任务 support 的标签、ID 和特征估计全部均值与头，旧类 membership 不参与目标或预测。全类等 K；class 和物理 ID canonical 排序只稳定算术，最终列恢复注册顺序。所有 K 只拟合一次，没有 CV、候选选择或历史适应状态。
2. **退化和状态正确性。** 特征完全相同情况下，平衡 centered onehot 的数学解为 W=0、b=0；显式零值只消除浮点均值尾数，没有容差阈值或数据反馈参数。一般解仍为同一 Cholesky primal/dual 系统。W/b 和 audit 不可变；实际数值头为 `8×737×C=5896C` B。K1 使用同一拟合公式，OOF=null 并明确无独立同类物理留出，不能将其称为已验证的 OOF 收益。
3. **checkpoint 与原始特征。** 新导出器经既有 source-only scratch 角色/继承检查、实际 SHA/epoch/seed 和 exact native loader 加载；不调用 source 样本 loader。每次原生 forward 严格只含一条 received 观察，eval/inference 且参数、buffer 逐值保持不变。identity 与辅助分支来自同一 aux，原 identity selector 代理检查不增加 forward。新 cache 不含标签或 query role。完整 received 缓存是本方法允许的只读推理输入，不是 support-only probe 缓存，也没有继承 probe 拟合状态。
4. **query 无拟合。** predictor 只以 `support_indices` 切片拟合；query 只经过相同固定特征公式及逐行 `xW+b`，面对全部注册类。query labels、truth、真实类别计数、配额或跨样本重新分配均不进入模型。NPZ 的 received 读取在 predictor 只取 IDs；score 接口不接收标签。实际 class 列、query IDs、有限分数、稳定物理 class ID 平局与独立 scorer 契约一致。
5. **启动和 smoke。** exporter 使用隔离包名加载本地核心，其相对纯数学依赖不会污染原生 `cvsrffi` exact loader。checkpoint 载入后的 PCG64 合成单样本先通过原生分支、历史 FFT 和 736 维转换，未先读取 received IQ。smoke 失败不写成功 marker；成功后计数分别记录 received N 次和 synthetic 1 次，总数 N+1，不把合成 forward 混入真实数据数量。该 smoke 不构成额外协议门槛。
6. **完成、评分和保护旧产物。** 新 feature/prediction 目录拒绝已有路径，预测 JSONL 逐任务记录后才写 completion；技术失败保留输出。runner 使用新 feature 路由，不重跑 support probe，不重新生成旧 baseline，不调用数据 builder 重验已验证 capsule。scorer 在所有注册模型的完整预测、ID/注册类、mode、argmax 和 marker 检查通过后才连接 truth；BranchRidge 平局规则与入口一致。实际新 run/release 命名区分既有数据与 baseline，publisher 继续拒绝冲突路径并发布已推送、无相关未提交变更的代码。
7. **成本与日志。** 完整 trace、compact JSONL/CSV 和单阶段日志记录物理损失质量、data/ridge/total、真实梯度与正常方程残差、阶段时间、真实 W/b 字节及提取计数。optimizer steps=0，LR/epoch/source validation 为 null 并给原因；没有将闭式求解称为梯度收敛。新增 source payload 和 ground statistics 均为 0；模型文件是完整训练 checkpoint 包，是否已部署和增量传输未知。单进程 RSS 与时间不能直接替代并发总量或整体墙钟。

## 验证证据与限制

核心负责人已报告 19 项新核心测试加 21 项复用 solver 测试共 40 项通过，覆盖公式等价、K1/C1、全零/近零、排列、逐样本 query 一致、只读状态、字节和负测。入口负责人最终报告 23 项 focused 合成测试通过，涵盖实际 core 相对导入、K1/K2 全 fit/predict、真 scorer 产物校验、当前 support 切片、query 单样本/重排完全一致、禁止 IQ member 读取、smoke 先后与失败边界、N+1 计数和 scalar-list ABI 桥接；py_compile/diffcheck 通过。审查期间指出的测试参数 tuple/dict 接口不一致已由负责人修正并验证，未发现运行实现 P0/P1。

root 已报告 61 项通用编排测试、24 项 summary 测试及 launch-ready/preflight 通过。以上测试和远端 preflight 均明确引用负责人证据，不伪称本审查独立重跑。审查者直接完成的是源码/接口检查、实际配置导入绑定和本地依赖存在性核对。

本审查没有远端 mutation、启动、真实 checkpoint forward、真实 query feature/truth/score 读取或提交；不证明真实运行已经成功。运行状态和真实产物仍由唯一 launch owner 独立读回。本轮数据为已评测复用基准，方法开发获得授权 support 探查信息，不能称为全新盲态泛化确认；性能结论必须等待完整预登记矩阵的独立评分。
