# ConditionalJoint 报告与训练诊断

本次新增两条独立只读接入：`report_d92_conditional_joint_support.py` 生成完整支持集报告，`collect_d92_conditional_joint_training_diagnostics.py` 描述全部训练机制与资源。两者不参与拟合、选模或重跑，不读取 query、历史评分或全局索引。本文件没有实际性能结论；本 worker 只创建合成测试，数值验证由 root 串行执行。

## 独立 summary 契约

报告与诊断只接受 `COMPLETE_CONDITIONAL_JOINT_PROBE_VERIFIED`，`summary_schema=d92_conditional_joint_support_summary_v1`，方法 schema 为 `d92_conditional_joint_local_ridge_v1`。query/source 使用数必须为 0，算法保持 CE-only、proximal coefficient 0、coordinate ball radius 0.5。

报告输入完整 summary 与独立分析 execution 绑定，CLI 为 `--summary --execution --report --interpretation`。输出必须不存在。报告明确区分 R0 与 R_CONDITIONAL_seq，保留全部 K=1/5/10/20、新增类 0/2/5/10/20 的 80 个路径×诊断矩阵单元，并保留全部 receiver/scene、model/cohort 分层。覆盖数从 summary 的实际 parent 数推导，不假设四个 row 或 160 个 parent；检查每个分层的完整矩阵和可测 parent 总数。

表格包括 A 旧、B 旧、C 旧/C 新、H、B−A、B−B0、注册旧类下降、绝对新旧差及相对 R0 的变化。A 实际来源缺失，因此 A、B−A 必须为 N/A；不拿 support 基线替代地面头。真实 K1 没有 held，所有 OOF/proxy 指标保持 N/A；新增类 0 的新类准确率/H/gap 保持 N/A。

H、decline、abs gap 使用独立 summary 的逐 parent 派生值，绝不从总体均值重算。Proxy 已在每个 parent 内遍历全部 anchor 再等 parent 汇总。报告完整保留矩阵和分层，不挑选最优单元。10/1/3 个百分点仅描述理想方向，不构成晋级、重跑或性能硬门槛。

## 完整训练读取范围

诊断沿用 snapshot/extract 两阶段。Snapshot 用白名单解析 summary 元数据，仅保留身份、算法、coverage、训练来源和归档元数据，不反序列化 statistics 或 outer 结果表。随后检查各 lane 的完整 `training_events.jsonl` 与 `training_events_compact.jsonl` 的身份和事件顺序，捕获全部训练引用及事件索引。

Extract 重新核对完整事件索引和训练引用，遍历每一个捕获的 NPZ，不抽样。允许的 namespace 只有 B_prepare、C_prepare、B_CONDITIONAL、C_CONDITIONAL_seq；必须匹配当前 run/row 和合法 support 作用域。OUTER_SUPPORT_HELD、query、目录逃逸及未捕获引用一律拒绝。不会打开 `fit_trace.jsonl`、outer held 特征归档、query truth 或历史结果。NPZ 只用 `allow_pickle=False`，核对所有数组的有限性、shape、dtype 和字节数。

原始训练事件和 NPZ 保持原位置。完整阶段的梯度、试探、接受步骤引用与真实 FINAL.audit 对齐，保留全部训练 objective 曲线和全部数组的数值统计。训练中的 inner support 监督属于训练记录；其损失不能当作独立性能证据。发现原 `probe_failed.json` 时保留它并拒绝将该 lane 重标为完整，不覆盖、不重试。

诊断 CLI：本机 `--summary-root --output`；两阶段使用 `--summary-root --snapshot-output`，再用 `--snapshot --output`。远端操作还需显式 `--ssh-host --ssh-config --remote-python`；远端只读 stdin/stdout，输出在本地独占创建。本次没有执行这些操作。

## CE-only 与资源口径

归档 `g_Z` 就是分类 CE 梯度。诊断不做 Affine 的 `g_Z−Z` 分解，不加入 `+Z` 或 0.5||Z||²。归档方向与规范化 CE 梯度比较；每个试探使用 `Z_trial−Z_current` 的实际投影位移，核对 0.5 硬球、梯度内积、Armijo 阈值与非增条件。比较容差沿用固定的机器精度规则 `128*eps64*max(1,|before|,|after|,|Armijo bound|)`，不从性能选择容差。

完整成功阶段分别核对 projection/residual 的两个因子与各两次 triangular solve，两组 adjoint 必须复用前向因子；三个谱诊断按实际 projection 头单列。每组 RHS 列数、元素数、`n²×nrhs` 工作代理单独保存。原始失败中的 partial ledger 不套用完整成功阶段恒等式。

阶段 audit 保留真实耗时、内存、数值缓冲区及梯度训练参数与解析头参数口径。数组范数、常数 v、旧锚点 residual 和完整数组字节是训练机制诊断；numeric bytes 不含 Python 对象开销或完整部署包，`n²×nrhs` 不是实测 FLOPs。嵌套阶段时间不得直接相加作为墙钟。旧训练锚点 residual=0 不能推成未来 query 的保旧保证，远处常数 v 仍属于部署函数。

输出包括完整 stages/curves/preparations/archives/strata 的 JSONL 与 CSV，以及 summary 和简要 Markdown。未测量项保留 null/N/A，不将预算上界写成实际资源。

## 安全传输与依赖

Collector 只复用 `collect_d92_affine_joint_training_diagnostics.py` 中通用 JSON 跳过器、有限数值 archive loader 和 CSV/JSON 输出 primitives；不调用该文件的 Affine 目标、梯度、头或阶段诊断公式。发布源包需包含这个 helper。

远端 stdin 脚本内注入同一只读 helper 源码；snapshot 使用 `ensure_ascii=False, allow_nan=False` 的 UTF-8 JSON，经 `gzip.compress(...,mtime=0)` 和 base64 传输，远端只用 stdlib 解码，不把嵌套 JSON 放进 Python 字面量。测试覆盖 Unicode、深层结构、压缩后超过 1 MiB、坏 gzip 和非有限值拒绝。

合成测试另覆盖动态 parent 数、完整矩阵/分层、A/K1/new0 不可伪造、parent-first H/gap、真实新 entry/core 产生的 K2/K3 全训练事件与所有 NPZ、CE-only 梯度、投影后实际 delta、完整工作量 ledger、来源边界、失败保留和独占输出。root 已在约定 `ssr-gpu` 串行执行两个新测试文件：**18 passed，18.42 s**。原始输出前缀为 `E:/type10-7/.codex_tmp/pytest_utf8_1790820313572737900`。本次验证不代表完整发布链已验证或已具备 launch 状态；由 root 合并 summary、实际依赖清单后另行确认，没有读取真实结果或启动远端诊断。
