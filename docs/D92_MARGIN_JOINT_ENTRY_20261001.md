# MarginJoint support 入口与发布能力边界

本文说明独立MarginJoint的prepare、preflight、run、publish四个入口及root负责的evaluate集成。五入口已通过34项合成集成检查，联合core接口已冻结。本次没有创建真实spec、选择实际资源预算或任务矩阵，也没有启动该方法实验。入口验证不等于独立分析链已可用，不能推出准确率改善。

## 输入与资源契约

prepare 只接收 owner 显式提供的 request 和代码 commit，不读取历史配置、索引、评分或运行报告。沿用现有 source-only support cache、capsule、checkpoint SHA、model seed、六类 seed 角色、显式 row 和完整 selection 绑定；不加载 checkpoint、encoder、源样本或 query。

`request.probe.qp_resources` 必须恰有以下两个字段，值必须是有限正整数，拒绝 bool、浮点、缺失值和默认补值：

- `max_transitions`：每次 QP head 求解允许的工作集转换循环次数。
- `max_factor_buffer_bytes`：每次 QP head 明确持有的因子输入/输出数组字节上限。

两个值原样进入各 cohort 配置的 `qp_resources`，由 evaluate 显式传给每次 core preparation。配置外形为 `algorithm`、`producer_matrix`、`selection`、`qp_resources`。preflight 回应、evaluate startup/complete 和 supervisor complete 均绑定相同值。数值由 owner 给出；测试中的数值仅是手工合成用例，不是正式预算。

因子 buffer 上限不是进程内存上限，不包括全部 K、旧点响应矩阵、返回状态、BLAS 工作区或 Python 对象。实际资源口径见 [QP head 组件说明](D92_MARGIN_QP_HEAD_IMPLEMENTATION_20261001.md)。峰值 buffer 按 max 汇总，实际累计工作按 sum 汇总；未测进程内存、设备开销、完整部署包与传输字节不能从配置或数组规模猜填。

## 五个入口

| 文件 | 职责 | 保留边界 |
|---|---|---|
| [prepare_d92_margin_joint_probe.py](../tools/prepare_d92_margin_joint_probe.py) | 生成独立 spec 和 cohort evaluator 配置，核对显式 QP 上限 | 预先检查所有目标不存在，再独占创建；不覆盖已有文件 |
| [preflight_d92_margin_joint_probe.py](../tools/preflight_d92_margin_joint_probe.py) | 生成/执行既有只读元数据可用性检查 | 仅 capsule/cache provenance/split 元数据和路径存在性，不反序列化特征数组 |
| [run_d92_margin_joint_probe.py](../tools/run_d92_margin_joint_probe.py) | 显式 row supervisor、命令、marker 身份和实际计费检查 | 唯一 owner、独占 run/row 输出、失败保留、无自动重试 |
| [publish_d92_margin_joint_probe.py](../tools/publish_d92_margin_joint_probe.py) | 显式源文件白名单与发布能力检查，复用既有推送后 archive/SCP 传输 | 不复制整 code；实际 helper 依赖明确入包，分析链缺失时拒绝 dispatch |
| [evaluate_d92_margin_joint_probe.py](../tools/evaluate_d92_margin_joint_probe.py) | 实际B→C顺序、support持出预测、评分、完整归档与日志 | 显式QP资源、逐样本全注册列、预测后truth join、失败成本与数值状态保留 |

权限和物理配对仍由现有协议限定：旧类 support 在同 receiver/scenario/K/support seed 的新增类轴上保持同一物理 ID 映射。任务数量从显式 rows 和各 cohort selection 推导，不使用全局四行除法，也不由 query 信息推断类数或配额。当前固定矩阵/外层预算契约来自既有设计和冻结 core；本工具不搜索或选择矩阵与预测参数。

## evaluate/core 对接要求

evaluate依次准备并拟合实际B，再将当次同run/row/split/scope/物理fold的B对象传给C；trial只是日志坐标，不增加继承权限条件。C包含全部旧、新物理训练记录和全部注册列，不压缩重复特征对应的标签约束。new0原对象复用B，不重新求解QP；其耗时或计算不能伪造为一次新head。

旧类 margin 阈值来自冻结的实际 B，在合法旧 train support 上计算并保留原值。held support 不进入约束。即使零核，非 new0 的 C 仍求解有自由截距的约束头；不能复制 Conditional 的零 residual 或旧点代表压缩规则。

固定预测包含 `M + L @ alpha + b`，上层不得重复加入 M。所有已注册类统一竞争，分数固定之后才进行独立 support truth join。K1 不制造 held 指标。A 和 B−A 在未明确接入匹配 Ground A 时保持 N/A，不能用 R0 或 B0 代替。

完整事件、物理 ID、实际 B 引用、原始数值数组和每次 QP 状态需通过生产 callback 归档。QP 失败由上层保留真实异常 audit 和数组；非规则活动集的 `UnsupportedJacobian` 是技术失败，不返回零梯度、无约束梯度或静默降级。

## 计费与失败语义

结构上界只用于核对 owner 声明的有限工作范围，不当作实际执行量。C 的单次 QP 最多包含一个 affine factor 和每轮至多一个 working factor，因此 runner 用显式转换上限推导保守 factor-attempt 上界。工作集独立性求解、最终方程读回与伴随求解另按实际 ledger 计费，不能套用“每个 head 两个 factor”“三次谱检查”或“triangular calls = 2 × factors”。

记录 forward/adjoint 两个家族的 factorization attempts/completed、condition estimation、谱检查、triangular calls/RHS columns/RHS elements/n²RHS 工作代理、transitions、完整约束扫描和 compact snapshot rebuild。伴随复用 forward factors，新增 factor/spectral 为零，但求解工作不免费。转换耗尽时的最后一次状态重建同样计数。

supervisor 成功 row 的累计工作相加，明确峰值字段取最大值。任一 row 技术失败时，`complete.json` 为 FAILED，`workload_complete=false`，`aggregate_scope` 明示仅成功 rows，`failed_row_workload=null`。失败 row 保存实际命令、失败产物与 state manifest 的位置；supervisor 不把没有读回的失败成本补成零。该层不替代 evaluate 中完整的已完成工作、失败当次 ledger 与 numeric snapshot。

已有失败产物、健康 lane 和目录均保留，不重试、不回退、不覆盖。缺失异常数组或无法归档的技术错误必须保留原始错误与具体限制，不能写出完成 marker。

## 发布状态与合成检查

当前独立 Margin summary/analyzer 尚未在本任务实现。publisher 将它们列为 `missing_analysis`，即使四入口和 core 文件齐全，也不宣称正式可启动，且拒绝 dispatch。文件 presence readiness 不是独立数学核验，也不代替真实 import 闭包检查。以后接入分析器时必须使用 Margin 的约束/KKT 审计，不能沿用 Conditional 等式头的数学结论。

纯合成测试入口：

```text
python -s -m pytest tests/test_prepare_d92_margin_joint_probe.py tests/test_preflight_d92_margin_joint_probe.py tests/test_run_d92_margin_joint_probe.py tests/test_publish_d92_margin_joint_probe.py tests/test_evaluate_d92_margin_joint_probe.py
```

测试覆盖显式资源传播、非法/缺失资源、动态 rows、QP 实际 ledger、峰值 max 汇总、失败工作未知范围、独占输出、metadata-only preflight，以及从白名单复制到隔离目录后的实际 import。隔离用例使用独立 Python 子进程，不能从当前工作区回退找到旧 Conditional 模块；所有 SSH/launch 都由合成替身替换。

冻结的 Margin core 实际从 `d92_conditional_joint_local_ridge` 复用 `project_coordinates`，该 helper 模块顶层又导入 `d92_conditional_affine_kernel`。因此这两个纯源码依赖明确进入白名单，不能按文件名删除，也不能依靠工作区隐式提供。隔离测试核对全部 `cvsrffi` 模块来自白名单复制目录，并核对实际 projection helper 绑定。它们的存在不表示 C 调用旧等式 head，也不允许复制 Conditional 的双因子/三谱计费或数学结论；Margin C 仍由独立 QP head 实现。未使用的旧 Conditional evaluator、summary、analyzer 和真实产物不进入该依赖集。

四入口作者仅做AST/UTF-8静态检查。root在已验证ssr-gpu环境串行执行上述五文件：**34 passed in 18.28s**，证据`E:/type10-7/.codex_tmp/pytest_utf8_1790835922424555700.stdout`及同前缀`.stderr`。除四入口覆盖项外，还验证真实callback/NPZ、同次B对象与跨row拒绝、K1缺失指标、truth-last、原QP失败接口、完整失败数值状态及成功/失败成本scope。

evaluate对普通成功档案要求有限数值；对明确失败状态可原样保存非有限NPZ坐标，JSON只写非有限数量及N/A摘要，不改写原数组。已存在的core失败ref复用，原始QP异常无ref时独立归档；没有审计或计费的异常记null和具体原因，不能补零。所有工作SUM，两个明确QP buffer峰值MAX。实际star资源、性能改善、部署完成均未验证。
