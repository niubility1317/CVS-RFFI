# PrototypeTransport-LocalRidge 技术恢复

当前状态：r02 四行、160 parent 已完整结束，主管及 worker 均退出。独立性能汇总及全日志分析已启动，性能结论尚未核实。

联合方法保留 BranchLocalRidge 最终分类器，在合法目标域 support 上优化 10 个原型条件适配参数（9 个约束自由度）。B 阶段仅用旧类 support；C 阶段继承 B 状态并注册新类。原距离与适配距离各占 0.5。固定地面 Phase1、practical residual，不读取源域样本、逐样本源域特征或 query。目标域 support 原型无需新增地面统计传输；实际常驻状态与计算成本仍需测量。

## r01 失败与修复

`20260930-phase2-d92-prototype-transport-support-m2-r01` 的四行均因完整 parent JSON 序列化失败而退出。独立 state/PID/argv 读回确认 FAILED、无活动子进程，未停止或重启健康任务。每行已有 5 个 K=1 parent 输出，但没有完整完成的实验行或可用于全矩阵比较的结果。K=5 内部拟合可能已执行，不能据此声称性能变化。

`_armijo_tolerance` 的 NumPy 浮点容差使 accepted 比较产生 `np.bool_`，公开 audit 字典没有统一转为 Python 标量。事件回调已有转换，因而事件日志成功、完整 parent 写出失败。修复仅涉及输出类型：容差与 accepted 显式转换，公开 audit 输出统一转换；入口严格转换 NumPy 标量与容器，保留布尔和数值类型，拒绝未知类型与非有限数，不使用 `default=str`。

修复验证：核心公开 audit 严格 JSON 测试 1/1 通过（2.33 s）；真实流式 JSONL、文本与 CSV writer 测试 1/1 通过（7.36 s）；入口全部 15/15 通过（9.96 s）。结合此前核心 27 项与编排 14 项，57 个不同相关测试已通过。保留全部原始 stdout/stderr，未重复无变化的全套检查。

## 同方法恢复

新 run：`20260930-phase2-d92-prototype-transport-support-m2-r02`。配置：[恢复预登记](../configs/d92_prototype_transport_support_recovery_20260930.json)。release：`d92_prototype_transport_support_20260930_r02`。

r02 与 r01 的数据、checkpoint 契约、权限、指标、方法、超参数、support 特征和 seeds 保持一致。恢复构造器逐项断言这些条件。完整运行同一 160 parent 矩阵，使用独立输出目录，保留 r01 全部失败记录；没有根据性能选择子集或改变参数。root 是唯一 launch owner。

输入可用性检查 `prototype_transport_preflight_1790764399351376600.json` 为 VERIFIED：四个原有 support cache 绑定有效，新 run/release/archive 路径不存在。检查没有读取 query 或源域，也没有启动 GPU 作业。这是路径与输入检查，不是方法性能证据。

后续先独立核实启动状态，再完成 support-only 验证。报告按 K×新增类数列出三阶段指标、旧类下降、新旧类差距、H 与资源成本；缺失地面 A 指标记 N/A，不能将原 support 分类器的 B0 当作 A。独立新数据验证按用户要求暂缓。

启动独立读回：`readback_1790764932355254600.json`，argv/CWD/commit 与预登记一致，未发现错误；完整矩阵尚未完成。

完成独立读回：`readback_1790766778553149900.json`。实际 3656 次接受更新、647 次拒绝试探、19821 次头拟合/分解。无训练技术错误；不把训练结束当成性能成功。
