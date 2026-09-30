# AJLR 完整结果报告入口

`tools/report_d92_anchor_joint_support.py` 只读取完整、独立核验的 AJLR support summary 与分析执行记录，不拟合方法，不打开 IQ、特征、query 或历史结果。当前真实试验仍在运行，本文件不包含性能结论。

报告完整输出两种诊断 × 两种方法 × 四个 K × 五个新增类数，共 80 行；旧类 6 个，新增 0/2/5/10/20 个，K1/5/10/20。逐行列出 A、B、C 旧/新、H、B−B0、注册下降、平均绝对新旧差与相对 R0 的 H 变化。A 和 B−A 保持 N/A，B0 不替代 A；K1 无独立持出性能。新增 0 不生成新类指标。

主总体是 96 个 new-present 可测 oof 配置，每个 K×新增类数单元有 8 个配对配置。receiver/scene 与 model/cohort 各保留四个完整分层，每层 24 个可测 new-present 配置。H、平均绝对差读取逐配置统计，不从总体旧/新均值重新计算；注册下降变化必须结合 B 与 C 分别判断。报告不自动选择较好路径，不实施晋级门槛。

常驻完整数值状态与推理必需数值缓冲区分开报告。实际训练/准备/推理时间、RSS、头/因子/primal/EDF/CE-adjoint 次数、地面统计及未测项按 summary 的实际字段读取；未测项写 N/A。CPU累计工作不当作墙钟，参数量不当作星载总计算量。两路径 AJLR 与三路径 FCR 的完整工作不同，不能用总运行时间直接证明微调模块速度改善。

完整 160 配置与独立分析结束后，root 单次执行：

```text
F:/App/miniconda3/python.exe -X utf8 tools/report_d92_anchor_joint_support.py --summary E:/type10-7/automation_reports/CV-SincNet/20261001-phase2-d92-anchor-joint-support-m2-r01/results/support_summary/summary.json --execution E:/type10-7/automation_reports/CV-SincNet/20261001-phase2-d92-anchor-joint-support-m2-r01/results/support_summary/analysis_execution.json --report docs/D92_ANCHOR_JOINT_SUPPORT_RESULT_20261001.md --interpretation E:/type10-7/automation_reports/CV-SincNet/20261001-phase2-d92-anchor-joint-support-m2-r01/results/support_summary/root_interpretation.json
```

输出采用独占创建，已有文件不得覆盖。输出状态为 `COMPLETE_SUPPORT_REPORT_RENDERED_DECISION_PENDING`，之后由 root 基于完整合法 support 证据解释共同改善和缺项；该状态本身不证明性能提高。

验证：10 个报告检查通过，证据 `.codex_tmp/pytest_utf8_1790792326814472900`。测试明确保护 parent-first H/绝对差、80 行完整矩阵、A/K1 缺项、完整分层与独立分析版本绑定，拒绝 partial、query/source 使用、捏造 A、缺失或重复单元。方法核心/入口/汇总/调度原 54 个检查不因新增报告器重复运行。当前健康 runtime 仍为 `a1a003f59e8ed14640a252ada8290a03af3420c5`，没有热修改或重启。
