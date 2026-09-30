# AffineJoint 报告器

[工具](../tools/report_d92_affine_joint_support.py)只读完整、已核验的 AffineJoint summary 与 analysis_execution，输出 Markdown 报告和机器可读解释。CLI 使用 `--summary`、`--execution`、`--report`、`--interpretation`；首次实际结果出来后由 root 单次执行。

报告保留 80 个 K×新增类数×路径×OOF/proxy 聚合行，以及 receiver/scene/model 分层。H 和新旧类绝对差先按 parent 计算再平均，不能以总体均值重新计算替代。旧类 A 与 B−A 缺失时标为 N/A；原 R0/B0 不是 ground A。true K1 没有持出准确率，不能补造。全部结果只是复用 support 的诊断，不是 query 或独立数据验证。

方法解释明确包含自由解析截距、完整 `g_b` 伴随、冻结实际本次 B prior、q gauge 边界、所有 RHS/intercept 费用，以及解析头参数与梯度参数的区别。资源代理不写成实测 FLOP，未测星载/GPU/传输为 N/A。无自动晋级、温度或参数选择。

[11 项合成检查](../tests/test_report_d92_affine_joint_support.py)通过（0.57 s，root 证据 `pytest_utf8_1790802979858337900`），覆盖完整正常渲染、schema/scope/status、完整矩阵与分层、A 缺失、资源字段和非法汇总拒绝。未读取真实 AffineJoint 性能，因为实验尚未开始。
