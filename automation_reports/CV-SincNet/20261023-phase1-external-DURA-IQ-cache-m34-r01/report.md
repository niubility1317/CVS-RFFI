# DURA 34契约角色：恢复IQ公共缓存

状态：本地验证后待发布，性能结果尚未生成。

范围及方法见 [固定实验方案](../../../docs/EXTERNAL_DURA_FROZEN_COMPARISON_20261023.md)。全部已有训练和 raw 预测直接复用；只新增同契约恢复与冻结分类读出，所有新视图固定后独立 truth-last 评分。

发布命令、环境、数据权限、种子、预算、停止规则及各行输出见 experiment.json。运行 owner 和实际代码 commit 以发布后 launch_verification.json 为准。
