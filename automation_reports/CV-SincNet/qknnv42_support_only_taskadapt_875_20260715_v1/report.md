# qKNNV42逐任务support-only适应875任务正式重跑

本文件是根目录正式报告`E:\type10-7\automation_reports\CV-SincNet\qknnv42_support_only_taskadapt_875_20260715_v1\report.md`的Git承载镜像。完整内容、运行协议、验证证据、N607命令与后续结果均以根目录报告为工作面，并在每次状态更新后同步到本文件。

## 当前摘要

- 状态：本地实现与dry-run验证完成，等待N607预检、同步与启动。
- Git提交：`08bce60 run: add task-specific qKNN support-only 875 matrix`。
- 矩阵：125个单qKNN+FFT96基线+6×125个逐任务support-only适应=875。
- 类别：6个旧类+2个已注册新类。
- 训练数据：只允许当前(receiver,seed,K)任务的目标receiver LEO support；禁止clean、source、proxy和query。
- 推理：单视图、FFT96、禁dense query、无角色Oracle、无类别配额。
- 本地验证：33项pytest通过，Python编译、Bash语法和875矩阵dry-run均通过。
- 结果：待运行完成后补充完整主表与行级artifact索引。
