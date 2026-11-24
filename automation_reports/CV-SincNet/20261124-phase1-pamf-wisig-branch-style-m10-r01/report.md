# PAMF SSL 的 WiSig 主对比分支风格干预

预登记编号为 `20261124-phase1-pamf-wisig-branch-style-m10-r01`。详细分支机理、匹配条件和后续路线见[设计报告](../../../docs/PAMF_WISIG_BRANCH_STYLE_20261124.md)，逐行矩阵见[实验登记](experiment.json)。

本轮固定种子 2026092701，复用同一 WiSig Main split 的合规三视图 E20。十行各新增 30 epoch、6660 步，保留原监督与 REPL 损失。训练冻结后逐行完成 clean 和六种 practical 的两群体预测，独立 CPU scorer 读取真值并计算指标。比较包含同种子对照、七景与 RX、TX、日期分层，以及相对原冻结 E50 的逐物理 ID 修复与损伤。

## N607 发布状态

2026-11-24 19:10（UTC+8）独立读回为 `VERIFIED / QUEUED`。发布归档校验与远端编译通过。调度器 PID 2933150、命令、工作目录及两次状态更新均已核实。10 行的训练、预测与评分共 30 个任务登记待发，调度器等待符合准入条件的 GPU。实际权重 smoke、训练、预测结果和评分待执行。启动读回见 [n607_launch_readback.json](n607_launch_readback.json)。

发布代码提交为 `8ec66542c21bfd9a08b9c799dd8cbc51740ad76c`。本地 20 项相关测试与独立 P0/P1 审查通过。
