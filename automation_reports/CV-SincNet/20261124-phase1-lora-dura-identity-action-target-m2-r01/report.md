# LoRa DURA 恒等作用闭式求解目标测试

## 预登记

使用固定 2,400 步的恒等作用权重和源 U clean 门限。同一权重、门限与推理覆盖 Full、Residual。半量为首要读出，四分之一量为次要读出，分别与该路线冻结原宽版的同剂量预测及匹配 DURA 对照比较。 [源训练定义](../20261124-phase1-lora-dura-identity-action-source-m2-r01/report.md)

固定 Tri-set 2026092702、15 个未见接收机、同 30,000 个物理包和每路线七景。独立 scorer 先核对双路各 210 份冻结预测以及历史对照的完整性、包序、门控和直通，再读取本地真值。报告逐景、逐 RX、clean、净纠错和配对区间。单种子结果用于筛查，最终三种子 LoRa 和 WiSig 要求保持。

源训练与另一载波作用试验使用独立不可覆盖的发布根和 run 根。GPU 按实际显存和进程状态派发。契约、真实无目标冒烟、预测覆盖及非有限值异常时保留已有产物，停止后续步骤。 [执行记录](experiment.json)

## 运行安排

独立本地会话已等待本源实验终态。冻结后自动发布目标候选及启动独立评分。 [源运行证据](../20261124-phase1-lora-dura-identity-action-source-m2-r01/launch-evidence.json)

## 发布回读 VERIFIED

真实源 U 冒烟通过。已核对 Full GPU 7 与 Residual GPU 3 的 PID、工作目录、命令行、GPU 与增长中的预测产物。本地独立 scorer 正等待全量两路冻结。 [运行证据](launch-evidence.json)
