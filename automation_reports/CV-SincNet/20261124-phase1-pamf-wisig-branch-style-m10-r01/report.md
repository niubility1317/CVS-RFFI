# PAMF SSL 的 WiSig 主对比分支风格干预

预登记编号为 `20261124-phase1-pamf-wisig-branch-style-m10-r01`。详细分支机理、匹配条件和后续路线见[设计报告](../../../docs/PAMF_WISIG_BRANCH_STYLE_20261124.md)，逐行矩阵见[实验登记](experiment.json)。

本轮固定种子 2026092701，复用同一 WiSig Main split 的合规三视图 E20。十行各新增 30 epoch、6660 步，原监督和 REPL 损失保持。训练冻结后逐行完成 clean 和六种 practical 下的两群体预测，独立 CPU scorer 读取真值。比较报告含同种子对照、七景与 RX/TX/日期分层及对原冻结 E50 的逐物理 ID 修复与损伤。

N607 发布以调度器、真实权重 smoke、训练及评分产物各自的读回证据为准。历史基准只作配对和重放核验。
