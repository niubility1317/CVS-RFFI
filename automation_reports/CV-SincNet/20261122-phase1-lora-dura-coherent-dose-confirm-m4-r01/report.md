# LoRa DURA 高相干双步幅三种子确认

预登记的单种子步幅矩阵已验证四分之一和半量修正均优于同种子的形状保护路径。按其跨种子条件扩展。追加种子 2026092701、2026092703 的同源高相干恢复器，每个从零训练 1,800 步，按固定训练池、独立 practical draw、源校准和留出诊断执行。种子 2026092702 复用全部冻结预测和源波形诊断。两档步幅均验证，不根据已知目标结果改变权重、门控、损失、源输入或针对接收机选路。训练不读取 F、TX 真值或目标。

每个种子接本种子的冻结 Tri-set。每景 30,000 个目标物理包，15 个未见 RX，clean 与全部六种 practical。比较 Tri-set 直通、固定 DURA、DURA 形状保护、高相干全量、四分之一、半量共六路。复用两个新种子的旧三路冻结预测，仅重算通过新增门控的三档预测。总 1,890 份盲预测数组和 126 个评分单元，其中 630 份是种子 02 的整组精确复用。全部预测齐全后独立评分，比较各 seed、场景、RX、clean 变化和星地净纠错。跨种子先检查两新增种子收益方向、最差接收机及 clean 保持，再报告平均；全量作为固定反证。

使用普通 N607、GPU0–7 允许调度、源 IQ 与真实 G 权重 smoke。由 `python -m experiments.pamf_lora_main.full_dura_publish --draw-submit ... --stack-submit ... --detail-dose-submit ... --confirm-dose` 执行，同一次 Git commit，工作目录 `CV-SincNet/releases/20261122-phase1-lora-dura-coherent-dose-confirm-m4-r01-release`，唯一输出 `CV-SincNet/runs/20261122-phase1-lora-dura-coherent-dose-confirm-m4-r01`。源训练并行，源审计后分别做盲预测。新增种子训练支持、各步损失、门控和源波形对照自动保存。来源、限幅、非有限、原预测身份或矩阵完备性为技术停止条件。
