# PAMF 普通 PL × G 数据权限：补齐既有 51 个冻结读出

本补充沿用完备性审计已登记的 `20261022-phase1-PAMF-PL-Gfactor-readouts-m51-r01`，只发布该子矩阵。未发布的 15 个细粒度训练及其额外计时继续暂缓。三核心 A/B/C 的已完成结果不改变本补充的配置、默认方法或顺序。

17 个原数据契约各有 3 个普通 PL 的 F。缺少的是它们连接同契约、全部祖先只用 L 的 G-L；已有匹配监督 F 的 G-L/G-LU 和普通 PL F 的 G-LU 全部复用。补齐后形成 F（监督/普通 PL）× G（L/LU）四格。PL 使用训练好的当前模型、固定阈值 .95、默认 sat 拼接 .68 和原训练预算，不能混入原生 T1 的选择性 U 门控。F0 使用输入绑定修复后的 controls-binding-m6，不能引用旧错位 F 分数。所有来源保留默认分类头，不另训头。

新增 51 predict、51 CPU score 和 1 CPU aggregate，共 103 个阶段任务；新增 F/G 训练及恢复缓存均为 0。每行只复用原 raw 类别决定，新增 G-L 七景决定；chrono 还保留原未见源日期诊断。G-L 的恢复 IQ 由已发布外部 DURA 缓存按同物理 ID/同契约计算一次，原 FP32 字节无额外归一化直接复用。缓存不含类别标签或 F 输出；加载前沿用原实际 checkpoint/完整祖先契约核对。尚未完成的原 F/raw 或缓存是队列依赖，不重训或强行读取部分输出。

预测器只接收既有 VALIDATED_ONCE received-IQ 句柄、模型来源、缓存和原 raw 决定，不持有 truth 路径。该行所有任务和七景预测冻结后，独立 scorer 才连接原 truth，输出 Accuracy、Macro-F1、每 RX/TX/date 及净纠错。原 raw 在新旧 PL 读出间必须一致。汇总只在 51 新评分及所有既有监督/PL 评分齐备后运行，重算 CM 和净纠错，先 seed 内汇总六景，再报告三个实际 F seed 的样本标准差。保留 PL−Sup、G-LU−G-L 的全部正负值和差中差交互。

监督 F+G-L 是全系统 L-only；监督 F+G-LU 与 PL F+G-L 都使用过无标签数据。PL F+G-L 不能称为严格监督。三个 F seed 条件于固定 G，不能冒称三个完整系统独立重复。F0 五源/七目标 168000 与源数量六目标 144000 保持分列，不新增五源数量扫点。缓存分类耗时不是在线 G+F 部署时延；G 祖先和一次共享缓存成本不按每个消费者重复记账，也不冒称为零。

新唯一资源 owner 为 `20261023-phase1-PAMF-PL-Gfactor-control-r01`，CPU 汇总为 `20261023-phase1-PAMF-PL-Gfactor-analysis-m51-r01`。原 controller 经已验证 handover 退休，全部健康 worker 原 PID 或自然完成接续；旧 job、科学配置、CWD、输出与每 job release 不改变。资源保留基础 3/card、全机最多 26、仅 GPU0/6 合规时第四 F、G-heavy≤1/card、free≥12000 MiB 的原策略，不增加并发上限。两次同指纹新技术错误停止新提交，保留已有产物与所有健康任务；低分、负效应不是技术停止理由。

实现位于 `pamf_gfactor.py`、`pamf_gfactor_extension.py`、`analyze_gfactor.py`、`publish_gfactor.py`。聚焦测试验证固定 51 来源/F0 替代、无 truth 预测配置、原任务保持、raw/ID 错配拒绝、预测后独立评分、完整已知交互及不完整输入拒绝。一次独立 P0/P1 审查的本地无 Torch 导入问题已定点修复；发布器第一步对真实普通 PL F 和同契约 G-L 做 L-only 无 query smoke，PASS 后立即 handover/派发，不增加封签或额外审批。

实际提交、精确命令、逐行配置与路径、发布及性能状态以三个 run 的 experiment.json、report.md、launch_verification.json 为准。本文件不声称新性能已经完成。
