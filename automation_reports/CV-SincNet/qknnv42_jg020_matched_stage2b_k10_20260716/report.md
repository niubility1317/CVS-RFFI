# JG_R8_LR020严格配对Stage2-B K10实验报告

Git镜像说明：主报告位于`E:\type10-7\automation_reports\CV-SincNet\qknnv42_jg020_matched_stage2b_k10_20260716\report.md`。本文件随代码提交同步更新，当前状态为`REMOTE_PHASE2_FULL_READY`。

实验覆盖5个target receiver×5个seed、固定K=10、3个`leo_*_weak`场景、单Query View。必须先逐row确认JG密封包与历史MRIOR/DADDA/ProtoNet K10 split的support/query ID顺序一致，并验证历史support/query View种子公式后，再运行锁定的`P4＋JG_R8_LR020`。本轮为Stage2-B old-only，不声明新类注册性能。

N607的25/25 cache/package/ID-View硬审计和runtime seal均PASS。三轮smoke后首行`20-1/713101/K10`完整通过：JG old_acc=78.8889%，相对direct +9.1667pp、相对P4 identity +2.2222pp，但相对matched MRIOR -12.7778pp，聚合最低旧类仅31.6667%。三轮回顾已重新核对`项目.md`、979条对话索引、历史三域适应、新类注册和K1报告；决定继续完成剩余24行作为用户要求的Stage2-B严格配对诊断，不用于query调参，也不替代Stage2-C新类注册。

详细设计、权限边界、风险、启动命令和结果表以主报告为准；每次N607状态变化后同步更新本Git镜像。
