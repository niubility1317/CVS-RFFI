# JG_R8_LR020严格配对Stage2-B K10实验报告

Git镜像说明：主报告位于`E:\type10-7\automation_reports\CV-SincNet\qknnv42_jg020_matched_stage2b_k10_20260716\report.md`。本文件随代码提交同步更新，当前状态为`REMOTE_PHASE2_SMOKE_RETRY2_READY`。

实验覆盖5个target receiver×5个seed、固定K=10、3个`leo_*_weak`场景、单Query View。必须先逐row确认JG密封包与历史MRIOR/DADDA/ProtoNet K10 split的support/query ID顺序一致，并验证历史support/query View种子公式后，再运行锁定的`P4＋JG_R8_LR020`。本轮为Stage2-B old-only，不声明新类注册性能。

N607已经生成25/25个cache、predictor package及detached seal，严格配对硬审计为`25/25 PASS`。第一次smoke的三场景request schema问题已由`3cd7bfd`修复；第二次smoke继续前进到候选锁校验，但runner错误读取了package中不存在的candidate lock member。`cfa058a`改为验证密封manifest中的预注册lock digest并通过33项相关测试。两次失败均在训练前且没有性能结果，日志与当时runtime seal均已归档；下一步同步、reseal并只重跑同一row。

详细设计、权限边界、风险、启动命令和结果表以主报告为准；每次N607状态变化后同步更新本Git镜像。
