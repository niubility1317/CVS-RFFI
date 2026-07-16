# JG_R8_LR020严格配对Stage2-B K10实验报告

Git镜像说明：主报告位于`E:\type10-7\automation_reports\CV-SincNet\qknnv42_jg020_matched_stage2b_k10_20260716\report.md`。本文件随代码提交同步更新，当前状态为`REMOTE_PHASE1_PREP_READY`。

实验覆盖5个target receiver×5个seed、固定K=10、3个`leo_*_weak`场景、单Query View。必须先逐row确认JG密封包与历史MRIOR/DADDA/ProtoNet K10 split的support/query ID顺序一致，并验证历史support/query View种子公式后，再运行锁定的`P4＋JG_R8_LR020`。本轮为Stage2-B old-only，不声明新类注册性能。

本地已完成`py_compile`、59项focused pytest、JG runtime primitive数值等价检查、offline exact split selector检查、25-row plan dry-run和`git diff --check`，均PASS。N607首个`20-1/713101`cache、bundle及严格配对审计已完成；support/query ID集合和顺序、1-view query、View种子公式、历史代码绑定均为`1/1 PASS`。下一步顺序补齐其余24个cache/bundle，执行25/25硬门禁和runtime seal，再放行Phase2 worker。

详细设计、权限边界、风险、启动命令和结果表以主报告为准；每次N607状态变化后同步更新本Git镜像。
