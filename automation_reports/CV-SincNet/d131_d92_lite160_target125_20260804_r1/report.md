# D131 D92-Lite160 Target125实验报告

本文件是`E:\type10-7\automation_reports\CV-SincNet\d131_d92_lite160_target125_20260804_r1\report.md`的Git承载镜像。实验状态、矩阵、资产、命令、验证和结果以根目录报告同步内容为准；提交前将完整同步。

- 实验ID：`d131_d92_lite160_target125_20260804_r1`
- 当前状态：`LOCAL_VERIFIED_COMMITTED_PENDING_N607_SMOKE`
- 协议：`p2_min_v1`
- 覆盖：125个outer row、375个scene row、单候选before/after共750个prediction surface
- before：同一规范化z_id160、同K的Phase1锁定qKNN
- after：K5/K10为D92-Lite160，K1为逐logit qKNN精确别名
- 禁止：query truth/role/quota/global reassignment/fit/update/selection
- Git基线：`0ad76d3a5568012c09ea66b6c5d8142f19766d16`
- 远端运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_target125_20260804_r1`
- 本地验证：26项聚焦/回归测试通过，`git diff --check`通过。
- 独立首轮审查：`P0=0,P1=3`；方法锁绑定与独立进程隔离已完成窄修复。
- 独立复审：`P0=0,P1=0,RELEASE_READY=yes`。
- 实现commit：`8d232fba89689e2e0e80cdc6eefdbbaa4340204f`。
- 方法锁SHA256：`6cfe8659390bf887bf1689edd24a17b6bed9ef103ccf6f5bfde4d36574725e15`；完整SHA已嵌入candidate身份，所有CLI命令均重新校验方法锁。
- 当前待完成：N607预检、真实checkpoint smoke和唯一runner交接。
