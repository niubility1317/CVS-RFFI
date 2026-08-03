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
- Git交接头：`a59aaa5942b166cafedb8b8e0a60b1fac7785573`。
- runtime归档：`E:\type10-7\code\snapshots\d131_d92_lite160_target125_20260804_r1_runtime_a59aaa59_v2.tar`，SHA256=`e213690d35776706827dd7f41009c4e654924bd73aacfb02d3cb41dbc72b4159`。
- N607正式只读预检通过：8张RTX3090空闲，本地SSH连接已清零。
- 方法锁SHA256：`6cfe8659390bf887bf1689edd24a17b6bed9ef103ccf6f5bfde4d36574725e15`；完整SHA已嵌入candidate身份，所有CLI命令均重新校验方法锁。
- 当前待完成：N607预检、真实checkpoint smoke和唯一runner交接。

## r1终态

- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 远端归档与core/adapter/CLI hash通过，但method lock在Windows Git archive中被导出为CRLF，远端原始字节SHA=`f6e8bed0e5993874fbcacbb1ea1efe17c21b52f3279f56ade0d0537270e16fc9`，与冻结LF字节SHA=`6cfe8659390bf887bf1689edd24a17b6bed9ef103ccf6f5bfde4d36574725e15`不一致。
- r1在prepare前停止；smoke、shard、merge、truth和score均未启动。远端run root完整保留，不续跑、不覆盖。
- landing证据SHA256：`ca8b49a3b58b365a10b07a51947022d34a553c359f46d57241a848e82ff095f1`。
