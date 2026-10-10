# 2026-10-10用户停止记录

香港时间10:25:50按用户明确请求停止下列两个run，10:26:08独立进程/GPU读回VERIFIED。数据与产物全部保留，无自动重启授权。

- [参考基线修复32行](../automation_reports/CV-SincNet/20261008-phase1-reference-repair-manysig-m32-r01/report.md)：30行源训练完成，1行partial，1行未启动；状态STOPPED。
- [Urban CE24行](../automation_reports/CV-SincNet/20261008-phase1-urban-ce-manysig-m24-r01/report.md)：24行均未启动；状态STOPPED。

新版状态条件作用控制器2534798及12个训练worker均未受影响。精确PID、start_ticks、操作及读回证据位于上述报告旁的evidence/user_stop_20261010.json。旧远端queue快照仅作历史证据。
