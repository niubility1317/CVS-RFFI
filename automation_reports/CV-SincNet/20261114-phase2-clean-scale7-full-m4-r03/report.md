# Phase2 clean 五卡七进程接管

按用户最新资源要求，将 `20261114-phase2-clean-non-issl-full-m4-r02` 的未启动作业接续至不可变后继 `20261114-phase2-clean-scale7-full-m4-r03`。原控制器先停止新增派发，在途作业自然结束；完整结果保留并核对，仅无输出且无启动日志的原计划作业进入后继。任何父任务失败或 partial 阻止接管，不重试或覆盖。

N607 GPU 0–4，每卡本任务至多 7 个训练/预测进程，实际启动前预留至少 6000 MiB；七进程模式对已启动进程按 3000 MiB 预算保留其尚未使用部分，每次显存快照每卡至多新增一个进程。实际显存不足时少于 7 个。其他 owner 进程被避让，不停止或修改。原数据、checkpoint、方法、seed、K/N 和输出权限不变。该队列仍不包含先前暂缓的 ISSL、MoPC、交叉组和 pilot 剩余点。

运行代码版本见下方；入口 `python -m experiments.pamf_phase2_practical.clean_scale_takeover --parent-config <原准备目录>/scheduler.json --plan-root <新准备目录> --run-root <新运行目录> --release-root <新发布目录>`。各 worker 保留原 source-only smoke、失败停止派发与自然排空。仅预测；全比较预测未固定、独立 truth-last 自动评分尚未接通，不得称完整交付。

实际接管核实：父队列自然排空，1065 项完成，失败 0；后继保留并跳过该 1065 项，接收 16278 项未启动任务。控制器 PID 3443754，已实际分发至 GPU 0–4；后继首次核实新增 15 项完成、失败 0，各卡进程数依次为 6、7、7、6、6，空闲显存均超过 16 GiB。单个进程曾观测约 4.1 GiB，超过预留估计；3000 MiB 是调度预算而非实际峰值保证，仍须监控显存与失败。真实运行代码 commit `82691cb5dbcf9974eb4384bdb09d7285ba3abe0f`；单归档 SHA 及远端 compile 已核实。
