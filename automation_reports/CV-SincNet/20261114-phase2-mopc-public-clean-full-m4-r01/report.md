# Phase 2 clean MoPC 全量与支持集验证

用户选择 MoPC-HR 公开仓库行为。原生 MoPC 和六类域适应 B→MoPC 使用预登记 clean support/query；共 1932 个场景及模型种子组合，每组 7 项，总计 13524 项。三项已完成原生 pilot 的 K5/N20 与 A0 预测按绑定验证后复用。已有其他方法预测、历史卫星实验及故障隔离均保留，不重发。实际 B 模型按 K 保存；各方法只能声称共享实际同一个 B 状态，旧 pilot 未保存 B 不与新 B 认作相同状态。

公开仓库版本不使用梯度裁剪，保留非有限保护。已完成六个合成种子、三项实际保存 B 的支持集检查及三池 clean pilot；旧论文语义真实故障未确定根因，不得声称历史故障已根治。新增 ProtoNet、DADDA、Tweak 三个仅源/clean 支持检查，不访问 query 或 truth；检查通过方可派发新矩阵。Tweak 保持 M1 单条 query 判决。

N607 环境使用既有项目 Python 和 clean 数据配对；GPU 5、6、7 每卡最多 7 个本队列进程，启动前余量至少 6000 MiB。一个新 run、一个 owner；发生非有限或系统错误停止派发、保留在途与部分结果，不重启隔离项。输出分别置于 `preparations/20261114-phase2-mopc-public-clean-full-m4-r01` 和 `runs/20261114-phase2-mopc-public-clean-full-m4-r01`。查询预测固定后才可独立 truth-last 评分；未完成其他组和评分前不声称完整交付。

2026-11-14 21:56 CST：固定执行版本 `e93004dbbf30260428d52f7ba1a703ae2229b663`，发布包本地与远端 SHA256 一致，远端编译通过。ProtoNet、DADDA、Tweak 的真实源／clean 支持集诊断各完成 160 次 MoPC 更新，未读 query 或 truth。控制器 Python PID 3693936，独立读回 CWD 为本次 release；GPU 5、6、7 各最多 7 个进程，首批真实预测增长，首个 Poster→MoPC 完成 16 个 K/N 预测并保存四个实际 B 状态。该时点调度状态 RUNNING、failed=[]。无真值访问。

本队列启动配置未接自动评分：全局比较矩阵尚有 ISSL、其他方法及补齐 pilot 的任务未完成，预测屏障仍关闭。已有独立 truth-last 评分入口，但需要在所有保留组预测固定后建立并核验完整矩阵，不能将预测完成当成评分完成。

2026-11-15 更新：队列完成 432 项后，`0061-hybrid-protonet_cda` 的 K10/N20 第 3 阶段出现 loss、CE、层级正则及梯度元素非有限；控制器停止派发并排空。该任务 11 份 partial 及 B-K1/K5/K10 保留。公开仓库版故障同样真实发生，先前抽样通过不能视为修复。自动评分等待进程另见 `20261115-phase2-clean-auto-score-m4-r02`，屏障仍关闭。
