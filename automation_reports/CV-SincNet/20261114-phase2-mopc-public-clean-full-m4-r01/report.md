# Phase 2 clean MoPC 全量与支持集验证

用户选择 MoPC-HR 公开仓库行为。原生 MoPC 和六类域适应 B→MoPC 使用预登记 clean support/query；共 1932 个场景及模型种子组合，每组 7 项，总计 13524 项。三项已完成原生 pilot 的 K5/N20 与 A0 预测按绑定验证后复用。已有其他方法预测、历史卫星实验及故障隔离均保留，不重发。实际 B 模型按 K 保存；各方法只能声称共享实际同一个 B 状态，旧 pilot 未保存 B 不与新 B 认作相同状态。

公开仓库版本不使用梯度裁剪，保留非有限保护。已完成六个合成种子、三项实际保存 B 的支持集检查及三池 clean pilot；旧论文语义真实故障未确定根因，不得声称历史故障已根治。新增 ProtoNet、DADDA、Tweak 三个仅源/clean 支持检查，不访问 query 或 truth；检查通过方可派发新矩阵。Tweak 保持 M1 单条 query 判决。

N607 环境使用既有项目 Python 和 clean 数据配对；GPU 5、6、7 每卡最多 7 个本队列进程，启动前余量至少 6000 MiB。一个新 run、一个 owner；发生非有限或系统错误停止派发、保留在途与部分结果，不重启隔离项。输出分别置于 `preparations/20261114-phase2-mopc-public-clean-full-m4-r01` 和 `runs/20261114-phase2-mopc-public-clean-full-m4-r01`。查询预测固定后才可独立 truth-last 评分；未完成其他组和评分前不声称完整交付。
