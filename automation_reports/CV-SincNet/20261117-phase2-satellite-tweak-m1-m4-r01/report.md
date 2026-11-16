# Phase2 星地 Tweak M1 单条 query

三池、七接收机、四模型种子，MIX01–08 的 K1/5/10/20 与 LS/MU/LU 各五次 K5，原生 A0/B 共 1932 任务。MIX 的 N 用于统一描述任务，但原生 Tweak 不注册新类，A0/B 每 K 各一行，不重复计算不同 N。source 使用 `20261029-phase2-external-threepool-m4-r01` 对应 seed 的 Tweak M1 固定源模型；目标只用当前星地 received 旧类 support，query 逐条独立决策，不经 DURA、不读取 query 标签、不更新模型。

旧 `20261029-phase2-external-threepool-m4-r01` 有一个 Tweak M1 星地 pilot 及评分，旧 runtime 的 `batch_size=128`，没有记录逐条 query 调用。当前用户要求一次仅一条 query，不能把旧 pilot 冒充本接口复用；旧结果保留。新输出标识 `tweak_m1_single_v1`，评分前检查 `query_records_per_decision=1`、完整 A0/B 和星地输入。

新根 `runs/20261117-phase2-satellite-tweak-m1-m4-r01`，准备与 release 同名；代码为包含本报告的提交。队列先等待原生域适应 r02 的 GPU worker 排空，再使用 GPU2/3/4，每 GPU 本任务最多 7、开始前至少 6000 MiB 空闲，其他 owner 进程存在时避让；与 r01 和原生 C 队列的 GPU0/1/5/6/7 不交叉。每任务源 checkpoint 无 query smoke，预测固定并验证后自动逐任务独立 truth-last 评分。系统性技术/协议故障停止派发并排空，低分不停止。数据、四 seed、source 绑定和逐任务路径见 experiment.json 与 scheduler.json。
