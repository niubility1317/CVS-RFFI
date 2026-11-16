# Phase2 星地 ISSL 长预算

使用用户选择的 repaired-v1 ISSL 长预算：每个新类注册配置 2000 个 SSL epoch 加 300 个 transfer epoch，源模型和合法 replay 沿用对应 seed 的既有固定状态；此为项目修复版，不冒称作者原始代码完全等价。三池、七接收机、四模型种子；MIX01–08 覆盖 K1/5/10/20 × N2/5/10/20，LS/MU/LU 各五次为 K5/N20，共 12012 个单 K/N 注册任务。C 从 A 直接开始，只用当前新类星地 support，query 使用同一星地 received、只逐记录预测，不用于训练或选模。

每个 K/N 单独输出并在该任务预测固定、身份和路径核验后独立 truth-last 评分，不等待其他 K/N。每个 capsule/seed/K 的 A0 只在 MIX 的 N2 任务或单景 N20 任务产生一次；其他 N 不重复 A0。输出标识 `issl_fixed_v1_long_v1`。本轮未发现该固定长预算版本已评分的旧星地任务；旧 ISSL 行及已取消任务原样保留。

新运行根 `runs/20261117-phase2-satellite-issl-long-m4-r01`，preparations 和 release 同名。按用户“先跑其他方法”的优先级，控制器等待修复版原生 C 队列及 Tweak M1 队列 GPU worker 均排空后再派发；GPU0/1/5/6/7，每卡本队列最多 7，启动前至少 6000 MiB 空闲，避让其他 owner。待前序未结束时仅控制器等待、不做预测。每个 worker 首步真实 source checkpoint 无 query smoke；有限梯度不作为停机理由，系统技术/协议故障停止新增派发、排空保留。实际配方、行矩阵、命令及输出见 experiment.json 和 scheduler.json。
