# Phase2 星地原生注册：修复版矩阵

三池、七接收机、四模型种子：MIX01–08 逐格 K1/5/10/20 × N2/5/10/20，LS/MU/LU 各五次 K5/N20。三种原生注册方法各 1932 任务；仅使用已固定星地 received support/query，C 从各自 A 源状态直接注册，不引入额外 B 或旧类目标标签。每任务预测固定并核对后独立自动 truth-last 评分。

固定版本：Orthogonal 源端伪标签修复版，标识 `orthogonal_tempered_pl_v2`；LoRa 等量源 gallery 与 `min(15,2K−1)`，C 标识 `lora_triplet_knn_balanced_v1`；作者行为 MoPC 配 FP64 全局范数和阈值 1000 裁剪，标识 `mopc_hr_clip1000_v1`。后者裁剪发生时会改变优化更新，不能冒称作者原始梯度完全等价。异常非有限 MoPC 单项隔离并保留 partial，有限梯度裁剪不会中断其他任务；源、输入、程序或资源系统错误停止新派发并自然排空。

LoRa 和 MoPC 旧星地 A0 的每个方法 2184 行已评分且预测存在，来源分别为原 Group1 源 r01 和作者 MoPC 源 r03；新 MoPC 使用继承同一四 seed 源状态的 r04。逐 capsule/seed/K 复用既有 A0；尚缺的 1764 条 A0 新算。修复版 Orthogonal 与旧源权重不同，全部 3948 条 A0 新算。旧 C 是不同方法版本，保留为历史，不冒充新 C。清点仅读元数据、固定预测文件存在和评分标记，不读取 truth。

run `20261117-phase2-satellite-native-c-m4-r01`，独立 preparations 和 release 同名。计划必须读取上述 A0 清点及两份原生 DA 队列计划。为不超每卡 7，本队列只在 DA r01 的 GPU worker 自然排空后使用 GPU 0/1/5/6/7；DA r02 使用不重叠的 GPU 2/3/4。未空闲时控制器等待，不干预其他 owner。启动前空闲显存至少 6000 MiB、每 worker 预留 3000 MiB；每任务首次使用真实源 checkpoint 进行无 query source-only smoke。运行命令、模型和数据根、预算及逐任务矩阵见 experiment.json 和 scheduler.json。
