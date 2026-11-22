# LoRa Tri 星地分类监督配对实验

使用种子 2026092702 做两组配对筛查。三种子原 Tri 的硬伪标签使用率约 29% 至 31%，冻结后核验的正确率约 99.5%。此前的独立星地观测实验只让星地波形进入特征一致性和候选集合损失，硬标签交叉熵仍使用原始接收波形。本轮检验同包星地波形的硬标签交叉熵，并用增强真实标签星地监督的对照辨别改进是否依赖 U。

| 方法 | 唯一新增损失调整 |
| --- | --- |
| `tri_set_practical_hard` | 将现有冻结教师已选中的硬标签配给同包的固定 practical 观测。对去啁啾相干度中位数至少 0.40 的观测计算按整个 U 批大小归一的交叉熵，额外权重 0.35，沿用八轮 U 渐入。 |
| `tri_set_practical_labeled` | 将现有有标签 practical 交叉熵权重由 0.35 提高到 0.45。 |

两组从同一 LoRa 源契约的 E140 纯监督权重开始，续训 28 轮、每轮 88 步。保留原 Tri 的标签及源域策略、全部已有损失、增强、损失归一、源样本顺序及最终冻结规则。每轮记录各损失、U 硬标签及集合曝光，新增 U 分支单记 practical 曝光、物理索引及伪标签，以便冻结后独立评分。源验证记录 clean、六景及逐接收机准确率。训练读取原有同包源星地缓存与 TX 隐藏的 U 元数据。

正式测试覆盖相同 30000 条记录、15 台未见接收机及 clean 与六种 practical。两组预测全部冻结后，由本地独立评分端读取测试真值，逐包与相同 seed 的冻结 Tri-set、原自适应 Tri、继续监督及普通 PL 比较，并独立核验 U 硬标签、候选集合与新增 practical 子集。重点比较六景均值、低链路质量城区及 clean，单种子结果用于筛查。既有历史结果已公开，此次是相同基准上的开发评估。

本轮由当前 Codex 聊天统一发布与监控。N607 普通账户，优先 GPU 1、4、5，其次 GPU 0、6、2、3。最多两个本 run worker，每卡最多一个，派发时单卡须有 12 GiB 空闲显存且新增后总进程数最多四个。运行目录为 `CV-SincNet/releases/20261122-phase1-lora-tri-set-practical-ce-m2-r01-release`，独立输出位于 `CV-SincNet/runs/20261122-phase1-lora-tri-set-practical-ce-m2-r01`。

从本提交执行 `python -m experiments.pamf_lora_main.full_adaptive_publish --practical-ce`，传入既有全量数据、监督、E140 阶段、原候选集合的 release receipt 和本地 N607 配置。启动器先做真实 E140 checkpoint 的源域 practical 烟测，核对源合同、损失有限及预算，发现异常时保留产物并停止。预期产物为每种方法的 `epochs.jsonl`、`accepted_U.npy`、`accepted_sets.npy`，新增分支的 `accepted_practical_U.npy`，及冻结权重和 210 个预测数组。独立端执行 `python -m experiments.pamf_lora_main.full_adaptive_score`，传入 `--controls` 和 `--set-control`，生成 `scores.json` 和 `u_offline_scores.json`。
