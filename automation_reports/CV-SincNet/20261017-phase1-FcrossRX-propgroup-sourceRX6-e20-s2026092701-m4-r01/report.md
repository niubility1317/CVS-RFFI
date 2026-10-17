# P20：F＋跨 RX 同 TX 的传播条件分组

状态：PREREGISTERED，尚未读取本轮真实 IQ、加载兼容 checkpoint 或训练。唯一模型种子 2026092701；目标测试访问 0。实际代码 commit 待填。

## 已有失败与本轮问题

P19 频带可用性方案已完整 E20，MASK 相对 RAW clean −1.111111 个百分点，low urban +0.079365 个百分点（1 个样本），固定方案关闭。旧 DC 交叉拟合诊断已失败：候选 DC 分量 p95=0.004160463488，低于已启用普通增强的边界 0.005012006679；本轮不重复该诊断。已有普通星地附加训练和 IQ 混合的 clean 收益也未跨 seed 重现。不能把单源 RX 或冻结模型响应作为泛化成功。

本轮唯一新假设：在不改物理边缘分布、不改网络和额外损失的条件下，按 step/RX/day/eq 共享传播状态，保留每包独立的噪声、相位、同步、跟踪及 RX 振荡残差，可能改善 practical 训练的身份证据学习，并保护 clean。分组是合成训练组织，不声称原录音同时采集，也不把联合依赖和有效独立环境数量的改变分开作因果结论。

## 固定配方与矩阵

增强前基线为 F＋跨 RX 同 TX SupCon：三路 SIG_TIME、统一 Sinc CENTER_UNITS、真实 joint160、原默认 cosine 头（scale=30）及正常联合训练。SupCon 保留原浅/中层 time_down、t2.pool，温度 0.2、权重 0.1、5 轮 warmup/10 轮 ramp。NoExplicitCFO 身份特征不变，原普通增强与内部补偿不删除。

|row|额外 IQ 对|场景与传播采样|
|---|---|---|
|RAW|raw|生成 IID 物理影子但不输入网络|
|IID|完整 practical(raw)|场景、传播逐包独立|
|SCENE_GROUP|完整 practical(raw)|组内场景相同，传播逐包独立|
|PROP_GROUP（主候选）|完整 practical(raw)|组内场景与冻结传播相同，包内接收随机流独立|

各组均从头 E20，原 13L＋7U、own EMA、旧门控、cosine LR、label_smoothing=0.01；lr=0.0002。共有原三对监督保持 .75 raw＋.25 ordinary，额外对权重 .25，总有标签目标权重 1.25。每 IQ 对的两次 RDrop 前向使用同一生成 IQ；每包共 8 次 L 前向。SupCon 只使用原 ordinary 两次前向。U 的 TX 真值始终隐藏，不新增 U/teacher 前向。不存在权重继承、新分类头、新部署参数或额外一致性损失。

四组各 4440 更新、4546560 L 前向样本；各生成 568320 次 native process、1136640 次 native 初始化，RAW 丢弃物理影子，其他组监督消费全部 568320 个合法 L 星地样本，无置信度筛选。组键不含 TX；硬件 session 为原 RX/day/eq，跨 epoch 稳定，训练实现按 epoch/step 刷新。

训练依旧使用原六场景均匀先验。通道 residual/post_sync、EQ=false（G=1）、reference_snr_db=30、reflect 边界，25 MHz、2×256；使用原 RMS 归一化（不去 DC）及原信道输入/输出归一化位置。保留 H→接收质量→补偿/噪声/AGC 依赖；不是弱化残余信道。共享属性仅 geometry/probs/state/z/ray_hz/ray_coef/f_doppler；总 CFO 重算为共享轨道 Doppler＋个人 RX CFO。SCENE_GROUP/PROP_GROUP 私有命名流相同，但依赖质量的实际噪声波形不保证相同。正式测试 IQ 与六种 practical 配置均不改变。

## 数据、执行和成本

复用 VALIDATED_ONCE split392005 原 carrier L6300/U56700/V27000。实际 fit RX1/3/4/8：L5040、U45360、V21600；源 RX6 原 L1260 从训练、U、V、校准和缓存全部排除，只在固定 FINAL20 后自动评价 clean＋六种原 practical。不是完整 L6300 或 L_train5580/L_dev720。无 target/query 构造。逐 row 全 7 视图预测后独立源域分析。

兼容烟测只用已核实 RX6 C0 scratch FINAL20/4440：/home/szu2070436088/2510044040/CV-SincNet/runs/20261016-phase1-FcrossRX-practical-source-confirm-s2026092701-m8-r01/C0-RX6-s2026092701/source。该 CP 无训练祖先、不接触目标；只做 2 个合成样本默认头前向，从不用于训练初始化或比较得分。其历史训练 E20/4440，历史 NN 数未知；本轮父初始化训练成本为 0。

本轮计划 80 model-epoch、17760 更新；训练＋源评价 NN 23293272，加兼容烟测 2，为 23293274。实际各类前向与墙钟按完整计数报告。已执行本地合成检查另记 4120 NN、4 优化器更新、模型检查 512 物理调用，物理单测 144 调用；无本轮真实 IQ/CP。

环境/CWD：ordinary N607 CVS-RFFI Python，/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_propgroup_sourceRX6_E20_20261017_r01。命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_propgroup_sourceRX6_E20_20261017_r01/experiments/cvs_environment_group20/runner.py
```

源与不可变依赖：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_additiveAs_RX6_source_E20_20261016_r01，commit 367b2f945ef65488e6d97eddd2357388d2477100；原 RX6 runner bootstrap 后运行新包。实际 launcher/leo_practical/native 模块 __file__ 随烟测/启动记录。输出独占 /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-propgroup-sourceRX6-e20-s2026092701-m4-r01；本地证据 E:/type10-7/local_artifacts/cvs_FcrossRX_propgroup_20261017。GPU 由已有共享资源管理按所有 owner 每 GPU≤3、空闲≥12000 MiB 安排，CPU2、loader0。先真实 CP no-query smoke，PASS 后立即四组训练，不设额外许可。

## 预先固定判据

只晋级 PROP_GROUP，FINAL20 固定，均为百分点。相对 RAW：clean≥0、clean macro-F1≥0、low urban≥+5、urban 三场景均值≥+3。相对 IID 和 SCENE_GROUP 分别：clean≥+1、clean macro-F1≥0、low urban≥+0.5、urban 均值≥+1。相对全部对照，每种 practical 和 seen-V 均不得低于 −0.5。各组完整 E20/4440、原默认头/角色/前向/As 覆盖，final common L objective≤0.5。

任何科学判据失败，关闭固定分组、随机流划分、剂量、RX、seed、预算和阈值；不从控制组挑候选，不补幸运 seed。全部通过也只支持一个源 RX6/模型种子的 As 组织，需另行预登记独立 seed；不宣称 Ag 有效、正式目标得分或联合目标达成。urban>60% 与 suburban 接近自身 clean 的全局目标仍未达成。

技术停止仅针对本 run 的协议/角色/来源/非有限值/输出冲突/预测非法等系统性失败，保存产物、无自动重试，不停止其他 owner 健康任务。源域涨跌本身不是提前停训理由。

## 已执行验证与交付

本地 physics：144 合成 native 调用、7 个负例，IID 与原实现、全部源域六场景输出一致；分组不含 TX、批次重排不变、全局 RNG 不污染、held RX6 不进入训练。model：4120 合成 NN、4 更新、512 物理调用，四组 206 个初始状态一致、900722 参数、0 BN，8 个输入轨迹与同 IQ 对正确，默认头/骨干梯度及联合更新正常，own EMA 不额外增强。protocol：31 个配置/判据负例＋原 RX6 6 个角色负例，U 真值读取 0。第一次本地模型入口导入了普通 code 路径，实际 NN/更新为 0；已修正本地入口先导入既有 overlay native，与生产 runner 保持一致。

一次独立 P0/P1 静态审查 PASS：传播/个人接收状态、RX6 角色排除、8L/4440 预算、own EMA、自动七视图源评价、容量及输出保护未发现直接 P0/P1；审查未读取真实 IQ/CP 或运行 NN。仅修正配置描述的 raw 份额与输入名称，不改执行。未新增重复审查、设计审批或哈希链。

预期产物：逐 row args/config/角色/初始与冻结记录、4440 步/20 轮完整训练曲线、L/U 序列、传播组/私有流/场景状态覆盖、clean＋六场景默认头预测与 margin/physics、固定 RAW clean 正确集合保持率、state/lock/质量/CFO/核/噪声/AGC/边界分层图、独立评分与收益归因、实际成本及全部负结果。实验矩阵和配置唯一登记于 experiment.json。
