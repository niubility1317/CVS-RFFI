# 源 Dropout 取点诊断：固定权重结果

真实源诊断支持继续单独验证描述子取点。冻结 SIG_TIME seed2026092701 权重、720 个固定 L 源包下，t2 末端 p=0.1 随机遮蔽回放的 feature-feature area 噪声／clean RMS 为 31.9940%，time area 为 21.7255%，协方差形状为 12.7249%。这是独立中间扰动的机制证据，尚未证明移动取点会提升识别性能。

| 描述子（四窗合计宽度） | clean RMS | 噪声 RMS | 噪声／clean | 8 回放均值偏移 RMS | 跨回放 RMS |
|---|---:|---:|---:|---:|---:|
| increment6（24） | 0.543464 | 0.135051 | 24.8501% | 0.048432 | 0.126068 |
| feature_area15（60） | 0.397484 | 0.127171 | 31.9940% | 0.046461 | 0.118380 |
| time_area6（24） | 0.316580 | 0.068778 | 21.7255% | 0.024598 | 0.064229 |
| SIG_TIME28（112） | 0.452902 | 0.116558 | 25.7359% | 0.042293 | 0.108614 |
| COV_SHAPE27（108） | 0.213123 | 0.027120 | 12.7249% | 0.010512 | 0.025000 |
| AREA_SHAPE27（108） | 0.361006 | 0.097323 | 26.9588% | 0.035738 | 0.090524 |
| added_residual160（160） | 0.107271 | 0.027129 | 25.2901% | 0.010368 | 0.025069 |

COV_SHAPE 与 AREA_SHAPE 在同一个 SIG_TIME 已训练 6D 投影上计算，未使用分别训练的 COV／AREA 权重。两个 RMS 的分母不同，不能由比值单独判断绝对噪声或迁移能力；完整分母与绝对量均保留。mask_mean_bias 是有限 8 次回放均值相对 clean 的偏移，并非无穷样本期望偏差。固定时间净增量常数 1 包含在 SIG_TIME28 RMS 分母，其本身不受扰动。

该观察说明：t2 输出虽在 eval 时与 t2.pool 完全相同，但训练时末端逐元素遮蔽会扰动非线性局部投影和二阶面积。下一步可只把描述子 hook 从 t2 移到 t2.pool，继续保持原时序主干 Dropout、描述子公式、输出维度、参数量和默认头训练不变；必须通过单独固定对照的实际训练检验。不会把整个时序主干改成无 Dropout，也不会与描述子替换或窗口层级堆叠。

## 执行与边界

run 20261014-phase1-source-dropout-descriptor-manysig-s01-m1-r01，代码 d34a7b91b91fb8672e622c3e39fb5142a1774dce，实际执行 b9144fee45b26342442e8a39bbe475bd19c03bd5；源 checkpoint 来源固定 20261014-phase1-moment-area-oldgate-cos-e50-clean-manysig-s01-m4-r01/SIG_TIME-s2026092701/source/final_ssdg.pth，执行512c10a267b5e654cfd2c983ab1debdeebc7f626。该来源是预定义最低种子的 SIG_TIME final50，依据报告路线三适用性固定，不按目标分数选权重。

L6300 中每个6TX×5RX×3day单元按 physical ID 排序前8，共720；同一物理划分、NoExplicitCFO及内部补偿保持。所有模型 eval/frozen，720源 feature forwards、8中间Bernoulli回放；0训练轮、0参数/缓冲变化、0优化器/分类头更新。8个mask样本不是8模型种子。

所有 eval 源包 pre/post t2 完全一致；实际 SIG_TIME 投影／tanh／四窗及 residual160 用真实权重计算。独立回放只注入这一处 Dropout；其他层/BN/参考响应保持 eval，不复现真实完整训练流、优化轨迹或 EMA 行为。不能以它解释本轮目标负结果的原因，也不能当作真实训练移位消融。

U IQ、U真值、目标、V forward均为0；复用源cache时V被materialize后discard。CPU2线程／CUDA_VISIBLE_DEVICES空，实际观测和重算 1.6405秒，不占GPU。模型状态及buffers逐值不变，源描述子保存后重读复算 VERIFIED，共63个overall/RX/day聚合记录；原始 observed_descriptors.pt 留远端。

一个archive比较、remotecompile、唯一直接P0/P1审查、真实 checkpoint两零IQ无查询smoke均PASS；PID2125054实际CWD/argv核实，最终completion ANALYZED、日志由0增长至206bytes、failure=null。当前无目标预测、性能提升、Phase2或新TX结论。

results.json 保留63分层记录；completion_evidence.json保留真实运行身份与零更新收尾。已完成源诊断，不重复访问源IQ或改变随机回放预算。
