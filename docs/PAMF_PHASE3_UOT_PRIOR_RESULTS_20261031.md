# Phase3 UOT 混合先验适应结果（2026-10-31）

**唯一主方法 `UOT_predicted_ratio` 不晋级（NO_PROMOTION）。** 相对原 `predicted_ratio`，星地已知正确接收完全不变，仅多拒对 1 条未知样本；clean 少正确接收 1 条。严格已知收益、绝对保护、拒识与完整场景条件未全过，保留原 predicted 重点候选与 pooled 冻结对照，完整目标 ACTIVE。

Run：`20261031-phase3-uot-prior-s2026092701-r01`；实际代码 `c0b7cc09a808e832d4d64a18080239e72e910250`。角色、信道和固定设计见[本轮计划](PAMF_PHASE3_UOT_PRIOR_PLAN_20261031.md)。

## 方法、权限与数值诊断

本轮为 UOT 启发的混合先验变体，并非[UniOT 完整复现](https://papers.neurips.cc/paper_files/paper/2022/hash/bda6843dbbca0b09b8769122e0928fad-Abstract-Conference.html)；采用计划列出的[非平衡传输对数域缩放形式](https://arxiv.org/abs/1607.05816)，不继承完整表征学习、未知发现或注册结论。原 F0/default6、source-only r03 signal、DURA、known90/OE32 的均值、方差及白化全部冻结。

仅适配前星地 U384 的原 r03 单位特征进入拟合：`C[j,i]=||w_i−μ_j||²/(160*v_j)`，源参考质量固定 known/OE 各 0.5、目标各 1/384；epsilon=0.05、rho_source=rho_target=1，完整执行唯一 200 次广义 Sinkhorn。已知 90 行质量跨六类汇为 15 个域权重，OE 汇为 32 个权重，各与原先验按 0.5/0.5 混合。47 个 FP64 数值、两个单纯形自由度 45，不改变几何、归一化或阈值；默认身份始终是原 F0 argmax，不用真实 TX/K/配额路由。

全部迭代 1 至 200 记录已读，数值有限；末 log-scaling 变化 2.2930457533e−9，KKT 驻点残差 2.1826436214e−9，log 总传输质量 −0.7433179638。known transport fraction 0.6356480368 是模型传输统计，不是真实已知比例；数值收敛不能代替未来识别收益。两组权重、源状态和旧阈值先冻结，再生成 source/clean/star 全部四列预测，独立 CPU 最后连接三份真值各一次，U truth 读取 0。

已有只读容量诊断显示，域权重/原均匀权重为 0.949452098 至 1.039400715，OE 权重/原权重为 0.515079559 至 2.366434069；由正混合比界，任意输入的新旧分数差处于 [−0.913254410, 0.702078218]。这说明已知域先验变化有限，不等于已经修正特征域偏移，也不能推断更大变化必然有效。

## 完整运行点

下表均为百分比；损失是同 DURA 输入的原 F0 闭集准确率减 A_K（百分点）。A_K/R_U 按真实 TX 宏平均，H 为二者调和平均，C_K 为已知接收率。

| 条件 | 方法 | A_K | C_K | R_U | H | F0 损失 | AUROC / OSCR |
|---|---|---:|---:|---:|---:|---:|---:|
| source | pooled_ratio | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 / 87.944738 |
| source | predicted_ratio | 89.744444 | 96.759259 | 61.936508 | 73.291437 | 1.237037 | 94.734250 / 87.937558 |
| source | UOT_pooled_ratio | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 / 87.944738 |
| source | UOT_predicted_ratio | 89.744444 | 96.759259 | 61.936508 | 73.291437 | 1.237037 | 94.734250 / 87.937558 |
| clean_DURA | pooled_ratio | 68.923611 | 85.416667 | 99.652778 | 81.487441 | 11.111111 | 95.991874 / 77.463710 |
| clean_DURA | predicted_ratio | 70.833333 | 87.326389 | 100.000000 | 82.926829 | 9.201389 | 96.158251 / 77.770544 |
| clean_DURA | UOT_pooled_ratio | 68.750000 | 85.243056 | 99.652778 | 81.365979 | 11.284722 | 95.969570 / 77.444420 |
| clean_DURA | UOT_predicted_ratio | 70.659722 | 87.152778 | 100.000000 | 82.807731 | 9.375000 | 96.144989 / 77.762105 |
| star_DURA | pooled_ratio | 63.020833 | 84.895833 | 76.388889 | 69.063927 | 9.895833 | 84.728250 / 62.604890 |
| star_DURA | predicted_ratio | 64.062500 | 85.590278 | 77.083333 | 69.972325 | 8.854167 | 85.027850 / 62.982253 |
| star_DURA | UOT_pooled_ratio | 63.020833 | 84.895833 | 76.388889 | 69.063927 | 9.895833 | 84.722825 / 62.603082 |
| star_DURA | UOT_predicted_ratio | 64.062500 | 85.590278 | 77.430556 | 70.115031 | 8.854167 | 85.030864 / 62.988281 |

同路径原 F0 闭集 source/clean/star 为 90.981481% / 80.034722% / 72.916667%，默认 logits/closed 不变。主 clean 接收 502/576，其中正确 407、错误 95；主 star 接收 493，其中正确 369、错误 124。已知接收率不等于身份正确率，未知拒对增加也不代表旧类身份改善。

## 配对、细分与判定

主对 selected predicted：source 正确/接收/未知拒对所有 gain/loss/net 为 0；clean 正确 0 增/1 失（净 −1），总接收 −1、未知不变；star known 所有事件不变，未知 1 增/0 失。星地 ΔA_K=0、ΔR_U=+0.347222、ΔH=+0.142706 个百分点；clean ΔA_K=−0.173611、ΔH=−0.119098 个百分点。不能将单条样本的 H 微增当作晋级。

主对 pooled：source/clean/star 正确净 −7/+10/+6、未知拒对净 +33/+1/+3；主对 UOT pooled：正确净 −7/+11/+6、未知净 +33/+1/+3。三套完整配对及逐 TX/RX/day/scene 分母保存在分析产物，不以较弱控制的局部差替换唯一主或所选基线。源五 RX 使用原先验，源四列重复旧两列，并非适应在源域也产生普遍收益。

主 source known12 通过，known42 未全过；30 个 RX×scene 诊断中 5 组原 F0 损失超 2 个百分点：RX3 high、mid urban、low urban，RX4 high urban，RX8 low urban。V27000 的额外已知校准监督及其阈值均来自祖先，本轮新 V 校准 0，不能将源诊断视作独立 known 盲测。

未知 19-15/19-16/7-1 的主拒对分别为 source 688/1050、512/1050、751/1050；clean 三类均 96/96；star 75/96、72/96、76/96。星地六旧类正确接收依序为 89、32、78、47、76、47（各 96）。全部 12 点的九 TX、源五 RX/三日与目标单 RX/单日细分、置信区间均保留于独立评分及私有分析。

主星地六景 A_K / R_U：high 72.857143 / N/A，mid 69.722222 / 94.444444，low suburban 66.240399 / 89.386261，high urban 75.330688 / 86.666667，mid urban 75.509259 / 75.555556，low urban 54.240154 / 57.858423（%）。high 未知三类数 0/4/2，缺 19-15，宏 R_U/H 为 N/A，不删除或补零认证。

最新 v3 要求 source known12、clean R_U≥95%、星地 R_U≥90%、C_K≥90%、同 G 原 F0 损失≤2 个百分点、必需场景完整覆盖，并要求主星地 A_K 严格增加且 R_U 不下降。主 clean R_U100% 但 C_K87.152778%/损失9.375个百分点失败；star C_K85.590278%/R_U77.430556%/损失8.854167个百分点及覆盖失败，严格 A_K gain=False。源未知指标仅诊断，不能取消目标测试或替代其门槛。

## 成本、执行与证据边界

新增目标先验拟合 1 次/200 步/47 个数值（376B）；连 schema 的三数组合计 504B，UOT.npz 文件 1298B。源 GMM/normalizer/阈值及 NN 更新均 0，真实 F0/G/signal 前向、源 L/OE IQ 读取均 0；每个真实 checkpoint 各 2 条合成 smoke 另计。原神经网络 source-only，但整套 U 派生适配状态 source_only=False、target_DA=True、REG=False，不能泛称无统计学习。

原 density11 聚合 364340B、部署 NN834665 参数不变；source 唯一 12 个 τ96B，四列重复 24 值192B，单主六 τ48B。F0 E50/11100、signal 既有2220（复制特征累计13320/MLP2220）、G E20/4940及原源统计/缓存成本继承；无逐 U/source 行 bank 或 coupling 部署。

流水线 ANALYZED_DEVELOPMENT，predict/score 均 COMPLETE/return0；实际 predict 24.286171s、CPU score 5.166273s、总流程 29.559577s，非 B1 包延迟。4 次观测独立捕获 driver1341003/predictor1341026 的 CWD/fullargv/CUDA1，GPU UUIDa2d039ff-1792-4222-2b47-140429e10f94 两次406MiB离散样本（非峰值），日志472→497→599B；CPU1341974只有pipeline gpuNone、代码CUDAempty及完成产物，未独立live捕获。终态live/ownedGPU空，foreign0。

最终11唯一项+33子项PASS（core4+10/0.92s、run-score5+16/8.52s、launch2+7/0.044s）；worker AST6/launcher import、唯一P0/P1审查PASS。持久测试日志保留实现前RED与最终GREEN，536条旧np.trapz警告仅在测试；三份完整生产stdout共7行，warning/error/traceback均0。三CP smoke PASS，697文件/data0发布包SHA `7a9a68324a69177757eedb8f8470982deb65b140cbf5e0d45d068b23656f956a`比较、remote compile/runtime VERIFIED。

数据承接 L6300/隐藏U_s56700/OE6134、V27000/NV3150、源五RX三日；star U384为RX24-1/2021-03-08，与源及未来Q物理互斥。未来同864条Q为03-15、old576/nonold288、九TX各96；三个测试全DURA、clean仅测试不学习、G10 MIX01固定单记录信道。NV/Q历史曝光、单seed2026092701/单目标RX、源日期重合；正式RX24-18未读。本轮无未知发现或匿名REG，A_N/H_OW为N/A，不构成正式盲测或完整开放世界成功。

复核产物：`local_artifacts/pamf_phase3_uot_prior_20261031_r01/score/results.json`（12点、全分组与三配对），`predict/adaptation/fit_diagnostics.json`（全部200记录），`prior_effect_bound.json`（既有权重变化界），`analysis.json`（全量紧凑统计/字节/日志），以及`observations.jsonl`、`pipeline_state.json`。
