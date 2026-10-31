# Phase3 共享方差密度候选结果（2026-10-31）

**唯一主方法 `tied_predicted_ratio` 不晋级（NO_PROMOTION）。** 星地正确接收与未知拒对相对原 `predicted_ratio` 均净少 1 条，仍未满足拒识、已知保护和完整场景要求。保留原 `predicted_ratio` 重点候选及 `pooled_ratio` 冻结对照，完整目标 ACTIVE；本结论只针对此次固定候选。

Run：`20261031-phase3-tied-density-s2026092701-r01`；实际代码 `9a80e2bef7eee4d8c2ec4d4725e0ca552552443b`。固定设计、角色、信道和权限见[本轮计划](PAMF_PHASE3_TIED_DENSITY_PLAN_20261031.md)。

## 方法与数据

原 F0/default6、source-only r03 signal、DURA v1.0、白化、known90/OE32 中心及权重全部冻结。原密度 11 数组逐值保留；新评分把原逐模式异方差统一为既有 `pooled_L_variance`：`s=0.54650927979218`，没有重新估计方差。分量项为 `log(weight)+w·μ/s−||μ||²/(2s)`，known 与 OE 的 logsumexp 相减；共有 query 二次项、Gaussian 体积与白化 Jacobian 抵消，但不保证任意接收机偏移不变。

新 pooled 使用全部 90 个模式；新 predicted 仅按原 F0 预测类选 15 个 RX/day 模式、权重各 1/15，背景 32 模式权重保持。真实 TX、query RX/day、未知 K 不参与路由。旧两列 raw/margin/τ/身份精确复用，新两列各用 V27000 的 known12 规则拟合六阈值；所有模型先冻结、24 阈值全冻结后才处理 NV 与两个 Q，完整预测后独立 CPU 连接三份真值各一次。

本轮不读取源 L/OE IQ。祖先 L6300 为原训练池 10%、隐藏 U_s56700、OE6134；V27000 是额外已知校准监督，也复用于源诊断，非独立 known 盲测。NV3150 的 19-15/19-16/7-1 各 1050，源 RX1/3/4/6/8、三个源日期。跨 RX24-1/index26、2021-03-15 的 Q 为同 864 条物理记录：六旧类各 96（576）、三未知各 96（288）；clean 与星地均经过 DURA，同物理复用不增加独立样本。G10 MIX01 固定信道，不重抽。

## 完整运行点

数值均为百分比；损失为同 DURA 输入的原 F0 闭集准确率减 A_K（百分点）。A_K/R_U 为逐真实 TX 宏平均，H 为二者调和平均；C_K 是已知接收率。

| 条件 | 方法 | A_K | C_K | R_U | H | F0 损失 | AUROC / OSCR |
|---|---|---:|---:|---:|---:|---:|---:|
| source | pooled_ratio | 89.770370 | 97.107407 | 60.888889 | 72.561330 | 1.211111 | 94.985255 / 87.944738 |
| source | predicted_ratio | 89.744444 | 96.759259 | 61.936508 | 73.291437 | 1.237037 | 94.734250 / 87.937558 |
| source | tied_pooled_ratio | 89.614815 | 96.740741 | 63.269841 | 74.172455 | 1.366667 | 92.308542 / 85.534670 |
| source | tied_predicted_ratio | 89.625926 | 96.388889 | 63.301587 | 74.198073 | 1.355556 | 92.533934 / 85.968489 |
| clean_DURA | pooled_ratio | 68.923611 | 85.416667 | 99.652778 | 81.487441 | 11.111111 | 95.991874 / 77.463710 |
| clean_DURA | predicted_ratio | 70.833333 | 87.326389 | 100.000000 | 82.926829 | 9.201389 | 96.158251 / 77.770544 |
| clean_DURA | tied_pooled_ratio | 71.527778 | 88.541667 | 97.916667 | 82.667350 | 8.506944 | 96.245660 / 78.162375 |
| clean_DURA | tied_predicted_ratio | 72.048611 | 88.888889 | 98.611111 | 83.262688 | 7.986111 | 96.712240 / 78.414352 |
| star_DURA | pooled_ratio | 63.020833 | 84.895833 | 76.388889 | 69.063927 | 9.895833 | 84.728250 / 62.604890 |
| star_DURA | predicted_ratio | 64.062500 | 85.590278 | 77.083333 | 69.972325 | 8.854167 | 85.027850 / 62.982253 |
| star_DURA | tied_pooled_ratio | 63.541667 | 83.333333 | 73.958333 | 68.355429 | 9.375000 | 83.445457 / 62.693504 |
| star_DURA | tied_predicted_ratio | 63.888889 | 83.680556 | 76.736111 | 69.725652 | 9.027778 | 84.069975 / 63.047960 |

原 F0 同路径闭集准确率 source/clean/star 为 90.981481% / 80.034722% / 72.916667%，候选闭集预测逐条不变。主方法星地接收 482/576，其中正确 368、错误 114；clean 接收 512，其中正确 415、错误 97。接收率与身份正确率不可混用，本轮净正确变化来自拒识集合，而非默认身份头改善。

## 配对、分组与门槛

相对原 predicted，主方法 source 正确接收 90 增/122 失（净 −32）、未知拒对 254 增/211 失（净 +43）；clean 为正确 18/11（+7）、未知 0/4（−4）、总接收 +9；star 为正确 10/11（−1）、未知 18/19（−1）、总接收 −11。星地 ΔA_K −0.173611、ΔR_U −0.347222、ΔH −0.246673 个百分点；clean ΔA_K +1.215278 伴随 ΔR_U −1.388889 个百分点，不能据局部收益换主。

相对 pooled，主方法 source/clean/star 正确净 −39/+18/+5、未知净 +76/−3/+1；相对 tied pooled，正确净 +3/+3/+2、未知净 +1/+2/+8。三套完整 gain/loss/net 和所有 TX/RX/day/scene 分母保存在私有分析及独立评分，不选择较弱对照宣称成功。

主方法 source known12 全过，known42 未全过：30 个 RX×scene 诊断中 7 组超过原 F0 损失 2 个百分点（RX3 的 high/mid/mid urban/low urban，RX4 high urban，RX6/8 low urban）。既有校准下一候选在 low urban（10460 条 known）损失 2.002729 个百分点而被拒；只读该已存绑定证据，没有重新扫描阈值。

主方法未知逐 TX 拒对数（19-15、19-16、7-1）为：source 598/1050、607/1050、789/1050；clean 96/96、96/96、92/96；star 61/96、81/96、79/96。星地六旧类正确接收依序为 89、31、85、46、74、43（各 96）；逐 TX/RX/day 完整指标和原有置信区间见下述产物。

主方法星地六景 A_K / R_U 为：high 72.857143 / N/A，mid 69.722222 / 73.148148，low suburban 69.097542 / 84.384384，high urban 75.330688 / 93.333333，mid urban 71.276455 / 79.444444，low urban 51.880919 / 64.870072（%）。high 未知各类数为 0/4/2，缺 19-15，R_U/H 为 N/A，不删除缺类场景或补零认证。

最新 v3 要求 clean R_U≥95%、星地 R_U≥90%、C_K≥90%、同路径原 F0 损失≤2 个百分点及必需场景完整覆盖，且主星地 A_K/R_U 不低于原 predicted。主 clean R_U 通过，但 C_K88.888889% 与损失7.986111个百分点不通过；星地 C_K83.680556%、R_U76.736111%、损失9.027778个百分点及覆盖不通过。source R_U63.301587% 仅开发诊断；source known12 通过不能等同 known42 或目标认证。

## 成本、执行证据与限制

新增 NN 更新、源密度/normalizer/方差拟合、真实 F0/G/signal 前向、源 IQ、U 读取均为 0；新增 V 阈值 12 个标量，旧 12 个精确复用，总24个FP64阈值192B，单主六阈值48B。原密度聚合364340B、NN参数834665（F0 232993+signal334721+G266951）不变，无逐源行部署 bank。F0祖先11100、signal既有2220、G4940及源统计/缓存成本继承；实际三 checkpoint 各2条合成 smoke 另计，不把真实数据前向0写成无祖先成本。

流水线 ANALYZED_DEVELOPMENT，两阶段 COMPLETE/return0；实际 predict 16.150911s、CPU score 7.875751s、总流程 24.114294s，均为过程耗时而非 B1 包延迟。4次独立观测中捕获 driver1323759 与 predictor1323782 的完整 CWD/argv/CUDA4；GPU UUID95d925a1-7218-ca7b-adef-164241f4f491 三次离散样本10/406/406MiB，非峰值。predict日志487→629B；CPU1324080仅有pipeline gpuNone、代码CUDAempty及完成产物，没有独立live证据。终态live/ownedGPU空，foreign0。

最终验证10唯一测试项+34子项PASS（core3+10/0.64s，run-score5+17/6.23s，launch2+7/0.034s），一次P0/P1审查PASS；持久测试日志保留实现前RED及最终GREEN，512条旧np.trapz警告属于测试，三份完整生产stdout共7行、warning/error/traceback均0。实际三CP smoke PASS，688文件/data0的单发布包SHA `c83b07b943c16b0a7bf3bacffa1a65beccea6d6310c9d56e6ad46b2106366c72`比较、远端compile/runtime为VERIFIED。

源NV/Q均有历史开发曝光，单seed2026092701、单目标RX、目标日期与源重合；正式RX24-18及目标U未访问。本轮无目标DA、未知发现、匿名注册，A_N/H_OW为N/A，完整目标保持ACTIVE；不推广为所有共享方差方法无效。

复核产物：`local_artifacts/pamf_phase3_tied_density_20261031_r01/score/results.json`（12点、完整TX/RX/day/scene/42组与三配对）、`predict/calibration/complete.json`（固定阈值及下一候选约束）、`analysis.json`（紧凑全量统计、成本及日志）、`observations.jsonl`和`pipeline_state.json`（执行证据）。
