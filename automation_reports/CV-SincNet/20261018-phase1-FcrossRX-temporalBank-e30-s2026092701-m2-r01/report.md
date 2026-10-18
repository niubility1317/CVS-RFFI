# P67：同物理样本跨轮确认与完整 practical 的匹配 E30

状态：LOCAL_IMPLEMENTING，尚未启动，不存在新性能结果。

本轮唯一假设：原 U 时间确认从批内不同样本的邻近同预测，改为同一物理样本跨轮稳定预测，能否在保留完整 practical 监督时减轻 clean 代价，并提高 mid／low urban。动机是 P66 原 U 最终通过率约 1.5%；通过率不是准确率，也不是必须提升的代理目标。原教师已经直接读取 raw U，没有“弱教师增强可旁路”的事实。无新损失、门控条件、额外教师或前向。

## 去重与固定基线

起点 commit fc7f941b6721f34ad77ae063dfaeedd98422e8a5，分支 codex/phase1-urban-sigtime-20261014。PCGrad、fullAs KL-only-off、正负噪声配对、终端 EMA 等近似方向已有真实实验，不以换名重开。原 epoch_bank 不是新算法：20261008 reference-repair E200 和 20261011 无星地 Stage2-SSL 已使用相近记忆。未发现当前 F＋CROSS_RX、full5RX／六 L 前向、完整 As 的匹配实训。另一对话当前研究 L ordinary→raw，本轮不修改 L。

模型固定 SIG_TIME_X3／统一 Sinc／CENTER_UNITS／joint160／默认 cosine30，900722 参数、206 状态键。默认头与骨干正常联合训练，不冻结后换头，不另拟合诊断头，无新部署模块。仅 eval 的参数无训练均值或头替换。

完整源 L6300／U56700（TX 隐藏）／V27000，RX1/3/4/6/8、day1/2/3、6TX，split392005。没有 held RX、L_dev 或角色移动。对应 P43 B00/B01 scratch20 的 student／ownEMA 是各自唯一来源：已核实全源角色、无祖先、固定 final_only／4440 更新、无目标选模。新行严格各自加载再 reset AdamW／cos10／时间缓存；累计 E30／6660。继承来源不跨 arm。

## 唯一变化与匹配矩阵

| 条件 | As | 时间确认 | 来源及新增预算 |
|---|---|---|---|
| NOAS30 | 无 | 原 batch_neighbor | 复用 P43 B00 FINAL30，无重新训练或评分 |
| AS30 | 完整 | 原 batch_neighbor | 复用 P43 B01 FINAL30，无重新训练或评分 |
| NOAS_BANK | 无 | epoch_bank | 对应 B00 own20＋10 |
| AS_BANK | 完整 | epoch_bank | 对应 B01 own20＋10 |

唯一 native 参数差是 pseudo_temporal_mode=batch_neighbor→epoch_bank。原 min_streak2／min_conf.8 保持。物理键为 (rx_i,day_i,eq_i,sig_i,base_index)，同 epoch 重访不增加 streak，仅上一 epoch 相同预测延续；缓存从 E21 空状态开始，不继承未知伪标签。不是按 TX 真值构建 session 或 memory。原 rx_day_quantile .86、tau .92 至 .97、无 domain gate、strong agreement、raw ownEMA U 教师／强学生 std.015、损失及归一化保持。

保留六 L 前向 768／step、总 L 1.25、既有 RDrop 和 ordinary128 跨 RX 同 TX SupCon .1。全部 64 个 fullAs 标签 CE 保留；U 学生仍是原 strong 视图，没有困难 urban 高置信度前置门。As 训练只用 low suburban／mid urban／low urban 各 1/3；源硬件按 RX/day/eq 稳定，不由 TX 定义。每 epoch／step 的训练信道刷新，训练／源读出／正式流分开。

实际信道复用 P43：IQ2×256、25 MHz，裁剪后逐包 RMS／center=false 在 practical 前，网络内部 CENTER_UNITS；train fc2.462GHz。实际 __file__ 将再次保存，真后端为 pinned20261015_r02/experiments/adv3b02_xuc/code/leo_practical，而非顶层 code。完整 residual/post_sync、EQ=false 且 G=1、quality adaptation／phase tracking／IQ compensation／AGC 开启、reference_snr_db30、quality reference20、reflect、AGC±30dB 保持。新增噪声比不是未知总 SNR，白输入核功率质量代理不变。正式 fc2.45GHz 与全部固定 received IQ 不变，不弱化信道、不删困难包。

该四格是“时间确认方式×As”的交互，不把时间记忆称为 IQ 泛化增强 A_g。计算 AS_BANK−NOAS_BANK 与 AS30−NOAS30，以及差分之差；同时报告两种 bank 相对各自旧参照。

## 源域决定与放弃条件

冻结累计 E30 学生后，原完整 V27000 clean 与同一固定 V1800×2 draw×3 practical 缓存读出。四条件同时满足才允许另行进入总计 E50：

1. AS_BANK fullV clean−NOAS_BANK ≥−0.1 个绝对百分点；
2. AS_BANK fullV clean−原 NOAS30 ≥−0.1 个绝对百分点；
3. AS_BANK mid／low 均值−原 AS30 ≥＋0.5 个百分点；
4. AS_BANK low−原 AS30 ≥0。

任一失败，关闭固定模式，不调 min_streak、阈值、缓存热身、seed 或窗口救跑。源决定函数不加载任何正式分数；主 owner 先读并保存源决定，再读新正式分数。科学失败不取消已授权的完整测试。仅 1 个模型 seed，两个 draw 不作为复现。E30 预算耗尽不代表收敛。

## 执行、成本及交付

启动首步：两份真实 own20 checkpoint，各用 student 和 ownEMA 的 2 包零 IQ 检查，共 8 NN；PASS 后立即训练，不读源 IQ 或 query，不另建许可。局部检查验证原生 bank 的键、跨轮、同轮、置信度和协议负例，0 真实 IQ／CP／NN。一个限定 P0/P1 审查。唯一 owner codex/root/temporal-bank67，普通 N607 CVS-RFFI Python，FP32／CPU2；每 GPU 总进程最多 2（计其他 owner），空闲至少 12 GB，不干预健康任务。

每新行仅 E10／2220 更新，native 预计3162962 NN；L1704960、原 U567000，As 行142080个完整 As，NoAs 0。没有额外 U／教师前向。每行源 panel12600 NN，固定正式7×168000＋smoke2＝1176002 NN；两行合计预计8703136 NN，含共享8 NN smoke。本地真实 NN 0。历史两 E20 父成本、旧两个 E30 控制成本与新增尾段分别披露，不把共享祖先重复累计。实际计数器和墙钟为准。

训练完成逐 row 自动 clean＋全部六 practical，七视图预测全部固定后由独立 CPU scorer truth-last 评分。独立完整分析包括：逐 row／seed accuracy、macro-F1、最差 RX、RX/day/TX、错误修复／新增、默认头 margin/30、固定参照 clean 正确集合保持率、源 LOS/PARTIAL/BLOCKED、locked/degraded/unlocked、实际新增噪声区间、有效覆盖、完整2220步／10轮曲线、bank 状态／采用率，以及四格交互。采用率和梯度健康不能替代实际识别收益。

命令：

```text
python -u experiments/cvs_temporal_bank67/runner.py
```

环境／CWD：既有普通 N607 CVS-RFFI Python3.10.19／Torch2.1.0+cu121／NumPy2.2.5，PROJECT/releases/cvs_FcrossRX_temporal_bank_E30_20261018_r01。不可覆盖输出 PROJECT/runs/20261018-phase1-FcrossRX-temporalBank-e30-s2026092701-m2-r01；本地 local_artifacts/cvs_FcrossRX_temporal_bank_E30_20261018_r01。精确可执行路径及实际启动读回保留私有运行记录。实际代码提交写入 experiment.json 和 release_commit.txt，单归档 SHA 仅比较一次，远端 compile 一次。

停止仅限本 run 的错误角色／来源、非有限数、输出碰撞、模式未生效、错误 bank 语义、无法合法预测等系统缺陷；保留 partial、不自动重试、不按低分停训练。正常行即使另一行技术失败仍按固定计划收尾。源／正式失败、完整成本与缺项全部保留。整体低 urban>60%、clean 可重复正收益及 suburban 接近 clean 仍未完成。
