# P70：固定 nominal 质量组风险，E30 最小匹配实验

本轮只检验一个假设：在相同有限 full practical 训练池、相同父模型和相同前向预算下，稳定每步实际质量组成并使用固定 nominal 组权重，能否改善 clean—urban 权衡。M3 困难加权、U practical 路由和公共传播单元不混入本轮；公共传播已在另一对话开展。

保留完整 practical residual、raw→practical 路径、F＋跨 RX 同 TX SupCon、SIG_TIME_X3/joint160/默认 cosine 头。P69 已完成且源两条件失败，不再延伸优化器状态。用户附件中连续 E30 技术失败的叙述是旧状态；授权恢复已经完成，应使用实际配对恢复结果。

逐行配置、父来源、六类 seed 和输出路径以 experiment.json 为唯一登记矩阵。两新行 M1=POOL_RANDOM、M2=QUALITY_STRAT 从同一个原 P43 As20 学生及自身 EMA 独立分叉，均重置 AdamW、使用原 cosine10 和 CENTER_UNITS。每行追加 E10/2220 更新，总 E30/6660；复用原 P43 NoAs30 和 As30 的冻结预测/评分，不重新训练或测试参照。

原 P43 As30 尾段 L 物理序列决定池生成。对 2220×64=142080 次出现逐一使用原 epoch/step/物理 ID 命名信道流，从合法原 IQ 生成完整 practical。共享只读池只生成一次，无 proposal、无样本删除；生成墙钟、缓存字节和实际调用单独报告。每条池记录携带 IQ、真实 L 标签、RX/day/domain、物理 ID、生成步和质量元数据。原 ordinary/raw/U loader 保持原流程，不能用原步标签监督重排后的 IQ。

四组按 signal_to_added_noise_db 划分：<3、3 至 <8、8 至 <15、≥15 dB。该量不是总 SNR。固定 p_g=N_g/142080，来源仅是生成于任何更新之前的有限 nominal 训练池，既不读取源 V 组占比，也不读取目标。所有 p_g>0 的组须 N_g≥2220（全池空组权重为 0，仅如实记录），否则保留池并在更新前技术停止，不复制、不丢弃、不静默改成另一目标。

M1：固定 seed 的全池随机一对一排列，每步64，原普通平均 CE。M2：各组每步分配 floor/ceil(N_g/2220)，每行总64、每组全程总数不变；组内随机一对一分配。As CE 为 .25×sum_g p_g mean_g CE；原同 IQ RDrop KL 仍 .25×逐包普通均值。ordinary/raw CE 与 KL、ORD128 SupCon、旧 U 资格与损失、教师前向和总模型前向数均不改变。

这不是只有重排的消融。M2 同时采用分层组织和固定 p_g 组均值估计。在固定参数、随机组内分配下，它对有限池风险无偏；floor/ceil 造成一次具体排列中个别观测权重略有差异。参数随更新变化后，不宣称条件梯度仍无偏；更不宣称有限池精确代表连续物理总体。每条原 epoch/view 生成的观测只使用一次，但全池顺序跨原 epoch 重排，这也是 M1 必须存在的原因。

As 总 CE .25、总 KL .25 保留，展开 L CE 仍 ordinary .75＋raw .25＋As .25，共1.25。实际 F 身份路径无 BatchNorm，保留已有 GN/LN；SupCon 仅 ordinary maps。两次 RDrop 对应同一个已生成 IQ，不新增跨信道 KL 或诊断头。

数据：L6300/U56700（TX 隐藏）/V27000，RX1/3/4/6/8、源 day1/2/3，全源训练，无 held RX/L_dev；split392005/原物理数据角色。输入 float32 B×2×256，fs25MHz；原逐包 RMS center=false 后生成 practical，网络沿原 CENTER_UNITS。训练 fc2.462GHz，固定正式 fc2.45GHz 的历史差异保留。完整 residual/post_sync/EQ=false（G=1）/reference_snr_db30/reflect、质量适配/相位跟踪/IQ补偿/AGC 均由原实际引擎决定，不弱化正式 IQ 或信道。

源推进只用两个预定比较：M2 全 V clean−原匹配 NoAs30≥−0.1 个百分点；两次固定 draw 的 pooled mid/low urban−M1≥+0.5 个百分点。状态/质量组、容易组、间隔、父 clean 正确集合保持率和梯度健康作解释，不增加替代门槛。源 FAIL 仍对两条固定行完成全部7正式视图；不依据正式分数决定重试、seed、超参数或晋级。M3 与 E50 不自动启动。

每行完成固定 FINAL30 冻结后立即进入 clean＋六种 practical 预测，全部7预测固定后由独立 CPU scorer 接真值。保留所有 row/seed/RX/day/TX 的准确率、macro-F1、最差 RX、修复/新错及曲线。单种子微涨不能称可重复双提升；low urban>60% 和最终 clean 正收益仍需实际证据。

成本：每条新尾段原生3162962 NN＋源12600 NN＋正式1176002 NN，共4351564；两行8703128，再加共享真实父学生/EMA烟测4，共8703132新NN。新增4440更新，CPU分析0NN。本地0真实模型前向/0IQ/0checkpoint。父训练成本与复用参照另列，不把共享父成本重复记为新训练。

执行：N607 既有普通非特权账户/既有 CVS-RFFI 环境，cwd=PROJECT/releases/cvs_FcrossRX_quality_strat_E30_20261018_r01；命令为既有 Python -u experiments/cvs_quality_risk70/runner.py。单 launch owner=codex/root/quality-risk70；每 GPU 最多2总进程（含他人），可用显存至少12000MB。根输出 PROJECT/runs/20261018-phase1-FcrossRX-qualityStrat-e30-s2026092701-m2-r01 和发布目录均不可覆盖。首次 smoke PASS 后继续，不新增审批或票据。

技术停止：错误数据角色/未知父/标签或domain错位/非有限数/输出冲突/原引擎或预算不符/稀疏质量组不可实现时，只停止本 run 对应失败行，保留全部产物；已健康行按计划收尾。无自动技术重试。保护其他健康任务。源科学失败关闭该固定候选，全部7测试仍完成。

状态：已登记，实现与必要检查进行中；尚无 P70 训练或性能结果。

发布前证据：实际训练代码提交 000a34f52dc4240c8d078254ada17e7761a27a84；35 项训练协议/池/接线检查、9 项 CPU 分析检查通过；唯一独立 P0/P1 PASS；真实 checkpoint 无查询 smoke 是 launcher 首步，PASS 后立即继续。一次归档传输 SHA 比较及远端23文件编译 PASS。未启动模型更新，未读取本轮正式成绩。


当前收尾：P70已ANALYZED；两新行与两个复用参照全部7环境齐全。源两条件FAIL，不扩E50。正式M2−同池M1 urban+.059722pp/low+.101190pp/clean+.071429pp；M2−原As urban−.022024pp/clean−.086310pp。对NoAs clean+.007143pp，本seed0.1pp阶段要求PASS，不能称跨seed双提升。详见[最终分析](final_analysis.md)。
