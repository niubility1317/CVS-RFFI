# 实际四步端到端训练的星地IQ恢复器

**六景0/6，合并clean保持，clean准确率86.575000%。** cross_rx clean净变化+41，cross_both净变化+20。完整目标未达到，保持 ACTIVE 和默认 DURA。[完整性能与分组](analysis.md)。

- run_id：`20261101-phase1-Restore-RolloutBridge-E30-s2026092701-r01`
- group_id：`restoration-actual-rollout-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（完整七景与独立复算已核实）

## 目的与对照

相对StateBridge同结构与从零初值，改为实际四步可微训练和最终一步损失；检验源端逐步退化，自动原Main七景。训练计算不匹配。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

原 Main168000×七景完整，六景0/6、合并clean保持。84单元与完整分组/配对复算PASS，详见analysis.md。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 源端依据与实际四步训练

固定源轨迹诊断（128cc49，原 Ls128/Us384×20draw，81920 条轨迹/96 组，0 TX/V/OE/target 更新）表明，StateBridge 低档城区的 F(clean) 一致率从第1步86.350975%降至第4步81.615599%，IQ NMSE .07612306→.11645158；不是TX准确率、Main因果结论或理论极限。后续步骤已在源端退化。配对训练state=(1−t)clean+t received与部署网络自行生成state不同，是本项可证伪假设，不据Main错例挑选超参数。

新G维持StateBridge/c50的4676898参数和全部从零随机初值，原Ls6300/Us56700/R07/20+10draw/order2026110102/modelseed2026092701/AdamW.001-warm247-cos2e-5/WD1e-4/B256tail24/globalclip1/单阶段30-7410及四步部署公式不变。received初始化z；remaining4/3/2/1各做一次state/time/坐标估计，z_next=z+(predictedclean−z)/remaining，最后直接predictedclean。训练也真实展开这四步，所有状态保留梯度，不detach、不输入clean、无插值teacher forcing或随机time，无新增真实观测、噪声或跨query统计。

只在第4步输出计算一次原损失：IQ NMSE+.05STFT+.1(outputF0cos+源方差标准化MSE)+.1(lastlatentcos+同尺度MSE)+.1(raw/同条clean RMS−1)²。原clean及冻结F0特征/RMS仅作loss，最终损失贯穿四步；不增加早期步损失或F0输出指导调用。F0/3ff8/defaulthead/源方差f57不更新，方法明确F-guided；0 TX/logits/CE/六类原型/新身份头/V/OE/novel/目标训练选模，OE仅冻结诊断。G无旧G/物理父/IQ教师祖先，State仅冻结比较与实际E0相同性核对，不初始化学生。

9项相关检查（新4+State5）/一次独立P0P1 PASS；真实Ls2的actual四步训练=部署逐值相同、四步梯度/E0严格重载为launcher首步，PASS立即训练。技术快照E01..30不作模型选择，固定final30自动原地面完整Tri-SIG-TIME SSL/defaulthead/Main126000+42000各七景，全预测固定后独立CPU truth-last/原共享锁，raw/DURA按同IQ-ID精确复用；六景每景至少140345正确=83.538690476%，两个总体分别保持clean。所有版本完整测试，不因源质量取消，默认DURA不改，历史基准曝光和单seed限制披露。

部署仍4×260633600=1042534400实数Conv/Linear/attention MAC，0FFT/CG；同函数/同结构复用State静态计数，不是新实测时延，F0/norm/activation/FiLM逐点/softmax/投影/更新/内存排除。训练由1core/记录变为4core/记录（7560000行），仍1890000在线F0输出特征与63000clean特征参考；30轮不等于相同训练计算量或全部系统历史成本。GPU4/6 UUID/free≥12000MiB/原compute≤2/add1总≤3，只新增一个本任务worker，其他owner保持，实际墙钟/峰值显存另测。源2smoke后的全批显存和第一轮真实耗时读回，技术失败保留产物/不自动重试。精确环境/CWD/命令/路径/停止规则和预期产物见登记矩阵。


## 实际四步展开训练已运行（2026-11-01）

`20261101-phase1-Restore-RolloutBridge-E30-s2026092701-r01`，实际代码`7250d0a88160d6ab949dafbdc73edb7f302e290f`，4676898参数、与StateBridge/c50实际E0逐张量一致、全G从零单阶段30/7410。训练和部署均为当前received起点的四步state-only共享网络，clean只作最终loss参考，最终损失贯穿全部四步、不detach；0clean混合训练输入/时间抽样/旧G祖先，源F0连续特征loss-only，0TX/V/OE/目标更新或选模，OE仅冻结诊断。

VERIFIED/RUNNING：9重点检查/一次P0P1、真实Ls2四步训练部署精确一致/四步梯度/E0重载PASS。E01/247耗时203.122173秒，模型/优化器/CPU-CUDA RNG独立CPU读回及同State实际E0/原加权loss复算PASS。GPU4准入free20452MiB/原2compute，只新增1条；PID/CWD/CUDA/GPU和日志核实，其他owner保持。训练core数4倍、epoch/更新仍30/7410，不称同训练计算量。固定final30后自动原地面Tri-SIG-TIME SSL/defaulthead/Main168000×七景，全部预测后独立truth-last；性能PENDING，六景逐个3pp+两个总体clean主目标ACTIVE，效率支线，默认DURA不变。


## 固定30轮源训练完成，正式预测已接续（2026-11-02）

VERIFIED。30/7410完整源训练、189000L/1701000U、7560000 Gcore行、训练进程exit0并退出、final与E30文件存在已核实。全部训练状态/真实source2与实际E0的完整核对仍按既有证据及最终复算，当前不把源末轮指标当性能验收。实际Main预测进程已存活，跨接收机clean126000完成日志已观察；这不是准确率结果。全部两总体七景prediction完成后才独立CPU评分。SoftPower继续只在CPU等整个Main技术完成，原训练计划/默认DURA/源权限和完整性能目标ACTIVE保持。

## RolloutBridge完整七景完成

VERIFIED / ANALYZED。全G从零单阶段30/7410，部署固定4次共享权重，两个Main总体各七景全部预测后独立truth-last；4阶段exit0/liveempty。84单元/2604组/1736raw配对/1302额外配对复算PASS，误差1.2e-14、raw/DURA判决差0。完整60轮/stdout/源顺序draw/实际四步训练/从零E0/最终加权loss复算/30快照E30-final核实。完整目标未达到，保持 ACTIVE 和默认 DURA。
