# F＋跨 RX：频率证据遮蔽 E20 源域反证终态

执行 VERIFIED；科学结论 **FAIL_FIXED_BAND_AVAILABILITY_CLOSED**。四个新 scratch E20 行和七视图全部完成，四个工作进程及 dispatcher 均自然退出。主候选 MASK 相对 RAW 的 held 源 clean −1.111111 个百分点、clean 宏 F1 −1.634608 个百分点，low urban 仅 +0.079365 个百分点（净增 1/1260）。未通过预登记的独立 Ag clean 条件，不推进第二训练 seed 或 Ag×As，不挑 GAIN/ORDINARY 控制替代主候选。

这是已多次用于研发的 held 源 RX1／L1260／modelseed2026092701，**不是正式目标测试或独立盲测**。全部正式目标访问为 0。源拟合域 guard PASS 与 seen V 约 98% 不代表跨 RX 身份提取有效；本轮没有证明新的 Ag、Ag＋As 联合收益或跨 seed 复现。活动总目标保持未完成。

## 固定起点与实际执行

起点 checkout 为 E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014，起点 HEAD 0c2989ac323f72584361cf777e5dc6dec9daac08。初版运行提交 5353a2490719d87c2fd2028d784c1bf051efcaa6 的 r01 在 guard 包导入时技术失败；只完成兼容 smoke（1 外部 CP、2 合成 NN），0 guard IQ/NN、0 新训练，原目录保留。修复后 r02 主运行提交 **a931a4d57aeebd836c4d973cae7b964034f490a8**；离线复核/图表工具 7d9c350b87f3e7238ce8e7d59f81868f5599f3ad、终态成本工具 95b90037f3ca4bdd7c16cc2b8390a867a13ffaca 均已提交并推送。

基线固定为 F＋跨 RX 同 TX：3×SIG_TIME、统一 Sinc CENTER_UNITS、真实 joint160、默认 cosine 6×160／scale30、NoExplicitCFO 表征；900722 参数／206 状态张量／0 BatchNorm。骨干与默认头正常联合更新，无新部署结构、诊断头、附加一致性损失或外部初始化。真实四组 initial_smoke 的 206 张量在全部 6 对中完全相同；独立比较只读取 4 个未训练初始文件，0 IQ/已训练 CP/NN。

数据为原 ManySig eq1 IQ256／25 MHz；native 每包 RMS 归一化在增强之前。数据集 eq1 与新 practical EQ 开关处于不同处理位置。固定 split392005；fit L5040＝RX3/4/6/8、6 TX、day1/2/3、72 格各70，U45360 TX 真值隐藏，seen V21600 只作源健康评价，held 源 RX1 L1260 只作固定 FINAL20 源读出。未将完整 L6300 与内部5580/720混用。新训练全为 scratch，旧 seed01/02 B00 同契约源权重仅用于冻结 guard／兼容，不作为学生或教师初始化。

四组共同保留原 13L＋7U、own EMA、旧门控、cosine LR、label_smoothing=.01、原 RDrop 和 ordinary-only 浅/中层跨 RX SupCon（warm5/ramp10，权重.1）；RDrop 不作为新 IQ 泛化增强。公共 .75 raw＋.25 ordinary 的绝对监督权重不变；额外 .25 同 IQ 的双前向 RD 对分别使用 RAW、匹配 RMS 的 GAIN、原已生成 ordinary 精确重放、MASK。总对权重1.25，每包8次L前向，所有组同预算且都生成相同 shadow；RAW 控制总 raw 权重实际为1。所有合法 L 均计算 CE，无学生置信度过滤；没有额外 U/教师视图、As 训练或未知 TX 硬件反演。

MASK 为 ±k/20（k 私有均匀1..6）对称 Gaussian 衰减，sigma=1/256、depth=.5；reflect128／FFT512／中心256，公共每包标量限 IQ 相对 RMS≤.1，不恢复 RMS。GAIN 精确匹配 MASK 输出功率。长反射参考仅在 guard 使用；不是训练视图。固定 STF 频率先验不识别或替换真实 RX，也不复活已关闭的33tap/DCT RX风格传递。NoExplicitCFO 指显式表征约束，原 ordinary 的 CFO 扰动基配置 p=.35 并未宣称关闭。原共享 ordinary 的 PA/AWGN/multipath 等沿既有强度日历保留，本轮未新增或独立放大。

实际导入：leo_practical.channel/residual、SSDG.train_ssdg、DataAugmentation 均来自原 immutable release 的 **experiments/adv3b02_xuc/code/**；完整 __file__ 在每行 actual_modules.json / actual_channel.json。六 practical 实际为 fs25MHz、residual、post_sync、EQ=false（G=1）、reference_snr_db=30、quality adaptation／IQ compensation／phase tracking=true、reflect。全部四组逐包物理状态与补偿元数据完全一致，未改正式 IQ、SNR、信道强度或删困难样本。新增噪声 SNR 与质量代理均不等同总 SNR。

## 七视图默认头结果

准确率单位为%；每格1260个同一物理样本，modelseed 均为2026092701，固定 FINAL20。

| 视图 | RAW | GAIN | ORDINARY | MASK | MASK−RAW（百分点） |
|---|---:|---:|---:|---:|---:|
| clean | 80.0794 | 80.0794 | 81.9841 | 78.9683 | -1.1111 |
| practical_high | 39.8413 | 39.5238 | 36.3492 | 41.8254 | 1.9841 |
| practical_mid | 39.6032 | 40.0794 | 37.3016 | 40.9524 | 1.3492 |
| practical_low_suburban | 36.0317 | 36.1905 | 34.4444 | 37.7778 | 1.7460 |
| practical_high_urban | 38.4921 | 38.8095 | 37.0635 | 40.0794 | 1.5873 |
| practical_mid_urban | 32.6190 | 32.3810 | 30.8730 | 33.0952 | 0.4762 |
| practical_low_urban | 28.0952 | 27.0635 | 28.2540 | 28.1746 | 0.0794 |

urban 三景均值：RAW 33.0688%；GAIN 32.7513%；ORDINARY 32.0635%；MASK 33.7831%。MASK 对 RAW/GAIN/ORDINARY 的 urban 均值分别 +0.714286/+1.031746/+1.719577 个百分点；但 clean 分别 −1.111111/−1.111111/−3.015873 个百分点，clean F1 条件均失败。seen V 差满足原保持条件，全部预算和数值健康条件 PASS。六 practical 不作为独立 Ag 的额外晋级门槛；本次失败是已预登记的 clean／F1 条件，未事后新增规则。

宏平均 F1（%）：

| 视图 | RAW | GAIN | ORDINARY | MASK |
|---|---:|---:|---:|---:|
| clean | 79.7518 | 79.7662 | 81.6475 | 78.1172 |
| practical_high | 29.1140 | 28.4612 | 27.8478 | 31.6365 |
| practical_mid | 29.4890 | 29.7386 | 29.7438 | 31.3046 |
| practical_low_suburban | 28.0472 | 28.0433 | 28.3938 | 30.6748 |
| practical_high_urban | 29.1817 | 29.3261 | 29.3168 | 30.9658 |
| practical_mid_urban | 27.2742 | 26.8253 | 26.7031 | 27.7972 |
| practical_low_urban | 24.7019 | 23.7362 | 25.2088 | 24.7660 |

MASK 的三 suburban 距自身 clean 分别 37.1429/38.0159/41.1905 个百分点，远未接近。相对 RAW 缩小差距的一部分来自 clean 下降，不能当作保持成功。只有一个 held 源 RX，无法报告正式最差目标 RX，也不能冒充三 seed 均值。

## 退化分层与配对归因

low urban 的同一状态/样本分层：

| 分层 | n | RAW（%） | MASK（%） | 差（百分点） |
|---|---:|---:|---:|---:|
| state: LOS | 384 | 42.1875 | 43.2292 | 1.0417 |
| lock: locked | 467 | 42.3983 | 43.0407 | 0.6424 |
| state: BLOCKED | 714 | 19.1877 | 18.9076 | -0.2801 |
| lock: degraded | 228 | 21.4912 | 21.0526 | -0.4386 |
| lock: unlocked | 565 | 18.9381 | 18.7611 | -0.1770 |
| state: PARTIAL | 162 | 33.9506 | 33.3333 | -0.6173 |

只在 LOS／locked 有小幅改善；BLOCKED、PARTIAL、degraded、unlocked 没有改善。quality<0 dB 仅1个净正确样本变化，其余低质量区间仍很差；完整 TX/day、质量代理、新增噪声SNR、CFO、核长、AGC和边界分层见 source_strata.json。分层为描述性源结果，不是独立机制因果证明。

MASK 对 RAW：clean 修复30／损坏44，low urban修复31／损坏30。固定 RAW clean 正确集合 n=1009 上，low 保持率28.245788%→28.146680%（−0.099108 个百分点）；MASK 的 clean 保持率95.639247%。因此 low 的单包净增并未保住相同 clean 身份证据。未保存逐包门控/全部分支响应，不据此归因于门控或某一骨干。默认头 cos30 间隔为真实运行保存值；准确率/F1由保存 hard prediction 独立重计，未用缺失全 logits 伪装独立 CE 重算。

冻结 fit-source guard 的两个旧模型虽通过（MASK accuracy −.143849/−.178571 个百分点、原正确保持均>99.8%），仍未预测 held RX clean 收益。该结果直接否定“拟合源身份保护足够即可推定 Ag 有效”的解释；并不否定所有选择性频率增强。

## 全量训练、覆盖与成本

独立复核全部35280 held-source判决、17760 joint/native/unit steps、23976 L/U序列记录、80 epoch 和全部stdout。四组完整 L/U顺序相同；私有 k 抽样逐步重新推导一致，存储的 mask/gain/ordinary MSE与全部变换统计逐项相同；统计相同不额外宣称逐点IQ重放证明。每行568320生成／5040唯一L／432 TX×RX×day×pair单元／30240物理L×pair全部覆盖。没有新增U视图。实际训练 IQ 相对RMS中位数 .100000、最大 .1000000048，97.5364%触发上限；功率匹配误差≤1.48983e−8。

全部活跃损失/梯度有限，0跳过更新、0零梯度。四组最终公共base目标 .310824/.310126/.311256/.311087，均≤.5；MASK 最后5轮 total L 为 .335295至.368468，seen V 为98.0370%至98.0556%。曲线仅表明本次预算内的训练轨迹，**预算耗尽不等于收敛**，不据中间分数改 checkpoint。每行 stdout 的完整单词 NaN 共200处，已全量定位为未启用的卫星/域/目标汇总栏位；简单子串计数260不等于260个训练NaN，活跃损失和更新无对应异常。

| row／seed | 更新 | 实际 NN | 训练耗时（s） | 启动到最终结果（s） |
|---|---:|---:|---:|---:|
| RAW-RX1-s2026092701 | 4440 | 5823318 | 1433.2822 | 1460.7487 |
| GAIN-RX1-s2026092701 | 4440 | 5823318 | 1484.3037 | 1511.9807 |
| ORDINARY-RX1-s2026092701 | 4440 | 5823318 | 1463.2180 | 1490.6587 |
| MASK-RX1-s2026092701 | 4440 | 5823318 | 1484.1722 | 1512.0797 |

新训练合计80模型epoch／17760更新；四行训练及源评价实际23293272 NN。各行计数扣除 own EMA deepcopy 继承的2个未执行 student smoke条目后，再加8820 held默认头前向。guard90720 NN、r02兼容smoke2及原失败r01兼容smoke2另列，合计本轮远端实际 **23383996 NN**。新学生父训练成本0；两个历史 B00 guard 父模型共40模型epoch／8880更新／9373356 NN另列，不冒充新增训练。当地合成模型测试4120 NN／4优化更新单列。

主dispatcher墙钟1589.4585 s（26.4910 min），工作进程到最终结果1460.7487至1512.0797 s。独立全量复核32.0246 s、0 IQ/CP/NN；全部四worker及dispatcher的原PID/start ticks已独立核实自然终止，其他 owner未改动。

## 已执行验证与命令

本地算子134合成IQ／134长DSP参考／5负例 PASS；真实固定CVS的四组合成验证4120 NN／4更新／4配置负例 PASS；科学/guard负例10 PASS；离线guard复核90,720预测及4负例 PASS；训练源评分/覆盖的12协议＋15科学负例 PASS。一次独立P0/P1静态审查和一次同启动边界定向复审已闭合；两seed完整隔离bootstrap零IQ/CP/NN验证PASS。每次实际发布先做真实匹配CP两合成包no-query smoke；r01失败目录保留，r02未改算法/随机流/预算/阈值。

原训练命令（CWD 为本run r02 release）：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u experiments/cvs_bandavailability20/runner.py
```

独立取回与复核命令（CWD 为当前工作树）：

```text
F:/App/miniconda3/python.exe experiments/cvs_bandavailability20/harvest.py --output E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final
F:/App/miniconda3/python.exe experiments/cvs_bandavailability20/analyze_training.py --input E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final
```

运行根：20261017-phase1-FcrossRX-bandavailability-sourceRX1-e20-s2026092701-m4-r02，单owner codex/root/FcrossRX-bandavailability-20261017。一次只读N607预检、一次本r02 archive SHA传输比较、remote compile以及启动后PID/CWD/cmdline/GPU/日志增长均 VERIFIED。134个固定产物已取回，不含IQ/cache/权重；152362766字节归档保留。Python tar提取弃用提示为非阻断日志，取回与完整文件数核验均PASS。

## 实际交付与边界

主代码/配置：[design.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/design.py)、[iq.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/iq.py)、[model.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/model.py)、[runner.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/runner.py)、[guard.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/guard.py)、[source.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/source.py)、[dispatch.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/dispatch.py)。独立复核：[analyze.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/analyze.py)、[analyze_training.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/analyze_training.py)、[checks_training_analysis.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/checks_training_analysis.py)、[harvest.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/harvest.py)、[cost.py](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_bandavailability20/cost.py)。完整配置保存在本登记 experiment.json 与每行 resolved_config.json／actual_native_args.json。

[全量独立结果](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/analysis_summary.json)、[所有分层](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/source_strata.json)、[80轮曲线数据](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/training_curves.json)、[实际成本及终态](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/actual_cost_and_terminal_processes.json)、[原始逐包guard复核](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/guard_independent_recount.json)。原始预测、物理ID及日志仅留私有产物，不入Git。

已生成并查看5组PNG＋PDF：

- [source_scene_results PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/source_scene_results.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/source_scene_results.pdf)
- [complete_training_curves PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/complete_training_curves.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/complete_training_curves.pdf)
- [low_urban_strata PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/low_urban_strata.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/low_urban_strata.pdf)
- [fixed_retention_and_paired_errors PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/fixed_retention_and_paired_errors.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/fixed_retention_and_paired_errors.pdf)
- [mask_dose_and_coverage PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/mask_dose_and_coverage.png)；[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_bandavailability_20261017/r02/final/figures/mask_dose_and_coverage.pdf)

另有先前2组guard图（身份保持/RX频率分层），不是新模型或target表现。固定频率遮蔽宽度/深度/中心/上限/权重/RX/seed/预算/阈值路线关闭；不通过改弱practical、筛样本、重跑幸运seed或选择控制补救。

本轮证明的是这个固定候选在同预算源域比较中没有独立 clean 收益；主要困难 low urban未缓解。没有取得 Ag＋As 因子交互、正式clean＋六practical目标成绩、跨RX最差表现或第二个新模型seed证据。两个旧guard seed不是两个新训练复现。low urban>60%、suburban接近clean、同一模型双收益及跨seed目标均仍未完成，不把源域反证收尾当成活动总目标完成。
