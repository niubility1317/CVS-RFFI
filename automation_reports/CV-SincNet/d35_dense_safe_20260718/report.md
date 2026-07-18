# D35-DSWR稠密安全winner条件注册实验

## 登记

- 实验ID：`d35_dense_safe_20260718`；operator：Codex；日期：2026-07-18。
- 状态：`REMOTE_COMPLETE_NEGATIVE_NO_PROMOTION`。
- 假设：D34的主要失败来自稀疏可见性而非原型量化；把所有新类改为全winner可达，并用旧support最大残差阈值保护每个winner，可去除约`-2`的新类LOO截断，同时维持旧fit support不退化。
- 比较：Z0、D25-C0、B3、D33-FAST、D35-A/B/C；7候选×3场景×5个独立held-rank折=105行。
- 本轮K10-only；K1/K5只是接口边界，不是执行或性能证据。

## 机制与协议

完整公式和实现追踪见`analysis/d35_dense_safe_registration_traceability_20260718.md`。同一已接收LEO_weak IQ仅生成一行288维`[z160,FFT96,RF32]`拼接描述；不新增信道overlay、physical sample、support row或K。query完全关闭；clean/source样本及未授权衍生信号不可达；无角色Oracle、真实batch类数、quota、global assignment或dense query图。

D35的所有新类对每个旧winner始终有有限score；winner只索引一个由旧support构造的安全阈值。A使用单mean原型；B/C保存最多2个确定性原型，但按`(old winner,new class)`support证据只选择其中1个，因此每个query、每个新类仍只计算一次288D int8 dot；C对旧floor winner加倍不确定度buffer。所有新类原型为int8+FP32 scale/inverse norm；旧FAST score前缀不修改。这里保证的是全局可见，是否可达仍必须由physical LOO margin证明。

## 成功标准

- full-K fit旧support逐类/floor不退化；15个outer held折旧类new intrusion全部为0。
- 三场景所有新类physical LOO margin_min>0，重点检查09f8和f608。
- 联合指标达到B3与D33-FAST门槛，且不转移旧floor损失。
- 0 optimizer step、active<=50k、状态<=50kB；相对identity qKNN、B3、D33报告MAC/延迟/状态Pareto。
- 即使D35注册成功，FAST注册前旧类82.22%仍低于正式92%目标；不得把Stage2-C成功描述为最终路线完成。

## 执行计划

复用D34同一密封support与receiver/seed/scenario，不新增数据准备。完成core、runner、launcher和测试后先Git提交，再执行N607直接preflight、live inventory、最小同步、SHA闭合和唯一输出检查；计划GPU0，唯一输出`runs/d35_dense_safe_20260718/output/support_screen_v1`。完成后回填105行、逐类/场景矩阵、old intrusion、新类LOO、完整日志、资源审计、RECEIPT和Git提交。

## 本地实现与验证

- 路线锁提交`005819f0`，winner条件selector修订`28264f03`；core+单测提交`2112a855`；runner+launcher+集成测试提交`7b48b223`。
- D35 core SHA`6a96b6641d40930a867d4b99fb335575daf7a47262e37f7578210e5e25c62c0c`；共享runner SHA`063dfcf6ea0182af825b4e5850a0e01d20cff9b61497a3f6948edb9498cf9c13`；launcher SHA`f2546a62b3aa0f5c06a56a827b5e631160d57fb29905491e53be459f1341ac1b`。launcher中的runner/core锁与实算一致，无占位符。
- D35、D34、D33、Fisher和共享CLI相邻回归77/77通过；core/runner`py_compile`、launcher`bash -n`、`git diff --check`通过。
- candidate set精确7候选×3场景×5折=105行。positive route除总体old/new/H/forgetting/joint floor外，还必须逐旧类、逐新类达到B3和D33-FAST两者中较高的均值阈值；09f8、f608、14-7和其他floor不能被总体均值掩盖。
- runner将全局可见与LOO可达分开；full-K10门继续要求outer held旧侵入为0、全部新类LOO转正、fit旧类逐类/floor不退化、资源/协议闭合。`selected_positive_route`来自full-K10后的最终decision。

20新类K10合成扩展回归仅验证资源公式：A/B/C的D35增量均为5,760MAC/query，与FAST合计7,776MAC，相对identity-only K10单qKNN的41,600下降81.31%；组合状态A为14,644B，B/C为20,584B；部署refit约1.602M/3.101M MAC，0 optimizer step。合成old LOO仍出现33/34/33次侵入，证明硬门真实生效；该回归不是性能证据。

当前D35只把目标新类原型压为int8；Stage2-B仍使用FAST旧头，没有声称目标旧support int8原型路线已经闭合。若D35注册层通过，下一轮必须加入目标旧类int8原型与授权Phase1 int8锚的轻量融合，专门提升注册前旧类与floor；若D35注册层失败，则先修注册再叠加旧头，不能用未实现的旧int8原型作成功叙事。

## N607计划

- 最小同步仅3个文件：`code/cvsrffi/stage2_d35_dense_safe_registration.py`、`code/scripts/run_d25_support_only_concat.py`、`code/scripts/launch_d35_dense_safe_20260718.sh`；不上传本地有他人改动的`stage2_diag_cosine_exploration.py`，远端必须继续核验固定SHA`14ec9193...1ca`。
- 计划cwd`/home/szu2070436088/2510044040/CV-SincNet`；Python`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；命令`D35_GPU=0 bash code/scripts/launch_d35_dense_safe_20260718.sh`；日志`logs/d35_dense_safe_20260718/support_screen_v1.log`。

## N607启动前闭环

- 09:30 CST直接preflight与live inventory通过：8张RTX 3090均0%/10MiB，训练进程0；目标输出不存在；`/home`可用7.6TB；SSH/TCP22均已退出。
- 只同步core、runner、launcher到约定同路径。远端实算SHA与本地完全一致：runner`063dfcf6...f9c13`、core`6a96b664...62c0c`、launcher`f2546a62...ac1b`；未同步diag仍为固定`14ec9193...1ca`，D33/Fisher依赖也闭合。
- 远端launcher`bash -n`、core/runner`py_compile`通过，唯一输出仍不存在。GPU0允许新增本任务1个进程；启动后回填PID、运行状态和最终artifact。
- 09:32 CST启动成功：PID`3803270`、GPU0；31秒探针仍在运行，CPU约346%，日志0B、RECEIPT未生成；SSH/TCP22均已退出。

## N607完成证据

- N607任务在42.3419s内正常完成；105/105个`candidate × scene × held-rank`联合行齐全。完成标记为`DEVELOPMENT_SUPPORT_ONLY_COMPLETE`，`query_opened=false`，没有正式query指标或部署成功声明。
- 远端唯一输出：`runs/d35_dense_safe_20260718/output/support_screen_v1`；完整日志：`logs/d35_dense_safe_20260718/support_screen_v1.log`。本地回收目录：`E:\type10-7\automation_reports\CV-SincNet\d35_dense_safe_20260718\remote_output_v1`。
- RECEIPT登记的5个核心工件SHA-256均与本地回收文件一致；候选、场景、fold主键精确覆盖7×3×5，没有重复或缺行。
- 自动选择结论为`selected_candidate=D25-C0`、`selected_positive_route=false`。这只是安全fallback，不表示D25-C0达到项目目标。

## 联合结果

下表每行保持同一候选的注册前旧类、注册后旧类、新类、调和均值和遗忘量共同出现；所有数值均为3场景×5折的support-only开发均值。

| 候选 | 机制 | 注册前old(%) | 注册后old(%) | seen-new(%) | H_old_new(%) | forgetting(pp) | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| B3 | 较强旧域头+基准注册 | 86.67 | 73.33 | 73.33 | 72.65 | 13.33 | 当前联合比较器，仍远低于正式目标 |
| D33-FAST | Fisher对角快速旧域适应 | 82.22 | 70.00 | 59.33 | 62.19 | 12.22 | 旧域头不足 |
| D35-A | 全局可见单int8新类原型 | 82.22 | 53.33 | 59.33 | 54.13 | 28.89 | 否决 |
| D35-B | winner条件双原型选择 | 82.22 | 53.33 | 56.00 | 52.55 | 28.89 | 否决 |
| D35-C | B+floor不确定度buffer | 82.22 | 55.00 | 55.33 | 53.17 | 27.22 | 否决 |
| D25-C0 | 旧fallback | 50.56 | 50.56 | 54.00 | 50.35 | 0.00 | 仅自动安全fallback |

D35三臂均保持fit-old support不退化和旧score prefix逐bit一致，但outer held旧类侵入分别为52/52/49次；新类不可达class-fold分别为63/68/68。由此可见，fit support最大残差阈值并不能泛化到held旧样本，`global_visible=true`也不等于physical LOO可达。

## D35逐场景结果

| 臂 | 场景 | 注册后old(%) | seen-new(%) | H(%) | forgetting(pp) | held旧侵入 | 新类不可达class-fold |
|---|---|---:|---:|---:|---:|---:|---:|
| A | clear | 56.67 | 58.00 | 54.80 | 25.00 | 15 | 21 |
| A | low | 46.67 | 56.00 | 50.28 | 30.00 | 18 | 19 |
| A | rain | 56.67 | 64.00 | 57.31 | 31.67 | 19 | 23 |
| B | clear | 58.33 | 56.00 | 54.08 | 23.33 | 14 | 24 |
| B | low | 50.00 | 52.00 | 49.58 | 26.67 | 16 | 22 |
| B | rain | 51.67 | 60.00 | 53.99 | 36.67 | 22 | 22 |
| C | clear | 58.33 | 56.00 | 54.08 | 23.33 | 14 | 24 |
| C | low | 50.00 | 52.00 | 49.58 | 26.67 | 16 | 22 |
| C | rain | 56.67 | 58.00 | 55.87 | 31.67 | 19 | 22 |

## D35-C逐类结果

| 类别角色 | TX/class handle | 准确率(%) | 观察 |
|---|---|---:|---|
| old | 20-15/`1f33` | 56.67 | floor失败 |
| old | 8-20/`33bb` | 70.00 | 仍低于88%下限 |
| old | 14-10/`75aa` | 46.67 | 严重floor失败 |
| old | 14-7/`8b02` | 43.33 | 最弱旧类之一 |
| old | 6-15/`a53` | 63.33 | floor失败 |
| old | 20-19/`f8df` | 50.00 | floor失败 |
| new | `09f8` | 33.33 | 仍是首要弱新类 |
| new | `1c2a` | 43.33 | 注册不足 |
| new | `b8fb` | 40.00 | 注册不足 |
| new | `d3af` | 83.33 | 相对可用但未达92% |
| new | `f608` | 76.67 | 较D34改善但仍未达标 |

full-K10下所有新类虽全局有限可见，但clear场景5/5类均不可达，low和rain均有4/5类不可达。旧类LOO侵入A为12/11/12、B为14/10/14、C为11/8/12(clear/low/rain)。因此不能通过增加buffer或再调单一阈值解决。

## 资源审计

| 臂 | 组合状态(B) | 部署refit MAC | 开发LOO MAC | query MAC | head均值延迟(ms) | head p95(ms) | support fit(ms) | optimizer/活动参数 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 9,709 | 1.128M | 19.063M | 3,456 | 0.108–0.115 | 0.123–0.129 | 560–604 | 0/0 |
| B | 11,194 | 1.287M | 32.008M | 3,456 | 0.108–0.115 | 0.123–0.129 | 723–728 | 0/0 |
| C | 11,194 | 1.287M | 32.008M | 3,456 | 0.108–0.115 | 0.123–0.129 | 723–728 | 0/0 |

5新类query MAC为FAST旧头2,016加5×288的新类dot，共3,456，与B3的dot-MAC相同；20新类合成为7,776MAC，相对K10 identity-only单qKNN的41,600下降81.31%。D35在适配计算和状态上比B3轻，但性能显著恶化；相对D33又增加状态和延迟，所以不构成性能—资源Pareto改进。

## 三轮强制复盘：D33–D35

本复盘在启动D36前完成。重新核对active objective与`项目.md`后，下一轮继续同时考核注册前旧域适应、注册后旧类、新类、H、逐旧类floor与遗忘；query仍是测试集，开发阶段不得打开。每个support/query仍只对应一个预先叠加的LEO_weak观测；`z160+FFT96+RF32`只是同一观测的确定性拼接描述，不增加physical view或K。预测器继续逐样本面向全部注册类，无query角色、真实batch类数、类别quota或global assignment。

三轮得到的机制性结论如下：

1. D33的对称注册改写了旧类决策面，注册后old仅70.00%，new仅59.33%；同时FAST注册前old仅82.22%，说明Stage2-B本身就是硬瓶颈。
2. D34用稀疏collision edge保护fit support，却让非edge新类分数落到约`-2`，造成66/69/68个class-fold不可达；outer held旧侵入仍没有消失。
3. D35取消稀疏可见性后，63/68/68个class-fold仍不可达，同时产生52/52/49次outer held旧侵入。稀疏二值门和稠密最大残差门是两个失败极端。
4. 下一轮不能继续微调hard visibility或单一max-residual threshold，而应使用support-only连续新旧margin校准器；其输入仅是可部署score几何，如`max_new-max_old`、old/new top1-top2 margin、半径归一化距离和不确定度，输出对所有新类共同的有限offset。所有新类对每个query仍保留有限score。
5. D36必须同时升级旧域头：优先从注册前old为86.67%的B3或其压缩近似出发，加入目标旧类int8原型、目标新类int8原型和可选只读密封Phase1 int8锚；用support jackknife选择不确定度融合权重，并对逐旧类floor施加硬约束。
6. D36开发门应同时要求：注册前old优于B3；注册后old/new/H均不弱于B3；outer held旧侵入显著下降；09f8、f608和14-7等floor类不能用总体均值掩盖。若支持筛选仍为负，不打开query、不扩张正式矩阵。

## 最终判定与下一步

D35是完成且可复现的负结果，不能promotion。根因已从“新类是否可见”收敛到“支持集上的新旧score分布校准不能泛化到held样本”，并且旧域头本身低于目标。D36将采用连续support-only margin calibrator与更强的旧域适应头共同优化；目标旧/新support原型均int8，授权地面int8锚只读，不使用raw/full-precision source信号。任何D36候选都必须先通过同一outer support held门，再进入K=1/5/10/20和多receiver/seed扩展。
