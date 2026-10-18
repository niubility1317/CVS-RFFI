# P77：物理新增噪声与 As 目标熵的匹配 E30 实验结论

状态：ANALYZED；两条新增训练行、各自 clean＋六种 practical、独立 truth-last 评分和完整分析均已完成，技术缺项为空。模型种子只有 2026092701。实际代码提交：cf49ff220dc926bac704f15b7f34a79d1ef60c27。

本轮没有找到优于现有完整 practical 增强的方案。物理对应 Q 比随机置换 R 的正式 urban 均值高 0.097619 个百分点，但比同父原增强 C00 低 0.066667 个百分点，六种 practical 均低于 C00。Q 的 clean 相对 C00 仅高 0.007738 个百分点。不能把这个微小单种子差异称为 clean 收益。

相对同种子、同总 E30 的 NoAs30 整配方参照，Q 的 urban 均值高 28.281944 个百分点、low urban 高 21.045238 个百分点；clean 低 0.039881 个百分点。单种子观察满足 clean 下降不超过 0.1 个百分点的阶段保持线，但大幅星地收益来自已有 As 配方，新增机制没有提供正增量。low urban 为 51.185119%，未达到 60%；没有可重复的 clean 与星地双提升证据。

## 比较与实际实现

唯一假设是：在固定 As 总系数和前向数内，将标签平滑系数按实际新增噪声比例对应到样本，能否优于随机对应并改善低质量 urban。固定全局 AsLS=.05 的旧失败不重试；本轮只检验物理对应关系。

Q 使用 ε=.01+.19/(1+10^(s/10))，其中 s 是 native signal_to_added_noise_db，范围 [.01,.20]；R 在同批 As64 内置换完全相同的 ε 多重集，使用独立命名随机流，两次既有 As Dropout 前向复用同一 assigned ε。该比值使用实际信号及新增噪声的 pre-AGC 功率，不是原录音总 SNR，也不是白输入接收质量代理。合法零功率的 None 边界依据实测新增噪声功率解释并保留，不删除样本。

两行严格加载同一个 SAT18 AS_CE20 student 与自身 EMA，从 E20 独立追加 E10，总 E30、6660 次更新；尾段重置 AdamW（lr=.0002、wd=.0001）和 cos10，使用原 CENTER_UNITS。C00 复用 P76 已完成的同父、同物理随机流、As ε=.01 控制；NoAs30 与原 As30 复用 P43 同种子总 E30 整配方结果。后二者父训练历史不同，不能称为同 AS_CE20 父的 NoAs 尾段对照。

F＋跨 RX 同 TX 对比、SIG_TIME_X3、统一 Sinc、joint160、默认 cosine30 分类头及骨干联合更新保持。每步 6L/768，As64 只原生生成一次，第二 As64 复制 IQ，两个 CE 各 .125，总 .25；L CE 总系数 1.25。ordinary/raw CE 与 KL、ORD128 SupCon=.1、原 U batch_neighbor 与自身 EMA 保持。实际 total L 用新 As CE 替换旧 As CE，没有新增一致性损失、诊断头或部署模块。Q−R 控制 ε 完整分布，不控制真实参数梯度范数；Q−C00 还包含平均平滑分布变化。

数据为 L_s=6300、隐藏 TX 真值的 U_s=56700、V=27000；源 RX 1/3/4/6/8、day 1/2/3、六 TX 全部使用，无训练 RX 留出，split=392005。IQ 为 2×256、25 MHz；原 RMS center=false，normalize_input 后执行完整 residual/post_sync，EQ 关闭，因此 G=1；reference_snr_db=30、reflect 边界，接收质量适配、相位跟踪、IQ 补偿和 AGC 保持。实际导入为固定 r02 的 experiments/adv3b02_xuc/code/leo_practical，运行记录保存 __file__。训练中心频率 2.462 GHz，正式固定 received IQ 与 2.45 GHz、六种 practical 配置保持。

2220 个尾段更新的原 ε、物理元数据和 L/U 序列均与匹配控制核对一致。每行 As 原生处理 142080 次，覆盖全部 90 个 TX×RX×day×eq 单元和全部 270 个单元×训练场景组合；状态及质量覆盖见 CSV。session 按 RX/day 定义，与 TX 标签无关。所有合法有标签 As 都进入 CE。U 实际接受 R=8790、Q=8798、C00=8808/567000；接受率不等于隐藏标签精度。

## 预先保存的源域决定

读取新正式成绩之前，已保存并由 owner 观察 SOURCE_READOUT_DECIDED：唯一候选 Q，source_qualified=false。全 V clean：NoAs30=98.359259%、C00=98.296296%、R=98.314815%、Q=98.322222%；Q−NoAs 为 −0.037037 个百分点，通过 clean 保持线。

两次固定信道 draw 合并的 mid/low 均值：C00=66.013889%、R=65.944444%、Q=65.986111%；Q 相对 max(R,C00,原 As) 为 −0.027778 个百分点，未达到预登记 +0.5。源 low：C00=59.027778%、R=58.916667%、Q=58.805556%。因此本候选关闭，不用正式分数救回 R，不扩 E50、不扩种子、不扫描 ε 或置换种子。两个信道 draw 不是两个模型种子。

## 正式默认头结果

每行每视图 168000 个固定测试观测，数值为准确率百分比。每行全部七视图 prediction 完整固定后，独立进程才连接 truth；三条既有参照不重预测、不重评分。结果不回流选模、晋级或后续参数。

| 环境 | NoAs30 | 原 As30 | C00 | R | Q | Q−R（百分点） | Q−C00（百分点） |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 85.247619 | 85.341071 | 85.200000 | 85.182738 | 85.207738 | 0.025000 | 0.007738 |
| practical_high | 42.685119 | 81.405357 | 81.279762 | 81.182738 | 81.253571 | 0.070833 | -0.026190 |
| practical_mid | 40.995238 | 78.661310 | 78.586310 | 78.455357 | 78.532143 | 0.076786 | -0.054167 |
| practical_low_suburban | 39.472619 | 75.861905 | 75.829762 | 75.640476 | 75.801190 | 0.160714 | -0.028571 |
| practical_high_urban | 41.439881 | 77.767262 | 77.700595 | 77.577381 | 77.670238 | 0.092857 | -0.030357 |
| practical_mid_urban | 35.132738 | 62.806548 | 62.791071 | 62.621429 | 62.702976 | 0.081548 | -0.088095 |
| practical_low_urban | 30.139881 | 51.244048 | 51.266667 | 51.066667 | 51.185119 | 0.118452 | -0.081548 |
| urban 三场景均值 | 35.570833 | 63.939286 | 63.919444 | 63.755159 | 63.852778 | +0.097619 | −0.066667 |

所有逐行宏 F1、最差 RX、RX/day/TX 分层和混淆矩阵见原 CSV/JSON；下表列同父基准及唯一候选，单位均为百分比。

| 环境 | C00 宏 F1 | Q 宏 F1 | C00 最差 RX | Q 最差 RX |
|---|---:|---:|---:|---:|
| clean | 84.924993 | 84.927459 | 78.845833 | 78.733333 |
| practical_high | 81.008518 | 80.982542 | 73.233333 | 73.079167 |
| practical_mid | 78.323371 | 78.268607 | 71.425000 | 71.229167 |
| practical_low_suburban | 75.562128 | 75.530727 | 69.095833 | 69.075000 |
| practical_high_urban | 77.424030 | 77.391466 | 70.470833 | 70.362500 |
| practical_mid_urban | 62.438589 | 62.337072 | 57.450000 | 57.320833 |
| practical_low_urban | 50.818555 | 50.718462 | 47.029167 | 46.933333 |

Q−C00 的 mid urban 修复 389 个错误，却新增 537 个错误，净 −148；low urban 修复 576、新增 713，净 −137。Q−R 对应净修复为 +137、+199，说明物理关联略优于随机置换，但不足以优于原 As。clean 对 C00 修复 208、新增 195，只有净 +13。

统一 cosine30 尺度后的默认头平均分类间隔，Q−C00：clean −0.000062153、mid urban −0.000986785、low urban −0.000594394。固定 NoAs30 clean 正确集合上的正式 low 保持率也下降 0.078204 个百分点。不能将尺度变化或微小 clean 准确率差当作判别改善。

## 退化分层与真实父模型集合

源域两 draw 的 Q−C00 净纠错如下，各维度重叠，不能相加：

| 分组 | mid urban 净纠错 | low urban 净纠错 |
|---|---:|---:|
| BLOCKED | +8 | −8 |
| unlocked | +4 | −5 |
| 新增噪声比 <3 dB | +2 | −3 |
| 新增噪声比 3 至 8 dB | +5 | −4 |

mid 的局部小幅修复没有覆盖 low 的主要困难组。本轮未证明任何 low urban 退化得到可信缓解。正式 IQ 没有可用逐包 state/lock 标签，不能把源域标签套到正式结果，不能作正式物理因果归因。

复用精确 SAT18 AS_CE20 student 既有合法源预测缓存，固定其 clean 正确 1767/1800 个物理样本；两信道 draw 的 satellite 分母为 3534。该集合与上文固定 NoAs30 集合分别记录，没有新增 NN。Q 的 clean 保持率为 1765/1767=99.886814%（C00 相同）；mid 为 2620/3534=74.136955%，比 C00 高 0.169779 个百分点；low 为 2103/3534=59.507640%，比 C00 低 0.254669 个百分点。Q 相对 E20 父的 low +0.848896 不能归给本机制，因为 C00 的同预算继续训练增量更高。该父集合没有正式版本，不新增测试去补齐。

## 成本、验证与命令

新增两条 E10 尾段，共 20 模型轮、4440 次更新。每行累计 E30/6660。新增 native 样本前向 6325924、源评估 25200、正式预测 2352004、共享真实 checkpoint no-query smoke 1540，合计 8704668；本地 NN=0。共同 E20 父的 native 成本 4797722 前向/1827.538 秒只计一次；三条历史参照和不同父历史分别列出，不能将五条 E30 曲线的 150 行当作 150 轮新增训练。

R/Q 新尾训练过程耗时分别为 1278.377/1312.472 秒，合计 2590.848 过程秒；源预测合计 3.648 秒、正式预测合计 100.046 秒。独立远端文件时间核实，从本轮启动到分析完成实际经过 1602.189 秒（26.703 分钟），并行过程时间之和不等于经过时间。同前向数不保证同墙钟，预算耗尽不代表收敛。历史加新增去重为 110 模型轮/24420 更新；已知历史加新增前向含复用父集合诊断为 36165128，历史共享 smoke 尚不能完整分摊。

已执行：协议 6 组/21 负例/5 源阈值案例；真实 hook 组件 13 负例及原 ε=.01 时 loss/grad/global RNG 与 C00 对齐；合法 None 边界 5 例；分析 18 项 CPU fixtures；定点语法检查；一次独立 P0/P1 审查并修复直接错误；一次普通账号 preflight、单归档 SHA 对比和 25 文件远端编译；launcher 首步真实父模型/自身 EMA/默认头两图 smoke 1540（无源 IQ、query、更新）；实际 GPU、CWD、参数和日志增长检查。全部 PASS/VERIFIED，无未解决 P0/P1。

独立只读结果核算 PASS：630 个正式和 1785 个源混淆矩阵、480 个分区汇总、55 个最差 RX、4830 个配对净纠错检查、1245 个真实父集合记录一致。4 张 PNG 已逐张查看，4 个 PDF 文件存在且头有效。最终所属三个运行句柄均已不存在，analysis status=ANALYZED、missing=[]。

可执行入口：在冻结 release CWD 使用既有 CVS-RFFI Python 执行 python -u experiments/cvs_quality_entropy77/runner.py；原输入/输出根不可复用。协议与组件检查入口为 experiments/cvs_quality_entropy77/checks.py、model_checks.py；分析检查实际命令为 E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 analysis/cvs_quality_entropy77_analysis_20261019.py --checks。真实 hook 检查通过 UTF-8 stdin 组合固定运行链，准确调用与输出保留于本地 core_model_checks_actual.json。本报告不授权重启已完成 run。

实现文件：experiments/cvs_quality_entropy77/ 下 entropy.py（物理 ε 与边界）、model.py（原 As CE 替换）、design.py（冻结配置）、observer.py（真实训练证据）、dispatch.py/runner.py（自动逐行收尾）及配套协议/评估文件；analysis/cvs_quality_entropy77_analysis_20261019.py 和对应 checks 文件。代码已在 cf49ff220dc926bac704f15b7f34a79d1ef60c27 提交。

## 实际产物与边界

全部本轮本地分析位于 [analysis_r01](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01)，远端仍保留独占 PROJECT/runs/20261019-phase1-FcrossRX-qualityEntropy-e30-s2026092701-m2-r01。训练权重、原始日志、IQ、逐样本原始预测和私有运行信息不入 Git。

- [全部逐行指标](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/formal_metrics.csv)、[配对结果](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/formal_pair_summary.csv)、[分层净纠错](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/formal_paired_effects.csv)。
- [源域分层](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/source_strata.csv)、[真实父集合](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/actual_parent_source_cohort.csv)、[完整训练曲线数据](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.csv)。
- [训练曲线 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/training_curves.pdf)。
- [源质量与状态图 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/source_quality_states.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/source_quality_states.pdf)。
- [正式七环境配对图 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/formal_all7_paired.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/formal_all7_paired.pdf)。
- [目标熵分配图 PNG](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/target_entropy_assignment.png)／[PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/retrieved_r01/analysis_r01/target_entropy_assignment.pdf)。
- [独立核算](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/independent_result_recount.json)、[启动到完成时间及所属句柄核实](E:/type10-7/local_artifacts/cvs_FcrossRX_qualityEntropy_E30_20261019_r01/terminal_state_observed.json)。

科学未完成项：low urban >60%、新增机制收益、可重复 clean 正收益、跨模型 seed 双提升均未达到。本候选没有技术缺项；不把科学目标未完成改写为运行失败，也不因单 seed 满足 clean 保持线宣称最终目标完成。历史失败、原增强三种子收益与其 clean 波动继续保留。附件提出的分层风险、U 路由和公共传播训练方向须结合已完成实验去重，不能用本轮 Q/R 的小差异为其作担保。
