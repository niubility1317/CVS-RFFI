# P74：冻结原默认头后的噪声与四贡献归因

状态：ANALYZED_SOURCE_ONLY / VERIFIED。两个冻结原 P43 E30 模型完成源 V1800、两个固定信道 draw、mid/low urban 的全部读出，168 组整体混淆矩阵与配对纠错独立复算一致，三张 PNG 已逐一查看，三张 PDF 已核对。无新训练模型或正式目标预测；当前总目标仍 ACTIVE。

**本轮不支持用分支屏蔽或单个整分支恢复作为下一轮默认方案。** 固定完整接收状态，仅移除新增噪声，As 模型源 mid urban 从 72.8056% 到 78.0833%（+5.2778 pp），low urban 从 59.0833% 到 68.1389%（+9.0556 pp）。但四个分支没有一个在全部 mid/low×两个 draw 中满足预先固定的局部噪声证据规则。它不证明局部训练永远无效，也不证明噪声是唯一损伤。

直接置零 PA、REF、SIG 的源 low urban 分别下降 6.4722、1.5000、6.2500 pp；SIG 置零还让源 clean 下降 6.8889 pp。SIG 的噪声特异替换低 urban 净增 4.2778 pp，但需要另一个不可在部署时获得的去噪观测，因此不能写成训练提升、可达上界或已修复部署退化。

## 固定模型、数据与真实路径

起点代码 542039ceb80144758f95814fcae89f17b654cf49；r02 真实执行代码 **b49877aba4095b15c2e9f01a3bf3b4d3d33ab626**。原 P43 B00-s2026092701/source/final_ssdg.pth（NoAs30）与 B01-s2026092701/source/final_ssdg.pth（As30）均严格加载；原父为各自 scratch20+own10、FINAL30、原 student/ownEMA，全 L6300/U56700 隐藏/V27000、源 RX1/3/4/6/8、day1/2/3、split392005。加载前沿原 own_model/parent_payload 核查完整来源、物理角色与源冻结依据；没有以新测试分数选 checkpoint。

F＋跨 RX 同 TX SupCon、SIG_TIME_X3、统一 Sinc、NoExplicitCFO、joint160、900722 参数/206 state项、原默认 cosine6×160/scale30，均保持。两个模型全程 eval/requires_grad false，参数/缓冲值与梯度不变；不拟合头、不增加部署分支。本轮只读静态逐维 PA/REF/SIG gains，未宣称新增动态门控或梯度因果证据。

复用已合法的 P24 V1800 物理面板：6TX×5RX×3day 每格20个物理包，两条既有命名物理随机流，缓存条件 full 与 no_noise/fixed。无 U TX 真值、无目标数据、无新增模拟或数据重验证。原源 IQ 2×256，25 MHz、eq1、原逐包复 RMS center=false；缓存诊断载频 **2.45 GHz**，区别于父训练 **2.462 GHz**。不把新加噪声比叫总 SNR。

实际导入打印的 __file__ 为 PROJECT/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/**experiments/adv3b02_xuc/code** 下 SSDG/train_ssdg.py、leo_practical/channel.py、leo_practical/residual.py。路径经过实际运行核对，未使用顶层 code/leo_practical。配置 residual/post_sync、equalization_enabled=false（G=1）、reference_snr_db=30、reflect、质量适配/相位跟踪/IQ补偿开启；AGC、残余核及数值延迟均沿既有缓存。固定去噪与 full 的传播状态、锁定、补偿频率、AGC 增益、残余核/边界逐字段一致。正式 clean+六 practical 的 IQ、SNR、困难样本、边界与评分规则不变。

## 本轮唯一假设与真实干预

唯一假设是：As 后噪声损失是否主要集中于一个真实加法贡献，或一个受损贡献是否持续误导默认头。四贡献为 TF=原 base_norm 输出；PA=tanh(head.gain)×pa_norm 输出；REF=tanh(response_gain)×response_projection 输出；SIG=整个 sequence_pool 输出（已含 gain/LN）。没有把单个 SIG 描述器当作独立160维加法项，也未拆开非线性 T/F fuse。

每次只将一个 full 贡献置零、换成同物理包 no_noise/fixed、换成同物理包 clean。按原加法顺序重建 joint，再调用原 classify 重新执行最终 L2。不是线性加减 logits；未更换头。clean 另做四个 zero 控制。所有完整 forward 的原 logits 与捕获重构 logits 保持一致，两个真实 checkpoint smoke 的最大 joint/logit 误差均0。

预登记解释规则没有改动：同一分支 swap_nonoise 在四个场景×draw 格都净增≥2 pp 且恢复整体 fixed-noise 增量至少50%（整体增量须正），才支持局部噪声证据；同一分支 zero 在四格都增≥1 pp 且 clean 代价≤.1 pp，才支持屏蔽研究。四分支均未通过两项解释规则；没有挑最好 draw 或将临界阈值事后下调。规则不自动筛除其他值得训练的新机制，不构成自动训练/E50门槛。

## 逐模型、逐 draw 源结果

以下是合法源域 V 诊断，不是正式目标测试；两个 draw 不是额外模型 seed。固定 NoAs30 clean 正确集合1772/1800，用于所有卫星视图保持率；不冒称共同 As20 父集合。

| 冻结模型 | 源场景 | draw | 条件 | accuracy % | macro F1 % | 最差源 RX % | 固定 clean 正确集合保持率 % |
|---|---|---:|---|---:|---:|---:|---:|
| NoAs | mid_urban | 0 | full | 42.9444 | 40.9578 | 35.0000 | 43.2844 |
| NoAs | mid_urban | 0 | no_noise_fixed | 47.0000 | 44.7791 | 37.5000 | 47.4605 |
| NoAs | mid_urban | 1 | full | 44.1667 | 42.2304 | 37.2222 | 44.4695 |
| NoAs | mid_urban | 1 | no_noise_fixed | 48.1111 | 45.5386 | 41.9444 | 48.3634 |
| NoAs | low_urban | 0 | full | 35.6667 | 34.5248 | 30.8333 | 35.7788 |
| NoAs | low_urban | 0 | no_noise_fixed | 41.3333 | 39.4295 | 32.2222 | 41.4786 |
| NoAs | low_urban | 1 | full | 35.5556 | 34.5233 | 29.7222 | 35.7788 |
| NoAs | low_urban | 1 | no_noise_fixed | 41.3333 | 40.1082 | 34.7222 | 41.8172 |
| As | mid_urban | 0 | full | 72.5556 | 72.4416 | 70.2778 | 73.4199 |
| As | mid_urban | 0 | no_noise_fixed | 77.3889 | 77.1011 | 72.5000 | 78.3296 |
| As | mid_urban | 1 | full | 73.0556 | 72.9935 | 66.6667 | 73.9278 |
| As | mid_urban | 1 | no_noise_fixed | 78.7778 | 78.5806 | 73.0556 | 79.5711 |
| As | low_urban | 0 | full | 60.1111 | 59.7500 | 56.6667 | 60.7788 |
| As | low_urban | 0 | no_noise_fixed | 68.1667 | 67.6355 | 63.8889 | 68.9052 |
| As | low_urban | 1 | full | 58.0556 | 57.7265 | 53.8889 | 58.4086 |
| As | low_urban | 1 | no_noise_fixed | 68.1111 | 67.6800 | 63.6111 | 68.7923 |

源 clean：NoAs 98.4444%，As 98.3333%，面板净−2/1800=−0.1111 pp。该有限源面板变化不能替代完整源 V 或正式未知 RX 上的0.1 pp阶段判定。

| As 贡献 | mid 去噪替换 Δpp | low 去噪替换 Δpp | low 置零 Δpp | clean 置零代价 pp | 四条件噪声规则 / 屏蔽规则 |
|---|---:|---:|---:|---:|---|
| TF | +0.3889 | +0.7778 | -0.0833 | 0.2222 | false / false |
| PA | +2.9167 | +2.4444 | -6.4722 | 0.4444 | false / false |
| REF | +0.6944 | +2.4722 | -1.5000 | 0.6111 | false / false |
| SIG | +2.3889 | +4.2778 | -6.2500 | 6.8889 | false / false |

low urban 整体 fixed 去噪修复539次错误、引入213次，净326/3600；仅 SIG 去噪贡献替换修复310次、引入156次，净154/3600。这3600是1800物理包的两次观测，不是3600独立物理样本。各分支增益不可相加，头归一化与证据相互作用是非线性的。

源 low 的 BLOCKED/unlocked 由30.3514%到45.9425%（1565次观测，+15.5911 pp）；LOS/locked由94.7170%到95.7547%（1060次，+1.0377 pp）。这支持新增噪声在困难遮挡/失锁观测中造成更大损失，但去噪后困难组仍较低，不能唯一归因于噪声或把其余损伤排除。PARTIAL/unlocked仅1次、空交叉格均完整保留并标记，不作总体解释。

4872行 source metrics 与4872行配对效应覆盖 state、lock、state×lock、signal_to_added_noise_db分箱、RX/day，含 accuracy/F1/最差RX/间隔÷30/保持率/修复新增错误。没有把噪声比当真实总 SNR，也没有使用正式目标状态给源样本贴标签。

## 正式性能与目标完成度：沿用既有证据，不混入本轮 oracle

本轮无新训练权重，正式成绩没有新增。既有完整 As 的[三种子匹配 E30 报告](E:/type10-7/automation_reports/CV-SincNet/20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02/final_analysis.md)明确：

| 正式条件 | 匹配 NoAs 均值 % | 完整 As 均值 % | 配对变化 pp |
|---|---:|---:|---:|
| clean | 84.9026 | 84.5815 | −0.3210 |
| high suburban | 50.872 | 80.327 | +29.455 |
| mid suburban | 48.547 | 77.635 | +29.088 |
| low suburban | 46.340 | 75.085 | +28.745 |
| high urban | 49.002 | 76.696 | +27.694 |
| mid urban | 40.090 | 62.035 | +21.945 |
| low urban | 33.4849 | 50.7698 | +17.2849 |

urban三场景均值配对+22.3079 pp；low F1由31.8445%到50.5353%，最差RX均值28.8153%到46.4681%。clean三seed配对为+0.0935/−0.1536/−0.9030 pp，均值−0.3210±0.5189 pp（样本SD），**跨seed不满足clean下降≤0.1 pp的阶段目标**。low三seed51.2440/50.4042/50.6613%均未到60%；没有可重复clean正收益，不能称双突破。

上一轮 P73 已完成：新概率均值 CE 相对同父同预算 mean CE 的 clean/mid/low/urban均值分别−0.0232/−0.0643/−0.0661/−0.0754 pp，固定候选关闭。其单seed相对NoAs clean+0.0446 pp不构成稳定双提升。P74不会用源oracle的+9.0556 pp改写这些实际训练结果。

## 历史去重与续轮边界

P24/P33为IQ损伤移除、P40为默认头/分支梯度；旧单SIG clean分支替换的结构与读点不同。未找到当前P43四postLN贡献的噪声特异干预实跑，本轮已补齐。

附件推荐的名义分层风险/P70、有界风险/P71及后续mask/P72、概率均值CE/P73已完成且未获机制增益，不原样扫描系数/seed/E50。另一对话已开展U practical资格路由，最新紧凑状态显示正固定共享AdamW与分离优化器状态的三行匹配矩阵（NoAs/As共享/As分离，各自身E20+10），源晋级条件在正式测试前固定；这些路线由另一对话负责，本轮不重复或热改。

公共传播跨TX机制并非全新：历史 propgroup-sourceRX6 E20四行已完成源域匹配评分且科学FAIL，PROP−IID的low+0.1587 pp、urban−0.0529 pp；后来的全源共享H提案已撤回。它与当前fullsource P43预算不同，但仅去掉heldRX或换组键不足以把同一已关闭假设重开。本轮没有再启动公共H实验，也未把源贡献替换直接升级成新对齐损失。下一训练方案仍须提出真正变化的假设和匹配对照；本轮结果不授权按正式分数挑选重跑。

## 实现、命令、成本与验证

实际文件均位于 [experiments/cvs_branch_noise74](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_branch_noise74)：__init__.py、design.py、runner.py、probe.py、capture.py、capture_checks.py、checks.py、analysis.py、analysis_checks.py。核心分别为[实际贡献捕获/干预](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_branch_noise74/capture.py)、[真实执行](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_branch_noise74/probe.py)、[分层分析](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_branch_noise74/analysis.py)、[固定配置](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_branch_noise74/design.py)。

本地执行20项协议/物理/预算检查、2项贡献重构/非线性干预检查、8项分析检查；9文件AST通过。一次独立P0/P1初评漏掉启动import缺陷，r01在CP/IQ/NN前退出；补齐Path导入后实际probe导入/路径构造检查PASS，唯一缺陷定点复核PASS，不重做全量评审。真实CP smoke包含在launcher首步，PASS立即继续，状态与预算核对通过。实际168组整体混淆矩阵、配对错误与父集合保持数独立复算一致，三PNG目视/三PDF头核对通过。

运行环境：N607既有普通账号CVS-RFFI Python3.10/Torch2.1/NumPy2.2、FP32、CPU2、CUDA_VISIBLE_DEVICES为空。一次同任务只读preflight，两个release各一次归档SHA传输对比及远端compile通过；未做成员hash/额外许可链。PID/CWD/argv/GPU与log增长独立核实，r02已自然退出且无failure，r01失败根保留，其他owner健康任务未改。

命令（CWD为PROJECT/releases/cvs_FcrossRX_branch_noise_sourceV_20261019_r02）：

~~~text
CUDA_VISIBLE_DEVICES= CVS-RFFI/bin/python -u experiments/cvs_branch_noise74/runner.py
CVS-RFFI/bin/python -u experiments/cvs_branch_noise74/analysis.py --root PROJECT/runs/20261019-diagnostic-FcrossRX-branchNoise-sourceV-s2026092701-m2-r02
~~~

新训练0epoch/0更新/0教师前向/0头拟合/0新增物理模拟。整网样本32404（两模型各16202，含各2包smoke），额外head-only样本219652（各109826，含各26smoke），分开记录，不冒充额外整网前向。原默认head包含在整网内的32404次不重复列作额外成本。

只读观察墙钟29.8733s，独立分析6.8874s；r02从launch到analysis完成产物58.2760s（含调度/人工衔接），不是纯计算时间。NoAs/As各row读出12.8583/13.1276s已包含于29.8733s，不再次相加。r01从launch到failure产物1.7168s，0CP/0IQ/0NN/0更新，单列保留。原两个E30模型历史训练成本共60model-epoch/13320更新；训练+验证各7960684NN、native训练+验证wall2146.190/2860.909s，已复用而非免费或本轮新增。原正式评分不重跑。本轮不新增E50，因为没有新的合格训练配方。

[全部实际分层表](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/metrics.csv)、[配对纠错](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/paired_effects.csv)、[逐条件规则](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/attribution_rules.csv)、[完整源诊断读出](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/readout.md)、[独立交付复算](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/delivery_verified.json)。原预测、物理元信息、日志、checkpoint与图片均不提交Git。

- [四贡献全部模型/draw变化图](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/branch_effects_all_draws.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/branch_effects_all_draws.pdf)
- [low urban状态分层](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/low_state_lock_fixed_swap.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/low_state_lock_fixed_swap.pdf)
- [clean删除贡献代价](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/clean_zero_cost.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_branch_noise_sourceV_20261019_r02/retrieved_r01/analysis_r01/clean_zero_cost.pdf)

没有本轮训练曲线，因为未训练。分层CSV/逐样本预测/三图均已实际生成；没有编造新模型、seed或性能增益。未完成项：low urban>60、跨seed clean≤.1pp、可重复的星地增强同时改善clean均未完成。正式星地识别收益已有三seed支持，新增机制的任务收益仍缺，当前目标保持ACTIVE。
