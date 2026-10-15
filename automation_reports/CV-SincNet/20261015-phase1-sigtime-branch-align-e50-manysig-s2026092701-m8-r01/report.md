# SIG_TIME：同分支位置对齐 E50 单 seed 小试

状态：八组小试训练／七视图独立评分／冻结后源机制均已ANALYZED／VERIFIED；仅model seed2026092701，正式三seed与稀疏组合尚未完成。本报告保留全部八组，不按目标成绩排序或选择性重跑。

七种新增约束均未通过预登记源推进条件；本家族不扩seed、不挑单组晋级。正式结果中A-S对A0的low仅+0.3673个百分点、clean−0.1030个百分点；A-J为low+0.3012、clean+0.6988个百分点，A-ALL为low−0.0232、clean−0.6137个百分点。全部low低于60%，没有达到预设大幅urban改善目标，也没有跨seed证据。

| 配置 | urban均值 (%) | low urban (%) | mid urban (%) | clean (%) | suburban均值 (%) | urban Macro-F1 (%) | 最差urban RX (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 64.1504 | 53.2714 | 63.4143 | 79.8060 | 76.8667 | 63.6986 | 58.2903 |
| A-T | 63.9609 | 53.1238 | 63.2399 | 79.2744 | 76.6274 | 63.5709 | 57.7028 |
| A-F | 64.1825 | 53.2708 | 63.4381 | 80.1262 | 76.9089 | 63.6972 | 58.2472 |
| A-P | 64.0238 | 53.3417 | 63.2952 | 79.9411 | 76.5196 | 63.4868 | 57.9847 |
| A-R | 63.8454 | 53.0185 | 63.2190 | 78.3726 | 76.4093 | 63.3730 | 57.5958 |
| A-S | 64.4821 | 53.6387 | 63.7548 | 79.7030 | 77.1286 | 63.7225 | 57.8069 |
| A-J | 64.5524 | 53.5726 | 63.7738 | 80.5048 | 77.3234 | 64.1112 | 58.7611 |
| A-ALL | 63.9597 | 53.2482 | 63.2512 | 79.1923 | 76.5093 | 63.4782 | 58.0972 |

| 相对A0（百分点） | urban | low urban | mid urban | clean | suburban | urban F1 | 最差 RX |
|---|---:|---:|---:|---:|---:|---:|---:|
| A-T | -0.1895 | -0.1476 | -0.1744 | -0.5315 | -0.2393 | -0.1278 | -0.5875 |
| A-F | +0.0321 | -0.0006 | +0.0238 | +0.3202 | +0.0423 | -0.0014 | -0.0431 |
| A-P | -0.1266 | +0.0702 | -0.1190 | +0.1351 | -0.3470 | -0.2119 | -0.3056 |
| A-R | -0.3050 | -0.2530 | -0.1952 | -1.4333 | -0.4573 | -0.3256 | -0.6944 |
| A-S | +0.3317 | +0.3673 | +0.3405 | -0.1030 | +0.2619 | +0.0239 | -0.4833 |
| A-J | +0.4020 | +0.3012 | +0.3595 | +0.6988 | +0.4567 | +0.4126 | +0.4708 |
| A-ALL | -0.1907 | -0.0232 | -0.1631 | -0.6137 | -0.3573 | -0.2204 | -0.1931 |

目标low urban60%的实际达到组：无。这些目标指标仅用于结果报告，不进入源推进规则。一个seed不能证明位置最优、稳定优于KL或跨seed复现。E50结束不是收敛证明。

A0与所有变体共同使用新内部源折：原L6300固定拆为训练5580／开发720；U56700的TX真值保持隐藏，V27000仅健康诊断。各TX×RX×day单元70条L中固定留8条，FOLD_SEED2026101503；开发IQ不参与优化器／EMA／归一化统计更新。该开发折曾被历史全L模型见过，并用于本轮源定标／推进，不能声称全历史未见盲测。旧全L B0／C3／KL／方向模型不是这个内部折的严格匹配对照；不能把新旧基线差归因于某一约束。

固定原SIG_TIME、内部门控、参考响应和默认cosine6×160／scale30；863730总参数，实际部署身份祖先195041参数／71张量。全部原参数仍由原CE／U训练联合更新，无新骨干、分类头、预测MLP或拟合诊断头。A0为普通clean＋星地拼接CE；全部有标签satellite都参与CE，clean:satellite=1:1分别均值，label_smoothing0.01。训练场景仅low suburban／mid urban／low urban；本位置矩阵课程eta=0，不与C3课程／间隔／新SSL叠加。原U旧门控33＋17安排保持。

前5轮均无对齐，epoch6至50仅启用各自一个方向目标。全部八组都有同等L clean ownEMA eval教师前向；仅教师clean默认头正确且urban时参与额外方向约束，不要求urban学生高置信。T/F为实际fuse输入（F含freq_stats_proj），P为完整pa_norm，R为可学习response_projection，S为gain前output_norm的160维，J为实际joint160的前向等值stop-gradient gain重组。学生图保留，教师停止梯度；不匹配raw29、错位112维、所有gain或不同分支。

一次静态强度校准使用A0自己的固定epoch5／ownEMA，其他组仍独立scratch，不加载A0预热权重。全局theta排除W、未接收梯度按0；lambda=.1×median||grad(.5CEsat)||／median||grad(Lb)||，不是层比值中位数。ALL先各分支定标、omega=.2平均，再对总强度独立定标。源静态10%不保证全训练／各层比例或AdamW更新相同，不在线追调lambda。

| 目标 | 冻结lambda |
|---|---:|
| T | 0.516316511097 |
| F | 1.476165015704 |
| P | 1.173301616579 |
| R | 0.112902236328 |
| S | 0.313356807098 |
| J | 0.647934640281 |
| ALL | 1.973158456105 |

FULL_FP32_POLICY的cudnn_deterministic=false：相同初始化规则／seed和增强不等于bitwise相同权重轨迹。前5轮的源小差如实保留，不借此改配置或重跑。

| row | E50成功更新 | 主训练前向 | L_dev面板 | 原V验证 | 墙钟 (s) | 裁剪步数 | 含测试总前向 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0-s2026092701 | 11100 | 6190200 | 70560 | 1404000 | 4593.410 | 155 | 8840764 |
| A-T-s2026092701 | 11100 | 6190200 | 70560 | 1404000 | 4886.580 | 163 | 8840764 |
| A-F-s2026092701 | 11100 | 6190200 | 70560 | 1404000 | 4592.898 | 197 | 8840764 |
| A-P-s2026092701 | 11100 | 6190200 | 70560 | 1404000 | 4798.450 | 139 | 8840764 |
| A-R-s2026092701 | 11100 | 6190200 | 70560 | 1404000 | 4768.287 | 121 | 8840764 |
| A-S-s2026092701 | 11100 | 6190200 | 70560 | 1404000 | 4833.575 | 166 | 8840764 |
| A-J-s2026092701 | 11100 | 6190200 | 70560 | 1404000 | 4811.223 | 193 | 8840764 |
| A-ALL-s2026092701 | 11100 | 6190200 | 70560 | 1404000 | 4793.529 | 152 | 8840764 |

全部父训练0／新增50／累计50，每组11100更新，八组88800更新。主前向各6190200：L clean／satellite／ownEMA clean各1420800，原U教师／strong学生各963900。各有14×5040=70560开发前向。原V各50×27000，并有final geometry与wrapper最终健康各27000，合计1404000；每组原生geometry共5300 cache hit、0miss，没有另加重复前向。每组scratch smoke2、最终checkpoint smoke2、正式测试1176000。另有共享dispatcher smoke2、A0源校准3600／6.332秒／0更新，冻结后机制46800／131.674秒／0更新。上述实际新增前向总样本数70776514，不是FLOPs；多GPU重叠的各行耗时之和也不是总墙钟。历史B0只作启动smoke，无权重继承。

全量解析88800步／400epoch及完整stdout，核对连续预算、有限训练标量、实际optimizer_step_applied=1、无非有限跳步和U真值暴露。source机制／梯度记录显示aux对默认W与三gain为0，原总训练仍更新五组身份特征／W／gain；原clip阈值5.0，实际裁剪如表。stdout中的inactive诊断nan与警告保存在审计JSON中，不混同实际训练非有限跳步。

| trace核对 | 增强输入数 | 实际L_train ID | 开发／其他ID | session数 | 相对A0损伤元数据差异数 |
|---|---:|---:|---:|---:|---:|
| A0-s2026092701 | 1420800 | 5580 | 0 | 15 | 0 |
| A-F-s2026092701 | 1420800 | 5580 | 0 | 15 | 0 |
| A-R-s2026092701 | 1420800 | 5580 | 0 | 15 | 0 |
| A-P-s2026092701 | 1420800 | 5580 | 0 | 15 | 0 |
| A-S-s2026092701 | 1420800 | 5580 | 0 | 15 | 0 |
| A-J-s2026092701 | 1420800 | 5580 | 0 | 15 | 0 |
| A-ALL-s2026092701 | 1420800 | 5580 | 0 | 15 | 0 |
| A-T-s2026092701 | 1420800 | 5580 | 0 | 15 | 0 |

trace使用source:<base_index>，登记使用opaque physical ID；通过原发布代码的L_s索引（仅6300合法有标签记录）做字段映射后遍历全部11366400个增强输入。索引重建按原生builder加载既有ManySig容器，仅遍历L_s metadata；不调用U/V/query样本的__getitem__或U标签张量，不构造query loader，不作模型前向／更新，不改原日志。这是样本使用层面的隔离，不声称原生容器加载没有读取任何非L字节。实际每个物理ID的session固定，15个RX/day/eq session均包含6个TX，90个TX×RX×状态单元有支持。匹配字段包含样本／session／场景／状态／锁定／质量／新增噪声SNR／CFO／残余核长／q；这是记录的生成状态匹配，不是IQ字节hash或模型权重轨迹相等声明。

正式测试每行完成后立即自动预测原clean＋六practical（high／mid／low suburban／high urban／mid urban／low urban），每视图168000记录，每行1176000预测／126指标；八组合计9408000预测／1008指标。只有同row七视图全部prediction固定后，独立exec scorer才接truth并独立复算。默认checkpoint为final50学生头，非另训头；原正式IQ／SNR／困难样本均不改。

| 源L_dev low urban | 准确率 (%) | 固定A0集合保持率 (%) | 修复A0错 | 新增A0错 | 归一化默认头margin |
|---|---:|---:|---:|---:|---:|
| A0 | 66.9444 | 68.2788 | 0 | 0 | 0.095764 |
| A-T | 67.0833 | 68.2788 | 6 | 6 | 0.096670 |
| A-F | 67.2222 | 68.5633 | 8 | 6 | 0.095452 |
| A-P | 67.0833 | 68.2788 | 6 | 6 | 0.097024 |
| A-R | 66.3889 | 67.5676 | 9 | 14 | 0.099663 |
| A-S | 67.5000 | 68.9900 | 16 | 11 | 0.095774 |
| A-J | 67.2222 | 68.2788 | 12 | 12 | 0.081752 |
| A-ALL | 67.7778 | 68.9900 | 13 | 8 | 0.101462 |

冻结后的固定参考为匹配A0 final50的clean正确集合703/720。源机制共46800前向，0optimizer／EMA／拟合头更新，模型及ownEMA state不变，V_forward0、无U/query读取。T/F/P/R/S/J真实节点方向、最近错误TX cosine、variance/norm、默认头修复／新错及margin均保留；最近错误TX比较只用720合法开发clean向量，不拟合决策规则。loss下降或配对cosine提高本身不证明身份判别提高，gain降低本身不证明鲁棒。

| 单项或组合 | 检查节点 | clean配对cosine | 最近错误TX cosine | 分支均方差 |
|---|---|---:|---:|---:|
| A0 | T | 0.830967 | 0.955869 | 0.026597 |
| A0 | F | 0.867154 | 0.931794 | 0.019972 |
| A0 | P | 0.913250 | 0.966242 | 0.200187 |
| A0 | R | 0.320002 | 0.658871 | 1.130055 |
| A0 | S | 0.357843 | 0.770885 | 0.933751 |
| A0 | J | 0.712107 | 0.895187 | 0.164988 |
| A-T | T | 0.977923 | 0.994408 | 0.017738 |
| A-F | F | 0.993171 | 0.996045 | 0.008428 |
| A-P | P | 0.974953 | 0.990546 | 0.070550 |
| A-R | R | 0.783205 | 0.883849 | 0.551354 |
| A-S | S | 0.617519 | 0.902711 | 0.652056 |
| A-J | J | 0.877745 | 0.949741 | 0.099571 |

真实约束已在对应位置生效：T/F的配对cosine分别升至约.978/.993，R升至.783，S升至.618。但最近错误TX也更近，T/F/P等方差下降，默认头收益没有同步扩大。不能把这些量单独解释为身份迁移成功或完整特征塌缩。A-J中实际R分支幅度／joint norm比约.110，A0约.237，同时J方向更近；融合分配的改变也参与了结果，不能据J不变性推断每个分支更鲁棒。

| 组别 | low源BLOCKED (%) | degraded (%) | unlocked (%) | 新增噪声SNR<3dB (%) |
|---|---:|---:|---:|---:|
| A0 | 43.2642 | 65.0350 | 38.5417 | 34.9515 |
| A-T | 42.7461 | 66.4336 | 37.8472 | 32.5243 |
| A-F | 43.0052 | 65.7343 | 38.5417 | 34.4660 |
| A-P | 43.5233 | 67.1329 | 37.8472 | 33.4951 |
| A-R | 41.7098 | 64.3357 | 37.1528 | 32.5243 |
| A-S | 43.0052 | 66.4336 | 38.1944 | 33.0097 |
| A-J | 43.0052 | 65.0350 | 38.8889 | 34.9515 |
| A-ALL | 44.0415 | 65.7343 | 39.9306 | 34.9515 |

状态、锁定和新增噪声SNR层相互重叠，表中按实际交叉单元样本数加权；新增噪声SNR不是总接收SNR。这些同流完整信道对比用于定位状态关联，不能独立断言解决了多径、Doppler、RX IQ或边界损伤。原四行total/fixed损伤移除诊断另行保留，不能把不同参考集合混合。

| 组别 | T/S全feature梯度cosine中位数 | 双非零共享参数梯度cosine中位数 | 共享参数张量数（并集） |
|---|---:|---:|---:|
| A0 | 0.052070 | 0.089824 | 13 |
| A-T | 0.046574 | 0.080960 | 13 |
| A-F | 0.052428 | 0.086170 | 13 |
| A-P | 0.054032 | 0.092758 | 13 |
| A-R | 0.054125 | 0.088784 | 13 |
| A-S | 0.029806 | 0.046285 | 13 |
| A-J | 0.026706 | 0.043039 | 13 |
| A-ALL | 0.049870 | 0.085703 | 13 |

上述T/S为冻结eval源mid/low各批raw目标梯度，不是实际Dropout／clipped AdamW更新。共享范围只含当前两个梯度norm均>1e-12的同名参数，排除各自独立投影；全feature角与共享角分别报告。P/R/S gain×分支norm与joint norm的比是幅度比，不是相加为1的贡献份额，抵消时可超过1；T/F仍是非线性fuse输入。

源家族推进条件在target前固定：至少一项对A0的final50 L_dev low≥+1pp、clean≥−0.5pp、urban≥0、最差urban RX≥−1pp。实际family_expandable=False，满足条件的源信号组为[]。
本家族未通过预登记源推进条件，保留负结果，不依据target放宽条件或挑某组扩seed。正式三seed与稀疏组合仍未执行。
三seed之后才按预登记源规则找b1/b2：各项3/3源low提高、均值≥+0.5pp、clean均值≥−0.5pp、urban均值≥0、最差RX≥−1pp；按low→urban→最差RX，0.2pp内才比少约束／成本。至少两个合格单项才做b1+b2／b1+J／b1+b2+J并重新总强度定标。不预设T+S，不把ALL、J或某个首seed的目标最高分直接晋级。

实际practical导入adv3b02_xuc/code/leo_practical/{channel,residual}.py，25MHz、IQ256、原native RMS后模拟器输入RMS，输出不再RMS；residual/post_sync、EQfalse（G=1）、reference_snr_db30、接收质量适配／phase tracking／IQcomp true、reflect。几何／传播／遮挡／噪声／接收状态／补偿依赖保持，未反演原录音未知TX硬件。

训练release commit152d8921c903fc6d5f782c199b075d0837840ece；一次源全局校准3600前向／6.332秒；源机制release commit0b5cc8b0451cfad03d298af3ba39102a665947af。11项训练／分支定点检查、4项源机制检查、真实checkpoint no-query smoke、一次限定训练P0/P1及源分析P0/P1检查完成；源recipe部分写入竞态按同目录原子rename修复并仅定点核实。普通N607账户、一次archive比对／远端compile和PID/CWD/argv/GPU／日志独立回读均VERIFIED。每GPU最多3总实验进程计其他owner；没有热改、停止或重启健康实验。

训练命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_urban_branch50.dispatch`，CWD PROJECT/releases/cvs_sigtime_branch_align_e50_20261015_r01。冻结后`-m experiments.cvs_urban_branch50.mechanism`，CWD PROJECT/releases/cvs_sigtime_branch_align_source_20261015_r01。唯一run/output、owner、GPU、stop rule和全矩阵仍见experiment.json。绘图：`F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_urban_branch50.figures --artifacts E:/type10-7/local_artifacts/cvs_branch_alignment_20261015`。
已执行的全日志获取／审计命令：`F:/App/miniconda3/python.exe -X utf8 E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/harvest.py`、同目录`source_index_aliases.py`和`audit_channel.py`。后两者分别翻译既有L metadata和遍历全部trace，0模型前向／更新；核心完整trace读码为experiments/cvs_urban_branch50/log_audit.py，可用`--run-root ... --row ... --index-map ...`复用。图像仅因本地Matplotlib的labels接口变化改用set_xticks后重绘，模型／预测／评分未重跑。

实际图表：
- [complete_training_curves.png](E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/figures/complete_training_curves.png)
- [source_learning_progress.png](E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/figures/source_learning_progress.png)
- [per_seed_formal_accuracy.png](E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/figures/per_seed_formal_accuracy.png)
- [actual_branch_identity_geometry.png](E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/figures/actual_branch_identity_geometry.png)
- [source_default_head_retention_margin.png](E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/figures/source_default_head_retention_margin.png)
- [gain_and_contribution_amplitude.png](E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/figures/gain_and_contribution_amplitude.png)
- [T_S_shared_gradient_angles.png](E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/figures/T_S_shared_gradient_angles.png)
- [source_state_quality_strata.png](E:/type10-7/local_artifacts/cvs_branch_alignment_20261015/final/figures/source_state_quality_strata.png)

完整证据：final/training_audit.json、channel_trace_audit.json、source_family_decision.json、source_branch_results.json、source_gradient_aggregate.json和source_delivery_r01/readback.json。IQ／权重／预测／原始日志／生成图输出不入Git。仅本轮阶段实现与小试已完成，完整三seed／条件稀疏组合／同信道中间视图／局部SIG分组教学尚未完成。
