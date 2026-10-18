# clean 保留式 As：三模型种子 E30 完整结果

状态：ANALYZED／VERIFIED。全部四个新增E30模型、七视图预测、独立truth-last评分、源/正式复算、三seed状态覆盖均完成；训练与分析七个所属PID均自然退出，无failure。

**真实 practical As 的 urban 收益在三个模型种子上重现，但 clean 保持不稳定，完整联合目标未完成。** low urban 从33.4849%升至50.7698%，配对增益17.2849±3.4169pp；mid urban 提高21.9448±5.1968pp；urban三场景均值40.8593%→63.1672%，配对增益22.3079±5.5014pp。clean为84.9026%→84.5815%，配对−.3210±.5189pp，seed03下降.9030pp。三seed low均未达60%，三个suburban与自身clean均值仍差4.255/6.946/9.497pp，不能称双突破或保护成功。SD为三个模型种子的样本SD(ddof=1)，不是置信区间；两个固定源信道draw不是额外模型seed。

## 固定设计与执行身份

Run `20261018-phase1-FcrossRX-As-replication-e30-s0203-m4-r02`；真实训练commit `63b7c8cd7bb4139a2d800e6f4c4a7f1f2163fd47`；独立分析commit `706fc9fedfe7145812331da5dba2ee76892773f8`。新增B00/B01×seed2026092702/03，seed2026092701的P43两个固定控制复用，未重训重测。预算采用用户后续明确授权的E30，覆盖目标文本旧E20。各模型scratch own20(13L+7U)+own10(10U)，累计E30、6660更新，固定FINAL30，不选最好epoch。own20 student/EMA来源及相同数据契约逐段核对；无E50/E200权重或其他父权重。旧AF50仅提供F架构工厂配置。首轮r01为导入前技术失败，0模型NN/更新；保留产物后r02仅修复namespace并已完成。

完整L6300/U56700隐藏/V27000，源RX[1,3,4,6,8]/day1,2,3，6TX，split392005，eq1，256点/25MHz，原RMS center=false。无训练RX留出、无L_dev。F/SIG_TIME_X3＋统一Sinc CENTER_25K/CENTER_UNITS＋NoExplicitCFO＋原内部门控/参考响应＋真实joint160/defaultcos30，骨干与默认头联合更新；0新分类头/部署分支。实际导入为旧固定release的adv3b02_xuc/code/leo_practical及DataAugmentation/SSDG，actual_modules逐row保留；不是顶层code。

每step6×128 L前向，.5commonRD+.5mixedRD+原SupCon+.25rawRD，L总系数1.25。B00 mixed=ordinary64+raw64，B01 mixed=ordinary64+原完整practical64；共同普通视图监督份额保持，额外raw槽两组相同。两次RDrop前向共享同一已生成IQ，原U/自身EMA/旧门控保持，无新U/教师前向。As直接从raw生成low_suburban/mid_urban/low_urban，0普通增强预叠加；全部合法有标签As均参与CE。真实残差路线residual/post_sync，EQ关闭G=1，referenceSNR30、fs25MHz/fc2.462GHz、reflect、接收质量适配/原相位与IQ残差/AGC保持，未变弱。RX/day硬件session不由TX定义，epoch/view命名流刷新；train/validation/receiver/正式流分离。

种子不只是日志标签：实际native.seed与F工厂seed均为对应02/03；真实合成smoke确认同seed两行整个初始state相同、不同seed默认头与局部投影不同。该工厂绑定修复不改网络结构。每row实际物理角色EXACT_MATCH；每seed两行全L/U物理出现序列(stage20共5994条、own10共4440条)及6660条ordinary MSE统计完全相同，U真值未解封。

## 逐 row／逐 seed 正式结果

每row固定clean+六practical，每view168000原样本；该row全部7view预测完成后才独立CPU scorer连truth。当前四行均自然完成，原dispatcher/worker进程已消失，completion/prediction/scoring齐备，无failure。已有seed01直接复用固定预测。独立复算756个正式混淆矩阵、源1428个矩阵，42个row×view及RX/day/TX/F1/间隔均保留。正式receivedIQ、信道、SNR、边界和困难样本不变。

| seed | arm | clean % | high sub % | mid sub % | low sub % | high urban % | mid urban % | low urban % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 01 | B00 | 85.248 | 42.685 | 40.995 | 39.473 | 41.440 | 35.133 | 30.140 |
| 01 | B01 | 85.341 | 81.405 | 78.661 | 75.862 | 77.767 | 62.807 | 51.244 |
| 02 | B00 | 84.686 | 57.873 | 54.865 | 51.976 | 55.474 | 44.048 | 35.886 |
| 02 | B01 | 84.532 | 80.035 | 77.290 | 74.891 | 76.310 | 61.582 | 50.404 |
| 03 | B00 | 84.774 | 52.058 | 49.781 | 47.571 | 50.093 | 41.090 | 34.429 |
| 03 | B01 | 83.871 | 79.540 | 76.955 | 74.502 | 76.013 | 61.717 | 50.661 |

| 场景 | B00 均值±SD % | B01 均值±SD % | 配对增益均值±SD pp | 各 seed 均为正 |
| --- | --- | --- | --- | --- |
| clean | 84.903 ± 0.302 | 84.582 ± 0.736 | -0.321 ± 0.519 | false |
| practical_high | 50.872 ± 7.663 | 80.327 ± 0.966 | 29.455 ± 8.454 | true |
| practical_mid | 48.547 ± 7.017 | 77.635 ± 0.904 | 29.088 ± 7.799 | true |
| practical_low_suburban | 46.340 ± 6.342 | 75.085 ± 0.700 | 28.745 ± 6.918 | true |
| practical_high_urban | 49.002 ± 7.080 | 76.696 ± 0.939 | 27.694 ± 7.897 | true |
| practical_mid_urban | 40.090 ± 4.541 | 62.035 ± 0.671 | 21.945 ± 5.197 | true |
| practical_low_urban | 33.485 ± 2.987 | 50.770 ± 0.430 | 17.285 ± 3.417 | true |

urban三场景均值：B00各seed35.5708/45.1361/41.8708%；B01为63.9393/62.7653/62.7970%。配对变化+28.3685/+17.6292/+20.9262pp。As绝对lowurban三seed为51.2440/50.4042/50.6613%，未挑最好seed，三个均低于60%。

| seed/arm | clean F1 % | mid urban F1 % | low urban F1 % | low 最差 RX % | low margin/30 |
| --- | --- | --- | --- | --- | --- |
| B00-s2026092701 | 84.980 | 33.447 | 29.185 | 25.750 | -0.041066 |
| B01-s2026092701 | 85.044 | 62.450 | 50.887 | 46.379 | 0.028076 |
| B00-s2026092702 | 84.414 | 41.471 | 33.943 | 29.008 | -0.030262 |
| B01-s2026092702 | 84.190 | 61.448 | 50.258 | 46.250 | 0.021666 |
| B00-s2026092703 | 84.427 | 38.264 | 32.406 | 31.688 | -0.031268 |
| B01-s2026092703 | 83.443 | 61.542 | 50.461 | 46.775 | 0.022530 |

lowurban macroF1均值31.8445%→50.5353%（配对+18.6908pp），最差RX均值28.8153%→46.4681%。每个seed的每个目标RX在六个practical均获益，clean逐RX正负混合；source clean门槛通过并不能保证未知RX clean保持。头尺度均固定30，间隔除30比较，没有将尺度漂移当收益。全7view F1/最差RX见[逐row CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/overall_by_row_seed.csv)。

## 源域判据与退化分层

原预登记：每seed两draw×mid/low四格均至少+5pp，完整源V clean下降≤.5pp、各源RX clean下降≤1pp。三seed全通过，独立计数与在线决定一致。source分析完整保存后才执行final truth-last复算；主分析先读取源决定，再读取正式分数。不是根据新目标结果晋级或调参。

| seed | 源 V clean 变化 pp | mid draw0/draw1 pp | low draw0/draw1 pp | 资格 |
| --- | --- | --- | --- | --- |
| 2026092701 | -0.104 | 29.611 / 28.889 | 24.444 / 22.500 | true |
| 2026092702 | -0.070 | 21.556 / 19.722 | 15.944 / 17.500 | true |
| 2026092703 | -0.067 | 25.111 / 24.944 | 19.944 / 18.722 | true |

源分层来自同一固定V1800及两个既有信道draw，以下每seed3600次观察，物理样本仍1800，不当作3600独立物理样本。各seed同状态分区，sourcepanel是既有诊断视图，不能等同正式receivedIQ；源模型/信道draw均未挑选。

| 状态/锁定 | 每 seed 两 draw 观察数 | B00 三 seed 均值 % | B01 三 seed 均值 % | 增益 pp |
| --- | --- | --- | --- | --- |
| LOS/locked | 1060 | 63.208 | 95.377 | 32.170 |
| PARTIAL/locked | 307 | 54.940 | 89.142 | 34.202 |
| PARTIAL/degraded | 253 | 43.610 | 77.075 | 33.465 |
| PARTIAL/unlocked | 1 | 33.333 | 0.000 | -33.333 |
| BLOCKED/degraded | 414 | 31.481 | 42.754 | 11.272 |
| BLOCKED/unlocked | 1565 | 20.958 | 29.734 | 8.775 |

收益在可用观测的LOS/locked、PARTIAL/locked/degraded更大；BLOCKED/unlocked仍只有约30%准确率，且余弦margin仍为负。PARTIAL/unlocked只有1次观察，完整表保留，图中不以单例作总体解释。这是分层识别关联，未在本轮重新做“移除损伤/固定接收状态”因果对照，不能唯一归因于噪声、多径、CFO、RX IQ或门控。正式逐包传播/lock标签未保存，不把源标签套给目标样本。

| signal_to_added_noise_db | 每 seed 两 draw 观察数 | B00 均值 % | B01 均值 % | 增益 pp |
| --- | --- | --- | --- | --- |
| <3 | 1127 | 19.373 | 27.122 | 7.749 |
| 3-8 | 794 | 27.246 | 38.875 | 11.629 |
| 8-15 | 303 | 42.134 | 70.957 | 28.823 |
| >=15 | 1376 | 61.434 | 93.944 | 32.510 |

上述分箱是signal_to_added_noise_db，不是总SNR，也不等于完整接收质量代理。低质量观测改善较小；不由此修改测试SNR、接收状态或删除困难包。完整RX/day/TX/状态/质量及margin见[源分层 CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/source/source_strata.csv)。原门控/全分支逐包轨迹本轮未保存，不能编造门控因果变化。

## 固定 clean 正确集合与错误归因

各seed固定使用B00 FINAL30 clean正确集合，对B00/B01共用，不按新模型重选。由于本轮从零训练，该参照是同预算控制，不冒称一个共同预训练父模型。

| seed | 固定 B00 clean 正确数 | B00 low 保持率 % | B01 low 保持率 % | 修复/新增错误/净修复 |
| --- | --- | --- | --- | --- |
| 2026092701 | 143216 | 32.807 | 55.633 | 44297 / 8842 / 35455 |
| 2026092702 | 142272 | 40.170 | 55.263 | 33240 / 8850 / 24390 |
| 2026092703 | 142421 | 38.132 | 55.468 | 36166 / 8895 / 27271 |

lowurban净修复三个seed均为正，但仍有约8800新增错误/seed；保持率从32.807/40.170/38.132%升至55.633/55.263/55.468%。因此是总体修复明显大于破坏，不是所有样本一致改善；clean代价仍须独立评价。

## 实际训练状态覆盖

| As row | L batch核对数 | As观察/唯一L | 非空TX×RX×day×scene×state格 | 最少格计数 |
| --- | --- | --- | --- | --- |
| B01-s2026092701 | 6660 | 426240 / 6300 | 810 / 810 | 75 |
| B01-s2026092702 | 6660 | 426240 / 6300 | 810 / 810 | 73 |
| B01-s2026092703 | 6660 | 426240 / 6300 | 810 / 810 | 72 |

三条B01共1278720次As，2430个分seed格全非空，每一batch的scene/state与原reference记录及native汇总一致。三条B00两阶段均As=0。只解码原L、U先跳过且不解封TX，不读取IQ或checkpoint、不生成新信道波形。覆盖核对的是实际命名传播状态，不冒充receiver lock/AGC/噪声波形重放或正式目标状态。

[完整状态格CSV](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/coverage/TX_RX_day_scene_state.csv) / [逐epoch状态格](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/coverage/epoch_TX_RX_state.csv)。

## 完整训练、成本与负结果

| row | 末轮原 L CE | 末5轮 L CE 范围 | 源 V final % | 原 U accepted | grad 均值/最大 |
| --- | --- | --- | --- | --- | --- |
| B00-s2026092701 | 0.2376 | 0.1997 至 0.2376 | 98.359 | 15118 | 1.3956 / 3.2774 |
| B01-s2026092701 | 0.4114 | 0.3733 至 0.4114 | 98.256 | 14568 | 1.4498 / 3.2245 |
| B00-s2026092702 | 0.2195 | 0.1949 至 0.2258 | 98.396 | 14953 | 1.4767 / 3.6583 |
| B01-s2026092702 | 0.3961 | 0.3728 至 0.4034 | 98.326 | 14377 | 1.5247 / 3.5850 |
| B00-s2026092703 | 0.2097 | 0.1897 至 0.2162 | 98.422 | 14949 | 1.4412 / 3.5197 |
| B01-s2026092703 | 0.3771 | 0.3654 至 0.3853 | 98.356 | 14427 | 1.4898 / 3.3369 |

每row6660步、10434条L/U出现记录全量复算，L6300/U56700均覆盖。三条As各426240次训练As，B00均0；U原规则accepted数只作行为记录，不冒充伪标签精度。有效训练loss/gradient及原生全epoch数值均有限，stdout error markers为空；禁用诊断项的null/NaN文本不作训练失败。曲线在预算末期仍有波动，预算耗尽不等于收敛，不延长或用bestV重新选模。

| row | 复用 | native 训练/验证 wall s | 预测 wall s | L NN | 学生 train/eval NN | EMA train/eval NN |
| --- | --- | --- | --- | --- | --- | --- |
| B00-s2026092701 | true | 2146.190 | 48.089 | 5114880 | 6078780 / 918004 | 0 / 963900 |
| B01-s2026092701 | true | 2860.909 | 49.650 | 5114880 | 6078780 / 918004 | 0 / 963900 |
| B00-s2026092702 | false | 2082.770 | 46.634 | 5114880 | 6078780 / 918004 | 0 / 963900 |
| B01-s2026092702 | false | 2818.066 | 46.947 | 5114880 | 6078780 / 918004 | 0 / 963900 |
| B00-s2026092703 | false | 2186.378 | 49.579 | 5114880 | 6078780 / 918004 | 0 / 963900 |
| B01-s2026092703 | false | 2862.159 | 46.552 | 5114880 | 6078780 / 918004 | 0 / 963900 |

每row source训练/验证7960684NN，冻结源读出12600NN，正式预测1176002NN，合计9149286NN；原学生train6078780/eval918004、EMAtrain0/eval963900。共同extra raw控制槽1704960NN已包含于总L5114880NN，不再加一遍。额外U/新教师/头拟合/VJP为0。source wall是两段native训练+验证之和，不含未记录的外部源读出耗时；预测wall单列，不伪称整体端到端耗时。

新增四模型120model-epoch、26640更新、36600218NN（含共享真实CP+四F合成smoke3074）；复用两个模型历史60model-epoch/13320更新/18298572NN单列，不算免费父模型。P43整个六行共享smoke4610未强分摊到所复用两行。本次4模型无外部训练父，own20成本包含在30内。独立source复算32.7418s、final复算23.3373s，0IQ/CP/模型NN/更新；源状态覆盖复算294.0086s，0IQ/CP/NN/更新；pipeline含等待墙钟1786.5231s，不能当成分析计算时间。

失败项明确保留：三个seed中clean只seed01为正（+.0935/−.1536/−.9030pp），源clean约−.1pp门槛未预示seed03的目标clean损失；新A_g仍未证实；lowurban>60未达；suburban距自身clean≤2pp未达；没有完整新A_g×A_s四因子或交互项，不能从两行As对比估计交互。已失败的BG/WHITE/HardLCE/Tukey/AMPM等路线不因本次As增益而复活。当前Wiener相位噪声已在普通增强中，不重复宣称新A_g。

## 实现、验证及实际产物

训练实现位于[As seed package](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_As_replicate49/design.py)；新增model工厂seed/observer/runner/dispatch/report及原source/evaluate继承路径随release记录。独立[分析器](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_As_seed_replication49_results_20261018.py)、[覆盖复算](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_As_seed_replication49_coverage_20261018.py)、[自动等待程序](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_As_seed_replication49_pipeline_20261018.py)和[发布器](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/analysis/cvs_As_seed_replication49_publish_analysis_20261018.py)已执行。训练8项边界/种子测试及AST、真实source CP/四F合成smoke、唯一P0/P1及一次namespace定向修复复核均通过；独立分析16项聚焦合成测试、本地模拟打包两远程AST、一次archive传输对比、remotecompile/import通过。没有新增审批或成员hash。

原训练命令`CVS-RFFI/bin/python -u experiments/cvs_As_replicate49/runner.py`，独立分析命令`CVS-RFFI/bin/python -u cvs_As_seed_replication49_pipeline_20261018.py --output <analysis_auto_r01> --truth <原truth.json> --code-commit 706fc9fedfe7145812331da5dba2ee76892773f8`；CWD分别为已登记训练和分析release，GPU4/5/6/7训练、CPU2线程独立分析、输出独占。未修改其他健康任务。

所有图已逐一读回检查，PNG/PDF真实存在，未提交原IQ/权重/日志/逐包预测：

- [逐seed七视图准确率](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_absolute_accuracy.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_absolute_accuracy.pdf)
- [三seed配对效应](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_paired_effects.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_paired_effects.pdf)
- [源lowurban状态](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/source/source_lowurban_states.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/source/source_lowurban_states.pdf)
- [完整180 model-epoch曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_full_training_curves.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_full_training_curves.pdf)
- [优化与原U行为曲线](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_optimization_curves.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_optimization_curves.pdf)
- [正式各RX表现](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_formal_RX_accuracy.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_formal_RX_accuracy.pdf)
- [suburban绝对差距](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_suburban_gaps.png) / [PDF](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/As_suburban_gaps.pdf)

[全量独立汇总](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/analysis_summary.json)、[逐row全分区](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/per_row_seed.csv)、[修复与新错](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/paired_repairs.csv)、[保持率](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/final/fixed_B00_retention.csv)、[源完整训练](E:/type10-7/local_artifacts/cvs_FcrossRX_As_replication_E30_s0203_20261018_r02/analysis_pipeline_publish_r01/retrieved/source/training_curves.csv)。合法原数据及已暴露基准背景保持，不称新独立盲测。

当前唯一已完成检验的假设：clean保留式真实As能否跨模型seed稳定提供urban适应——六practical均为正、As适应可重复；clean全seed保持则被否定。最小实验是已执行的B00/B01三seed，未以新的目标分数挑seed/配方。下一科学候选尚无新的源域证据，不能把本次基线确认替代完整A_g/As联合目标；任务保持ACTIVE。
