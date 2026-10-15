# SIG_TIME＋配对RDrop：E20/E50三种子主验证

状态 RUNNING，12行已启动且独立核验VERIFIED。上一轮四条件源域门槛PASS，只决定固定反相关配对可进入本次验证。相对原RDrop只净少错5/1260（修正21、引入16），mean CE下降.0296376；交互项−40包含双CE对照恶化35个，并非主路线少错40个。收益在day3/TX0上不均匀，不能据单个源折宣称统计显著、目标提升或全球首创。

本试不新增假设，固定使用13处原生p的“不重复丢弃”配对：第一视图已丢弃位置第二视图必保留，其他位置以p/(1−p)丢弃，每次边际失活率保持p；只作用成对L。U、teacher、domain与eval保持原生，真实增强后L128复用，不重抽增强。原双CE均值和对称KL不变，CE对照alpha0、RDrop alpha1。具体数学与实际13处位置见[源域结果](../20261015-phase1-SIGTIME-rdrop-paired-masks-sourcefold-s2026092701-m4-r01/report.md)，直接复用其代码提交`c803f2b13e74bd6d721c3795e64cb1c12ec377cc`。

| 条件 | E20三种子 | E50三种子 | 本轮动作 |
|---|---|---|---|
| INDEP_CE | 原TWO_CE | 原TWO_CE | 复用6个完整历史产物 |
| ANTI_CE | 3次从零 | 3次从零 | 新增6个模型 |
| INDEP_RDROP | 原RDROP | 原RDROP | 复用6个完整历史产物 |
| ANTI_RDROP | 3次从零 | 3次从零 | 新增6个模型 |

固定模型seed2026092701/02/03；E20=13L＋7U、E50=33L＋17U，每轮222步、独立从零，不把E20当E50的中间checkpoint。恢复完整源RX1/3/4/6/8与day1/2/3，L6300/U56700hidden/V27000。源折checkpoint的RX训练集合不一致，禁止拿来初始化、教师或EMA。默认6×160 cosine头scale30/eps1e−4与骨干原流程联合训练；RAW_HZ、普通增强、LS.01、batchL128/U256、AdamW lr.0002/wd.0001、cosine、原门控自身EMA、NoExplicitCFO均不变，不另训头。

历史12行已对当前实际config、native args、物理角色和完整预测/评分产物只读匹配VERIFIED；三行来自历史技术恢复root，其余原root，详见historical_input_verification.json与登记reused_fixed_rows。历史原run不完整/观察失败记录仍保留，按实际row定位合并，不覆盖。输入核验未读目标IQ、truth或分数值、未加载模型；复用的是完整固定结果，不从这些模型继承参数。历史原SIG_TIME不加RDrop的对照及全部负结果仍保留在原报告。

所有12行固定终轮冻结student checkpoint，V只作健康诊断。每row冻结后同worker自动clean默认头逐包预测168000个既有物理样本，不等其他row；该row全部clean预测写完后，新CPU进程才连接独立truth评分。只测clean，星地工作后置；既有VALIDATED_ONCE clean capsule、物理ID、split和六TX映射复用。没有新盲测、新TX、卫星或Phase2声明。target结果不会改变任何健康训练、参数、剩余seed、停止规则或选模。

主判据发布前固定：E50_ANTI_RDROP三seed clean均值≥84%；相对同预算原INDEP_RDROP均值严格为正且至少2/3 seed正差；E50三seed平均错误交互项(E_ANTI_RDROP−E_INDEP_RDROP)−(E_ANTI_CE−E_INDEP_CE)<0。E20是预算比较，不根据目标分数挑预算或改配方。完整报告逐seed配对差、样本SD、正差种子数、MacroF1、最差RX/day、修正与新增错误、源曲线和掩码审计。旧预测未保留目标CE，不补做不对称目标CE前向。

成本：新增12次从零，共420 model-epoch、93240更新、11934720额外L身份前向、8164800 U观察、2016000新clean决定/216同row指标。复用12个控制此前也花费420 model-epoch，比较总24行/840 model-epoch/4032000决定/432指标；另单列已完成源探索80 model-epoch，不混作父训练或隐藏。每个新模型骨干仅20或50轮。首步own初始CP无query smoke共24合成包，预测前另24合成包；CPU/CUDA局部检查各1024合成包，0真实IQ/CP/优化/头拟合。

CPU/CUDA各144协议负例、4资格否决、198相同初始张量、真实t1/t2/SIG/defaultW有限非零梯度PASS。复用已验证原算法，不重做不变掩码全套检查。一次主验证新增接线P0/P1审查、实际Git提交与远端OID、一个immutable root、一次ordinary N607 preflight、一个archive的本地/远端SHA比较、remote compile、首步own CP smoke后立即训练，之后核验PID/CWD/argv/GPU与日志增长。无额外权限链或门槛。

环境ordinary N607 CVS-RFFI；CWD PROJECT/releases/cvs_SIGTIME_rdrop_paired_masks_main20_50_20261015_r01；命令`CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_masks_main/runner.py experiments/cvs_sigtime_rdrop_masks_main/configs/matrix.json`；root PROJECT/runs/20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r01；唯一owner `codex/root/SIGTIME-RDrop-paired-masks-main-20261015`。全部owner/preCUDA计入每GPU最多3实验进程，CPU2/worker、loader0，scorer的CUDA为空。技术失败保存产物并停pending，不杀健康/其他owner，不自动重复Popen。低分不重训。代码/config身份在release_commit.txt，初末CP、原始日志/逐包数据留远端，不进Git。

本次目标仍是验证一个具体RDrop训练机制能否稳定改善实际联合特征与默认头泛化。反相关Bernoulli本身已有ARM/DisARM先例，配对覆盖假设不等于降低非线性梯度方差，也没有匹配KL或AdamW位移；有限文献检索不支持全球首创声明。

<!-- MAIN_REVIEW -->
一次主验证新增接线next-run P0/P1检查PASS：全源域预算/seed、3个历史恢复row定位、source/默认头、逐row自动预测与独立CPU truthlast、24行汇总及launch_once/shared cap3/不可变输出。未复审不变mask算法或重跑全套检查。

<!-- MAIN_LAUNCH_VERIFIED -->
执行提交`0e11057172191b61470599bf5a52010bf385ab89`已推送且远端OID匹配。单archive传输比较与remote compile VERIFIED；12行进程身份、实际GPU、首轮日志增长、own初始CP无query smoke、完整源角色、原配方和13处mask审计已核实。全部owner当前GPU进程计数≤3。模型仍在训练，尚无本轮目标结果。见launch_verification.json。

<!-- ROW_REPORTS E20_ANTI_CE-s2026092701,E20_ANTI_CE-s2026092702,E20_ANTI_CE-s2026092703,E20_ANTI_RDROP-s2026092701,E20_ANTI_RDROP-s2026092702,E20_ANTI_RDROP-s2026092703 -->
同row分析新增：[E20_ANTI_CE-s2026092701](row_reports/E20_ANTI_CE-s2026092701.md)、[E20_ANTI_CE-s2026092702](row_reports/E20_ANTI_CE-s2026092702.md)、[E20_ANTI_CE-s2026092703](row_reports/E20_ANTI_CE-s2026092703.md)、[E20_ANTI_RDROP-s2026092701](row_reports/E20_ANTI_RDROP-s2026092701.md)、[E20_ANTI_RDROP-s2026092702](row_reports/E20_ANTI_RDROP-s2026092702.md)、[E20_ANTI_RDROP-s2026092703](row_reports/E20_ANTI_RDROP-s2026092703.md)。这些完整单row结果不构成整批三seed结论，其他健康训练保持冻结方案。

<!-- E20_COMPLETE_BUDGET -->
E20六个新模型已完成同row独立分析，和六个历史控制形成完整三seed次要预算比较，见[E20汇总](E20_summary.md)。E50主矩阵仍在进行，不据E20分数改变任何训练。

<!-- E20_SOURCE_DIAGNOSTICS -->
完整E20源训练损失、KL和U接受量对照见[源机制统计](source_diagnostics_E20.md)，只作描述性源证据，不调整任何当前训练。
