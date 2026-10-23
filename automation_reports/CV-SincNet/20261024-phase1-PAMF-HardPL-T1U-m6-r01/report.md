# 普通硬伪标签加权与 PAMF 监督机制扩展

状态：VERIFIED_HPL2_RUNNING_HPLT1U_QUEUED。153项聚焦检查通过；一次独立P0/P1审查发现启动中GPU进程漏计P1，已修复并定向复查关闭。两方法真实原F21 E30 source-only检查均PASS；HPL2三个正式进程已在GPU5／6／7训练，HPLT1U三个模型等待UVB卡位。尚无新版目标成绩，目标仍ACTIVE。

用户明确要求先检查普通伪标签质量，高则提高权重并叠加监督机制，低则适度提高门控。本版按这一要求固定原普通PL F21 E30父权重，两个独立匹配变体均追加20轮到E50/11100，保留13L＋37LU、原默认头联合训练和原T1有标签六前向。

## 普通伪标签质量

三seed各完整源V27000条，实际原F21 E30学生eval前向、硬argmax、置信阈值；模型/EMA/原型/归一化均无更新，未读取U真值或target。原.95接受率94.9037%/95.1815%/95.1296%，接受后正确率99.8868%/99.8366%/99.9027%，错误29/42/25条。提高到.98接受率92.6926%/93.0556%/92.8259%，正确率99.9680%/99.9403%/99.9721%，错误8/15/7条。按诊断前固定的各seed接受正确率≥99%规则保留.95，避免额外丢弃正确样本。完整分TX/RX/day见source_pseudo_quality.json。来源是源V估计，不能冒称全部历史U伪标签实测正确率或目标效果。

## 两组方法

附属文件source_view_reliability.json来自原RXPSM E30自身EMA的固定同记录三视图诊断，不是普通F21或本版新模型结果，也未用于本版门控判定。物理视图上的教师错误不代表clean伪标签失效；其梯度统计仅为logit层诊断，不是参数梯度或Adam步保证。

HPL2：仅把原普通硬PL masked CE从1提高到2，eval当前学生产生硬标签、阈值.95、分母仍为完整U批。

HPLT1U：同一hard PL门控与权重2；为U生成普通Gaussian.015视图，原生Practical residual/post_sync/noEQ作用于交替半批（low suburban/mid urban/low urban）。ordinary/mixed/raw各两次真实学生前向；ordinary/raw RDrop系数1，As两次CE且独立KL0，LS.01；仅通过门控的样本进入time_down/t2.pool、同伪TX不同源RX的原监督对比。拒绝样本不计CE/KL或对比统计，原网络训练前向行为保留。整体U目标为2×[.5ordinary-pair＋.25mixed-ordinary-pair＋.25As-pair＋.25raw-pair＋.1CRIC]/1.25，使总CE剂量与HPL2可比；LS与RDrop/CRIC属于显式机制差异。U噪声视图不冒称与L原生增强完全相同。

每U记录一次当前学生eval，无EMA、软原型或G训练；HPL2一次学生U前向，HPLT1U六次。父U曝光963900，新20轮1134000，累计2097900（重复使用，uniqueU56700）；HPL2累计学生U视图2097900，HPLT1U7767900。机制组新生成physicalU567000，两次复用形成1134000学生物理视图；训练时间和成本单列。提高剂量可能放大剩余错误，源V高纯度不保证目标泛化收益。

## 固定矩阵与发布

六模型：HPL2/HPLT1U×seed2026092701/02/03，GPU5/6/7一seed一卡，最多两条本任务/card。HPL2接替同seed UDAP全部14项GPU预测完成的卡位；HPLT1U等待同seed UVB及UDAP GPU工作完成、HPL2已派发。计入已注册以及本控制器已派发未CUDA的子进程，含其他owner总数最多3、free≥10000MiB；不停止、重启或热改健康任务。

代码入口control及hard_pl_worker，发布命令见experiment.json；Windows本地Git承载与N607普通账户CVS-RFFI Py3.10/Torch2.1环境，FP32/TF32off。不可覆盖root为PROJECT/releases/20261024-phase1-PAMF-HardPL-T1U-m6-r01及PROJECT/runs/20261024-phase1-PAMF-HardPL-T1U-m6-r01。一次Git代码版本、一次发布archive本地/远端SHA比较、一次远端compile；首GPU顺序检查两方法的真实同seed F21 E30默认头、L768/U256、权重2、六U前向、浅中层真实梯度、As KL0及无EMA，PASS立即训练；probe丢弃，正式各seed重载原F21 E30。连续两次同技术失败停止新派发，健康进程和产物保留；无性能停止或目标反馈调参。

每row冻结最终E50学生后自动PAMF raw及同一固定DURA v1.0各clean＋六practical，共14项168000条预测，完整后独立truth-last。单CPU scorer在UVB终态后串行。本版6row共84个正式预测cell，F20/F21两readout原同seed对照复用。DURA仅冻结参考，G+F20不称全流程L-only。最终报告每景/seed/TX/RX/day、净纠错、接受率及增量/累计时间。raw clean至少+1pp、六景严格+1.5pp须同时超过两对照，DURA不替代raw主目标；目标仍ACTIVE。历史测试暴露如实披露，不宣称新的独立盲测。

预期产物：smoke/state、六行source/segment50/final_ssdg.pth与hard_pl_state.json、source完整记录/step日志、source_V_diagnostic、prediction/source_execution_check及14项决策、独立scores和完整汇总。真实PID/CWD/完整命令/实际GPU/日志增长须启动后核实；本地通过和排队不等于实际训练或收益。


## 发布后实际排队证据

执行commit `8624df76b2c2c8f9c731e246afe0227c9c84f793`；一次发布包本地／远端SHA对照与远端编译PASS。独立读回实际CPU控制器PID／CWD／完整命令、存活状态及同版本state，六row均QUEUED，无本run GPU子进程。源E30真实检查尚待GPU5上UDAP完成其全部14项预测后释放卡位；检查两方法PASS后继续，无需二次许可。原UDAP／UVB及其他owner保持，未将排队当作训练或目标收益。详见queue_verification.json。


## 实际源检查与加权组训练

VERIFIED_HPL2_RUNNING_HPLT1U_QUEUED：原F21 E30真实source-only检查同时覆盖两方法并PASS，probe丢弃。L768／U256、门控.95／权重2、无EMA、默认头真实梯度均正确；机制probe实际执行1536学生U视图、生成128物理样本并复用256视图、ordinary/raw RDrop、AsKL0和浅中层跨RX对比，hooks移除，真实两层梯度有限且非零。

正式HPL2三个worker已分别重载原同seed E30，在GPU5／6／7接替UDAP空位；PID／CWD／完整命令／实际GPU及全部已刷盘step前缀增长独立核实，每卡HPL2＋UVB两条本任务，其他owner保留。三个HPLT1U仍QUEUED，待UVB释放卡位；未冒称六个新模型均已训练。实际执行commit `8624df76b2c2c8f9c731e246afe0227c9c84f793`，详见source_activation.json；目前只有源检查与训练前缀，尚无新版目标成绩。


## HPL2 冻结 E50 的伪标签源域健康检查

三个固定最终学生各覆盖完整源 V27000 条，原默认头 eval 硬 argmax；无 GPU worker、参数／教师／原型／归一化更新、U 真值或目标读取，不改变健康训练或门控。此项只报告该方法的源域健康，不把源精度当作 U 实测正确率、完整本批测试成绩或目标达成。

| seed | E30 接受精度（%） | E50 接受精度（%） | E30 覆盖（%） | E50 覆盖（%） | E50 接受错误数 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 99.886825 | 99.831547 | 94.903704 | 96.740741 | 44 |
| 2026092702 | 99.836570 | 99.846943 | 95.181481 | 96.792593 | 40 |
| 2026092703 | 99.902667 | 99.881167 | 95.129630 | 96.618519 | 31 |

上表门控均为 0.95；0.98 对照及完整 TX／RX／day 分层保留在 source_pseudo_quality_E50_HPL2.json。不依据微小源域变化选 checkpoint 或修改当前配方，完整预登记 raw／DURA 目标矩阵继续。


## 监督机制扩展三种子已实际训练

VERIFIED_HPL2_SCORED_HPLT1U_RUNNING：HPL2 三种子已完成固定双七景预测及独立评分，尚未读取部分目标成绩；HPLT1U 三种子已接续 GPU5／6／7，真实 PID／CWD／完整命令／实际 GPU 与全部已刷盘源 step 前缀增长均核实。实际六次 U 前向、物理半批、RDrop、As KL0、浅中层锚点与计数符合预定；每卡与对应 noUAs 健康 worker 并行，最多两条本任务、含其他 owner 总数不超过三条。源训练未全部结束，全六模型矩阵仍未齐；不把局部完成冒称目标达成。完整前缀证据见 supervised_U_activation.json。
