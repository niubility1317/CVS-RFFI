# SIG_TIME＋RDrop：双视图失活配对

状态 RUNNING，四行启动 VERIFIED。当前只优化 SIG_TIME＋RDrop，重点改变 RDrop 两个随机子模型的配对关系。历史原版 E50 三种子 clean 均值84.340079%保留；差分梯度和跨RX混淆证据两项源域资格失败均关闭，不继承其模型，也不据目标成绩挑父模型。

假设：独立 dropout 会在两次前向中同时丢弃部分身份坐标；固定每次失活率，改用不重复丢弃的配对，可能让一致性约束覆盖更互补的特征证据。它也可能制造过强的视图差异，妨碍真实身份学习，因此必须与原版直接比较。可选的连续相关性扫描会引入额外调参；本次只固定一个边界方案并做2×2消融。

第一视图保留原生 Bernoulli dropout。第二视图：第一视图已丢弃的位置一定保留，其他位置以p/(1−p)丢弃；所有13处p<.5，故每个视图边际失活率仍为p，双重丢弃率为0。融合p=.45时独立双重丢弃率理论为20.25%，本方案为0；两视图不一致率从49.5%变为90%。这不是完全相反的两个掩码，也不保证非线性梯度方差更低、KL强度相同或AdamW更新长度相同。

实际13处身份路径为t1/t2/t3（.1）、f1/f2/f3（.05）、freq_stats_proj（.1125）、pa_b1/b2/b3（.08）、pa_proj/pa_stats_proj（.1125）、fuse（.45），包含浅中层与最终融合。仅成对L前向启用；domain骨干、原U、教师、V、推理均使用原生dropout。每处每次仍只消耗一次原生失活随机抽样，增强后的实际L128不重抽；两次结束后掩码缓存立即清空。无新增可训练参数、无优化器额外步、无头拟合。

| 条件 | 第二视图采样 | KL系数 | 损失 |
|---|---|---:|---|
| INDEP_CE | 独立 | 0 | 两次CE均值＋旧U |
| ANTI_CE | 不重复丢弃 | 0 | 两次CE均值＋旧U |
| INDEP_RDROP | 独立 | 1 | 两次CE均值＋原对称KL＋旧U |
| ANTI_RDROP | 不重复丢弃 | 1 | 两次CE均值＋原对称KL＋旧U |

原SIG_TIME、RAW_HZ、普通增强、LS=.01、原6×160 cosine默认头scale30/eps1e-4、AdamW lr.0002/wd.0001、cosine日程、旧门控自身EMA保持。默认头与骨干照原流程联合训练。CE对照不是纯CE基线：它仍保留本次旧SIG_TIME的U机制。四行均从零E20（13L＋7U），固定第一预定义seed2026092701，222步/轮；不选择历史源域checkpoint、不拿RX1训练。

有效fit RX3/4/6/8的L5040/U45360hidden/V21600；RX1在模型、U、教师、calibration、cache和初始化前排除。最终1260合法held L仅fixed final20后用checkpoint内默认头预测一次，并同时保存逐包CE、真类概率、margin，供同row分析；无target或另训诊断头。每行144个fit L anchors由复用helper读取但未用于训练。source V只健康诊断。

科学资格固定：所有行final L CE≤.5；ANTI_RDROP在held RX1错误数严格少于另外三行且mean CE均不高于对照；错误交互项(E_ANTI_RDROP−E_INDEP_RDROP)−(E_ANTI_CE−E_INDEP_CE)<0。后一条避免把单纯双CE采样收益当作RDrop改进。实际13处边际失活率与原p差≤.005，anti双重丢弃严格0，计数与真实4440步一致且缓存为空。PASS仅允许另行登记主实验；FAIL关闭固定配方，不扫耦合、层、alpha、种子或轮次救跑，不接触target。

预算4×20=80 model-epoch、17760更新；每行568320额外L身份前向、396368循环U观察，四行2273280及1585472；held决策5040。另8合成包用于own初始CP无query smoke。每设备最终实现检查1552合成包和每种p的262144掩码元素，0真实IQ/CP/优化/头拟合。此前本地失败诊断不作为性能证据。

CPU/CUDA各58协议负例、5资格否决、198相同初始张量、实际增强IQ复用、浅中层与SIG/defaultW有限非零梯度、掩码边际与缓存清空、U/eval原生行为、教师控制器隔离PASS。独立采样的输出/损失/RNG逐位复现；CPU梯度逐位一致。GPU原版自身重复有约4.47e−8归约波动，最终对照梯度最大绝对差4.47e−8、相对范数<2.76e−7，通过atol1e−7/rtol1e−5且relativeNorm<1e−6；不声称GPU全网梯度逐位一致。

最小发布只执行一次聚焦P0/P1、实际Git提交与远端OID、一个新root、一次ordinary N607 preflight、单archive传输SHA比较、remote compile、worker首步own CP无query smoke后立即继续。全部owner及preCUDA计入每GPU≤3、CPU2/loader0；技术错误保留产物并停pending，不自动重发Popen，不杀健康或无关进程。科学低分不重跑。原始CP、日志与逐包预测不进Git。

环境ordinary N607 CVS-RFFI，CWD PROJECT/releases/cvs_SIGTIME_rdrop_paired_masks_source20_20261015_r01；命令 `CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_masks20/runner.py experiments/cvs_sigtime_rdrop_masks20/configs/matrix.json`；唯一owner `codex/root/SIGTIME-RDrop-paired-masks-20261015`；root `PROJECT/runs/20261015-phase1-SIGTIME-rdrop-paired-masks-sourcefold-s2026092701-m4-r01`。实际提交写release_commit.txt；原模型代码依赖b3ac442e2d33a5b315fb3251c30e0ad7083566bc，仅代码复用，不继承训练状态。预期产物见experiment.json。

文献边界：[R-Drop](https://arxiv.org/abs/2106.14448)已有同包双随机子模型对称KL；[ARM](https://arxiv.org/abs/1807.11143)与[DisARM](https://arxiv.org/abs/2006.10680)已有用于随机二元变量梯度估计的反相关采样。不能把反相关Bernoulli本身宣称创新。本试待证点是固定失活率的成对特征覆盖如何影响RDrop泛化；不声称全球首创，有限检索不能证明不存在相同先例。

<!-- INDEPENDENT_REVIEW -->
一次独立next-run P0/P1审查PASS：掩码阶段与缓存、原生反向、教师deepcopy、U/eval隔离、四行配置、RX1全路径源折与默认头、启动依赖、不可变输出、全部owner共享cap3准入。未重跑全套检查，未增加发布门槛。

<!-- LAUNCH_VERIFIED -->
执行提交 `c803f2b13e74bd6d721c3795e64cb1c12ec377cc` 已推送，远端 OID 独立匹配。单次发布 archive 比较及 remote compile VERIFIED。四行各通过 own checkpoint 无 query smoke，独立进程身份、实际 GPU、配置、角色和后续日志增长均已核实；13处掩码边际、配对计数与缓存为空均已核实。源域训练尚未完成，未访问 target。完整启动证据见 launch_verification.json。
