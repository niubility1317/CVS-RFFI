# TIME 覆盖范围与分支非线性：最小预登记

run_id：20261014-phase1-time-factorial-oldgate-cos-e50-clean-manysig-m15-r01。状态 ANALYZED。主候选 LONG_SILU，默认联合训练分类头 clean 三固定 seed 均值84%为研究目标，不保证达到。此路线来自用户最新说明，另一对话的归一化恢复/分支竞争实验已自然结束；本轮只改时间特征提取路径。

|条件|三个depthwise分支(k,d)|分支激活|seed/预算|
|---|---|---|---|
|COMPACT_LINEAR|(3,1)/(5,1)/(5,1)|无|2026092701/02/03；各scratch50/11100updates|
|LONG_LINEAR|(3,1)/(5,1)/(5,2)|无|同上|
|COMPACT_SILU|紧凑|每branch后SiLU|同上|
|LONG_SILU|长覆盖|每branch后SiLU|同上|
|OLD_GATE_COS50|历史原TIME|原结构|匹配33+17三seed fixedfinal50与原default决定复用，不初始化新模型|

四新条件均237697 identity参数，比历史原177025多60672；只改变t1/t2/t3，保持原GroupNorm/外ReLU/pool/dropout、PA/frequency/reference、NoExplicitCFO/internal per-packet compensation和默认ResidualPhysicalHead6×160/scale30。初始state参数逐项一致，branch SiLU/dilation不增加参数，初始函数不同。模块私有CPU种子modelseed+62103，不修改主CPU/CUDA RNG；本地真实builder检查4组postbuildCPU RNG一致。历史TIME builder主RNG实测不同，因此不复用TIME_MS50为匹配LONG_LINEAR。历史OLD为外部配方基线，主因果归因仅由新4条件的固定配对检验支持；不把参数核rank当joint160rank。

旧门控＋EMA＋cos配方33label+17pseudo、50轮/222step/11100updates固定；AdamW LR.0002→1e-6 cosine50/WD.0001，旧pseudoquantile.86、tau.92-.97、batchneighbor2/minconf.8、strongagreement、EMA.999/versioned/nostartupaverage、lambdaU.16/entropy.01/labelsmoot.01/ordinaryaug保持，无satellite、无额外loss/headwarmup/单独headfit/prototype/episode/support tuning。默认头只随原损失联合学习，独立head updates0。

先首seed4条件；所有4个固定终轮最后222step labeledCE≤.5且来源/训练技术有效后，才放行其余2seed。阈值仅sourceL；V健康诊断不排名、不更新；target不参与门槛。每个已经训练的row自动冻结并预测/独立评分，不等待整批。若source门槛失败，保留首seed实际结果，报告partial；禁止依据target有选择地晋级或重跑。全矩阵上限12新模型600epochs/133200updates，历史基线150epochs单列复用。原pipeline21新TIME参数tensor实际optimizer步进/EMA非零更新、实际joint160导出/defaultlogits精确相同已在合成IQ验证；这是接线验证，不冒充真实训练收益。

数据角色仍L6300/U56700/V27000，训练池标注10%/全source7%，source split392005、RX1/3/4/6/8、day1/2/3、原旧6TX/物理ID保持；U TX真值−1隐藏，reuseVALIDATED_ONCE。既有真实TIME3 checkpoint各source角色/无祖先/scratch/final50核验后只读核统计及每TX/RX/day按物理ID排序前8个L包，共720/seed。9个实际preGN等效核rank4、外±4能量3.4%-4.8%、doublefold误差≤3.34e-15；全部分支撤除影响joint160与默认true-class margin，但不是独立训练消融，也不推断目标收益。统计见source_time_diagnostic.json。该诊断零参数/buffer/head更新，无U/target IQ。

本轮源机制记录：每epoch t1/t2/t3真实参数运动；33/34各一次源V/RX健康额外前向，共54000V包/新模型，eval/no-grad，并恢复原模式/Python/NumPy/Torch/loader RNG；V数据不进入梯度或选择。U门控原函数调用局部变量只读接收率、预测类别分布与RX/day计数；不打开U真值、不新增U前向。student强视图与EMA弱视图agreement标明视图差异，不能称伪标签准确率。

clean-only固定168000包/row，旧6TX×7targetRX×4day；15rows/2520000默认决定，baseline504000复用。每row预测与模型冻结完成后独立truth-last scorer，270条CM/accuracy/MacroF1及RX/day/TX/三seed样本SD、8固定配对和覆盖×非线性交互；汇总纠错与新增错误。目标结果不得回流结构/参数/父权重/停止/排序，exposedbenchmark非新盲测；无Phase2/newTX/LEO部署声明。

执行：普通N607/CVS-RFFI Python/fullFP32；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_time_factorial_20261014_r01；python -u -m experiments.cvs_time_factorial.dispatch；owner codex/root/time-factorial-20261014。实际Git commit由publisher用HEAD写入release_commit/registration执行记录，代码/config身份不另加hash。root \home\szu2070436088\2510044040\CV-SincNet\runs\20261014-phase1-time-factorial-oldgate-cos-e50-clean-manysig-m15-r01不可覆盖。每GPU总实验进程≤3，计入全部其他owner，共享容量锁/free≥12000MiB。expected paths见experiment.json，native source/step/epoch/源诊断、default_frozen/NPZ/complete/scores/analysis/completion完整保留；代码/config/聚合结果同轮Git交付，不提交权重/IQ/原日志/预测。

发布依次：focused negatives与唯一一次独立下一run P0/P1审查；一个代码/config commit；单release archive一次本地/远端SHA比较、remotecompile与真实baseline noquery checkpoint smoke；PASS即启动。一次PID/CWD/argv/GPU/commit/loggrowth核验。系统技术错误关闭本owner新调度，保留已运行健康row及所有输出；不停止/重启其他owner，无性能失败重试。通用designapproval/hash/seal/report-onlyreview均NONBLOCKING/REJECTED_EXTRA_GATE。

唯一一次独立下一真实run P0/P1审查PASS：TIME整包、shared defaultPipeline、PA参数化source/predict/score/dispatch、历史来源快照、scratch50/33＋17、RNG保持、默认头零独立训练、源首seedCE门槛、U/V诊断、truthlast动态15行、发布依赖/入口核对无直接P0/P1。审查仅只读，没有真实IQ/truth/remote动作；native实际解析为已打包adv3b02_xuc/code/SSDG/train_ssdg.py，15rows/12new、checkpoint路径空、hookNone和ORIGINAL_HEAD一致。focused模型与优化器/EMA/joint160验证PASS，完整15行270指标/8配对/交互/固定重入及16预测负例PASS。未新增设计审批、报告review或额外hash。

实际发布后核验 VERIFIED：执行Git 351e35cd0d298a6832c0865628bba28ab5a14cbe，单archive SHA比较、remotecompile、真实默认checkpoint noquery smoke PASS。首seed4worker PID/CWD/argv/GPU/commit完全一致，epoch0→4、日志增长；4份真实initial_smoke全部200参数/buffer tensors一致。GPU3/4/1/6各1个本run进程，所有owner每GPU最大2；3历史OLD默认决定已经复用并独立评分，failure0。其余8模型等待固定source门槛；暂无新目标分数，不作收益或完成声明。

全部15行已完成并独立核验：主LONG_SILU81.6417%，目标未达；完整源/预测/特征/paired与负结果见final_analysis.md及results.json。原RUNNING/epoch0→4为历史发布证据，当前allworker自然退出，无健康任务待恢复或重复启动。
