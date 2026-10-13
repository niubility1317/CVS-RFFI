# 功率化前复频谱关系与同容量功率对照

- run_id：`20261013-phase1-cross-frequency-legacycos-clean-manysig-m9-r01`
- group_id：`cross-frequency-feature-20261013`；类别：`cvs`；阶段：`phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原旧门控/EMA/cos不变，新增类无关跨频点关系分支；POWER/COMPLEX各3scratch50，baseline3三读出完全复用。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

最小预登记（用户路线四）

候选/矩阵：LEGACY50 三 seed 权重与三种固定读出完整复用 20261013-phase1-episodic-feature-legacycos-clean-manysig-m9-r01；POWER_RELATIONS 与 COMPLEX_RELATIONS 各三 seed 2026092701/02/03，从零50epoch×222=11100更新，32label+18pseudo。共6新模型300model-epochs；历史baseline150model-epochs单列。主指标固定COMPLEX_RELATIONS ORIGINAL_HEAD clean三seed均值≥84%，V仅健康，不设性能门槛/排序/选择性重跑。新原型及普通cosine换头是特征诊断，不替代主目标、不冒充Phase2。

结构：原时域/功率/PA/reference路径完整保留；附加整包256复FFT，按每包谱均方功率+eps1e-4归一化。固定k→k+1、k→-k、k→k+16三种边（mod256）。complex取归一化crossproduct实虚部，power取tanh logratio/logmean；两者6观察通道+2固定频率sin/cos，共享Conv8→32→32/kernel3/pool8/Linear256→160/LN，45344参数，外加gain160（初值.01，tanh）共45504。新表征参数221569，类W960不属于fθ。整个新分支输出增量加到实际joint160，旧cos头结构/scale30不变。新分支privateinit=modelseed+51503，两模式state_dict逐参数相同，保留主初始化RNG。共同常数相位数值不变仅指新关系支路；幅度缩放近似不变，不能宣称任意RX/FIR/时移/CFO不变。

训练：完全保留旧rx_day_quantile.86/tau.92-.97、batch_neighbor window2/minconf.8、strongagreement、EMA.999 versioned/no startupaverage、lambdaU.16/entropy.01、labelsmoot.01/ordinaryaug。无satellite、无新的辅助损失/episode或分类器机制。AdamW LR.0002→1e-6 cosine50、WD.0001。新学生与teacher均ownscratch，不用旧weights初始化；最终固定final50。

数据/权限：原ManySig raw/post_sync/noeq，source split392005 L6300/U56700/V27000；source物理RX/day/TX角色未改，U标签隐藏，V不训练，checkpoint来源与继承规则按PROJECT_PROTOCOL核对。target仅既定VALIDATED_ONCE clean168000观测，旧六TX、7RX/4days，训练完全不构造target。该基准已暴露，不声明独立盲测、真正新TX或Phase2。未变数据句柄不追加hash/validator。

读出：新模型featurefreeze后丢弃W，真实joint160与原features逐元素相等；仅L拟合新ordinarycosine（Xavier modelseed+60013，scale30/无bias/eps1e-4，Adam.01/WD0，20epoch×25batch=500Wupdates）和原型；θ追加0updates。所有读出冻结后才打开targetIQ，三列168000决定全部fixed后独立truth评分。baseline三列原NPZ/sourceheads完整复制并验证已完成父记录，不再推理/refit；baseline复用1512000决定，新模型3024000决定，总固定4536000，486同row aggregate指标。完整负结果保留，不以target调参。

发布：本报告与实际code/config一次Gitcommit（publisher按实际HEAD记录release_commit.txt）；无design/sourceplan/report SHA。命令python -u -m experiments.cvs_cross_frequency.dispatch；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_cross_frequency_20261013_r01；普通N607/CVS-RFFI Python/fullFP32。owner codex/root/cross-frequency-20261013；每GPU总实验进程≤3含其他owner、free≥12000MiB、共享capacity lock。单不可覆盖run根 /home/szu2070436088/2510044040/CV-SincNet/runs/20261013-phase1-cross-frequency-legacycos-clean-manysig-m9-r01。首步真实baseline checkpoint零IQ no-query smoke及focusedchecks PASS后立即launch；单archive SHA本地/remote比较、远端compile/preflight、PID/CWD/argv/GPU/loggrowth各一次。

停止/产物：owned数据角色/来源/预算/非有限loss或gradient/模型边界/预测评分技术失败关闭新调度，健康行完成，保留旧产物，不触碰其他owner；无性能stop。逐row sourcefreeze→三列prediction→独立score→report/register/Git。expected paths与matrix见experiment.json；不覆盖已存在root。

Focused证据：relations数学误差≤5e-16，FP32共同相位输出误差complex4.77e-7/power7.15e-7，power同功率异相位相同而complex改变，逐包独立/zero/tiny/weakbins/梯度及11negative PASS。真实scratch骨干接入：featureexport maxdiff0，旧W扰动不影响export、gain0精确baseline、featureparams221569，CE新分支梯度.22824/.23407、gain梯度.83523/.83536，旧classifier联合CE保留，两模式参数完全同初始化，5config negatives PASS。source换头/预算/角色/未固定评分24negatives PASS；无真实IQ/target读入。一次独立下一run P0/P1审查待返回，remote真实权重smoke待publisher首步执行。

REJECTED_EXTRA_GATE：通用design/spec审批不在项目Exclusive Minimal Experiment Workflow的八项白名单内；按现有用户授权与本最小报告执行，不增加审批。

完整合成汇总：9行/9summaries/9paired/486records PASS，发现并修复旧EPISODE arm名称造成的KeyError。code.commit在实际Gitcommit产生前保持null，正式release由release_commit.txt绑定实际已推送HEAD，dispatch在RUNNING登记写入相同code.commit与execution_commit；不伪造尚不存在的自引用Git身份。launch-ready字段检查在实际commit产生后执行。

一次独立P0/P1 review：发现唯一直接P1（汇总遗留EPISODE名称），已修复；仅针对此项复查，真实finish处理九行合成数据、486指标、9paired与analysis生成PASS。最终P0/P1 PASS，无真实数据读取或launch，未增加重复全review。

发布后独立读回 VERIFIED：实际release/code 9c171f9f13c7675eb06a4f2fc82d31105919becf，remote compile/真实baseline checkpoint no-query smoke/relations/integration/readout negatives均PASS。6新scratch行真实PID/CWD/argv/visibleGPU匹配，epoch从2增长到10至12，各stdout持续增长；dispatcher live、pending0、failure0。3baseline行三读出完整复用并已闭环。当前各GPU总实验进程最多2，含其他owner，小于等于3；未热改/停止其他任务。证据保存local_artifacts/cross_frequency_delivery/post_launch_verified.json。当前状态RUNNING，未读新方案targetmetrics，无性能结果。
