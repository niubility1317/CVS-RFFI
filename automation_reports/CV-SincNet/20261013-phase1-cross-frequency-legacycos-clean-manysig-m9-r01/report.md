# 跨频特征路线：默认分类头完整结果

状态 VERIFIED / ANALYZED。本轮两项特征结构均低于旧门控＋cos 基线，未达到 clean 三种子均值 84%。分类头为原默认 ResidualPhysicalHead（6×160，scale 30），随骨干联合训练；本结论只使用 ORIGINAL_HEAD。

|结构|seed 2026092701|seed 2026092702|seed 2026092703|均值 ± 样本标准差|与基线配对差|
|---|---:|---:|---:|---:|---:|
|LEGACY50|80.7417%|78.5071%|82.6875%|80.6454% ± 2.0918 个百分点|—|
|POWER_RELATIONS|77.8881%|74.1226%|78.5667%|76.8591% ± 2.3940 个百分点|-3.7863 个百分点|
|COMPLEX_RELATIONS|79.5185%|78.7411%|79.6577%|79.3058% ± 0.4940 个百分点|-1.3397 个百分点|

COMPLEX 相对 LEGACY50 为 −1.3397 个百分点，仅 1/3 seed 为正；POWER 相对 LEGACY50 为 −3.7863 个百分点，3/3 为负。COMPLEX 相对同容量 POWER 为 +2.4466 个百分点，3/3 为正，但仍未超过原基线。复关系与功率关系的差异不能替代与基线的对比。

完整默认头准确率、macro-F1、最差 RX、逐 RX/day 混淆与配对统计见 [results.json](results.json)。默认头 macro-F1 均值依次为 80.1381%、76.1218%、78.7188%；最差 RX 准确率均值依次为 67.8611%、62.2708%、63.6903%。

## 实现、训练与验证

执行代码为 9c171f9f13c7675eb06a4f2fc82d31105919becf；实现与配置在 experiments/cvs_cross_frequency。POWER/COMPLEX 的三条固定跨频边、通道数、卷积与池化容量相同，每个 fθ 新增 45504 个参数。默认分类头对象、形状和 scale 不变，损失保持旧门控、EMA 与 cosine 配方。六个新模型均从零完成 50 轮、11100 次更新，累计 300 model-epochs / 66600 次更新。三 seed 基线是已核实来源的固定 final50，同 seed 权重和默认头预测完整复用，历史 150 model-epochs 单列。

逐行重新读取完整 source proof、450 条 epoch 记录、六个新模型全部 66600 条 step 记录和真实 final checkpoint，确认 epoch 顺序与预算完整、loss 有限、没有非有限跳步、EMA 连续且 U_s 标签隐藏。每个新模型的 9 个新增参数张量均有限且相对 initial_smoke.pt 实际改变；真实默认头零 IQ no-query smoke 通过。主训练确实更新了特征分支，不是只更换分类器。

9 行默认头各 168000 个独立物理包，共 1512000 个 clean 决定，其中基线 504000 个复用。预测固定后由独立 truth 侧评分，本次逐行重新计算默认头 162 条整体/RX/day 混淆、准确率和 F1 记录，与已保存分数一致。所有队列和进程均已结束，失败项为空。源 V 仅检查健康；没有按目标分数调参、选模、重排或选择性重跑。

数据为原 ManySig source split392005，L6300/U56700/V27000，7/63/30；目标为 clean 旧六 TX、7 RX、4 天。训练没有目标访问。本基准已暴露，不声称新独立盲测、真正未见 TX、Phase2 或 LEO 部署效果。原始权重、IQ、预测、训练日志只保留在不可覆盖 run 根；Git 仅交付实现/配置、聚合结果与报告。对应路径和冻结矩阵见 [experiment.json](experiment.json)。

## 历史单独分类头诊断

本轮此前的自动流程已完成 source 全 L 新 cosine 头与原型诊断；最新用户要求到达时这些步骤已经结束，不能写成已取消。历史诊断没有更新骨干或原默认分类头，final checkpoint 保持原值。额外 324 条聚合记录全部保留在 [historical_readout_results.json](historical_readout_results.json)，明确不计入 84% 目标，也不据此宣称特征迁移成功。

后续保留网络默认分类头，仅随骨干联合训练；不再冻结骨干后单独拟合分类头、不再换头或使用 support-only W 训练。后续 Phase1 直接用原默认头推理、独立评分，优化对象继续是浅层/中层和实际 joint160 特征提取器。删除未使用的 support_cosine 接口和新增默认头纯推理入口的政策实现已推送 4a8235a4dab2fc9971e95d7b76ca5a22ff670732。

## 已执行的原预登记（历史记录）

以下文字反映发布当时的实际流程，后续分类头权限以上述最新用户约束为准。

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
