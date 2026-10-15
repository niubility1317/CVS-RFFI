# E20 真实证据责任：默认头训练机制验证

状态 **PLANNED**。主攻机制是：真实证据加入后若增加身份CE，就把修复梯度送回该component，其他context不承担该项修复。原默认头及所有部署特征仍按主CE正常联合训练；辅助W/3gain停止梯度。创新候选是CVS实际四项加性证据的责任路由，尚未证明完整泛化收益，也不宣称首创。完整公式及等值控制见[源原型](../20261015-phase1-SIGTIME-evidence-responsibility-sourceL-s2026092701-r01/report.md)。84%默认头clean三预定义seed均值目标active，当前未达成。

## 合格源依据与新矩阵

source `20261015-phase1-SIGTIME-evidence-responsibility-sourceL-s2026092701-r01` 执行34b8309871822c2756169f39183133d1a4af839a，24cells完整终态、INITIAL3/4通过、FINAL0/4。初期ROUTED heldACTUAL/PRE_W CE相对ZERO/JOINT均四组下降；不同梯度及真实joint/浅中层改变成立，ordinal37的ROUTED/JOINT cosine .951916超过固定.95，维持失败。晚期不同方向存在但实际identity收益不足1e-5。source只支持新scratch20验证，不支持E20+20或直接宣称DG。源资格没有依据target分数、旧权重最高分或本main结果。

固定三seed2026092701/02/03×四条件，12freshmodels，每20epoch/222steps/4440updates，共240模型轮次/53280更新。先首seed四行源健康及实际信号，再另两seed8行；逐row训练冻结即自动clean预测/独立truthlast，不等全矩阵。

| arm | 训练阶段 | 辅助 | 额外L网络前向 |
|---|---|---|---:|
| BASE_CE |20L/0U，真实关闭U/EMA|无|0|
| BASE_SSL |13L+7U，原oldgate/ownEMA|无|0|
| JOINT_GAP |13L+7U，原oldgate/ownEMA|同loss/mask，四component共同反传|32400|
| ROUTED_GAP |13L+7U，原oldgate/ownEMA|每term只负责component边反传|32400|


所有相同SIG_TIME/CENTER25K、residual_fusion、真实fourterm joint160、原默认cos6×160/scale30/eps1e-4、NoExplicitCFO保留内补偿。AdamW原188参数、lr2e-4/wd1e-4/clip5、cosmin1e-6，L128/U256。三SSL条件保留oldbatch_neighbor/ownEMA .999、confidence.8/neighbor2/strongagreement、quantile.86/lambdaU.16/entropy.01，U强视图原Gaussian保留；普通L增强/星地增强关闭。BASE_CE的U迭代/teacher前向和EMA更新实际关闭，不仅loss权重为0；LengthOnlyUnlabeled只保留原222步预算。

辅助日程在任何main真实结果前固定：epoch1至13每16step，共180calls，每call当前四L180bank之一循环，720unique current、每bank45calls，总32400网络packet前向/active模型。900L载入含held180，但held绝不参与main梯度；不做current主L排除，允许同一合法训练包重复曝光，不能称独立heldout。模型eval额外capture、保留rawfeature梯度、恢复flags/RNG/caches、不进入native Lcontext、不更新teacher。只在EMA deepcopy后绑定callback。后7epoch辅助不运行，原半监督继续。基线不建立bank、不执行额外IQ任务；成本差明确披露。两辅助arm标量、active、分母、calendar、bank和读出相同，差异是component边的梯度去向；共享上游参数仍可收到来自不同组件的梯度。

四actualstep探针16/896/1776/2880在main一次真实更新前对比base与base+aux临时AdamW，auxW/3gainNone，main默认W/gain正常。实际与shadow逐element atol2e-7/rtol2e-6对照，保留额外8tempsteps/model成本，不把大unused参数的全局容差当紧精度，也不声称bitwise。实际步不增加；完整step/epoch损失与branch/RX/active/真实Sincinitial-final保留。探针均在label阶段，不能声称实际观测了U阶段辅助更新；后7epoch辅助实际关闭，U完整曝光由native字段和计数核对。

## 实际参数差异与历史边界

本地检查发现parent.make_args实际返回label_smoothing=.01，虽旧config.execution_overrides声明0。只读核对历史sharpness20全部15行actual_native_args均为.01；native train_ssdg.py9102/9104使用该实际参数计算L CE。完整[只读证据](historical_smoothing_readback.json)保留。旧产物不覆盖、不重训、不倒改历史配方。新入口明确设置a.label_smoothing=0，四arm统一，make_args/配置及未来actual_native_args/provenance都检查0；本矩阵因此拥有新的匹配CE/SSL对照，不把旧M15分数当LS0对照复用。source INITIAL是untrained自有随机快照，资格不受父late训练的平滑设置影响；source FINAL实际来自.01模型且0/4，保留这项限制。普通phase1数据角色/物理ID未变，超参差异不是数据契约污染；preflight旧CP仅作独立shape/noquery smoke，绝不进入新scratchworkers。

L6300/U56700hidden/V27000、RX1/3/4/6/8、day1/2/3、六TX及原physicalsplit不变。VALIDATED_ONCE数据handles复用，训练对照实际roles严格匹配；不读取U TX或以V更新梯度/EMA/原型/持久统计，不拆V。每个新模型从零构造，ownEMA只复制自己的新学生。四arm初始化代码/参数结构相同，bank构建保留RNG，训练中的额外eval也保留主随机流；实际初始state可从各自initial_smoke对照，未读取前不宣称实际跨arm逐tensor一致。

## 源扩种子、冻结与测试

首seed四行均final20/4440更新、LCE≤.5；BASESSL/JOINT/ROUTED的源V整体/最差RX较BASESSL下降不超过.01/.02；active两臂scheduled180/eligible≥95%、至少2个固定探针aux非零、positivegapcalls≥1%。完整sourceproof在任何预测模型CP load前检查ownscratch来源、roles、args、finalfreeze、counter/probe；BASECE/SSL scheduled0仍保留4个零aux真实步probe。只按这些源规则扩02/03，不读取target分数，不因score尚未完成增加扩种子门槛。若科学失败，只保留首seed4行并完成其已授权测试，不把partial当完整三seed。负准确率不停止健康任务。

每row固定final20后，clean168000 originaldefaulthead独立逐包预测全部固定，再freshCPU scorer连接truth；18CM/row、216全12、2016000决定。报告mean/样本SD/MacroF1/最差RXday、ROUTED−BASESSL和ROUTED−JOINT及其他预定配对、完整纠正/新增错误与缺项。无新headfit/RX头/独立身份分类器，源diagnostics只保存冻结机制记录、无额外拟合或forward。该基准已经暴露，不称独立盲测；target结果不影响其他row、参数、stage/parent选择或选择性重跑。研究限Phase1/cleanonly，不转Phase2或卫星部署声明。

run `20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r01`，owner `codex/root/evidence-responsibility-train20-20261015`，actualcode/config commit为release_commit.txt，命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_evidence_responsibility_train20.dispatch`，普通N607/CVS-RFFI Python，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r01`，不可覆盖output `/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-SIGTIME-evidence-responsibility-e20-clean-manysig-m12-r01`。具体12row配置、所有路径、6类seed含义和产物唯一维护于[experiment.json](experiment.json)。动态GPU分配要求≥12000MBfree、每卡≤3TOTAL实验进程含其他owner；不干预旧健康工作。一次focusednegativechecks/新actualmaincandidate唯一P0P1集成审查、单realCP noquerypreflight、单归档SHA比较/compile、PID/CWD/argv/GPU/loggrowth核对后运行。停止规则：Input/role/provenance mismatch, forbidden Utruth/targetfit/headfit, outputcollision, nonfinite, wrongcallback/actualstep or inability to close legalprediction/scoring: stop newdispatch, preserve partial and healthyactive rows, reconcile before retry. Sourcehealth/signal failure freezes firstseed4 scope and still completes their clean predictions/scores; no seed02/03 expansion. No targetscore feedback or healthyprocess termination.

历史结构/CENTER25K正收益与相关负结果保留；数据、模型、原始预测与日志留远端既定路径，不上传Git。代码/config/本报告及必要scalar分析登记按Git交付，sourcePASS、启动或本地检查不等于84%完成。


发布前聚焦验证已PASS：48个matrix负例、144个source/recipe/CP/probe负例与14个完整矩阵负例；四臂完整4440 callback/late synthetic日程、16次mock实际AdamW/32temp shadow、原objective四真实探针fixture、auxW/3gainNone、基线零bank/IQ/teacher与U隔离、角色/计数/恢复负例均PASS。四个实际CPU模型构造也PASS：各863730总参数/188参数张量/200state张量、原joint160/6class和四项exact，68auxfeature+4protected；对应首seed CPU初始state逐tensor相同。该结论限localCPU构造，不能替代尚未读取的真实CUDAworker初值。新actualmaincandidate唯一P0/P1集成审查PASS，无直接缺陷。源assay既有loss/代码不重复审查。
