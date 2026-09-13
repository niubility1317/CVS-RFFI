# XUC15发布与监控记录

状态：RUNNING / VERIFIED。15/15行已在N607启动，实际PID/PPID/CWD/argv/GPU/scratch全部核对，连续日志增长已验证。每小时监控ACTIVE。正式结果未完成。

唯一launch owner：线程01a096a4-aa1c-7ac3-8a58-414b9ba959cc；实施期间监控不得重复提交。hourly heartbeat `xuc15`已ACTIVE，周期1小时，已读回automation.toml。

## 固定方案及输入

执行包：`experiments/adv3b02_xuc`；配置`configs/matrix15.json`；15行M00—M14不变。M00为本次ADV3B02 CORE90对比基线；M01—M08为grid承载及X/U/原C2消融；M09/M10为原生A1无X/有X；M11为普通CORE90＋原C2；M12为主候选X＋U＋C*＋暴露课程；M13/M14分别去课程/去C*动作。M05保留被动审计。

数据按三个引用线程及实际源码核对：ManySig.pkl、equalized=1；source RX[1,3,4,6,8]、day[1,2,3]；target RX[0,2,5,7,9,10,11]、day[0,1,2,3]。6TX，source90000条，按TX/RX/day/eq内sig_i顺序连续划分L/U/V=0.07/0.63/0.30，即6300/56700/27000。共享source contract由source-only builder记录物理角色，CORE90及原生A1实际loader逐项比较。U接口隐藏TX标签及TX元数据。未改变的数据沿用既有角色配置，不新建数据审批/封存链。

全矩阵seed392005、FP32、scratch E200，不加载任何历史checkpoint。13行CORE90每epoch49次主更新，合计每行9800；原生A1使用U256、每epoch222次，合计每行44400。名义主更新216200次。原生X效应以M10−M09计算，不把原生与CORE90差值归因于X。13行CORE90统一使用独立ticket随机流，保留完整原目标及阶段日程，不声称与历史run逐步随机轨迹相同。

X为clean分支跨RX triplet，λ=.05、margin=.2；U为逐record单位化后K均值的交互项，λ=.01，P4Q4K2×4。真实MixStyle模块消费donor mask；M/lite_d无BatchNorm，冻结BN记N/A。

C*使用source L完整capture容器隔离的恢复fit/monitor，私有online-head副本40步AdamW；fit改善及分组paired bootstrap上界共同判可信。无效恢复标UNKNOWN。校准观测0/250/500、之后3次有效几何确认；动作要求2次唯一新观测、年龄≤10步、冷却250步；明确低lag≤.005且方向失衡才允许EG，.005—.01中间区不误触发。课程窗口≤10epoch并切开阶段边界，权重/mask/channel seed随origin ticket，优化器/EMA/controller按真实已接受更新单次推进。

## 本地验收

16项聚焦测试（最终以xuc_acceptance.xml为证据）：E1/E80/E131完整CORE90目标数值一致；E131 X/U＋U_s＋sat完整EG两次求值；两个X/U梯度非零、4个有效U块；AdamW一次持久更新；唯一票据消费；C*中间区/重复/过期/无效恢复负测；passive审计恢复模型、模式和Python/NumPy/Torch RNG；课程票据样本、场景、mask、seed和阶段权重多重集一致；native参数及实际模块来源；GPU尚未CUDA初始化的子进程名额预留；恢复拒绝健康行重训；15行命令映射。

M00/M08/M12真实入口CPU一步训练均成功，checkpoint由本次synthetic source训练产生，随后strict重建并完成6类forward；无query接触。证据execution_acceptance.json。此处仅证明实现可执行及受控机制路径；正式训练自然触发、长期效果和最终指标仍待正式日志/评分。

一次独立P0/P1审查发现并修复C*中间lag区误触发与matrix rows/runs键错误。恢复入口定点修复：扫描旧run所有残留进程、阶段PID持久化、禁止健康E200重训、冻结预测引用跨恢复保留。修复后的定点复核记入后续发布状态。

|ID|验收要求|状态及证据|
|---|---|---|
|T01|实际L/U/V角色与U隐藏|实现、接口与远端实际source contract通过；native两行EXACT_MATCH|
|T02|完整CORE90目标|E1/E80/E131数值一致通过|
|T03|X/U真实梯度及EG闭包|真实M/lite_d、两个非零梯度、4块、两次完整field通过|
|T04|V2 grid、donor mask、BN|采样与真实mask路径审查通过；BN不存在为N/A|
|T05|原C2|独立原分支；M08真实入口训练及checkpoint通过|
|T06|C*证据和动作|可信/无效/滞回中间区/重复/过期负测通过|
|T07|暴露课程|票据多重集与权重、seed、mask保持通过|
|T08|完整事务|EG一次AdamW、一次票据；EMA/prototype accepted后更新审查通过|
|T09|15行及native|argv/架构/seed/E200/scratch/source-only验证通过|
|T10|scratch、truth-last|本次自生成checkpoint通过；训练不构造target；统一predict独立score已实现|
|T11|远端发布与运行|VERIFIED：15行PID/PPID/CWD/argv/GPU/scratch匹配且日志增长|
|T12|每小时监控恢复|heartbeat xuc15 ACTIVE，1小时一次已读回|

## 发布、运行及恢复

本地Git工作树：E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-xuc-fusion-design-20260913；分支codex/adv3b02-xuc-fusion-design-20260913，起点702c3440。发布提交由publish_xuc_release.py读取实际HEAD并独立比较origin。archive只含执行包，SHA本地/远端比较一次，远端编译一次。

普通N607账号szu2070436088，项目/home/szu2070436088/2510044040/CV-SincNet，Python/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python。release为releases/adv3b02_xuc15_<commit前10位>，CWD为该release。正式run ID：phase1_adv3b02_xuc15_s392005_20260913_r1；权重/状态在runs/<run-id>，日志在logs/<run-id>，dispatcher日志logs/<run-id>.dispatcher.log。preflight synthetic产物在logs/<run-id>_preflight。

实际命令模板：`<python> -u <release>/code/scripts/dispatch_xuc15.py --project <project> --run-id phase1_adv3b02_xuc15_s392005_20260913_r1 --commit <HEAD>`。launcher先做一次M12 CUDA synthetic checkpoint smoke，PASS后直接启动。GPU按当前compute PID与未初始化子进程并集动态分配，每GPU最多2个，启动至少有6500MiB空闲。其他健康任务不改动。

目标包复用runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/target_inputs，已读回168000条、不含labels、IQ文件344064128字节。训练全结束后统一以CORE90通道配置、seed392005、batch256生成每行clean＋三leo_*_weak共672000预测。15行共10080000预测全部固定，再由独立scorer连接target_truth/truth_sidecar.json，逐scene验证opaque ID覆盖。checkpoint/输入只在推理重建时读取；无target反馈调参/选行/重跑。

用户授权的技术恢复：确认明确执行/协议/数值/产物故障，保留全部历史产物及健康进程；本地复现、定点修复、相关验证、提交push并读回后以新run/release恢复。若旧owner/健康行还活跃，先保留其运行，不重复发布。待旧owner退出后以`--recovery-from <旧ID> --retry-rows <失败行>`仅重训失败行；健康E200仅复用于推理。预测/评分故障可空retry-rows恢复冻结产物。接口拒绝健康E200重训，并保留跨恢复prediction引用。无低分驱动重跑，无强制激活，合法排队/自然不激活不算异常。同异常指纹修复一次仍复发则通知用户、停止盲目重启。

完成边界：本次发布任务以正式进程及日志增长读回为RUNNING；15行E200、四场景冻结预测、独立评分闭合后才为ARTIFACTS_COMPLETE，不能提前声称融合有性能提升。每小时heartbeat在无实质变化时安静，完成/失败/修复重发时通知，全部闭合后暂停。

## 发布结果与当前交接

最后读回时间：2026-09-13T02:54:57.159522+08:00。执行提交：`2a71f5d92d8830b9880415f4e45dafa5e6032598`，GitHub分支已独立ls-remote核对一致，无ahead/behind。release：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_xuc15_2a71f5d92d`。dispatcher PID：3316498，PPID=1，实际CWD及argv与发布记录匹配。

正式15行均RUNNING，13个CORE90行已完成E2—E6范围内的epoch，两条native均完成E1；所有行连续进度检查通过。GPU0—7各2个compute PID，其中GPU4的原任务PID612456保留，本任务使用其余15个名额。没有确认的训练故障，无需重发。

|行|PID|GPU|已完成epoch|
|---|---:|---:|---:|
|M00|3316514|0|6|
|M01|3316590|1|5|
|M02|3317046|2|5|
|M03|3317143|3|5|
|M04|3317218|5|4|
|M05|3317294|6|4|
|M06|3317425|7|4|
|M07|3317501|5|4|
|M08|3317574|7|3|
|M09|3317646|0|1|
|M10|3317657|1|1|
|M11|3317668|2|4|
|M12|3317743|3|3|
|M13|3317817|6|3|
|M14|3318296|4|2|

实际source contract已读回L6300/U56700/V27000、6TX、15个RX/day域；M09/M10初始化记录source_roles=EXACT_MATCH。CUDA smoke已从自生成checkpoint strict重建为4×6 logits，source-only。X开启行128个合法锚点，U开启行4个有效块、0个不可用块。该计数只证明路径在真实训练中可用，不能证明性能提升。C*尚处早期校准，未自然触发动作不属于技术故障；不为强制触发调整阈值。

独立P0/P1审查及恢复定点复核全部通过，当前无未解决P0/P1。16项本地测试及3行真实入口/自生成checkpoint检查通过；已完成一份发布归档SHA比较和一次远端编译。

后续唯一监控：heartbeat xuc15，每小时一次，已更新到实际run/commit并读回。主发布已结束，后续由该heartbeat沿本报告执行健康检查和授权的技术恢复；先核实旧owner及所有关联进程，禁止重复提交。原生A1使用final_only，在E200以前没有latest checkpoint属于预期；现有监控helper对超过20KB的native单行telemetry可能标partial_write，此时用完整metrics_epoch.jsonl或EPOCH-END日志确认，不能据此判训练失败。

证据：launch_identity_verification.json、progress_growth_verification.json、remote_source_smoke.json、remote_snapshot_1/2/3.json、automation_verification.json、delivery/landing.stdout。训练尚未E200，prediction/scoring尚未执行；最终评分由dispatcher按已登记链路接续，全部闭合后暂停heartbeat。

## 每小时监控：2026-09-13 03:15—03:16

VERIFIED_HEALTHY：15/15行保持RUNNING，实际PID/PPID/CWD/GPU/scratch核对通过，每GPU仍为2个compute进程，所有行日志相对前次读回增长。CORE90行已完成E29—E35；native M09/M10已完成E12/E13。已扫描全部可用actions/logs/metrics_epoch结构化记录和完整训练日志，CORE90非有限loss/grad与未接受更新均为0，未发现Traceback、CUDA OOM或异常退出。native非有限skip计数未暴露于现有epoch字段，不将缺失计数写成0；完整日志无致命标记且epoch正常推进。native尚无checkpoint符合final_only。

原C2比较分支已自然出现CORRECT，C*未触发仍按既定门槛运行，不为激活改阈值；这不是性能结论。未干预、未重发、未访问target truth。保留每小时监控；证据heartbeat_20260913_0315*.json及heartbeat_scan.py。

## 每小时监控：2026-09-13 04:16

VERIFIED_HEALTHY：15/15行仍在原PID运行，PID/PPID/CWD/GPU/scratch全部匹配，8张GPU每卡2个compute进程，日志较03:15检查均增长。CORE90行完成E91—E107，native M09/M10完成E28/E29。扫描当前全部结构化日志及完整训练日志，CORE90非有限loss/grad、未接受更新均为0，无Traceback或CUDA OOM，无异常退出。native TEST/JOINT-METRIC中的nan是未构造target测试loader时的预期占位，不是训练数值异常；native final_only在E200以前不保存checkpoint也属预期。

保持原训练、参数、seed及数据契约，未修复/重发，未读取target truth。C*无自然动作不作为异常处理。heartbeat继续每小时检查；本轮无需要通知的状态变化。证据heartbeat_20260913_0416*.json。

## 每小时监控：2026-09-13 05:17

VERIFIED_HEALTHY：15/15行保持原PID运行，PID/PPID/CWD/GPU/scratch均匹配，每GPU仍为2个compute进程，所有日志相对04:16均增长。CORE90行已完成E150—E166，native M09/M10完成E42/E43。全部CORE90行已进入含U_s的后段训练，实际日志仍持续推进；扫描所有可用结构化记录和完整训练日志，未发现非有限主loss/grad、未接受CORE90更新、Traceback、OOM或异常退出。native未构造target测试的nan占位及final_only无中途checkpoint按既定边界处理。

仍处TRAINING，尚无最终预测/评分。未停机、未改参、未重发、未读取target truth。继续原每小时监控；本轮无需要通知的故障或任务状态变化。证据heartbeat_20260913_0517*.json。

## 每小时监控：2026-09-13 06:17—06:19

VERIFIED_PARTIAL_TRAINING_COMPLETE：12行（M00—M08、M11—M13）完成E200；独立读取最终checkpoint，确认epochs=200、step=9800、seed392005、FP32、scratch-only、source物理角色EXACT_MATCH、模型状态全部有限且target_contact=false。M14完成E197后继续，native M09/M10完成E55/E57后继续。3条活跃训练进程及dispatcher身份、CWD、GPU绑定核对通过；每GPU最多2进程，其他健康任务未动。

全量可用结构化日志和完整训练日志未发现非有限主loss/grad、未接受CORE90更新、Traceback/OOM或异常退出。M12/M13最终完成记录为CONTROL_NOT_ACTIVATED，两行C*整程0次CATCHUP/0次CORRECT；这是自然激活缺失的科学证据，不属于授权重跑条件，也不能宣称三机制协同已验证。M05本为passive，仅记录审计，不将其无动作视作缺陷。原C2行已有实际CORRECT。

总体仍处TRAINING，等待全部15行最终checkpoint后才进入统一预测与独立评分。未改参、未停健康进程、未重发、未读取target truth。临时只读checkpoint核验脚本首次缺少release导入路径，补上后12行全部通过；该问题不属于实验训练故障。保留原错误输出，后续监控直接使用修正脚本。证据heartbeat_20260913_0617*.json及verify_completed_checkpoints.py。

## 每小时监控：2026-09-13 07:18

VERIFIED_CORE90_TRAINING_COMPLETE：13个CORE90行均完成E200。新完成M14的checkpoint独立读取通过：200epoch/9800主更新/seed392005/FP32/scratch-only/source角色EXACT_MATCH/模型状态有限；前12行已验证checkpoint的size/mtime未变化，沿用06:19结论，不重复加载。M14完整9800条actions及200条epoch记录无未接受更新或非有限主loss/grad。

仅native M09/M10继续训练，分别完成E71/E75；PID/PPID/CWD/GPU/scratch及日志增长核对通过，两行全部可用epoch记录和完整训练日志无致命异常。当前每GPU至多1个compute进程，GPU4原有健康任务保留。M14本来关闭C*动作、仅保留暴露课程，其无控制动作按设计处理。

矩阵整体仍为TRAINING，等待两条原生对照完成E200后统一预测/评分。未停机、未改参、未重发、未读取target truth。只读监控脚本新增--rows筛选，以后仅扫描仍增长的行并沿用未变完成产物的验证。原每小时监控保持ACTIVE。证据heartbeat_20260913_0718*.json。

## 每小时监控：2026-09-13 08:18

VERIFIED_HEALTHY_UNCHANGED_PHASE：13行E200完成状态及最终checkpoint的size/mtime保持不变。M09/M10分别完成E87/E92，原PID/PPID/CWD/GPU/scratch匹配，日志相对07:18均增长。对仍增长的两行扫描完整可用epoch记录和训练日志，无致命异常；native skip计数未提供，不将缺失当0。每GPU至多1个compute进程，调度器正常等待两行完成。

总体仍为TRAINING，尚未进入预测/评分。无异常修复、无重发、无target truth访问；每小时监控保持ACTIVE，本轮无实质状态变化，保持安静。证据heartbeat_20260913_0818*.json。

## 每小时监控：2026-09-13 09:20

VERIFIED_HEALTHY_UNCHANGED_PHASE：13行E200完成状态和最终checkpoint的size/mtime未变。M09/M10分别完成E104/E109，原PID/PPID/CWD/GPU/scratch匹配，日志较08:18继续增长。两行完整可用epoch记录和训练日志无致命异常；未暴露的native skip计数继续记为不可用。每GPU至多1个compute进程，调度器保持正常等待。

总体仍为TRAINING，预测/评分尚未开始。未改参、未干预健康进程、未重发、未读取target truth；每小时监控保持ACTIVE，本轮无实质状态变化，保持安静。证据heartbeat_20260913_0920*.json。


## 用户授权提前测试：2026-09-13 09:45

用户明确要求没有测试就执行测试。对已完成13个CORE90最终checkpoint先行执行同配置四场景预测，13组预测全部固定后独立评分；不等待M09/M10，但禁止根据结果改变仍在训练的配置或选择。原调度器和训练保持运行。独立evaluation_id=phase1_adv3b02_xuc13_early_eval_s392005_20260913_r1；输出/日志分别位于N607项目runs/和logs/同名目录，预测入口及模型实现继续使用不可变release adv3b02_xuc15_2a71f5d92d。新增控制脚本test_completed_rows.py负责最多4个推理进程，沿用每GPU最多2进程的资源检测，拒绝覆盖输出。原调度器后续仍会运行完整15行测试，早期结果与该链路分别标记，不覆盖或合并不一致数据。

技术失败保留全部产物和活跃PID，不盲目重启；本次不加载其他checkpoint，不重建数据，不修改训练进程。验证包括冻结模型与source角色负测、脚本编译和独立P0/P1审查。早期评分结果不得反馈本矩阵训练/选模/重跑。


提前测试发布VERIFIED：控制提交0e03073d6b9867ede89f47f89f14111eb34c4299已push且独立OID匹配、ahead/behind=0/0；单文件传输校验/远端编译通过，冻结角色正负测及P0/P1定点复核通过。提前测试owner PID3490321，首批M00/M01/M02/M03分别PID3490335/3490348/3490361/3490374，GPU2/3/5/6，CWD及argv已独立核实；原dispatcher3316498、训练3317646/3317657和其他健康进程保留。预测中，尚无score。证据early_eval_launch.json、early_eval_identity.txt。

自动任务xuc15本轮读回为PAUSED（updated_at=1789263569376），保留该实际状态，未擅自恢复；已通过automation_update把提前测试追踪路径加入原prompt。后台测试本身独立运行，不依赖heartbeat。
