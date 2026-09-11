# CORE90交叉响应复验与正式实验发布

完整实现与全部实验总报告：[Markdown](CORE90_CROSS_RESPONSE_FULL_REPORT_20260911.md)｜[HTML阅读版](CORE90_CROSS_RESPONSE_FULL_REPORT_20260911.html)。覆盖20项实现追踪、10变体、2000轮日志、源机制检验、全部接收机/类别评分、成本、完整配置和证据索引；以该总报告为综合解释入口。

## 最终状态（2026-09-11，14:44，UTC+8）

VERIFIED：全部10项完成E200、final checkpoint、四场景固定预测及独立评分。相对最早三项快照新增7项；相对11:58五项快照新增U3、U4_additive、U4_bilinear、U5、permanent_detach五项。全部2000轮日志和7920000条预测完成核对，1480组评分计数复算一致。完整数据见[十项最终测试报告](completed_detail/all_results.md)、[1480行评分CSV](completed_detail/all_scores_long.csv)及[对照差值](completed_detail/all_contrasts.json)。下文保留各时点历史快照，以本段及最终报告为准。

总体clean最高为U3的79.9268%，三LEO均值最高为U1的63.1215%。U4_additive匹配双因素差值为clean+0.4848、LEO均值+0.1394个百分点，但没有超过最强单任务/采样对照。正式梯度按各变体定义实际激活，未因性能较低修改或重发实验；单seed不作统计显著性或晋级结论。半小时自动任务core90-3已读回确认为PAUSED。

日期2026-09-11。用户授权：再次对照设计报告及计划检查实现和参数/机制激活，验收后发布实验；seed392005；允许突破每GPU两个实验的限制。唯一launch owner为本任务主Agent，其他审查/实现Agent不启动远端进程。

## 预登记矩阵与边界

冻结10个变体：U0、U1、U2、U3、U4_additive、U4_bilinear、U5、Ux、head_only、permanent_detach。每项E200，label130+pseudo70，从零训练，固定final_only。模型seed和本次正式配置的data_seed均392005，所有行共用数据划分/结构采样seed；两个字段仍独立，不使用合成测试的392002。

正式响应梯度门保持源验证16个有效块、误差/常数≤0.95、连续3次，不移植测试用宽松阈值。默认FFT8维、读出16维、双线性rank4、lambda_resp/dec/cross分别0.01、gradient_cap0.1、K菜单[2]、交互重加权0。事件、自相关、IQ统计及增K为已测可选能力，未纳入本次初始矩阵，不能报其默认启用。对照中关闭响应/决策/身份梯度属于定义，不是失活故障。

纯双因素联动使用U4_additive−U2−U3+U1。U4_bilinear相对加性U2同时改变容量，只能作为含容量差异的描述性比较；单seed不作显著性或晋级结论。U5相对U4_bilinear分析调度效果及实际计算成本。

## 输入与来源

远端项目`/home/szu2070436088/2510044040/CV-SincNet`，数据`Dataset_WigSig/ManySig.pkl`，equalized=1。source RX1,3,4,6,8、day1,2,3；target RX0,2,5,7,9,10,11，heldout day0。按现有named seen-day/unseen-day输出联合覆盖目标日期，不把source日期全部排除。L_s/U_s/V=.07/.63/.30、单一V，实际IDs/每格独立K由392005源preflight确认。

没有baseline、resume、teacher或其他外部checkpoint；EMA只继承本run的随机student。source-only scratch checkpoint smoke仅验证保存/严格加载及真实源样本前向，其权重不进入任何正式row。旧CORE90测试选模与污染权重不复用。当前checkpoint输入权限包含实际文件身份、L/U/V物理ID、RX/day、source split角色和配置。

## 验收与发布状态

状态：RUNNING_VERIFIED。设计附件与既有CR01—CR20计划重新对照；本轮独立P0/P1审查覆盖实际配置可达性、输入权限、真实源K、终态prediction/scorer与新launcher。原实现提交54f2d94c7f7841199b1dec880251146b8820700a，本轮发布提交在验收后固定。

本轮发现并修正旧U_s路径仍把真实标签用于pseudo_correct统计的问题：正式交叉响应路径在数据接口返回y=-1，仅保留RX/day/eq/signal物理定位元数据，禁用真标签伪标签正确率统计；真实标签不进入损失或诊断。U0和U4_bilinear实际CUDA两轮验证覆盖label与pseudo阶段，均通过，证据为`analysis/cross_response_unlabeled_recheck_20260911/synthetic_verification_summary.json`。这是合成正确性验证，不是正式性能结果。

发布前另外校正训练与调度器CWD/入口为当前不可变release的CODE路径，防止落入远端项目根目录旧代码；源preflight按TX/RX/day/eq逐格核对K和完整验证块可行性。未修改正式门槛以适配测试。math完成唯一独立P0/P1审查及上述修复的定点验收，结论可发布、无新增确定阻断。最新dispatcher/U_s/analysis回归13项通过，原runtime17项通过；合成source preflight确认L/V各16/16完整块、strict-load前向精确一致及正确梯度路由。U_s两轮验证刻意省略heldout评分，终态HELDOUT_EVAL_INCOMPLETE符合测试边界；本次只证明训练路径通过，四场景闭合由先前完整合成评分测试覆盖，正式结果仍待E200。

本地规范解释器`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe`。N607普通用户szu2070436088，direct SSH preflight已通过；环境解释器`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。不使用管理员、不安装或更改服务。

## 资源、命令与预期产物

初次preflight有34个计算进程，8张3090均有健康任务，单卡显存约6.5–10GB/24GB，磁盘余7.1TB。保留全部现有任务。本次使用GPU UUID绑定，最多每卡新增2行；实时显存不足则排队，启动中的未注册CUDA进程计入预留。用户已覆盖旧的总进程2/GPU上限；不设置无限并发，不因空闲推导其他实验授权。

run ID=`core90_cross_response_s392005_20260911_r1`；输出`runs/<run-id>`，日志`logs/<run-id>`，CWD为本次不可覆盖release。调度器根目录/锁排他创建，既有目录拒绝重新启动。每行保存实际argv、PID、UUID、log路径和状态。启动参数为新dispatcher的`--project-root`、`--run-id`、`--dataset`、`--config`、`--python`、`--devices`和`--seed 392005`，最终命令随发布记录保存。

预期每行有完整metrics、source评估/激活报告、训练状态、final_ssdg.pth、四场景固定预测和独立scorer结果。prediction必须覆盖该行全部注册类和物理记录，预测文件全部固定后才读取truth；准确率不反馈本次训练、阈值、选模、排序或选择性重跑。E1日志不构成完成或性能证明。

## 技术停止规则

协议/数据契约错误、输出碰撞、错误CWD/参数、无法执行或无法生成合法prediction属于技术失败。失败row不自动重启、不覆盖产物、不回退权重。两个相同预测前异常指纹停止派发后续pending行，已运行健康行继续。持续无有效更新、系统性非有限、OOM等需核对所属PID/日志后处理；不会因低准确率、门未自然达标或负收益杀进程。若正式门始终不达标，报告联合梯度未激活，不能放宽本次冻结门槛。

只允许终止验证属于本run的错误进程，不使用广泛pkill，不干预其他实验。启动结果不明先只读核对原PID/root，不重复提交。后续运行完成按同row评分、资源和机制证据判断，单seed不晋级。

## 发布与启动证据

2026-09-11约02:14（UTC+8）完成发布并启动。代码提交`54afc2bce417de321bd639059f8087ab51436f8e`已push至`codex/core90-cross-response-20260911`，独立ls-remote与本地HEAD相同。Git代码归档SHA256为`a96d2f2db7f717d823d646c6df73173028e98980a0494d9f5673b7fa58c82f83`，远端读回一致、定点compile通过。实际训练release为`/home/szu2070436088/2510044040/CV-SincNet/releases/core90_cross_response_392005_54afc2bc`。完整冻结命令见本目录`publication.json`。

真实source preflight为VERIFIED：L_s6300、U_s56700、V_s27000；L/V逐TX/RX/day/eq格满足K=2，源训练与源验证均16/16完整块。候选搜索按冻结上限128执行、并非穷尽搜索。实际源前向4个有效块，响应/决策损失有限；响应头梯度范数0.012250、域分支0.016015、身份响应梯度0，源门保持关闭。随机模型保存与strict-load前向精确一致，没有构建target loader或执行target IQ前向；smoke权重没有作为训练输入。详见`source_preflight.json`。

独立读取/proc和nvidia-smi确认10/10训练PID存在、PPID=1496875、逐字argv（包含空参数）匹配、CWD均为上述release/code、CUDA_VISIBLE_DEVICES与实际GPU UUID相同。每行均`seed392005/from_scratch=true/baseline_ckpt=''`，并有`init=scratch`日志。首次检查器过滤了argv内部空字符串，已修正只去掉末尾NUL，并以只读stdin重取，未变更正在运行代码。核验结果见`startup_verification.json`，原始证据见`inspection_2.json`。

|变体|PID|GPU索引|
|---|---:|---:|
|U0|1497827|1|
|U1|1497828|0|
|U2|1497829|4|
|U3|1497830|7|
|U4_additive|1497831|5|
|U4_bilinear|1497832|2|
|U5|1497833|3|
|Ux|1497834|6|
|head_only|1497835|1|
|permanent_detach|1497836|0|

这属于发布和启动验证，不代表E200完成。首轮结果与实际激活计数见后续启动观察；门未自然满足时不能称身份联合梯度已激活，未进入计划epoch的基线正则项也不能称默认失效。

## 首轮训练观察（02:18，UTC+8）

第三次只读观察确认10/10仍RUNNING，所有日志相对首次观察增长，全部写出E1或E2，实际optimizer_step_applied=1、skipped_nonfinite_grad/loss均0。U0/U1/U2/Ux/head_only已记录E2；其余行记录E1。证据为`inspection_3.json`和`initial_training_progress.json`，只作启动健康判断，未分析或反馈目标准确率。

U4_bilinear的E1响应头/域分支梯度范数为0.011521/0.010726，决策身份梯度16.773690；U4_additive、U5和permanent_detach亦记录各自非零响应与决策梯度。U2仅响应路径非零；U3仅决策路径非零；Ux身份交互梯度11.076079；head_only仅辅助头响应梯度0.012377、域/身份响应梯度0，均符合本行定义。U1有完整块但关闭附加损失，U0关闭交叉响应。

所有启用行当前每批4个完整块、K=2；响应身份联合门仍关闭，共享前端及身份响应梯度仍为0。后续必须由冻结的源验证标准自然开启，不能把当前预热等同于已联合激活。本次未停止或重启任何已有进程；未完成项为E200、终态四场景预测/独立评分与全程激活/稳定性/成本证据。

## 半小时检查（2026-09-11，02:53–02:55，UTC+8）

VERIFIED：10项持续RUNNING，独立/proc完整argv、PPID、CWD、GPU绑定仍匹配，日志均增长。完整读取各行现有metrics及日志，进度为U0 E23、U1 E22、U2 E23、U3 E14、U4_additive E13、U4_bilinear E12、U5 E13、Ux E24、head_only E21、permanent_detach E12。没有Traceback、Error或OOM；最新轮均正常更新且非有限损失/梯度跳步为0。全程曾有少量非有限梯度跳步（每行epoch平均率之和0.081633，此值不是事件数），最低轮更新比例0.959184–0.979592，随后正常推进；不属于持续无更新，不据此重启。

关键变化：U4_bilinear从E6、U2/U4_additive/U5从E7记录joint_open=1及非零身份后部响应梯度。完整历史中共享前端和身份前部响应梯度保持0；head_only/permanent_detach身份响应始终0，符合对照定义。已从“配置可达/预热”进入真实联合梯度激活阶段；这不证明性能收益。保留所有进程，未修复或重发布，继续原半小时检查。

证据：`heartbeat_20260911_0253.json`、`health_full_20260911_0253.json`及读取脚本`check_full_health.py`。全量记录中的非有限跳步汇总为epoch平均率求和，不能当成次数或整段平均率。

## 已完成结果快照（2026-09-11，09:39，UTC+8）

U0、U2、Ux已完成E200、四场景固定预测和独立评分；本次读取三项全部600轮日志及2376000条预测，每项148组评分计数复算完全一致。当前详细测试数据见[已完成三项报告](completed_detail/completed_results.md)及[全部评分CSV](completed_detail/scores_long.csv)。这是固定快照，不代表其他7项此后状态；自动检查继续覆盖完整矩阵。

clean准确率U0/U2/Ux=77.3005/77.8970/78.3717%，三LEO均值61.6369/61.9975/61.9860%。U2的clean未知日期＋未知RX子集从U0的74.0214%下降至71.2071%，雨衰总体亦略退化；Ux的该clean子集74.9143%，但部分LEO子集及类别仍有退化。单seed结果仅描述，不作晋级，不反馈调参或重跑。所有训练继续沿原冻结矩阵执行。

## 新增完成与ETA（2026-09-11，11:55–11:58，UTC+8）

新增U1和head_only完成，共5/10项。新增两项及U0参考均再次逐条读取四场景预测、先验证完整覆盖再读既有truth，148组/项评分计数全部一致；两项E1–E200及stdout完整解析通过。详细数据见[五项测试报告](completed_detail/five_completed_results.md)、[740行评分CSV](completed_detail/five_completed_scores.csv)和`completed_detail/new_prediction_audit.json`。

U1的clean/三LEO均值79.3389/63.1215%，head_only为77.6712/62.7311%。U1四场景总体均高于其他已完成项；U2相对U1的clean/三LEO均值下降1.4419/1.1241个百分点，Ux下降0.9672/1.1355个百分点。目前不支持附加损失优于完整块运行对照的结论，也不能把U1−U0全部归因于单一采样因素；单seed不作晋级。

剩余U3 E164、U4_additive E171、U4_bilinear E169、U5 E172、permanent_detach E162，均处pseudo阶段并正常更新。最近10轮均时给出训练剩余约0.97–1.61小时，更长窗口最慢约1.9小时；考虑终态预测/评分，估计14:00–14:30完成全部闭合，非保证时间。未修复、重启或改变训练；继续半小时检查。
