# P45：仅 L 标签平滑的匹配 E30 消融

状态：VERIFIED_RUNNING，两条E30已启动并完成首个epoch；无新正式成绩。实际矩阵唯一登记于 [experiment.json](experiment.json)。联合 Ag＋As 目标仍 ACTIVE；本项不是新 IQ 增强，不把源梯度现象当作训练收益。

唯一问题：保留当前 clean 监督份额时，将有标签 L 的 CE 从 LS=.01 改成硬标签 CE0，能否保留 clean 并改善原完整 practical 适应。P40 的源 L 导数显示 CE01 的反向数混入高置信样本校准目标，而 CE0/默认头间隔没有相同冲突。这仅支持一次损失定义消融；不声称已定位训练主因，不复活已失败的梯度修正、投影、课程、FIR、噪声或一致性方案。

最新完整 FIR 训练报告 20261017-phase1-F-crossRX-spectralIQ-firsttrain-e20plus10-s2026092701-m2-r01 已 ANALYZED：它不再只是冻结输入诊断。其固定新增训练 clean 与 urban 增量失败，继续关闭，不以“尚未训练”为由重开。共享 RX 响应统计在引用对话独立进行，本分支不重复启动或改动其代码/产物。

## 固定模型、IQ、U 与预算

F/SIG_TIME_X3、统一 Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、既有内部门控/参考响应、真实 joint160 与默认 cosine30 正常联合训练。900722 参数/206 state，不换头、不另拟合头、不冻结骨干单训头、不增部署模块。原数据合同 all5RX=[1,3,4,6,8]、day1/2/3、6TX、L6300/U56700 TX隐藏/V27000、split392005、256IQ/25MHz，native perpacket RMS，center=false。没有训练 RX 留出或 L 内部分折。U 真值不解封，V 不反向或更新任何持久状态。

新增 H00=无 As/硬 L CE，H01=有 As/硬 L CE；只读复用 P43 B00/B01 的匹配原 LS=.01 FINAL30 控制，构成 loss定义×As 四格，而非 Ag×As。两个新模型各从零20(13L+7U)＋自身student/EMA续10(10U)，累计E30/6660步，L贯穿30轮；尾段重置 AdamW/lr.0002/WD.0001/cos10，原普通增强固定父E13，原SupCon按累计epoch运行。没有加载旧源权重作为新训练初始化；真实CP烟测仅用合法P43B01，不进入新训练。自身parent加载前核对原roleIDs、无目标接触、scratch来源、fixedfinal和ownEMA。

每步6×128 L NN：原 ordinary RD对权重.5、混合ordinary/As-or-raw RD对权重.5、原raw控制RD对权重.25，总 L尺度1.25。混合槽中64完整As对应总.25监督；clean份额保持。SupCon仍原ordinary128、两层两pass、温度.2、最大.1、warm5/ramp10。所有已生成有标签As参与真标签CE，不按学生高置信筛选。每个RD对共享同一生成IQ。

只在有标签 L hook 的三次 terms 调用中替换 CE，原KL原样复用；finally恢复原函数后才返回。native参数 label_smoothing 始终.01，原U路径、旧门控、ownEMA、普通增强、软监督及其损失保持；无新增 U 前向或教师。native先算CE01，再由既有hook的 total−identity_loss 抵消，只留下新完整L目标，首步烟测检查此值。日志同时保存 CE0、同logits CE01、原KL、加权差与函数恢复标记；不把 CE0 损失数值变小本身当识别改善。native 的 labeled CE 诊断列仍可能记录初始 CE01，实际 L 目标以 hard_L_steps/reference_steps 为准，不混用二者作训练曲线。

practical输入仍raw，经原真实 residual/post_sync，EQ关闭G=1、ref30、reflect、质量适配/相位跟踪/IQ补偿开启。训练low_suburban/mid_urban/low_urban，RX硬件按RX/day/EQ session稳定，传播/噪声/状态/补偿依赖不变、epoch/step刷新。没有新IQ算子或重复叠加普通AWGN/多径。正式测试保持既有clean＋六practical固定IQ，不改SNR或困难样本。

## 固定判断与收尾

source条件全部满足才记本消融资格通过：H00 clean−B00≥.2pp；H01 clean>B00且不低于B01；同一既有V1800、两draw的mid/low四格各比B01≥.5pp；source low_suburban两draw及各源RX clean相对B01不降超过.25pp。全V27000 clean和冻结源panel只读，无梯度/拟合/更新。旧panel fc2.45GHz是已有源代理，训练/正式fc2.462GHz保持，差异披露。任一失败关闭此固定L0控制，不改平滑值、损失权重、RX、seed、轮数或父模型救跑。通过也不证明Ag或跨seed，不能根据目标微增晋级。

每个固定row FINAL30后先源只读、自动完成原7×168000 prediction/logits，再独立CPU truth-last评分。源读取技术失败保留failure并仍执行已登记正式评价；源科学负结果不跳过测试。Agent在source_matrix_decision固定前不读取新正式成绩；逐row结果不改变另一健康训练。保存全部RX/TX/day、macroF1、最差RX、urban均值、low/mid、suburban绝对值/clean差距、同ID修复与新增错误、cosine间隔÷30、源状态/质量分层和loss定义×As交互。现有3个目标仍是clean净正、low>60%、suburban各距clean≤2pp探索目标，不是保证。

## 最小执行与成本

本地7项聚焦检查PASS：CE0数值与解析梯度、原KL相等、退出/异常后U函数恢复、输入/RNG不变、非法shape/标签/alpha拒绝、源条件正反例、checkpoint边界。Torch/NumPy旧ABI警告保留，本测试未走tensor.numpy接口。18个Python文件AST通过，0真实IQ/CP/NN/更新/target。本候选唯一一次限定P0/P1审查PASS，未发现直接缺陷，未读取真实IQ/CP/target或改动文件。launcher首步1个真实合法checkpoint/noquery合成2NN＋两行实际F完整L hook合成1536NN，PASS立即进入训练，没有额外许可产物。

两个新模型共60modelE/13320更新，L NN10229760；每行5114880。原U每行963900个输入，student/teacher实际NN、源健康、冻结源12600NN、测试1176002NN及墙钟分别记录；0新teacher/U前向。P43控制原成本单列为既有成本，不复训。完整训练曲线、全日志与覆盖/配对结果在完成后分析，预算耗尽不称收敛。

普通N607账号；每GPU所有owner合计最多3个实验进程、新slot至少12GB空闲，容量锁保护，不停旧任务。唯一owner codex/root/hardLCE45。不可覆盖run/output、CWD、命令、六类seed和预期artifact见experiment.json。发布入口为 experiments/cvs_hard_lce45/publish.py 的transfer/launch；远端入口 experiments/cvs_hard_lce45/runner.py。使用现有普通账号SSH配置，一次N607 preflight、一次归档SHA比较、一次remote compile；启动后独立PID/CWD/argv/GPU/log增长核实。

只因所属角色/来源、非有限、shape/预算、输出冲突、不可执行或无法闭合合法prediction的技术故障停本run并保留，不自动重试、不按低成绩停健康训练。协议 p2_min_v1/capsule/split 对本Phase1不适用，复用未变VALIDATED_ONCE输入句柄，不建额外SHA、receipt、签名、重复审查或审批。代码/config由实际Git commit固定，发布时记入release_commit.txt和登记；报告文字不成为额外启动门槛。


## 实际发布与首轮读回

执行 commit `62b9bb6247b29fc98bf1379b15f3c262f9a495dc` 已推送并由独立GitHub compare核对一致。普通账号preflight、单归档传输比对与remote compile PASS。真实checkpoint/noquery2NN＋两行完整L hook1536合成NN均PASS后，同一dispatcher直接进入训练。两行各857原生参数，仅candidate_id/output_dir/run_id与对应P43控制不同；native/U平滑仍.01，实际L配置0，scratch无祖先、无外部权重、target_contact=false。

独立读回dispatcher 961519、H00 962003/GPU0、H01 962080/GPU1，CWD/argv/startticks/CUDA归属一致；两日志从6372增至11504bytes，stderr空，无failure。两行均完成E1/222更新/LNN170496；H01实际As14208包，H00为0，额外U前向均0。仅作为发布健康证据，不是源资格或识别收益。后续继续按固定E30/all7，不停健康任务或改配置。详见 [execution_verification.json](execution_verification.json)。


## 完整分析已部署，源响应候选关闭

分析代码`1928b5ec55e584f4a93c77124fe1dfd7541c7e6f`已提交推送并独立OID一致。本地与N607各7项纯合成检查PASS，单分析归档比较及remotecompile PASS，0IQ/CP/NN/更新。仅source模式的分析进程967127已独立核实PID/CWD/argv/startticks并在等待所属训练；尚未读取新正式成绩。它完成后先固定/提交源结论，再启动正式四格复算。详见 [分析范围](analysis_preparation.md) 与 [执行记录](analysis_execution_verification.json)。

同一批模型最新读回：H00 E11/2442更新/LNN1875456，H01 E8/1776更新/LNN1363968、实际As113664包；原dispatcher与两worker均live，无failure、stderr空。未重启或改配置。

引用对话的新固定RX响应检验已 [ANALYZED并关闭](E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-F-RXresponse-sourceCells-manysig-m1-r01/source_analysis.md)：响应SSE2.944634，相对加性控制.732224变差，0/18折与0/5RX满足改善条件。本分支不从该固定模型生成IQ；先均值/逐包归一化的后续归因属于另一个尚未执行的问题，不把本FAIL翻为Ag依据。当前本分支唯一实际候选仍是已启动的L-only CE0消融，它不替代Ag研究；联合目标保持完整、尚未完成。


## E20→自身E10阶段核验（部分矩阵）

H00已完成E20/4440更新，真实默认头随骨干联合训练且head_fit_updates=0、target_access=false；已恢复严格自身student/EMA，source_roles=EXACT_MATCH。优化器state_entries=0、reset=true、lr.0002/WD.0001/cosine10，native/U平滑仍.01，L为0，6L/1.25尺度不变。当前累计E21/4662更新，追加段仍有实际L前向170496，未因label_epochs=0而漏掉L监督。H01当前E16/3552更新，尚未进入续训，不能将H00核验冒充两行均完成。

原dispatcher和worker PID均live，日志持续增长，无failure；源分析PID967127独立核实live、stderr空，继续只等待源完成。正式prediction/scoring均未完成，Agent未读取任何新正式成绩。此次只读元数据核对0IQ/CP/NN/更新，见 [阶段核验](own20_transition_verification.json)。这是已完成阶段与健康等待证据，不是方法有效、收敛或联合目标达成证据。


## 源结论已固定：L-only硬CE候选关闭

在任何新正式数值读取前，独立源分析完成并与原固定criteria六项逐项一致：2 PASS／4 FAIL，SOURCE_REJECTED。全V clean：B00=98.359259%、H00=98.325926%（−.033333pp），B01=98.255556%、H01=98.214815%（相对B01−.040741pp、相对B00−.144444pp）。H01−B01的两draw×mid/low差为[[+.222222,−.111111],[+.111111,+.333333]]pp，0/4达到+.5pp。source suburban及各RX保护条件通过，不能抵消clean与urban失败。

全部四格各30轮／6660步完整日志重算通过，无错误标记；两新行共13320条hardL记录逐步恢复CE0、同权重CE01及原KL，最大误差≤4.7684e−7，全部U函数恢复且0额外NN。源分析墙钟26.975677s，0IQ/CP/NN/更新/target。原B00/B01是复用成本，不冒充新训练或新seed。

关闭这一固定L0损失控制，不扫描平滑值、权重、RX、seed、父或预算救跑，不据新正式微增翻案；不把源H00部分受损视图改善替代本轮完整条件。正式7环境收尾仍照原矩阵执行，其数值尚未读取。完整源证据见 [source_verdict.json](source_verdict.json)。联合Ag＋As目标保持ACTIVE。
