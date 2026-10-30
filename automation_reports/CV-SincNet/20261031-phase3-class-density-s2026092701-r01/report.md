# PAMF 预测身份条件密度：三条件 DURA 测试

状态：本地实现及聚焦合成验证已准备，本轮启用该固定候选，尚无真实运行结果。用户优先要求的 `dura-best` 配对矩阵已经完成；后续独立 `mixture-uda` 未取得收益。本次沿用原 prepared 的四列、唯一主和源端校准规则，不消费其目标适配器或按目标测试结果更改候选。实际发布身份和启动状态另记 run 报告。

Run `20261031-phase3-class-density-s2026092701-r01`。唯一主 `predicted_ratio`。检验同一冻结 P1 开放表征中，全六类模式求和是否允许其它旧类提供代偿支持；只改变 known likelihood 的类别条件，不增加网络、训练、统计拟合或目标访问权限。

## 固定候选与来源

四列顺序为 pooled_known、pooled_ratio、predicted_known、predicted_ratio。前两列源侧精确复用 raw-open r01 的 P1_known12/P1_ratio12 控制，actualcode `617681c8b2f24e7d05c1e8f5a48cb74d99a1c8cb`；其源宏 R_U 分别46.888889%/60.888889%，仍未达标。实际开放网络为 signal-open r03 / code `dc01a402582f93e452dbaa3ef59a9fbb44151818` 的 P1 signal checkpoint，不是 raw-open 的 P0 checkpoint。比较实际加载身份时取父 `signal_reference_identity.model_identity`，不能将其与 P0 主 identity 混淆。

原身份仍为同一 Tri-SIG-TIME SSL F0 E50/11100/default6head，源合同392005；独立开放信号232033＋MLP102688＝334721，源端新增2220更新已经完成，本轮新增0。原F0、DURA v1.0及独立P1分支来源和历史曝光如实继承，不用其它模型或更高控制替换。

复用固定 FP64 known90/background32 的11项聚合状态及 shared whitener，不重新读 L/OE IQ 或逐行训练特征来拟合。known mode 的原顺序为 class→RX→day，每类15个，原全类权重1/90；条件 known 只取原F0默认头预测类的15个 mode，权重重新归一为1/15。均值、方差、Jacobian、OE背景与其权重不变，保留每域多模态及体积项，不按 query RX/day 选模式，不用真实 query TX 路由。

设 pooled p_K 是90模式已知混合，p_K(z|c_hat)是预测类15模式条件混合，p_OE是既有32模式背景。四 raw 分数依次为 log p_K、log p_K−log p_OE、log p_K(z|c_hat)、log p_K(z|c_hat)−log p_OE。两条件分数与两旧分数使用同一 unit160/FP64单位化及白化。每条 query 独立，不用跨 query 状态或标准化；已知身份只由原F0输出。

## 三类测试全部接入 DURA

| 条件 | 输入与模型路径 | 固定数据 |
| --- | --- | --- |
| source_seen_receiver | 既有星地 received→原冻结DURA→F0/P1 signal；本轮复用相同预测/特征 | V27000＋source novel3150；Main源5RX/3day |
| clean_DURA | 既有 clean public→冻结DURA v1.0→原F0和同一P1 signal | RX24-1 future Q864 |
| star_DURA | 同角色固定G10星地 public→同一DURA→原F0和同一P1 signal | RX24-1 future Q864 |

clean 也经过 DURA。两个跨RX条件都重新前向原F0，闭集保护基线来自与候选完全相同的 DURA 输出，不复用历史绕过DURA的clean成绩。source/clean/star均实际DURA路径，源侧缓存已含原DURA身份；新Q的G固定R69 v1.0、Graph-RR、FP32/TF32off、reuse=True/graph=True/native=False。DURA参数、BN和推理预算不更新。

源 L6300/OE6134只作为已冻结统计/网络祖先，新读取/拟合0；原Us56700、目标U384、正式RX24-18、本轮新目标适应/发现/注册均0。source V旧六类各4500，只用真实已知y校准两套新阈值；source novel三TX 19-15/19-16/7-1各1050，仅冻结预测后truth-last。源RX索引1/3/4/6/8为1-19/18-2/19-2/2-19/3-19；日期2021-03-08/15/23均已见。四源物理角色共42584互斥，新物理记录0。

两个跨RX public复用现有开发池evaluation角色，RX24-1(index26)源外、future日2021-03-15，U为3月8日且物理互斥。本轮不打开U。每个Q旧六类各96、未知三类各96，仅builder/scorer知道角色和真实TX；预测不读取K/配额。两条件是同物理角色的配对对照，不能当1728个独立物理记录。源NV和开发Q都有历史曝光，不能称新盲测或正式RX24-18确认。clean公共输入为现有phase3_clean_min_v1，星地为现有G10 MIX01公有三文件，匹配capsule/split；复用VALIDATED_ONCE，不重新造数据或全IQ验证。

星地固定G10MIX01：600km/25MHz/2.45GHz、residual/post_sync/noadditionalEQ/refSNR30dB/reflect、郊区城区各0.5及原六面积权重，receiver2027/scene2026103001/channel2026104001，一次输入unitRMS、输出不再RMS。不从clean给星地提供逐条参考、参数或真值。clean仅冻结测试，训练/适应/发现/注册仍要求星地观测。

## 冻结顺序、校准与验收

首先严格加载实际 r03 signal、原 F0 和 DURA，合成G10两行经过G→F0/signal的无query smoke，三个模型state不变；PASS立即继续。父 source11聚合、两旧阈值和全部NN在V前冻结；两新阈值只在V27000上用既定known12 exactFraction common-tail规则确定。两旧τ/raw/margin精确复用。总体＋5RX＋6scene共12组保护真实旧TX宏C_K≥90%、原F0同P1闭集减A_K≤2pp；30RX×scene完整诊断，不把12通过写成42通过。

四套六阈值全部冻结后，才读取任一source novel预测或任一crossRX Q IQ。**三个条件共享同一源星地阈值与模型**，不为clean或目标另拟合normalizer/密度/τ。依次完成source known/novel、clean_DURA、star_DURA四列预测，再唯一独立CPU scorer先重建全部三个条件的raw/margin/原F0身份，之后分别读source novel、clean Q、star Q真值各一次。所有已登记测试均执行，源侧失败不取消跨RX测试；不评分partial，不据Q调参/换主/选模/选择性重跑。

主报逐TX宏 A_K、R_U、H-score，伴随C_K、原F0闭集、损失pp、AUROC/OSCR、逐TX/RX/day/scene和覆盖。source R_U90是开发诊断，source known12是源τ保护，不是取消目标的前置门槛。跨RX clean整体R_U≥95%、跨RX星地整体及预登记六主景R_U≥90%，C_K≥90%、同DURA输出原F0 closed−A_K≤2pp；缺任一必需TX样本则对应值N/A，不能认证通过。当前小Q已知存在场景覆盖不足，必须报告，不能事后重抽或把整体通过当全景通过。

只有唯一主两种跨RX条件和各自所需主景/已知保护/覆盖全部通过，才记development gate通过；仍非完整Phase3成功，因为这是历史开发池、单模型单RX，且未执行无标签适应、发现、未来注册。完整星地跨RX、零目标TX标签DA和未来匿名注册（A_N80%、旧类注册损失≤2pp、后续未知保护）继续ACTIVE，A_N/H_OW本轮N/A。控制不能替主。

## 成本、执行和停止规则

本轮新NN/默认头/signal/projection/G优化更新0，L/OE拟合读取0。新source真实IQ/F/G/signal前向0；两个Q各864，合计G/F0/signal真实前向各1728；首步synthetic各2。重复处理不计新增物理数据。父原F011100、signal特征11100＋2220、MLP2220以及DURA既有训练单列，不因本轮0更新抹去。部署原F0＋一个P1 signal活跃567714，加G算术834665；固定11数组364340B、四套六FP64τ192B。保存全实验和单主部署成本，批量时间不冒充B1。

N607单GPU producer后单CUDAempty CPU scorer；启动核实actualfree≥4096MiB、每卡新增本任务≤2，允许与其它owner共卡，绝不停止/修改/重启其工作。一次actualcode commit、focused解析/条件模式/真值隔离/三模型合成CP、一次直接P0/P1 review、最小登记、不可覆盖root、单preflight/archive SHA/compile、独立PID/CWD/argv/GPU/log增长及终态。非法来源/角色、非有限、输出碰撞或不能产生合法预测时保留，无partial score/自动重启；UNKNOWN先核原PID/root。科学失败不停止健康或其它owner任务。

## 实际发布预登记

Actualcode `4af0668331682636ee212077b38cd912857a1b31`，唯一owner `codex-phase3-class-density-20261031`；四列×三个DURA条件固定矩阵在experiment.json。环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-class-density-s2026092701-r01-release`，不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-class-density-s2026092701-r01`，预测predict、独立score、日志logs。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_class_density_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-class-density-s2026092701-r01-release/runtime.json`。P1 r03原CP/config身份保留dc01；raw父仅取P1控制，P0权重不读。

Focused验证：{"producer": {"tests": 3, "subtests": 4, "seconds": 7.27, "stdout": "run_tests.log"}, "scorer": {"tests": 5, "subtests": 8, "seconds": 7.36, "legacy_numpy_trapz_warnings": 2568, "stdout": "score_tests.log"}, "core_and_launcher": {"tests": 15, "previous_session_console_pass": true, "persistent_stdout": false}, "test_items_total": 23, "additional_subtests": 12, "AST_files": 8, "mirror_matches": 8, "production_unchanged_since_prepared": true, "real_IQ_weights_truth_reads": 0}；本候选一次直接P0/P1审查PASS。单次N607preflight只核父完成/来源元数据、三CP和public文件存在、两个864的capsule/split/count和实际free显存；IQ/private truth实值0，复用VALIDATED_ONCE。

首步actual三checkpoint syntheticG10两行→DURA→F0/P1signal，state保持、PASS立即继续。全部统计NN在V前冻结、全部四τ在任何NV/Q前冻结，三个条件共享源星地τ，两个Q都经DURA且重新算同输出F0闭集。全矩阵pred完成后独立CPUtruthlast；源侧失败不取消两跨RX测试。无U/适应/注册/clean训练或目标参数拟合，缺scene覆盖保留N/A。全部原始IQ/权重/日志/逐行预测不入Git。当前尚未启动／无新性能。
