# PAMF Phase3 冻结身份的无标签二元图适应

## 预登记（2026-10-30）

Run `20261030-phase3-star-binary-graph-s2026092701-r01`；code `bb9f5823f53bdbba6dd3e65f9f58dfe70bf86cdb`；唯一主DA1_REG1，四态DA0/1×REG0/1。完整机制/数据/G10/祖先/成本/晋级与技术停止定义见`docs/PAMF_PHASE3_STAR_BINARY_GRAPH_PLAN_20261030.md`及`configs/pamf_phase3_star_binary_graph_20261030.json`，矩阵唯一维护在experiment.json。

原F0 E50/11100/default6head/232993、DURA R69/266951以及oldneighborhood/source(codecb58217de28eb2cb5536d2392aef268b79006877)的P0/P1geometry/四个3233读出/6τ全冻结。source当前IQ/逐行feature读取0、F/G/head/readout/τ更新0，不继承旧targetadapter。原source L6300/Us56700/V27000与OE6134成员监督仍是祖先。本次只用U384/Mar8真实无标签池构造unit160 mutual10NN二元图，pos(m>=0,MSP.95)、neg(m<0,MSP<=.5)，一次FP64 solve(I+L+10D)f=tanhm+10s，无神经优化器；任一seed空精确skip=DA0。原P1六类身份在所有状态完全不变，不把gate收益解释为身份学习。

Ugraph/registry全冻先Q864/Mar15，Q每行独立到固定U10NN插值，无Q-Q统计/拟合；明确保存U384 unit160/sigma/f/opaqueIDs的合法无标签memory，source逐行bank0。两DA各自主K component min16/boundary.05/LOO95radius，REG oldgate先行匿名不能覆盖已接受旧label。全部四态prediction固定后独立CPUscore公有gate→Utruth两个Hungarianmaps全冻→Qtruth，AN全arrival含漏发现、unmatched匿名acceptedwrong、主REG0三unknown/REG1later分母区分，缺类场景N/A。目标TX/K/配额输入0，formalRX24-18读取0。

同一个G10 MIX01 received及P1冻结G，source五RX1/3/4/6/8与devRX24-1互斥，receiver2027/scene2026103001/channel2026104001保持；U/Q物理互斥/VALIDATED_ONCE capsule+split原句柄，不重抽信道或物理记录、不新clean路线。不称新RX/devices/Q盲测。主报宏AK/RU/H附CK/F0closed总损/AN/laterRU/REGretain/HOW与各TXsceneRXday/CI和配对。

DA1REG0完整v3七组CK90/RU90/F0总损2pp+positiveAKgain，DA1REG1 AN80/intendedregisteredCK90/laterRU90/F0protect2pp/REGretain2pp及覆盖；DA0绝对pass只诊断，不加source90前置，不用其他状态替换主。科学失败不重启/改τ或K，不自动扩seed/formal；fullgoalACTIVE/finalfalse。

- 环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-binary-graph-s2026092701-r01-release`，owner `codex-phase3-star-binary-graph-20261030`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_binary_graph_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-binary-graph-s2026092701-r01-release/runtime.json`。
- 不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-binary-graph-s2026092701-r01`；13公有predictfields与U/Qtruth/targets严格分离。预期launch/pipeline/configs/logs、predict{smoke,complete,predictions,Q_evidence}、adaptation{complete,U_evidence,graph,DA*registry/diagnostics}及score{scores,complete}/observations。私有IQ/逐行状态/原始logs不提交。
- N607卡0..7最高actualfree>=4096MB，最多2新增own/GPU，本run1predictionworker后独立CPUscore；foreign不停止/改动。单preflight已核source/F/G/capsule/split及实际free13036–22420MB，未读UQ/IQ/truth值。
- 预期realF0full2496/G1248、synthetic4/2、默认头feature2另计，四prior均实际算real4992/synthetic8MLP行，神经驻留512876。U384SPD/图、逐Q距离、匿名聚合、memory数值/IDbytes另记，无B1时延或峰值显存结论。
- 技术fail仅直接协议/祖先/非有限/冲突/不能合法prediction等，保留partial；emptyseed/K0/低分正常结果。UNKNOWN先核原root/PID，不重复launch/误伤foreign。

聚焦集成：26 passed, 632 warnings, 24 subtests passed in 7.51s；8AST/实际runtimePASS，唯一fresh独立直接P0/P1 PASS。一个code/configGit提交和独立remoteOID读回；一次archiveSHA比较/remotecompile；真实F0/G合成G10checkpoint smoke首步在U前PASS立即继续；实际post PID/CWD/argv/GPU/log证据及独立truthlast自动收尾。无额外seal/审批/数据重验。

## 原预测完成、评分前技术失败及单次恢复

原四态预测和真实F/G smoke已完成；独立CPU公有gate因f7.77e-16舍入差拒绝，未读任何truth，原FAILED_PRESERVED保持。模型/阈值/预测和接受符号无变化；一次定向修复在新`20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01`仅补独立CPU评分，不重训或重预测，score code`ca05fae1fbdd9b0174e8fb0804ffc2645c5516c5`。当前尚无本方法正式性能结论，待该恢复完成后同row汇总。

## 实际结果：独立评分已补齐，完整目标未达（2026-10-30）

原producer code`bb9f5823f53bdbba6dd3e65f9f58dfe70bf86cdb`四态预测、U图/registry、真实F0/G合成smoke完整；其pipeline仍FAILED_PRESERVED。首个独立CPU在任何truth前因FP64 exactgate误拒，unit/ID/sigma逐值同，fmaxabs7.77e-16/residualdelta1.776e-15/jointmax4.44e-16、全部预测/closed/接受符号改变0。一次定向修复abs1e-12/rtol0只用于浮点重建，原数值/模型/τ/数据/四态pred全部保持。

新独立CPU恢复`20261030-phase3-star-binary-graph-score-recovery-s2026092701-r01` code`ca05fae1fbdd9b0174e8fb0804ffc2645c5516c5`已VERIFIED/ANALYZED_DEVELOPMENT；公有gate→U两个匿名映射全冻→Qtruth，唯一主DA1_REG1=NO_PROMOTION。scores保留原producer身份，recovery另记执行版本；不把原failedpipeline改称成功。source新IQ/逐行feature0、目标标签/K/配额输入0、Qfit0、formal0、新NN参数/更新0。

| 状态 | 宏A_K | 旧类C_K | 当前R_U | H-score | 未来A_N | H_OW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DA0_REG0 | 65.972222% | 88.368056% | 76.388889% | 70.799458% | N/A | N/A |
| DA1_REG0 | 68.229167% | 90.798611% | 57.291667% | 62.283887% | N/A | N/A |
| DA0_REG1 | 65.972222% | 91.493056% | 65.625000% | 65.798153% | 4.687500% | 12.308736% |
| DA1_REG1 | 68.229167% | 93.750000% | 15.625000% | 25.427019% | 19.270833% | 22.979986% |

REG0 R_U为所有未来非旧6/7/8，REG1主R_U为U未出现later8，不能直接以两态H/RU差宣称注册影响。主intended旧六+全部两到达类接收82.031250%<90，不能以old-only C_K93.75%冒充全注册接收达标。原F0同P1闭集420/576=72.916667%，两DA闭集逐值保持；主开放393/576=68.229167%，纯误拒损27/576=4.6875pp>2，无身份漂移。

DA1REG0相对DA0REG0：旧正确380→393净+13（21得8失），接受509→523净+14（25得11失），未知拒对220→165净−55（2得57失）。旧C_K总体90.798611%通过单项，但R_U57.291667%和原F0保护均失败。later拒对64/96→16/96，REG后15/96；同DA REG旧正确保留393→393损0，但旧接收523→540多17不代表身份改善。未来新类主仅37/192=19.270833%，远低80；U簇数DA0=1/DA1=2由算法自主而非真K输入。

二元U种子正136=128old+8unknown，成员精度94.117647%；负27=17old+10unknown，成员精度仅37.037037%，其中旧class4有12。这些是全部预测/独立评分后truth-side诊断，未回算法。低MSP并不能当可靠unknown标签，冻结identity和精确求解本身不足以保护未知。U图1197边，energy128.742233776→37.923038253、残差1.24345e-14、f在[-1,1]，一次solve数值成功不等于科学成功。

数据沿原source L6300/Us56700/V27000/OE6134监督祖先、五sourceRX与目标RX互斥；本轮当前U384Mar8/Q864Mar15均RX24-1、物理互斥、原G10 MIX01与同一固定received。所有postPhase1新IQ均星地，P0received/P1冻结DURA，旧source原权重祖先保留，不读source逐行bank。已暴露devRX/设备/Q，不是新盲测/未见日期或正式RX。high suburban缺必需unknown6，N/A/覆盖false，不重抽样凑指标。

实际F0 full2500=2496real+4smoke、G1250=1248real+2smoke、额外原head smoke2；四个3233读出每次都存/算，real4992/synthetic8MLP行，神经驻留512876无新adapter。合法U memory数值251904B+IDs59904B=311808B，npz312784B/json488B；新CPU恢复仅公有图重构验证和独立评分，无部署新拟合/原预测变动。源训练/恢复器/聚合缓存祖先成本保留。无B1/吞吐/峰值显存或提速证据。

完整4stdout已读取：原predict931B/driver17B、旧score2804B包含已记录技术traceback，恢复score837B无异常。原predictjob9.218162537s/internal5.381406529s、原失败pipeline12.263982534s，CPU恢复score0.501980066s，不能拼成连续端到端或B1。原driver3286708独立live CWD/argv核实，predict3286801/GPU4与旧score3286873只jobs/log支持，未抓liveGPU；恢复CPU3410600独立live正确CWD/argv/CUDAempty，终态live/GPU空、foreign干预0。

初始26tests/24subPASS7.51s和一次freshP0P1；直接数值修复3受影响PASS5.28s/真实公有gatePASS、CPU恢复执行3PASS4.36s与唯一问题定点复审PASS，不累计成同一次32test全suite。原397file/data0 archive d0244e8fdab1199d1fe4da1193108e731e861aaf3e099d884449c2798d72cc1c；恢复399file/data0 archive8d1417794f1a48914a2faf147b52435c7a6682b329287713dca6d24e1963798e，各一次SHA/compile/runtime核实，无重复训练/预测或新数据。

详细四态/24scene/主逐TX与CI、U组件纯度/匿名映射、数据权限、全部成本和失败恢复证据见`docs/PAMF_PHASE3_STAR_BINARY_GRAPH_RESULTS_20261030.md`。本轮不晋级、调τ/seed/K/复制控制，不扩大正式/多seed；全Phase3 goalACTIVE/finalfalse。下一独立科学问题是可靠unknown种子及跨日类别覆盖，不能把低置信度旧样本直接定义未知。
