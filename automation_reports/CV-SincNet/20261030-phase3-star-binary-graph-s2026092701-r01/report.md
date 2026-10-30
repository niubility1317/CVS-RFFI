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
