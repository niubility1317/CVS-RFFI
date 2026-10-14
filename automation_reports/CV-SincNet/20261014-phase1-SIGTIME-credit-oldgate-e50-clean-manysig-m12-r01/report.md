# SIG_TIME训练机制匹配矩阵预登记

唯一main训练矩阵 `20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01`：BASE、普通SupCon、停止anchorORDER、实际AdamW跨RX增量CREDIT，各固定seed2026092701/02/03。全部相同SIG_TIME架构与native默认6×160 cosine头，freshscratch50（33L＋17U），每模型11100实际更新。原L/U sampler、oldbatch_neighbor/ownEMA/entropy/cos、NoExplicitCFO和无星地训练保持；不堆叠PA、R/E、Fishr/MixStyle或Sinc。旧正结果各自配方保留；不据目标挑父权重，0checkpoint继承。代码/config身份release_commit.txt（提交后记录）。

预先固定aux margin/coefficient/SupCon temperature=.1，normalizeeps1e−4、贡献容差1e−6；warmup5轮，第1111步以后every16，共624scheduledcalls/model。输入为同一次Lforward的实际pre-gain160；currentL positive/negative stop，anchor graph，full合法triplet平均含满足0项。只有visibleL真标签，U旧配方保留。缺源RX则整次辅助skip并记录，不改sampler或重新归一化。

Lhook只返回0；完整nativebase（含当时U/entropy）组装后，按同actualAdamWstate分别clip5临时比较base与base+RXaux，samepre-step默认W在otherseenL RX评估。平均CE改善>1e−6且各otherRX>=−1e−6才接受。所有4控制同cohort、gradient/shadow/extraevalforward日程，BASE0、SupCon普通loss、ORDERungated、CREDITacceptedRXsum；只late加一次并真实更新一次。固定步骤1120/4448/7776/11088验证真实梯度、W/gainNone与shadow/actualstep。额外sourceevalIQ和10temporary步骤/call＋8probe步骤逐模型记录，不声称零开销。

首seed4行全部sourcefixedfinal50；LCE<=.5、相对本次BASE源V下降<=1个百分点/worstRX<=2个百分点、eligible>=95%、ORDER/CREDIT至少2实际梯度probe非零、CREDITacceptedgroups>=1%才source扩展其余8行。规则仅源证据，目标分数不参与排序/改参/重跑。每row固定defaulthead自动clean168000预测，完整后独立CPUtruth-last评分，不等待全批、不再等用户许可；216CM完整复算。原benchmark已暴露，不能称新盲测。科学失败保留已训练首4行及其预登记测试，剩余如实pending。

普通N607项目Python，CWD `releases/cvs_SIGTIME_credit50_20261014_r01`；发布 `python -X utf8 E:/type10-7/.codex_tmp/native_polyphase_runner.py experiments.cvs_crossrx_credit50.publish --output <local_artifacts/cvs_SIGTIME_credit50_20261014_r01>`。每rowcommand见experiment.json，输出唯一 `\home\szu2070436088\2510044040\CV-SincNet\runs\20261014-phase1-SIGTIME-credit-oldgate-e50-clean-manysig-m12-r01`；已有root不覆盖。GPU共享lock，所有owner总进程每GPU<=3、free>=12000MB，GPU按实际容量分配并记录。一次realcheckpoint无querysmoke＋聚焦negative/actualcallbackchecks后直接启动，单archiveSHA/compile、独立PID-CWD-argv-GPU-loggrowth。仅预定确切runowned系统技术失败停止，不影响健康旧任务，保留partial，不自动重启。

预期checkpoint/完整step和epoch/source机制/4actualupdates/默认头prediction/独立metrics与rowcompletion/fullanalysis。native源码已兼容旧callback，只optin传Linputs并消费新late辅助字段；所有旧返回保持。首次sourceprobe已证明末期变化低于容差而停止，不能把该证据当早期收益；这轮检验完整训练作用，84%仍active。方法与原负结果见[机制记录](../../../analysis/phase1_crossrx_increment_credit_20261014.md)。唯一独立P0/P1review PASS，未发现直接使下一实际训练或合法预测错误/越权/破坏/不可启动的缺陷；聚焦nativeAST与actualSIG/FakeNative更新checks VERIFED，shadow误差<=2.98e−8。其他非阻塞文档不延迟实验。


## 实际发布与训练启动

VERIFIED：执行提交 `fdf95db20b707c4bebac70323c0937e3b497a203` 已推送，远端发布档案比对、编译、真实 checkpoint 无查询 smoke 和一次独立 P0/P1 检查通过。首种子 BASE、SUPCON、ORDER、CREDIT 均从零启动；两次独立读回确认进程、CWD、命令、GPU 与日志增长，四组已完成前 2 轮。每 GPU 总实验进程数未超过 3。

训练中只检查源域机制和健康状态。固定第 50 轮默认头冻结后逐行自动生成 clean 预测并独立 truth-last 评分；首种子源条件通过才扩展其余 8 行。当前尚无正式性能结论，84% 目标仍在进行。公开启动证据见 `launch_evidence.json`。


## 中期源训练信号

VERIFIED：第 21 轮固定 step 4,448 的 CREDIT 选中辅助梯度范数为 0.01119972，相对完整主损失更新的全参数增量范数为 2.80581e−5。时序分支与实际 t2 参数的增量范数分别为 2.15263e−5、1.06615e−5；实际一次更新与临时预测的最大差为 7.45058e−9。辅助 W/增益梯度为 None，原默认头继续正常联合训练。

截至 CREDIT 第 23 轮的源日志快照，250 次调用均具备五个 RX，75/1,250 个 RX 分项通过固定贡献条件，57 次选中辅助梯度非零。四组全部已记录调用的接受判断、计数、状态恢复与固定实际更新点经过独立标量复算，新增 IQ 读取、网络前向、训练更新和目标访问均为 0。详见 `source_signal_progress.json`。

这些检查点仍位于 L 阶段，当时没有 U 主损失。后续还需固定 U 阶段更新点、第 50 轮源条件和独立 clean 评分；当前未完成三种子矩阵，未确认泛化收益，也未达到 84% 目标。
