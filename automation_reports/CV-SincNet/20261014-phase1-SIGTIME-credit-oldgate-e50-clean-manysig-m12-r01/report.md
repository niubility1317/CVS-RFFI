# SIG_TIME 中层排序与实际 AdamW 贡献：首种子完整结果

本轮首种子四组已完成固定 50 轮、默认头 clean 预测及独立 truth-last 评分。CREDIT 的固定实际梯度检查仅 1/4 点非零，未达到预登记至少 2 点条件，因此不扩展其余 8 行。该决定来自源记录；没有技术失败，也不依据目标分数修改矩阵。当前只有 1 个 model seed，不能计算三种子均值，84% 目标未完成。

|方案|clean 准确率|相对 BASE（百分点）|净判对变化|Macro-F1|
|---|---:|---:|---:|---:|
|BASE|83.3625%|+0.0000|+0|0.828965|
|SUPCON|83.3655%|+0.0030|+5|0.828677|
|ORDER|83.2613%|-0.1012|-170|0.827839|
|CREDIT|83.2726%|-0.0899|-151|0.828097|

model seed 为 2026092701；每行 168,000 个固定物理样本，6 个 TX、7 个接收机、4 个日期，仅 clean，原默认 ResidualPhysicalHead 随骨干联合训练。72 个分层混淆矩阵的样本数、准确率和 Macro-F1 已独立复算。SupCon 的 +5 个净判对不构成稳定收益；ORDER 与 CREDIT 在此 seed 均未胜过匹配 BASE。未启动行与完整分层记录见 `result_evidence.json` 和 `metric_records.csv`。

## 机制与源条件

四组均有 624/624 次合法五 RX 辅助调用。CREDIT 接受 78/3,120 个 RX 分项（2.5%），60 次选中梯度非零，全部位于 L 阶段；U 阶段的 236 次调用、1,180 个分项全部未通过接受条件。只有固定 step 4,448 的选中梯度非零，源调用覆盖率、接受量及默认头健康条件均通过，固定多点梯度可用性条件失败。不能把这解释为完全没有学习信号，也不能把同批已见 L 的贡献评价称为留出接收机泛化验证。

16 个固定实际更新点、2,496 次调用的接受判断与状态恢复记录已复算；实际一次更新与临时预测最大差为 2.98023e−8。U 点有 256 个 U 样本及非零 U 损失，辅助项基于完整当时主损失评价。默认头正常联合训练，辅助 W/增益梯度为 None，0 另训头、0 额外真实优化器更新、0 U 真值读取。第 21 轮 CREDIT 对浅层 time_fuse、t1、中层 t2、时序分支的参数增量均非零。所有 16 点的 Sinc 频率参数虽有非零梯度，记录的主更新和辅助增量均为 0；这只证明所检查点的更新现象，原因尚未确认，不能外推全程或直接归因于单位/精度。

完整 200 条源 epoch 记录已检查，实际主损失、L/U 损失、源准确率和学习率均有限；原始遥测的 nonfinite 计数也如实保留在 `source_curves.csv`，不把原始遥测计数报告为 0。实际成本为 4 个 scratch50 模型、44,400 次主更新、2,555,904 个额外源评价 IQ 前向样本、24,992 次临时 AdamW 步和 672,000 个 clean 决定。其余 8 行从未启动；4 个 worker 与 dispatcher 均自然退出。

## 后续依据

本版 CREDIT 停止扩种子，不降低接受容差、不增大 margin 制造梯度、不按此轮目标结果选择性重跑。SIG_TIME 的匹配 BASE 首 seed 为 83.3625%，历史正收益按各自原配方保留，仍不能加总不同路线收益。下一步先用代码和合法冻结参数解释 Sinc 的“梯度存在、实际更新为零”，同时保留此前独立 SincCENTER 正结果；在原因和有效特征更新证据得到核实前，不追加新的主训练矩阵。

本轮为 Phase1 已暴露测试基准研究，不称新盲测、不称部署完成。方案、固定阈值、源条件和实际发布的历史记录如下，保留原貌。

---

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


2026-10-15 实现归因补记：native Sinc／CosFace 的 eval 参数缓存以 tensor._version 为键；旧 `credit.assess` 的 functional_call 使用不同但可能同版本的临时张量，且未逐分支清空缓存。实际 native CosFace 聚焦检查复现了不同 W 产生相同陈旧输出，因此旧贡献链未能保证每次读取当前更新前默认 W。它是贡献选择器的实现限制，不能把旧 CREDIT 的负分数完全归因为抽象跨 RX 训练机制无效，也不代表修正后已有效。旧训练、default-head clean prediction、truth-last score 和 4/12／8 未启动状态保持，不重跑或覆盖。新 `20261015-phase1-SIGTIME-center-heldcredit-sourceL-manysig-s2026092701-r01` 逐次清空恢复缓存，独立 source-only 检查仍为 4 个单元各 0/20 accepted，源条件未通过。

固定 BASE/final50 只读参数分析另核实：primary identity Sinc 的低频名义末轮 moment 位移最大约 3.09e−7 Hz，48 个低频／带宽坐标应用名义位移均无 FP32 可表示变化，25 个坐标 sqrt(v) 不高于 AdamW eps；仅该保存状态的公式分析，不是全部历史步骤回放。BASE 低频参数最大构造差 8.58e−5 Hz，带宽与构造值完全相同；无额外 NN 前向／数据／优化／头拟合。源 CENTER 假设步骤能改变 12 至 13 个核，但并未产生超过预定容差的跨样本 CE 贡献。
