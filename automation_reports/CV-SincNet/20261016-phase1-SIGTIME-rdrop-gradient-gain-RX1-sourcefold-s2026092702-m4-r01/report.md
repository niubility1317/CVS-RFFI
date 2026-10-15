# SIG_TIME＋RDrop：梯度强度自调源折确认预登记

状态PLANNED。上一goal turn为实质进展：四个固定E20及完整日志/CP/默认头源读出已ANALYZED/VERIFIED，代码/报告已push，结果改变本轮行动。本批继续只优化SIG_TIME＋RDrop，保留原双向KL和默认分类头联合训练；数字84%均值里程碑已由原版E50三seed达成，但更有创新性的稳定改进尚未验证，goal ACTIVE。

## 为什么确认这条路线

[上一RX3源折](../20261016-phase1-SIGTIME-rdrop-batch-center-RX3-sourcefold-s2026092702-m4-r01/report.md)主CENTER相对FULL多7错，固定判据FAIL关闭；普通POINT多1错但CE降低，跨折未确认。NORM_MATCH只作为探索性控制：对FULL修正118/新增94，净少24错（+1.904762个百分点），CE−.233749；对POINT净少25错。只有2/6TX、2/3day准确率改善，不能据此声称稳定泛化或对类定制规则。它不改变全空间P的梯度方向，只用中心化C提供当前图中的梯度范数参考。该控制不能事后替代上一主候选，本批是另行固定的源折确认，不救跑CENTER/NULL。

一个核心假设：在SIG_TIME＋原完整RDrop上，以当前特征几何为参考动态调整P的力度，是否比单位权重和单纯较大固定权重更好。对应三条研究选择为：继续直接中心化（已有反例，关闭）、普通POINT（RX1正迹象/RX3未确认，保留对照）、固定原NORM_MATCH动态力度（本批主候选）。不新增其他方向或叠加模块。

## 固定算子与对照

对复用同一实际增强后L128输入的两次原生独立dropout前向，实际部署特征为z1,z2。A=normalize(默认W,eps1e−4)，pbar=(softmax(logits1)+softmax(logits2))/2，T=900×[sum_k pbar_k||A_k||²−||pbar A||²]，double计算后转FP32并detach。P=.5×mean[T||normalize(z1)−normalize(z2)||²/160]。mu=.5×mean(z1+z2)，C=.5×mean[T||normalize(z1−mu)−normalize(z2−mu)||²/160]，mu可微。

gP/gC只对当前实际joint160对(z1,z2)求导，r=stop(||gC||/||gP||)。不遍历至参数、不积累grad槽、不做优化步。两范数都0取r1；gP为0而gC非0则技术失败，r不截断、不加救援阈值。所有组都保留原pairedCE＋full symmetricKL＋旧U，附加项如下：

| 组 | 附加项 | 用途 |
|---|---|---|
| FULL | 0 | 原版RDrop |
| POINT | P | 原单位特征一致性 |
| FIXED11 | 11P | 一次固定较大力度 |
| DYNAMIC（主候选） | rP | 保持上一NORM_MATCH原算子 |

11来自上一RX3 NORM_MATCH所有4440个**训练图**r的均值46868.09065139294/4440≈10.555876，按最近整数取整。常量依据是源训练统计，不由held/target指标拟合，不扫描任何权重。它只提供平均尺度参照，不匹配各自训练轨迹、瞬时参数梯度、clip或AdamW。r只匹配当前图中特征边界的P与C梯度总L2；DYNAMIC的实际方向始终来自未经中心化的P，不能声称其共享平移不变。完整KL/CE对默认W继续更新，辅助项对W直接梯度None。

数学核心直接复用已发布center20.mechanism，CPU/CUDA已验证DYNAMIC与原NORM_MATCH的loss及特征/W梯度逐位一致；新增FIXED11只放大P并保持可导原KL。评估复用既有默认头读出，仅覆盖initialize/evaluate/native_scope，元数据/完成逻辑使用本批设置；过滤复用原生RX1.fold.prepare，避免误走旧RX3 data。CENTER_RELEASE为代码依赖5c60deca3f42c02ebf82f5402a205bd9db81c89f，无发现权重/EMA/optimizer/feature继承。

## 数据、训练、判断与预算

四臂均seed2026092702、scratchE20=13L＋7原U、每轮222步共4440。保留原SIG_TIME RAW_HZ、普通增强、LS.01、默认6×160cosine scale30/eps1e−4、AdamW lr.0002/wd.0001、cosine、旧batch_neighbor门控和自身EMA，无星地增强、增强重抽、额外模型前向/参数/更新。源RX1在模型/教师创建前从训练L、probeL、U、V、calibration五loader以及所有继承路径排除，fit RX3/4/6/8、day1/2/3、六TX，保留原15域静态编号。有效L5040/U45360hidden/V21600，U过滤只读RX不读TX，V仅监测。只有本行从零权重和自身EMA，固定final20学生，不按V最高点或目标选模，不冻结骨干/替换/另训头。

固定终点默认头预测1260原可见RX1 L，保存逐包ID/标签/预测/CE/margin/真类概率及day/TX、MacroF1、配对修正与新增错误。RX1与RX3都属于已有开发源域；新训练折不代表新的研究独立盲测，不声称目标、新TX或卫星结果。额外144fitL anchor由共用helper读取但无训练或拟合，成本披露。源角色原handles复用，不另做数据重验。

**固定主资格**：全部4行finalLCE≤.5；DYNAMIC至少90%实际L步骤辅助项为正；DYNAMIC相对FULL/POINT/FIXED11都严格少错且meanCE不高。数学/协议检查通过后才判断科学资格。FAIL关闭当前固定gain配方，不改ratio/weight/cap/eps/seed/epoch/temperature救援，不把FIXED11等副对照替换主候选。PASS仅支持另行预登记全源E20/E50三模型seed与原版RDrop的正式配对比较，冻结后逐row自动clean预测及独立truth-last；本批无target队列。

4scratch20=80 model-epoch/17760更新；各568320额外L身份包次、396368循环U观察、1260held默认头读出、1404额外L读取（含144unusedfit），合计2273280/1585472/5040/5616。每行首步自己的真实随机初始checkpoint严格加载2合成包noquery smoke，PASS立即继续；nonFULL每步另2次边界autograd，实际walltime单列。不把重复U计数当独立物理样本/模型seed或伪标签精度。

## 执行证据与文献边界

CPU/CUDA各59协议负例、固定科学资格、198同初态、实际CE/logits/nativeRNG、FULL原版梯度及部署t1/t2/SIG辅助梯度/W直接None检查PASS。DYNAMIC原算子逐位等价、FIXED11辅助梯度、解析T、共同平移参考梯度、norm匹配和置换/对称/零视图均通过。各1280合成包次、0真实IQ/CP加载/optimizer/headfit。实际args、runtime配置、源码路由与一份独立P0/P1下一运行审查结果记录review.json。无重复完整审查或额外报告审批。

唯一owner `codex/root/SIGTIME-RDrop-gradient-gain-20261016`，ordinary N607 CVS-RFFI，CPU2/loader0，全部owner/preCUDA每GPU实验进程≤3。CWD `PROJECT/releases/cvs_SIGTIME_rdrop_gradient_gain_RX1_source20_20261016_r01`；命令 `CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_gain20/runner.py experiments/cvs_sigtime_rdrop_gain20/configs/matrix.json`；不可覆盖root `PROJECT/runs/20261016-phase1-SIGTIME-rdrop-gradient-gain-RX1-sourcefold-s2026092702-m4-r01`。实际Git身份见release_commit.txt，模型代码依赖b3ac442e2d33a5b315fb3251c30e0ad7083566bc、center代码5c60deca3f42c02ebf82f5402a205bd9db81c89f及明确FOLD default-head-only trainer。预期逐row初末CP/args/角色/冻结/完整step及epoch日志/rdrop_gain_progress/result/source_readouts和matrix completion/qualification，详见experiment.json。一次ordinary preflight、一个archive一次本地远端SHA比较、remote compile；独立PID/CWD/argv/GPU/log增长。技术故障保留所有产物、停止pending，不重发或停止健康/无关进程。

[R-Drop](https://arxiv.org/abs/2106.14448)和[LR-Drop](https://aclanthology.org/2024.lrec-main.891/)已有预测/隐藏状态一致性；[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)已用梯度量级动态平衡多任务损失。因此“按梯度调权”本身不是全球新颖。本候选待证的是：以同包特征的可微共同中心作为**仅力度参考**，保留未经中心化的方向与有效fullKL，优于一个训练统计固定的大权重。只有固定源折及后续完整验证通过才增加有效性证据；不继承GradNorm或VICReg的任务/防坍塌保证。

REJECTED_EXTRA_GATE：通用brainstorming/spec额外审批不在用户最小流程白名单，作为NONBLOCKING不执行。方案、矩阵、来源、停止规则在本报告，无额外签名/数据重验/审查。goal ACTIVE。


## 启动独立核实

VERIFIED / RUNNING。实际代码提交 `9e3034698962f6a5c61b37f92f63a80a1a82ae03` 已push并独立核对远端OID；一次release archive SHA比较和remote compile均VERIFIED。四行均已通过自己的真实初始checkpoint noquery smoke并完成首轮，独立PID/CWD/argv/GPU及实际compute app匹配，日志均增长，无技术异常。实际RX1全部五loader排除，fit3/4/6/8、L5040/U45360hidden/V21600；旧13L＋7U/LS.01/普通增强等实际args一致。新增信号与梯度数值审计在真实训练通过。全部owner当前GPU compute app数量≤3，preCUDA容量在每次launch前计入。详见launch_verification.json；原始进程/命令/日志与私有运行细节只留local_artifacts。尚无final20结果或目标预测，不能宣称机制有效。
