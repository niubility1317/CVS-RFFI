# 固定 SIG_TIME：源接收机风险训练首种子对照

状态：PLANNED；实现已验证，尚无新训练或性能结果。代码提交 `341d0dc0bdab44fce88e0af3d8fc29eb50b39912`。

固定模型seed2026092701，三新模型从零训练，各50轮/11100更新（33有标签＋17旧门控半监督）。网络SIG_TIME、joint160部署特征、既有默认cos6×160头及原联合训练不变；学习率.0002、AdamW WD.0001、cos日程、ownEMA.999、UCE.16、entropy.01、smoothing.01均不改。旧门控＋cos历史基线只复用评估，不使用纯CE替代基线。新训练总150模型轮次/33300更新，另列历史50/11100。

| 行 | 源身份监督损失 |
|---|---|
| OLD_GATE_COS50 | 复用原历史默认头，只做固定评估 |
| SIG_TIME | 原始整批CE，直接返回原loss对象 |
| RX_BALANCED | 每个出现源RX的平均CE，再对出现RX等权平均 |
| RX_DRO | 五源RX风险的动态加权平均；logq每步加.01×detached当前RX均值CE，softmax后使用 |

分组只使用合法L_s的RX=1/3/4/6/8，不拟合接收机分类器；缺组时风险为0且logq不增加，应用权重只在本批出现RX上归一化。从epoch1每个原生L步骤执行，不增加采样、增强、模型前向、头、优化器或额外训练样本；q为5个停止梯度的训练状态。默认头随正常身份监督与fθ联合更新，不另训头。U损失返回原对象，不叠加已有Fishr/特征一致性/SAM。

依据：[GroupDRO原论文](https://arxiv.org/abs/1911.08731)与[作者实现](https://github.com/kohpangwei/group_DRO/blob/master/loss.py)的指数群权更新，固定step_size=.01。这里是源接收机分组的适配，不是原论文数据/正则化实验复现。作者明确指出直接GroupDRO可能失效；本轮保留原WD以隔离风险权重作用，不以论文结果预判本项目收益。现有“风险学习”包针对动作选择，不属于此源群监督目标。

数据角色与VALIDATED_ONCE/物理ID/划分沿用原契约：L6300、U56700隐TX、V27000仅健康诊断；不访问U真值，不用V拟合或选模，不新增目标数据权限、星地增强或显式CFO坐标。所有模型固定final50，不用目标或V选epoch。

首seed源证伪规则：三臂最后5轮原始未加权L CE均<=.5；两个候选相对SIG的真实L720部署特征差>=1e−6，并在17/33/50的实际t1/t2/SIG参数上有非零风险增量梯度；RX_DRO最后5轮逐RX汇总训练CE的最差值<=RX_BALANCED；DRO最后5轮mean(max q)>=.2001。精确11100步/963900U、默认头/导出/梯度证据必须满足。通过后才将三臂固定原样扩02/03，先保存并提交/推送源决定，再读取本run新numeric target摘要；失败保留全部固定行结果、不扩展，不按目标调系数/换父/挑seed重跑。该判据只用于源机制证伪，不是泛化证明。

每行训练后直接默认头clean168000预测；全部声明预测固定后独立truth-last评分，无分类头重训。保留完整50轮、所有逐步群风险/q/所用损失、每16步诊断且只在1/17/33/50保存的52梯度批/208层对象记录/624向量（含正常默认W），由独立分析过程复算。三新合计624层/1872向量。没有额外训练前向；source特征检查及smoke属于诊断成本，另列。

普通N607既有CVS-RFFI、FP32、每worker CPU2线程。CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_receiver_dro50_20261015_r01`。入口：`python -u -m experiments.cvs_receiver_dro50.dispatch`。唯一owner：`codex/root/receiver-dro50-20261015`。不可复用输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261015-phase1-receiver-dro-oldgate-cos-e50-clean-manysig-s01-m4-r01`；训练日志`<run>/logs/<row>.log`，本地`E:/type10-7/code/worktrees/phase1_feature_challenge_20261013/local_artifacts/receiver_dro50_delivery_r01`。原共享调度器在GPU0..7中按实际进程数/可用显存分配，每GPU计入所有owner最多3总实验进程，发布时至少12000MiB空闲；不停止或改动健康旧任务。

技术停止规则：仅本run输入/角色/头/原生U契约错误、非有限loss/gradient、固定更新/关键产物缺失、重复启动失败或输出冲突；保留现场，不自动重启。科学资格FAIL不取消该矩阵健康行或自动clean评分，只阻止扩种子。所有原始IQ、权重、预测、日志与PT向量留运行/本地证据目录，Git只提交实现和必要聚合报告。

本地9负例、真实调用点执行、数学/缺组/停止梯度、三真实网络同初态和骨干＋默认W非零联合梯度、入口/生成脚本编译通过。唯一独立P0/P1审查PASS无发现。发布器一次普通预检、单归档SHA比较、远端编译、真实source checkpoint10个synthetic IQ无query smoke后直接启动；不继承该CP到训练。发布后核实实际PID/CWD/argv/GPU/日志增长，训练完成不等于实验完成。

84%仍未达成，首seed不能完成三seed目标；只有固定默认头完整评分可说明实际提升，无Phase2/新TX/卫星部署声明。
