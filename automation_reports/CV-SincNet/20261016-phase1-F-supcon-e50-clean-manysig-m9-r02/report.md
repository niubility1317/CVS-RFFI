# F＋浅中层监督对比：目标 clean 三种子固定矩阵

状态：PLANNED。CPU/CUDA真实网络检查PASS，尚未启动本候选训练或产生本候选目标成绩。总目标继续ACTIVE。

用户要求继续优化F（三路SIG_TIME＋统一Sinc＋RDrop），主要提高实际f_theta并在测试集目标域衡量，默认分类头随骨干原样联合训练，每模型不超过50轮。上一轮固定Sinc全局/频带缩幅均负，已完整交付78799a4d；本轮不改其倍率或选择seed救跑，保留F原完整物理Sinc更新。

历史依据：20261013-phase1-crossrx-contrastive-clean-manysig-m6-r01在旧30轮UNIFORM_SSL条件下，ALL_TX与CROSS_RX_TX默认头相对其旧匹配基线分别+0.3577/+0.4661个百分点，3/3正；两者互比仅CROSS_RX在1/3正，不能预设跨RX专属作用。历史不同配方不作为本轮匹配baseline，不继承其权重或原收尾诊断。只复用其明确的损失定义/温度/权重/启用日程，并将当前F＋RDrop组合当成新假设，不假设涨幅相加。MixStyle完整结果无稳定增益，本轮不增加MixStyle/Fishr或结构分支。

固定9行：BASE＝已完成AF r02的F三seed，只核实并复用其final50默认头prediction/独立评分；ALL_TX与CROSS_RX各新增seed2026092701/02/03，从零50轮=33L＋17U、每轮222步，每行11100更新。6新模型300模型轮/66600更新，历史F150轮/33300更新单列。所有三seed在任何新目标汇总前固定并上传，不设源性能资格门槛，不按新目标分数改变参数、候选顺序、seed、epoch或父权重。

结构不变：三个独立局部投影及SIG_TIME描述子共同进入实际joint160，不投票、不增加MEAN_VAR/COV分支。900722总参数，部署f_theta232033，默认ResidualPhysicalHead6×160/scale30的960个W参数继续主CE＋RDrop联合训练，不冻结、不重训、不替换，也不拟合RX/day诊断头。

新增训练损失只读当前真实L前向中的time_down[128,48,128]及t2.pool[128,96,32]。每层按channel拼时间mean与population std，再用本批detached mean/std（floor1e−4）标准化，每包L2 normalize（eps1e−4）。统计仅用于训练损失，推理不新增描述子、统计库或参数。两个原RDrop随机前向各算同样两层目标，四项等权平均；不增加网络前向、IQ曝光或优化器步，不重抽增强。

每anchor i，ALL_TX正例为其他同TX包，分母为所有非自身包；CROSS_RX正例为同TX不同物理RX，分母为这些正例与其他TX包。同TX同RX包不作为CROSS_RX负例。temperature固定0.2，loss_i=logsumexp(sim_ij/T over denominator)−mean_positive(sim_ij/T)−log(number_positive)，保留旧实现的常数修正，不改变梯度。只接真实L标签，RX来自extra.rx_i并与原15个RX/day域映射核对；不访问U真值。

保留原native随机L128采样，既不平衡重采样也不额外抽包。缺正例或缺其他TX负例的anchor被排除并完整记录；整批无合法anchor时返回有限零梯度，不把sameTX/sameRX误当负样本。相比历史平衡L实验，该处理明确属于当前F随机批次组合，不冒充旧实验精确复现。

权重固定0.1×clip((epoch−5)/10,0,1)，E1至5为0，E6为0.01，E15至50为0.1。原RDropalpha1/LS0.01保持，两个L前向使用同一次实际普通增强IQ。特征钩子只在E6至50的主L源调用临时安装，第一/第二前向捕获后立即移除并清空图引用；U、ownEMA teacher、eval和export没有这些钩子。直接辅助梯度只进入特征参数，不连接默认W，原主目标继续训练W。

Sinc保持F完整原统一方案：p=p0+25000q、q初始0、Fs25MHz，lr_q=alphaFs/25000、eps_q=epsilon25000/Fs；按物理Hz与所有其他参数共同clip5；物理origin衰减q-=alpha lambda(q+p0/s)恰一次，q原生WD0，然后执行完整Adam自适应步，不缩幅。student/ownEMA/推理及导出同坐标，沿用实际叶参数缓存版本。

数据角色保持SOURCE L6300/U56700隐藏/V27000、RX1/3/4/6/8、day1/2/3、原6TX及split392005；旧batch_neighbor伪标签门控、ownEMA、cos、NoExplicitCFO与内部补偿不改，无星地训练增强。V只健康诊断，U不新增特征对比目标。每新模型仍963900隐藏U观察，共5783400；每模型原RDrop额外1420800个L身份包前向，新增SupCon额外模型前向0。

每新行记录11100个原生RDrop与SupCon调用，其中9990次权重非零、39960个层/前向目标；保存每层validanchor覆盖、正例范围、rawloss、权重和完整epoch/step、实际Sinc梯度/更新见证。源闭合只核实合法执行与默认头/真实f_theta导出，不把源V涨分当作成功。固定final50后自动168000目标clean默认头prediction，完整后新起CUDA为空的CPU scorer连接truth，逐行报告18分层；最终9行1512000逻辑决策、1008000新决策、162CM。

固定比较ALL_TX−BASE、CROSS_RX−BASE、CROSS_RX−ALL_TX；报告逐seedaccuracy/MacroF1/最差RX及day、同物理ID修复与新增错误、均值/样本SD(ddof1)/正差数及实际时间成本，全部负结果保留。已有F84.270238%±0.275383个百分点只作匹配baseline，84%以上不自动代表新提升。

本地CPU/CUDA检查均PASS：10个协议负例、历史公式等价、invalidanchor有限零梯度、warmup0与原RDrop损失差0；真实F两已有前向四组浅中层捕获，辅助Sinc梯度非零、W辅助梯度None且主W梯度非零；教师/eval不激活、临时hook清空、900722参数/三独立视图及W移除导出逐元素一致。检查只用合成IQ，0源/目标IQ、0参数/optimizer更新、0头拟合。

入口CVS-RFFI/bin/python -u experiments/cvs_f_supcon50/runner.py，CWD PROJECT/releases/cvs_F_supcon50_clean_20261016_r02，root PROJECT/runs/20261016-phase1-F-supcon-e50-clean-manysig-m9-r02。唯一owner，共享跨owner容量锁，每GPU含preCUDA最多3总进程、启动前≥12000MiB；保留其他健康任务。一个Git代码/config提交、一个独立下一真实运行P0/P1审查、一次archive传输比较、remotecompile，真实匹配F checkpoint no-query smoke作为首步PASS即执行；每新行再以自身scratchCP smoke启动。

技术停止仅角色/来源/输出碰撞/非有限/预算或合法prediction问题，保留产物、停止新pending，健康worker继续，无自动重试或分数停止。执行后核实PID/CWD/argv/GPU/log增长，评分后同row分析及Gitpush/远端OID/根镜像。原IQ、权重、原始日志/预测及私有机器状态不提交。

边界：已暴露ManySig clean默认头测试，不声称新独立盲测、新TX、Phase2或卫星部署收益。新辅助损失进入f_theta不等于性能一定改善；最终以测试集目标域配对分数回答。REJECTED_EXTRA_GATE：brainstorming的额外设计审批与本项目Exclusive Minimal Experiment Workflow及用户既有自动研发授权冲突，按NONBLOCKING记录，不另问许可。

技术失败：r01 的 6 个新增模型完成 5 轮后，在第 6 轮首步辅助损失读取 RX 元数据时退出。原生 extra 为 (domain, metadata)，实现误读为字典。新增目标预测为 0，没有性能结果；6660 次已完成更新和全部输出保留。CPU 原生 unpack_batch 回归已复现同一错误，修复后以新 r02 从零恢复，配方及种子不变。

恢复说明：r02 只修正原生 (domain, metadata) 的 RX 读取格式；与 r01 完全相同的科学配方和三种子矩阵，从零训练，每个模型仍为 50 轮。失败尝试的 30 模型轮次、6660 次更新单独计成本；无新目标结果、无旧权重继承。

完成结果：本轮 9 行（6 个新增模型、3 个 F 匹配复用）全部完成固定 50 轮、目标域 clean 预测与独立 truth-last 评分。ALL_TX：84.823214% ± 0.402545 pp，相对 F +0.552976 pp，3/3 正差；CROSS_RX：85.113889% ± 0.379881 pp，相对 F +0.843651 pp，3/3 正差。全部 162 个混淆矩阵、66600 次新增更新和 450 个 epoch 已独立核算。详细结果、成本和限制见 [final_analysis.md](final_analysis.md)，代码执行提交为 `4df7b94d0fd8ce29056bd8236364e45afa92e07f`。技术失败 r01 成本另计并保留。
