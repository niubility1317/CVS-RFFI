# 旧 SIG_TIME 叠加 RDrop：E20/E50 三种子匹配验证

状态 **RUNNING，已启动12/18行；E20首seed3行已评分，其余6行已扩展运行；E50首seed3行继续训练，其余6行待源资格。独立核验 VERIFIED**。用户要求旧 SIG_TIME 与 RDrop 有机结合；本次原样保留旧 SIG_TIME 的 RAW_HZ、标签平滑 0.01、普通训练增强、NoExplicitCFO、默认 cosine 头联合训练、旧门控和 ownEMA。18 个独立 scratch 模型：E20/E50 × ORIGINAL/TWO_CE/RDROP × 2026092701/02/03；E50 原 33L+17U，E20 固定 13L+7U，不继承旧权重，不使用 CENTER25K/cleanL/LS0 替代旧配方。

| 条件 | L 前向/更新 | 原平滑 CE | 一致性 |
|---|---:|---|---|
| ORIGINAL | 1 | 原生 CE，原路径无辅助 hook | 无 |
| TWO_CE | 2 | 同一增强 IQ 两次随机前向 CE 均值 | 无 |
| RDROP | 2 | 同 TWO_CE | 原两 logits 的对称 KL，alpha=1 |

旧版先增强成 x_l_main，而通用 native hook 提供的是增强前 x_l。为保证两次仅更换 dropout，本实现只在真实 L 前向 context 内临时注册 root prehook，捕获已经增强的实际 IQ；第一次前向后移除，callback 消费并清空，再做第二次 identity-only 前向。增强不重抽，教师/U 不进入此 context。两次 CE 都显式使用原 0.01 标签平滑；native 已含 CE1，仅补 .5(CE2−identity_loss)+alphaKL，避免双计 CE。所有特征和默认 W 正常联合学习，无新参数或另训头，实际部署仍为原 joint160。

每预算先运行首 seed 三行。完整来源、更新、默认头冻结后，仅依据源域 finalLCE≤0.5、RDrop 源 V 均值/最差 RX 相对两个控制不低于1/2个百分点、至少1%调用KL为正，放行该预算其余6行；两个预算独立，任何target指标不进入此判定。固定对照每row训练冻结后立即自动 clean168000预测，全部决定固定后 fresh CPU CUDAempty truth-last评分，保留18分层 CM/accuracy/F1和配对纠错/引错。主条件固定 E50_RDROP，预算差包括cosine终点和13+7/33+17分配，不等价于同轨迹续训。

此前历史 SIG_TIME 83.0393% 与近期 RDrop E50 82.1030% 的配方不同；近期实际是 LS0/cleanL/CENTER25K。本次重新训练的 ORIGINAL 才是匹配控制，旧83.0393%只作完整保留的非配对背景，不把其高分当父权重选择，也不保证逐位复现旧 CUDA 随机轨迹。仍用原full5RX的6300 L、56700隐藏U、27000 V和固定合法数据句柄；V不更新参数/EMA/归一化，不读取U真值，测试分数不回流选模或重跑。普通训练增强在用户本次明确指定的旧配方范围内，星地训练及测试不加入本clean目标。

全计划630model-epoch、139860更新、11934720额外L前向、12247200 U重复观察和3024000 clean决定/324CM；主L前向17902080次，另36个合成CPsmoke包。实际普通增强、每轮V、预测和评分耗时分别记录，不宣称独占GPU加速倍数。每launcher先保存/重载own初始CP并对2合成包无query预测，PASS立即继续。

ordinary N607 CVS-RFFI；CWD `PROJECT/releases/cvs_originalSIGTIME_rdrop20_50_20261015_r01`；命令 `CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_legacy/runner.py experiments/cvs_sigtime_rdrop_legacy/configs/matrix.json`；owner `codex/root/original-SIGTIME-RDrop-20261015`；唯一run root `PROJECT/runs/20261015-phase1-originalSIGTIME-rdrop-e20e50-clean-manysig-m18-r01`。首6row就绪，其他12由源规则独立放行；现有shared receiver_residual_capacity4.lock内查all8 GPU/all-owner含preCUDA，空闲优先、cap3/GPU、free≥12000MiB，CPU2/worker、loader0，初始估计4.5GiB/worker、启动后实测。不停止/迁移/重启任何健康任务。技术故障或错误数据/来源/预算/预测/scorer/输出碰撞保留产物、停止pending，已启动健康行完成，无自动重试；源科学失败只关闭该预算未启动的6row，无阈值/seed/轮次救援。

本地当前39项数学/配置/context负例与22项来源/预测权限负例、8个源资格否决边界通过；CPU/CUDA实际模型初始198state与旧构建器相同、增强后输入复用、两个paired条件RNG一致、时域/t2/SIG/W梯度有限非零。原先“0×CE辅助项逐位梯度相等”的CUDA检查观察到最大5.96e-8舍入差，现 ORIGINAL 完全不安装辅助hook；不是修改旧训练或根据target调参。唯一独立P0/P1审查已通过，代码已提交推送，普通N607 preflight、release archive SHA比较、remotecompile和独立启动读回均已通过；这些证据不代表实验涨点或已达84%。

RDrop为已有方法，[原论文](https://arxiv.org/abs/2106.14448)。本次研究它与旧SIG_TIME的实际训练匹配，不改名声称全球创新。完整18row矩阵、命令、seed、数据角色、停止规则和期望产物唯一维护在experiment.json；已有正负结果及未发布的残差交换原型保留，后者不叠加本试。REJECTED_EXTRA_GATE：额外设计审批/重复报告审查/哈希seal链不进入发布流程。


<!-- ORIGINAL_SIGTIME_RDROP_VERIFIED_LAUNCH_20261015 -->
执行 commit `b3ac442e2d33a5b315fb3251c30e0ad7083566bc` 已推送并独立核实远程 OID。首 seed 的六行覆盖 E20/E50 和三个条件，GPU0至5各1进程，实测每进程 3346 至 3348 MiB；快照内所有 owner 均未超过每GPU3进程。CPU2、loader0、dispatcher/scorer CUDA为空。六行 own初始CP无query smoke均通过，真实L/U/V为6300/56700/27000，原标签平滑0.01/use_aug=True/RAW_HZ与默认头联合训练已核实，paired机制实际复用增强后IQ，CWD/argv/PID-start/GPU及两次读回日志增长均 VERIFIED。没有技术失败；逐row自动clean预测和独立CPU truth-last尚待训练冻结后执行。其余12行只由两个预算各自的源资格判定放行，测试成绩不进入判定。详见 [启动证据](launch_verification.json)。


<!-- ORIGINAL_SIGTIME_E20_SOURCE_EXPANSION_20261015 -->
E20源资格已按预登记规则PASS：3条件末轮LCE均≤0.5；RDrop全部4440次KL调用为正；源V均值/最差RX相对ORIGINAL为+0.077778/+0.222222个百分点，相对TWO_CE为−0.096296/−0.166667个百分点，均在固定1/2个百分点容差内。独立重算判定与远程source_gates一致，仅读取源完成证据，不读取目标指标。E20首seed三个模型已各自动固定168000个clean决定并完成fresh CPU CUDAempty评分，共504000决定/54分层记录；本次核实闭环产物而不解读部分测试成绩。其余两seed的6个E20模型已启动、ownCP smoke和首epoch通过；另3个首seedE50仍在训练，共9个活跃GPU进程，all-owner计数≤3/GPU。E50六个后续模型仍只等待其自己的源资格。详见[E20扩展证据](E20_expansion_verification.json)。

兼容旧配置时继承的非执行描述字段updates=11100和cost_claim=21 new scratch50...仍保留在不可变config中；前者对E20、后者对本矩阵已过时。实际native epochs、训练器UPDATES=epochs×222、进度和completion核实E20为4440更新、E50为11100更新，本18行均从零。此说明不改变执行、冻结配置或历史产物。


<!-- ORIGINAL_SIGTIME_FIRST_E20_SOURCE_LOGS_20261015 -->
E20首seed三个已完成轨迹的源日志分析 VERIFIED：完整13320次step/60轮epoch及三份stdout，实际损失重构最大误差2.38419e−7，全部有关损失有限，ORIGINAL无额外hook，两个paired条件均实际复用增强后同IQ且4440次KL调用为正。ORIGINAL/TWO_CE/RDROP末轮CE1依次0.275006/0.270163/0.273353；源V依次97.792593%/97.966667%/97.870370%。每条件7轮U共396900次重复观察，接受次数依次6143/6203/5611。RDrop没有通过增加接受观察次数改变训练；这不证明其U标签更准，也不用于推断目标收益。该分析只读取已冻结源日志/元数据，未读目标指标、真值或IQ，未加载checkpoint、更新参数或拟合头；其他健康行不变。完整逐epoch源曲线和损失统计见[first_E20_source_analysis.json](first_E20_source_analysis.json)。
