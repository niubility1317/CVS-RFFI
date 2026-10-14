# E20 跨 RX 特征脆弱性矩阵预登记

运行 `20261015-phase1-SIGTIME-center-crossRX-sharpness-e20-clean-manysig-m15-r01`。固定 BASE_CE、BASE_SSL、CLEAN_CE、GLOBAL_GAP、CROSS_GAP × modelseed2026092701/02/03，15 行全部从零训练20轮/4440实际更新。全部相同 SIG_TIME、固定 CENTER_25K、原6×160默认分类头联合训练，0新参数/分类头/额外真实优化步骤。BASE_CE为20L且U/EMA前向和迭代实际关闭；其余4行为13L＋7U，保留原batch_neighbor/ownEMA/entropy/cos，U TX真值隐藏。关闭普通L随机增强和星地训练，保留原U专用strongGaussian。13＋7预先固定为E20内原33＋17时序的压缩近似，不照搬超出预算的E50。历史原E20纯CE、CENTER和SIG_TIME结果保留，新匹配基线用于本矩阵的机制比较。

固定ρ=.05、coefficient=.1、eps1e−12。只选择17个部署参数张量：primarySinc/time_fuse/t1/t2/SIG_TIMEsequence_pool（除gain）。当前L每RX梯度生成detached归一方向；原seenL180（90cells×ordinal68/69）排除当前全部base_index/完整物理ID，在其他4RX计算relu(pertCE−cleanCE)，两项均对features留graph；默认W/gain/其他参数仅在辅助中detach，原主损失正常联合训练。GLOBAL用完整当前L一条方向，其余掩码相同；CLEAN用同bank额外CE；ZERO保持相同6functionalforwards成本。不是未训练holdout或未见RX测试。

前444步辅助关闭，之后every16步，共250scheduledcalls/model。缺当前/heldRX或空bank整次skip，固定5分母，采样器不改。functionalstate克隆buffer、临时eval、逐次清空并恢复参数cache，保留RNG/live模型状态。辅助加到完整nativeL＋实际U/entropy损失一次，真实优化一次。固定448/1776/3104/4432步核对feature梯度、辅助W/gainNone、同实际AdamW/moments/clip5的BvsB+aux以及唯一真实更新。记录原始完整epoch/step、额外L曝光、Sinc初末kernel、实际梯度和默认头健康。

先跑首seed5行；sourcefinal20/LCE≤.5，研究行相对本次BASE_SSL的Vmean下降≤1个百分点/worstRX≤2个百分点；scheduled250/eligible≥95%；GLOBAL/CROSS至少2个固定probe梯度活跃且positivegapcalls≥1%，才扩展另10行。仅源证据控制扩展，无target排序/调参/选择性重跑。每行冻结final20后自动clean168000默认头预测，固定后独立CPUtruth-last；18CM/row，完整15行为270CM/2520000decisions。科学条件失败保留已训练首5行及测试，剩余如实未启动。已暴露benchmark，不称新独立盲测；84%默认头三seed目标尚未达到。

代码/config唯一身份为release_commit.txt。CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_SIGTIME_center_crossRX_sharpness20_20261015_r01`，普通项目Python，immutable输出 `\home\szu2070436088\2510044040\CV-SincNet\runs\20261015-phase1-SIGTIME-center-crossRX-sharpness-e20-clean-manysig-m15-r01`。发布命令 `python -X utf8 E:/type10-7/.codex_tmp/native_polyphase_runner.py experiments.cvs_crossrx_sharpness20.publish --output <local_artifacts/cvs_SIGTIME_center_crossRX_sharpness20_20261015_r01>`；逐行命令见experiment.json。唯一launchowner `codex/root/SIGTIME-center-crossRX-sharpness20-20261015`；GPU按共享容量锁分配，所有owner总进程每GPU≤3/free≥12000MB。首步聚焦协议negative＋一次合法真实checkpoint无querysmoke后直接启动；旧E50仅独立smoke使用，不进入任何训练worker。一个archiveSHA比较、remotecompile、独立PID/CWD/cmd/GPU/loggrowth核实。唯一独立P0/P1审查只覆盖下一次真实运行。只按登记的确切runowned技术失败停止，保留产物、不自动重启、不动健康旧进程。

跨RX源原型4/4通过，只证明信号可用与GLOBAL不同；仍未证明识别或泛化收益。SAM/SAGM/DISAM/DGSAM已有相关思想，当前候选的feature-only/fixedreadout/off-diagonalRX/current-packet-disjoint区别需要此消融检验。完整结果与方法依据见[机制分析](../../../analysis/phase1_crossrx_feature_sharpness_20261015.md)。不把历史CENTER与SIG_TIME增益相加。

独立 P0/P1 审查：PASS（sigtime_feature_path）。只读核对下一真实scratch20默认头训练/动态预算/完整U损失/辅助边界/唯一AdamW/源规则/15行发布/独立truth-last，未发现直接缺陷；0数据或权重字节/target指标访问，未追加重复测试或阻塞条件。本地结构核验PASS：参数总量、默认头初始权重、初始物理滤波器均保持，13＋7/U专用Gaussian=.015、BASE_CE实际U/EMA关闭、scheduled250与语法/发布入口一致。聚焦source mock更新/shadow检查和150项metadata负例PASS；真实checkpoint烟测由远端launcher首步执行。
