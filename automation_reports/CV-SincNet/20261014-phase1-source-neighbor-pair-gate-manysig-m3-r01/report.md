# 保留旧门控的物理邻包成对采样源诊断

状态：LOCAL_VERIFIED，尚未真实运行。run `20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01`；唯一 owner codex/root/source-neighbor-pairs-20261014。完整矩阵和实际配置只在 [experiment.json](experiment.json) 维护。

已完成源证据：r01三个旧默认学生/EMA的V完整门控覆盖3.4099%，confidence+strong覆盖92.2815%、精度99.8221%；uniformB256/w2物理邻包几何V期望3.6978%，U56700/cell630仅用协议常数推导1.7807%。这是合法L/V观察及采样几何，不是U精度或目标收益。下一假设：随机批次缺少邻包限制了U身份梯度；成对shuffle能让原门控看到相邻包，并可能在保留精度时增加进入真实fθ的训练信号。此run只诊断，不开始训练。

三个固定历史OLD_GATE_COS50-s2026092701/02/03，原33＋17/scratch50/11100默认cos学生和各自实际EMA。来源契约与原checkpoint继承通过复用source_proof核验；仅推理，不能作未来训练的父权重。比较RANDOM/PAIRS，两个arm用相同模型、完整L6300/V27000、B256、同一source split392005和物理ID。U IQ/ID/标签与target均不读取。L占训练池10%；本次没有训练采样率。默认6×160/scale30头、NoExplicitCFO及逐包内部补偿不改，无新头拟合或RX读出。

RANDOM保持r01私有generator shuffle2026101410/20。PAIRS仅读取RX/day/eq/sig/base_index五个允许字段，排序base_index后验证每对同RX/day/eq、sig与base_index差一致且1–2；随机shuffle不重叠的pair并随机pair内朝向。全部包每arm恰一次，B256和偶数尾批保持pair。没有TX/标签字段、同类真值配对或类别配额。强学生原Gaussian .015和train-mode identity dropout、EMA weak clean、原confidence quantile、batch_neighbor与strongagreement全保留。每sampler/role重置既有strongnoise/dropout种子2026101411/21；物理包重新排序后随机扰动随批次位置改变，不宣称每包强视图完全相同。teacher/student参数和buffers逐项保存/核对，身份路径无BN，V无梯度/optimizer/EMA/state更新。

固定科学反证：每个seed的V FULL_OLD覆盖≥50%、精度≥99.5%，且相对同seed RANDOM的精度下降≤0.25个百分点，三seed均满足才记source_criterion_pass。源标签只在gate mask固定后用于观察；没有U真值精度估计。阈值在本run真实访问前冻结，不按target修改。全部3seed诊断均完成并保留正负结果；失败不调参、不重跑，也不自动发布训练。通过只支持下一独立预登记的scratch50采样训练对照，不能自行宣称特征或DG提升，不叠加Sinc/TIME。

输出overall/RX/day/source类四类gate覆盖/接收正确数/错误数/precision与rejected-correct，3×2samplers×2roles×60=720条分层计数；报告model-seed均值/样本SD与配对差，物理sample数不当独立模型seed。成本：33300源包×2前向×2samplers×3模型=399600包次、1572批前向，CPU2线程顺序执行，CUDA_VISIBLE_DEVICES为空，GPU训练/模型/头/optimizer更新均0。复用已验证L/Vcache，不重验数据、hash或seal。

本地10个协议/metadata/配置负例与实际原门控合成smoke PASS：完全一次访问、尾批pair、标签字段拒绝、RANDOM原shuffle逐位一致、私有RNG不修改globalTorch状态。合成结果不冒充真实source性能。唯一独立下一真实run P0/P1审查按已写代码执行，只限直接非法/错误/破坏/不能运行缺陷。发布代码/config Git commit+push并独立比较remoteOID；一归档SHA本地/remote比较、remotecompile、一实际历史checkpoint零IQ无query smoke，PASS立即运行。无需额外审批、签名、源设计SHA或未来训练工具。

精确入口 python -u -m experiments.cvs_source_neighbor_pairs.audit；CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_neighbor_pairs_20261014_r01；普通N607用户既有CVS-RFFI fullFP32 CPU，OMP/MKL/OpenBLAS/Torch线程2。执行身份是release_commit.txt内实际Git HEAD。不可变run/output根 /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-neighbor-pair-gate-manysig-m3-r01，每row独占output，已有时核实、不覆盖。一次独立postlaunch PID/CWD/argv、CPU限定和log增长检查。技术stop：所属来源/metadata/非有限/状态改变/默认头不符时写failure.json、停止当前诊断调度、保留partial，无自动retry、无其他owner操作。只有完成同row分析与Git交付后记ANALYZED，原IQ/权重/log留原存储。

唯一独立 P0/P1 审查 PASS，未发现直接阻止本次源诊断的缺陷；另核对真实L/V规模的合成尾批156/120、全包访问与CPU发布入口。无真实IQ/target读入。
