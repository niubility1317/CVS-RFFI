# 默认特征网络的源梯度路径诊断

状态 ANALYZED，首种子三行已完成，实际代码/config d4c1e9280854a3bfc988626f98224e737347260b。结果见 [完整分析](final_analysis.md)。以下保留预登记。run 20261014-phase1-source-feature-gradient-manysig-s01-m3-r01。已完成模型的源机制诊断，没有新训练或目标测试。逐行配置见 [experiment.json](experiment.json)，唯一直接 P0/P1 审查和入口修复 targeted review 见 [review.json](review.json)。

依据是完整旧代码与源记录：U CE 对已接收样本取平均，完整66600 source updates中 RANDOM/PAIR 的加权U/L逐步损失比均值约1.4%至1.7%。源RX3/8强弱一致性低于其余RX，排序改变几乎不影响这一比例。标量损失比例不能推导梯度比例，故先测量同一路径的实际导数，不直接扩大λ、放松门控或增加结构。没有使用目标IQ、错误分组或分数来选探针、种子或阈值。

矩阵固定首种子2026092701：OLD_GATE_COS50、RANDOM_ORDER、PAIRED_ORDER的 actual final50/11100 学生与各自真实EMA。来源、原契约、物理ID、祖先、固定终轮、NoExplicitCFO/内部补偿和默认头均由现有 source_proof 核验；不加载最优目标权重、不恢复训练。source L_s6300全部恰一次，完整PAIRS允许base_index/rx/day/eq/sig物理排序，batch256、尾156、25批；shuffle seed2026167702、strong seed2026177702，固定后不依观察改变。既有source loader会materialize V缓存，立即丢弃；V前向、V求导、U IQ/真值/目标访问均为0。无新数据验证或额外签名/hash链。

学生和EMA全程eval，strong视图仅使用原噪声0.015，没有dropout；这是固定源eval导数，不冒充原训练梯度历史。三项与原损失同形式：clean L真标签CE mean/smoothing .01，强视图以弱EMA和原完整confidence/batch_neighbor/strong agreement mask计算 .16 pseudo CE mean，全部强样本 .01 entropy mean。合法L标签进入CE和正确率观察前，门控已固定且未接收标签。L/U代理用同一source batch并非真实训练L128/U256组合，解释限定为源机制证据。

默认cosine6×160/scale30保留，单独分类头拟合/训练/替换/冻结训练均0。autograd.grad只请求实际fθ176065参数和 time_fuse、t1/t2/t3、f1/f2/f3、pa_b1/b2/b3、joint160激活；只排除class W960，保留全部norm/gain/reference。不调用backward/optimizer、不累积.grad、不更新参数/buffer/EMA。输出11个参数路径内的加权U/L norm比和cosine、各源RX逐样本激活敏感度；Sinc Hz独立分组，不将不同路径原始单位作为重要性排名。mean reduction归一化明确记录。

成本是读取3个已有50轮模型，无额外骨干轮数/optimizer/head更新。每模型6300合法L、18900包次前向（teacher weak＋student clean＋student strong）、25批，总56700包次；新训练0。不是完整三模型种子实验，不验证84%/新TX/Phase2/卫星或U正确率。

执行：普通N607既有CVS-RFFI CPU/fullFP32，CUDA_VISIBLE_DEVICES为空，Torch/OMP/MKL/OpenBLAS2线程。CWD /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_source_feature_gradient_20261014_r01，命令 python -u -m experiments.cvs_source_feature_gradient.trace。代码/config身份为实际发布Gitcommit，写入release_commit.txt并由运行记录回读。不可变输出 /home/szu2070436088/2510044040/CV-SincNet/runs/20261014-phase1-source-feature-gradient-manysig-s01-m3-r01；逐行resolved_config/results/completion，整轮completion或failure，release预检/dispatcher日志/submit保留。

停止：任一来源/角色/默认头/shape/nonfinite/接线/完整计数/参数buffer变化/梯度积累缺陷立即停止后续诊断，保留产物和旧模型，不自动重试、不动其他owner。没有额外科学性能筛选或后续种子放行，不用target决定停止。

本地6协议负例、真实模型结构零IQ导数/形状/finite/无积累/状态不变检查PASS；一次独立审查发现OLD入口误走禁止重训分支，已改为读核实的实际saved native args，三row预检覆盖入口、唯一targeted复核PASS。尚无真实运行失败。发布前一Git代码/config提交，一个归档一次SHA对比，一个remote compile/真实CP零IQno-query导数smoke，PASS后立即开始；启动后独立PID/CWD/argv/CPU/日志增长核对。除此不增加发布阻塞。
