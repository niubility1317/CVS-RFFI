# MILKR 源类内噪声投影：K1优先先导

预登记R17，完整Tri-SIG-TIME SSL基座与DURA v1.0、神经默认头冻结。保留R14 ridge/nested支持头，仅改变源端训练后冻结的两路投影权重。命令、环境/CWD、合同/祖先、路径、owner、GPU、stoprule见experiment.json。

源L6300物理记录在既有模拟ep91/92形成12600raw/G配对；按源六TX类均值去中心，仅在源阶段估计八个CCA16块的pooled raw/G类内残差协方差W。T=[(1−s)trace(W)/16 I+sW]^−1/2，s0/0.25/0.5/0.75/0.9。将同一T右乘原两路CCA投影Praw/PG，再单位化，与先单位化后做metric等价。s0精确R14。最终只保存融合编码器参数source_class_cca.npz（仍36448参数），原CCA副本保留，不上传W、类均值/原型、源labels/examples或source feature库。目标只加载冻结encoder，符合5.3冻结模型输入，不在目标重建源统计。

五源RX留出：每个shadowCCA与metric的训练均排除heldRX全部两epoch记录；只在固定ep91评估，六类随机次序old3/new3，每类52，support前K≤20、query后32，物理互斥。K1/5/20覆盖低中高shot，targetK10为中间预算外推；五strength×五RX×三K×四路共300个源H。六类均被F见过，不是真novel验证；源3景增强与目标6景不同，保留局限。

源门槛：四路平均H较s0≥0.5pp，K1两个S的Q0各≥s0+0.5pp，所选全部K/S Q0差≥−0.25pp，每个K/S Q1−Q0≥0且每个S均值>0。通过者源H最大，平局较小s；无通过source-only科学停止target0。通过才用全部sourceL训练最终metric并融合原CCA权重，F/G更新0、U/V真值0。额外源监督只复用既有L标签，不增加标签权限或物理观测。

目标首RX1-1/P01/MIX01/seed2026092701，全部K1/5/10/20×N2/5/10/20、B/C共80四路评分行，query全补集。严格S0/Q0只raw、S1/Q1只真实G，query路径共享同一support拟合，K标签额度不变。全部80预测固定后独立truth-last；匹配原版/R14各80行复用。用户已明确每格≥5pp：两个S每个KN的Q0较原版≥5pp且Q1>Q0，逐格报告四路各自+5pp和旧新/F1/floor，均值不能替代失败格。保留原Q0均值旧新各≥原版−1pp guard；只有全部16KN通过才原参数扩其余六RX，不再mean-only晋级。

唯一独立P0/P1审查PASS。合成噪声抑制、s0恒等、fused checkpoint重载/逐query独立/rawG相等、checkpoint无cov/classmean/labels、坏s拒绝测试PASS。真实checkpoint smoke为launcher第一步，通过即继续；源GPU4、targetGPU0一worker，actualfree检查，不动其他owner。技术失败保留产物，不盲重试。
