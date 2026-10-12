# 报告机制：普通 cos、三种子 30 轮、只测 clean

用户取消源域验证性能门槛，要求报告机制继续优化并以测试集证明84%目标。本轮提前固定12个训练arm×三个primaryseed2026092701/02/03，共36条独立从零30轮轨迹。交叉RX反馈的CE_MATCHED/UNIFORM_SSL/META_SSL共9模型及SWA3模型纳入正式全源clean比较，不因旧sourcepilot负结果取消测试。其余8个独立机制是子带调制、多项式、仿射IQ、log谱、标量闭环、矩阵闭环、切向约束、C4轨道平均。没有串接按目标分数挑选的旧变体、没有继承oracle80/旧教师。

每模型6300合法训练L，实际L/U/V7/63/30，标签占训练池10%。U仅两个SSL变体打开原56700无标签缓存，始终隐藏TX真值；源V无前向、无候选性能资格门槛。30轮×222步，每模型6660更新，总239760更新/1080模型轮。普通30×cos(joint160,W6x160)，无RX专家头、路由或margin；Phase2可丢弃W另拟合。六个统计分支用128隐藏MLP、末160 LayerNorm及2sigmoid通道增益（初始1）与native表示相加，使新统计获得实际优化通路；整体含native表示不能据此声称严格不变。C4用四个旋转的非线性特征均值，推理仅有限群严格不变；不用rawIQ平均。切向项.05约束归一化表示对两个固定数字代理动作的中心差分，每步轮流一个动作、额外两次deterministic前向，不更新分类权重；最终源L的两动作有界距离只作诊断，分类仍cos。

全部实现公式、固定尺度/阈值、退化处理及适用条件详见experiment.json notes与frontends.py。多项式是四阶乘积而非累积量；仿射满秩/良态才有效；子带局部平坦假设与深衰落边界保留；log谱仅删除固定平滑基；闭环包内8片段共同接收链假设尚未实证，保留零/秩1/常数闭环内容诊断。数学动作测试不能代替clean识别性能证明。Gaussian有效Fisher已在独立投影组作明确工作模型诊断，不作为离散TX梯度损失。

全源元反馈每Lbatch五RX26/26/26/25/25、day/TX完整遍历平衡，轮流反馈RX25/26；U只其余四RX各32。4组梯度与反馈梯度使用冻结当前AdamW二阶状态度量，排除普通cos权重，正向余弦平均归一后cap2；UNIFORM相同梯度开销、权重1，CE不打开U。warm5/ramp5/conf.95/完整相邻两pass/noise.015/KL.25。此为一阶近似，不声称原论文精确AdamW双层优化；与历史只按RX/day采样CE的差异明确记录。SWA固定前20cos2e-4至5e-5、后10交替1e-4/5e-5，平均21..30各末轮67个真实optimizer张量，保存10份紧凑快照，最终buffer和未训练参数不平均；实际模型无BN。其他arm原30轮cos2e-4至1e-6，AdamW WD1e-4、clip10、batch128。训练无星地增强。

每row训练并冻结后直接clean预测，SWA的FINAL/SWA两个读出均完成后才独立exec真值评分。总39组读出×168000=6552000决策、39×18=702条same-row指标，保存每seed/RX/TX/day、混淆、MacroF1及完整三seed均值/SD。仅clean，不追加practical/LEO。全部矩阵与参数在本次target访问前固定，所有负结果保留，不选择性重跑；已有benchmark暴露，不宣称新独立盲测。84%以实际完整三seedclean均值判断，尚未达成。

单owner、普通N607账号、不可覆盖run/release/output，共享GPUcapacitylock每GPU总实验进程≤3/free≥12000MB。一次各候选独立P0/P1及聚焦负测、真实合法checkpoint无querysmoke、实际Git代码/config提交、单archiveSHA、remotecompile/N607preflight、postlaunch PID/CWD/argv/GPU/loggrowth。仅runowned系统技术、权限、数据或非法预测失败停止并保留产物，不按分数停止、不修改健康旧任务。commit、命令、环境/CWD、路径、GPU及产物在experiment.json和release/readback中记录，完成后独立重算、登记及Git交付。
