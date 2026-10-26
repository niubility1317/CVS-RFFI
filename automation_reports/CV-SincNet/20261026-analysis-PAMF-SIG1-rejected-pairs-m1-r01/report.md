# 单视图 REPL：低置信样本的同包恢复一致性

本次仅分析已完成源域缓存，不训练、选模或修改在跑的三个学生。唯一矩阵、权限、原 checkpoint、CPU2 环境、CWD、命令、独占路径和失败停止规则见 [experiment.json](experiment.json)。发布时的 Git commit 是代码身份；runtime 与实际读回记录它。

原 E20 默认 cosine 头为 6×160、scale30。读取其原始权重和原 source14views 的 logits/单位 joint160 特征，用原头代数重建所有 logits；L2 实际权重/缓存 smoke PASS 后立即继续分析，无另训头、模型实例化或 IQ 读取。保留原 L6300、按 RX 固定选择的 U6300；U 文件必须无 y。相同 source 物理 ID、角色和原 checkpoint 来源已核实，本次不改变数据。

比较三个已固定规则：raw soft 作为两个 G 的目标；Ga/Gb 互为 detached 概率目标；两个 cosine 分数（logits/30）的均方差。每个角色三景，分别同景不同随机实现和相邻不同景，共12个单元，再按 raw95 接受/拒绝、RX 分层。L 额外按原判对/判错分层，记录共同错误、均值预测的纠错/损伤及相对真实 LS.01 CE 的方向；U 真 TX 与真实精度始终未知。

Ga/Gb 概率一致性的 logit 梯度和两个相同真标签 CE 梯度的点积恒等于 ||pa−pb||²，分数均方差的对应点积非负。但这只是 logit 代数；共享错误均值不能靠一致性自动纠正，也不等于真实骨干/AdamW 方向。额外报告原冻结头在单位特征叶子上的梯度投影，明确与原未归一化 joint160 和参数梯度的区别。L 是已见训练样本，不把其方向当作 U 或未见 RX 保证。

0 新 IQ/F/G 前向、0 optimizer/新头、0 V/target/Utruth。CPU2，CUDA_VISIBLE_DEVICES 为空，不占 GPU 或阻挡健康训练。任一原契约、标签可见性、原头重建、有限性或代数恒等式失败，只停止本 CPU 进程并保留产物，不自动重试或改变其他任务。

独立6项数学/负向运行检查通过，编译通过，一次独立 P0/P1 审查无阻挡问题，见 [local_verification.json](local_verification.json)。一份发布包传输 SHA 比较、远端编译和独立进程/产物读回随实际发布补入。本分析没有扩种子/目标测试选择规则，也不改变三角色学生已有源域门槛。结论必须等待完整12个单元，不从部分缓存统计宣布新方法有效。
