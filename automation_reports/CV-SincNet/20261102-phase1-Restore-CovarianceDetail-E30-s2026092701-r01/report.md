# 物理估计与IQ细节双分支：全源特征指导、单阶段30轮

- run_id：`20261102-phase1-Restore-CovarianceDetail-E30-s2026092701-r01`
- group_id：`restoration-independent-physical-and-IQ-detail-gradients`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

5481223参数scratchG，CovarianceMAP物理估计与MSAT非线性IQ细节分支。物理只simulator监督，IQ与原冻结F0连续特征梯度只到细节；全63000每轮曝光/独立梯度裁剪/同一优化器，30后自动原Main七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


本候选大幅改造恢复器：5481223参数，物理估计1230341参数与MSAT IQ细节4250882参数共同从零开始，单阶段30轮、7410步。物理分支初值与原BF相同，细节输出初始为零；不加载旧G、不分阶段预训练。原源63000的mu/fullC统计固定。物理Action/noise只受模拟参数损失监督；真实后验与lognoise上下文detach，IQ和原冻结F0连续160特征梯度只到细节分支。两个分支独立norm1裁剪后同一个AdamWstep，避免细节梯度改变物理更新幅度。

每个原Ls/Us记录都执行实际FP64后验、IQ细节和F0特征损失。30轮1890000次源曝光、59070次微批后验调用，另63000条cleanF0缓存；比旧首微批采样显著增加成本，不以效率为主线。本版是结构和完整曝光共同改变的复合候选，不作纯梯度路由消融声明。IQ损失为NMSE+.05多分辨率STFT，特征损失固定.01(cos+源方差MSE)；无TX/CE/logits/原型/新分类头、无V/OE/novel/target训练。

原R07的20draw+重复1..10、sourceorder/LR/30预算及地面Tri-SIG-TIME SSL/Main保持。真实source2检查含完整G/F0、物理与细节梯度隔离、初始零细节、状态保持和严格重载；PASS立即全30，最终30固定后自动完整Main双总体七景，独立truth-last。所有源和目标结果均报告，不因源质量取消测试，不按目标选模或选择性重跑，默认DURA不变。

源V上一诊断仅提供机制动机：low urban净增9/1800不构成稳定提升，同时物理误差加大；不能据此宣称该结构有效或Main唯一原因是域变化。新网络参数、总FP32状态22977820B（其中源统计1050624B）、额外F0知识及实际训练时间/峰值显存均报告；新细节分支的完整MAC与延迟尚待实测，不沿用DURA时延。

## 实际启动核实

VERIFIED：actual code `1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6`，release archive一次SHA一致、远端compile通过。controller 1041063 / training 1041142，GPU 4，准入free 24110 MiB、existing compute 0。实际PID/CWD/命令/CUDA/GPU和日志增长已核实。真实source2完整G/F0、两分支梯度隔离、初始零细节与严格E0重载通过，自动继续原30轮和完整Main七景。当前仅为RUNNING，尚无新正式识别结果，目标未完成。

## 首轮训练产物核对

# 首轮实际训练核对

状态：PASS；训练整体仍为 RUNNING，尚无新版 Main 结果。实际执行代码 `1fcbc3eb3cc16e7577ecc3cbedda99174b4c5cc6`。

已核实保存的 E0／E01 权重及 optimizer：物理分支 E0 全部张量与固定 BF 对照相同，细节输出初始为零，全部可学习初值等于本版 scratch factory。源 IQ 先验保持不变，两分支均更新，optimizer 有限。源 order/draw 与 BF 固定对照完全一致，30 轮日程不变。

实际完成 247 步、63000 条源样本（Ls6300／Us56700），每条均参与真实后验、细节修正和 F0 连续特征指导；1969 次微批后验调用，没有旧版首微批采样。记录加权目标由各损失项复算，绝对误差 4.00251431998e-09。

首轮耗时 230.688593 秒（3.844810 分钟）。这是首轮实测，完整 30 轮成本尚待统计。增大模型并增加特征曝光的成本如实保留，不能称为完整训练提速。

物理分支 E01 相对 BF 同轮权重的最大绝对差为 0.000418007140979；同初值、源 order/draw、损失和预算不等于逐值相同训练轨迹。本核对不把轨迹相等设为门槛，也不改动运行。当前只能将本版视为结构、梯度路由和完整特征曝光共同改变的复合候选，不能将后续收益单独归因于某个分支。差异的具体数值来源尚未定位，未证明新的协议或数据错误。

真实 source2 检查已验证 IQ／特征损失不向物理参数传梯度、物理损失不向细节传梯度；两分支独立 norm1 裁剪，然后执行一次 AdamW 更新。F0 参数与持久状态保持冻结。此处仅检查产物，新增神经前向、IQ 或 truth 读取均为 0。

首轮在线源均值仅用于检查训练，不是冻结恢复性能、源 V TX 准确率或 Main 结论。最终 E30 自动进入原 Main 双总体七景，全部预测固定后独立 truth-last；六景逐个距 raw clean≤3 个百分点并保持两总体 clean 的目标仍未达到，默认 DURA 不变。
