# 固定 E20：R-Drop 源域学习信号检验

状态 **ANALYZED / CPU 源检验 VERIFIED，3/3 固定父通过**。浅层 t1、中层 t2 和 SIG_TIME 的 KL 导数均在五个平衡批次非零；跨批平均后与两次 CE 均值的导数方向均相容。这只建立源端训练信号资格，尚未证明实际训练或泛化增益。

| 固定父 seed | t1 cosine | t2 cosine | SIG cosine | paired CE 均值 | symmetric KL 均值 |
|---|---:|---:|---:|---:|---:|
| 2026092701 | 0.457382 | 0.317435 | 0.561274 | 0.02528145 | 0.04159344 |
| 2026092702 | 0.212014 | 0.459611 | 0.731514 | 0.02541033 | 0.04022252 |
| 2026092703 | 0.787462 | 0.740636 | 0.664013 | 0.02986488 | 0.05227931 |

执行代码 `37edd17d35279fa7731b4666f7b33644a58f2d3d` 已提交、推送并独立核对远端 OID。单次 archive 传输比较、远端编译及三个真实 checkpoint 无查询 smoke 均通过。原 PID 已自然结束，完整 22 行 stdout 无致命标记，三行 completion/independent_recount 与根矩阵一致；launcher 捕获过准确 PID/CWD/argv/空 CUDA 环境及日志增长，后续独立读回核实终态，未赶上短作业存活期间另做实时捕获。CPU 2 线程、GPU 分配 0，没有重复发布或重跑。

实际同一组 900 个不同 L 物理包由三个父分别观察，共 2700 包观察、8100 次源 packet 前向、15 个梯度批、60 个参数层记录及 120 个保存向量；另有 6 个合成零 IQ checkpoint-smoke packet。新进程只复算保存产物，新增前向/梯度均为 0。每个父的参数和 buffer 逐位不变；0 新训练 epoch、0 optimizer 更新、0 头拟合。父模型既有 3×20 epoch/13320 更新另计。

固定 alpha1 下，三处骨干的 KL/CE 平均导数范数比约 1.47 至 3.01；这不是可直接解释为学习率或训练收益的比例。本轮不据此扫描系数。joint160 两次随机前向余弦均值约 0.39 至 0.42，而默认头多数仍同类，说明类别输出一致性与所有特征坐标一致性并不等价；本轮没有增加特征对齐损失。

下一步按源端资格进入一个单 seed、真正留出 RX 的独立训练证伪：匹配“两次默认 CE 均值”对照与“该均值 + alpha1 对称 KL”，保留原生 dropout、默认分类头和固定预算。需要先单独固定训练安排与停止条件；当前结果不自动授权目标测试扩种子，也不使 84% 目标完成。NONE 的正式三 seed +0.270833 个百分点以及旧 E50 源阳性仍保留，不据已暴露 target 成绩换父或重排候选。

发布前冻结方案：

状态 **PLANNED / 已登记**。复用原固定 BASE_SSL final20 三个父；本轮0训练更新，仅检验同一clean L输入的两个原生随机子网络输出是否提供兼容身份CE的浅层、中层和SIG梯度。先执行seed2026092701；若固定源资格失败，保存失败并停止其余父。若通过，继续固定02/03，三者全部通过才称源端确认。不会由本轮直接发布训练矩阵或访问target。

公式为两次默认CE均值（LS0）与两可微默认概率的对称KL，固定alpha1；比较原t1.pw、t2.pw、实际single SIG local_projection及原W的导数。两个分支均保留梯度，无teacher、投影头或头拟合。第三次eval前向只给同输入参考。原native dropout参数、CENTER25K/NoExplicitCFO及内部补偿、joint160/defaultcos保持。已有[R-Drop（NeurIPS2021）](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)约束两个dropout子模型输出；当前alpha以**平均两CE**为基准，明确本实现归一化，不宣称全球创新或复现论文其他任务。

这是新source检验：旧阳性来自DRO final50/LS.01/strongL及multiview SIG的views.0，本轮是原BASE_SSL E20/LS0/cleanL/single SIG。旧报告指定run实际仅有seed01，02/03父不存在；没有补E50训练或静默换父。父固定为原BASE_SSL，不选择已暴露target成绩更高的NONE/CLEAN_SIG。旧900L源梯度阳性、所有负结果及NONE正式三seed+0.270833pp均保留，各自配方和证据边界独立。

复用VALIDATED_ONCE源角色L6300/U56700hidden/V27000、RX1/3/4/6/8、day1/2/3和六TX，训练池10%标注率保持。共享源loader一次读取现有L/V cache，V立即丢弃，禁止U cache/tensor、V forward/gradient、target及teacher前向。沿原固定cohort种子2026210712，对每TX/RX/day70包做置乱，前10个L90秩两两合并，5个L180共900不同L物理包；所有父使用同一900包。每父2700源packet前向、5梯度批、20层记录/40向量；最多8100前向/120向量，另每父2个合成零IQ真实CP smoke。跨父观察不是新增独立物理样本。原父已有3×20epoch/13320更新成本单列，本轮0新epoch/optimizer/headfit。

源资格固定：五批KL均>0，t1/t2/SIG的KL导数各5/5非零（norm>1e-12），先跨五批平均导数再算其与pairedCE的cosine，每个≥0；模型全部参数/buffer不变、.grad不累积。W只观察，0拟合。真实CP来源检查在反序列化前执行；每row加载后2零IQ smoke PASS直接继续，同一模型用于后续源测量。保存logits/真实joint160/导数与合法L元数据后，新CPU进程只对保存产物复算，不新增前向、梯度或truth连接。

ordinary N607/CVS-RFFI，CPU2、CUDA_VISIBLE_DEVICES空，0 GPU分配；CWD `PROJECT/releases/cvs_SIGTIME_rdrop_source20_20261015_r01`；命令 `CVS-RFFI/bin/python -u experiments/cvs_rdrop_source20/runner.py experiments/cvs_rdrop_source20/configs/matrix.json`；不可覆盖root `PROJECT/runs/20261015-phase1-SIGTIME-rdrop-source-e20-clean-manysig-m3-r01`；owner `codex/root/SIGTIME-rdrop-source20-20261015`。依赖原R02不可变代码release，Git代码身份以本次release_commit.txt/运行resolved execution_commit固定。发布前本地聚焦负例和实际模型合成检查、唯一独立P0/P1；一次N607身份/工作负载/路径/disk preflight、单archive SHA比较与remote compile，发布后独立PID/CWD/argv/CPU环境/GPU无成员/log增长或终态核实。技术停止和产物以experiment.json为准，不干预任何其他owner，不覆盖或自动重试旧输出。

通过只说明固定冻结源父上的梯度兼容性，不能替代真正heldRX训练或84%目标。后续训练对照需保持两次L前向/平均两CE，另以源证据固定；当前无训练发布。设计与矩阵在本报告/experiment.json中完成；REJECTED_EXTRA_GATE：通用独立设计审批、spec链、重复report-only审查及额外hash不进入本次最小流程。

本地聚焦检查与实际模型合成输入检查 PASS，运行矩阵和登记逐项一致，7个Python文件及2段嵌入远端Python语法检查 PASS；唯一独立P0/P1审查 PASS，未发现下一次实际运行阻断缺陷，证据见review_evidence.json。尚未读取真实IQ或checkpoint，真实CP smoke、N607执行和源科学资格仍待发布核实。
