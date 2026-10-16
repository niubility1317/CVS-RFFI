# Sinc 与 RDrop：源 L 梯度分解

状态：PLANNED，尚无真实源梯度或性能结果。用户要求继续优化 Sinc 与 RDrop 的配合，本次先做最小无更新机制诊断。全部后续性能结论仍须默认头匹配训练与独立 truth-last 评分，本诊断不能替代。

固定四状态：A/E、seed2026092701、各自原始 initial 和 fixed-final50 student；A 为原 RAW 标准优化器，E 为统一居中＋物理单位更新。复用 A–F 执行提交09554c3a与已有空祖先、物理角色、默认头、固定终轮来源检查；不因目标数字选择 checkpoint。原7/63/30与 NoExplicitCFO 不变。

每状态完整读取同一6300个已见过的合法 L_s，一次固定 shuffle，50批（49×128＋28），原普通增强按E1或min(E50,E33)日程配置。每批第一个完整原生 forward 与第二个 identity forward 复用同一次增强后的 IQ，模拟原 RDrop 调用；loss=.5(CE1＋CE2)＋KL，LS0.01/alpha1/defaultW6×160。分别求 CE/KL 对实际部署 f_theta 和 W 的梯度；E 的 Sinc 坐标梯度除25000转回物理Hz。按Sinc48参数、time_middle、SIG_TIME、其余特征、默认W统计范数、夹角和冲突率，不拟合参数。

总成本：25200次 L 观察、50400包身份前向，模型/optimizer/EMA/head更新0；U/V/target IQ读取0、U真值保持隐藏。四个状态均在诊断后逐位核对全部参数和 buffers；autograd.grad 不累积 .grad。原始Sinc梯度向量和日志留私有输出，Git只保存必要聚合、完整表和结论。

科学判读：先报告四状态的完整源梯度，不预设梯度冲突。如果 final50 E 的Sinc有至少10/50批负 CE·KL，且Sinc相对冲突/强度结构提示需要保护监督频带学习，则下一步只研究针对 Sinc 的 CE/KL 更新协调，并设置原 RDrop 和缩幅控制。若不满足，否定该冲突假设，不把放大或投影作为已证实方案。该阈值仅决定是否立项源端候选，不是性能或扩种子门槛；新候选须另行固定源泛化资格与训练矩阵。不得用本次目标分数调参，本次不生成目标预测。

精确逐行配置和六类seed见 experiment.json。运行入口：CVS-RFFI/bin/python -u experiments/cvs_sinc_rdrop_coupling/runner.py --output PROJECT/runs/20261016-phase1-sinc-rdrop-gradient-source-s2026092701-m4-r01；CWD PROJECT/releases/cvs_sinc_rdrop_gradient_source_20261016_r01。CPU2/loader0，跨owner共享容量锁，最多3总进程/GPU、至少12000MiB余量。首先真实checkpoint no-query smoke，PASS立即开始L诊断。

技术停止：来源/角色错误、模型状态变化、非有限或缺失梯度、碰撞或进程失败则保留全部产物，不自动重试、不改健康任务。一个不可变root、一个归档与一次传输SHA、一次远端编译、一次PID/CWD/argv/GPU与日志核实。固定code/config Git提交后发布；独立P0/P1仅检查会让本次真实诊断跑错、越权、破坏、无法启动或丢失合法产物的问题。

边界：这是已训练源 L 上的冻结模型操作诊断，不能称源留RX泛化、未见样本证据、训练因果结论、新TX迁移或卫星部署收益；未测U/pseudo梯度。后续目标仍是获得真实三seed匹配性能提升，本目标保持ACTIVE。

本地验证：CPU/CUDA 两种真实网络 A/E 均通过，五组 CE/KL 梯度非零、48 个 Sinc 分量及参数 buffers 保留，3 个负例通过，compile PASS。GPU 观察超时未作为检查失败或运行依据；核实原检查进程结束后，保存重读检查日志并核实 PASS。本地源 IQ/CP/optimizer/head 更新均为0。
