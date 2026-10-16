# F＋跨 RX 同 TX：I/Q 非圆对称的源域证伪

状态：PLANNED，已完成本地合成/权限验证；尚未读取本轮源统计或启动。

唯一问题：固定参考窗的中心化非圆对称系数幅值，在排除同一 TX 和日期后，是否仍有稳定的 RX 关联。它可能同时含 TX、RX、信道及已有均衡处理信息，本轮不恢复硬件参数，也不把 RX 统计可预测当作身份保持或泛化收益。固定 additive As 已因第二种子 clean 失败关闭，本轮不是该配方的系数/种子/RX 救跑。另一对话的 RX6 实验保持独立，不修改其健康进程。

## 数据和观测

复用现有 VALIDATED_ONCE source_contract 及 source geometry cache：原 L6300 中只处理 RX3/4/6/8×day1/2/3×6TX 的 L5040，每单元原 70 包，split392005。载体的其他 U/V/特征及 RX1 的1260个L虽随缓存反序列化，但不访问其标签/特征或加入任何统计；RX1 在观察前排除。不是完整 L6300 训练，也不是 L5580/Ldev720。U TX 真值始终不使用，无模型或 checkpoint 继承。

实际原生输入为 equalized=1、25 Msps、256 点、逐包 RMS。固定窗口 [80,160)，沿用当前 ExcitationResponse 的 lag20、atan2(cross.imag,cross.real+epsilon²)/20 补偿公式，epsilon=1e-6。仅作为统计观察，部署与训练输入不变，NoExplicitCFO 和必要内部补偿不修改，不输出 CFO 作为身份特征。

令 v 为补偿后窗口减去自身复均值，主量为 kappa=abs(mean(v²))/(mean(abs(v)²)+epsilon²)。它在理想公共复增益/相位下不变，但不是纯 RX IQ 参数。原窗口 kappa、补偿后奇/偶周期 kappa 与 coherence 只作说明，不成为失败后的替代赢家。旧 ordinary PA 链已有概率镜像项；不能把同一类随机项改名就算新的 A_g。

机制背景只采用广义线性 TX/RX 镜像模型及其混淆边界：[Korpi 等原文 II-A](https://arxiv.org/html/1402.6083)。此论文研究全双工自干扰，不是本数据的标签保持增强或识别收益证明。此前镜像子空间结构失败保留，不重开新网络分支。

## 最小矩阵与判据

单行 IQ_CIRCULARITY_SOURCE_CROSSFIT；无模型、分类器、教师、checkpoint、优化器、训练或目标预测，CPU 2 线程、GPU 无。未来若进入训练仍固定 F＋跨 RX 同 TX、三路 SIG_TIME、统一 Sinc、joint160、默认 cosine 头正常联合训练及 E20；本轮 0 epoch 仅指源测量，不冒称完成训练验证。

全部 72 个 TX/RX/day 单元均取原70包均值。18 折同时排除一个 TX 和一个 day，在其余5TX×2day估计各 RX 均值，预测留出 TX/day 的4RX单元。对照为 pooled mean 和所有9个无固定点的错误 RX 指派；另计算全部10个无序3TX/3TX分组的 RX 中心方向一致性。

固定条件同时满足才记统计 PASS：总均方风险减少≥5%；18折至少14折正收益；各RX风险变化均≥0；优于最佳错误RX指派至少2个百分点；全部10个TX分组的中心 cosine 最小值≥0.8；RX中心功率／奇偶周期RX中心差异功率≥2。零基准风险视为未建立证据。完整各折/各RX/各单元/全部对照保留，无困难包删除或按最好子集挑选。

失败关闭这一固定 circularity 源依据，不改窗口、坐标、补偿、阈值、RX或seed补救。通过也只支持另行固定身份保持/默认头检查，及匹配 E20 的 B10 验证；不自动训练或测试 target。

## 工程验证与发布

本地已通过4种已知生成情况：RX为PASS，TX-only/DAYFLIP/TXFLIP为FAIL；10个数据/角色/非有限负例、TX/RX/day置换不变、公开解析镜像kappa误差约6e-14、零输入与窗口边界检查。合成描述量20160个，实际合成IQ观察7包；0真实IQ、0 NN、0更新。G0不消费模型或checkpoint，真实CP no-query smoke不适用；启动的第一计算为合成观察/权限smoke，PASS立即继续。

唯一独立 P0/P1 审查只覆盖本次6个模块与实际源权限/运行闭合，结果发布前记录。不增加设计审批、hash/receipt链或数据重验。实际代码身份为发布 commit、release_commit.txt 与 resolved_config；登记回填实际执行 commit。

owner：codex/root/F-IQ-circularity-source-20261016。普通 N607 现有 CVS-RFFI Python；CWD PROJECT/releases/cvs_F_IQ_circularity_source_crossfit_20261016_r01；命令 CVS-RFFI/bin/python -u -m experiments.cvs_iq_circularity16.run；CUDA_VISIBLE_DEVICES空，CPU/OMP/MKL/OPENBLAS线程2。

不可覆盖输出 PROJECT/runs/20261016-diagnostic-F-IQ-circularity-source-crossfit-manysig-m1-r01；日志 release/source_circularity.stdout.log。一次 N607 preflight、一次发布归档及单次本远端 SHA 对比、remote compile、启动后 PID/CWD/argv/CPU环境/日志或终态产物独立读回。缺输入、错误角色、输出碰撞、非有限、异常保存 partial/failure，不自动重试或干预其他任务。

产物：active、resolved_config、synthetic_checks、circularity_crossfit（18折/9wrongRX/10TXsplit）、72cell JSON、私有 fitL IDs 和 observations.npz、completion/failure、stdout。完成后独立复算全部5040描述量与72单元/判据，生成图、报告与登记，提交推送并核实远端OID。无身份预测，truth-side target评分不适用；原始IQ、逐包描述、权重/日志/机器状态不入Git，公开小型聚合。


独立 P0/P1 审查 PASS，未发现使本次 G0 越权、算错或无法执行的缺陷；未要求新增模型、CP smoke 或报告审批。仍未启动源测量。
