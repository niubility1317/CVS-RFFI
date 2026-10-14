# 固定 SIG_TIME：软身份关系的最小源诊断

仅检验新半监督训练目标的源端可用性，不训练新模型。部署SIG_TIME195041／joint160和原6×160默认cosine头保持；分类头无单独拟合、重训、替换。前序逐实例EMA特征一致性的实际骨干梯度较小且接近正交，尚未给出稳定增益，因此本次改变关系学习目标，不扫描其权重或RSC遮挡。

Run：`20261015-phase1-source-soft-relation-manysig-s2026092701-r01`；代码提交：5ab6d390d5306a787c0efbf4a1532df861f0b877。固定首模型2026092701，原DRO矩阵SIG_TIME控制的自然final50和同checkpoint ownEMA0.999（原执行977d4fb609436bac23621ef8e32c2017495acef7），不按目标成绩挑父。加载前复用source_proof核对实际源契约、scratch谱系、final50与11100次EMA成功更新。历史目标分数已知，本设计仅据已有合法源梯度及公开方法提出，不声称全新独立盲测。

## 问题、公式与固定对照

检验：原teacher类别概率能否给出身份纯度足够的跨源RX软关系，并对实际t1PW/t2PW/SIG参数产生与同strong图身份CE兼容的非零梯度。借鉴[CoMatch的语义与表征关系相互约束思路](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_CoMatch_Semi-Supervised_Learning_With_Contrastive_Graph_Regularization_ICCV_2021_paper.pdf)，仅采用关系目标思想，不声称复现其投影头、图平滑、记忆队列或图像实验。

原EMA weak/eval输出detached q_i；学生沿用原strong噪声0.015和train/dropout，取实际joint160并归一化z_i。ALL_GRAPH排除自边；XR_GRAPH额外只允许源RX不同的边。T_ij与q_i·q_j成正比，在允许边内逐行归一化；S_ij是z_i·z_j/0.2在同允许边的逐行softmax。损失为平均有效行KL(T||S)，诊断有效系数0.16，与原UCE系数相同而非根据本次梯度调权。无有效边的行贡献0；不增加置信阈值、投影、网络、头、教师或优化器。纯关系函数不接受TX真值，教师q不反传，关系损失对原分类头W的梯度必须None。

合法L真值只在冻结关系构造后用于观测teacher边质量和同strong图默认头CE（smoothing0.01），不进入关系目标。ALL_GRAPH是同批／同模型关系对照，不作为待择优的第二训练候选。

## 数据与成本

复用原VALIDATED_ONCE／split392005源角色L6300/U56700隐藏TX/V27000。固定既有L180前五cohort，每cohort为6TX×5RX×3day×2，共900个唯一L物理包。teacherweak+studentstrong两次前向，共1800包×模型前向／10批；每cohort保存CE、0.16ALL、0.16XR在t1PW3456、t2PW6912、SIG投影576、默认W960上的实际导数，5批／20参数层／60向量。没有模型优化、headfit、U IQ／U真值访问、V前向或V梯度、query或target前向。加载器沿用合法现有L/V句柄；V返回即丢弃，不为方法变化重新验证数据。

## 预定判定与后续

源PASS须900唯一物理包、图/损失/向量独立复算和不变参数/buffers/RNG边界全部完整；XR的平均teacher软边真实同TX质量至少0.90，t1/t2/SIG各5/5批非零、五批平均向量相对同图CE的余弦均不小于0。FAIL保留全部结果，不扫temperature／系数／父权重。PASS仅允许另行登记不变02/03源复核，没有自动完整训练晋级。final50训练L代理不能证明U伪标签精度、早期训练或DG/84收益。

## 运行、停止与产物

入口`python -u -m experiments.cvs_source_soft_relation.audit`；普通N607原CVS-RFFI／FP32／CPU2threads，CUDA_VISIBLE_DEVICES为空，无GPU实验分配。CWD为`PROJECT/releases/cvs_source_soft_relation_20261015_r01`，唯一不可变输出`PROJECT/runs/20261015-phase1-source-soft-relation-manysig-s2026092701-r01`，owner`codex/root/source-soft-relation-20261015`。一次归档传输SHA比较、远端compile和真实checkpoint no-query smoke后直接运行；一次postlaunch进程/CWD/argv/日志增长核验。

仅本run的数据/继承/默认特征或头边界错误、非有限loss/梯度、固定cohort缺失、真实graph或recount不一致、输出碰撞为系统技术失败；保留产物、不自动重试、不碰其他健康任务。不依据目标分数停止、调参或选模。本次明确只做source诊断，不训练或测试新模型。

预期：preflight.json、submit.json、resolved_config.json、actual_source_gradients.pt、逐cohort图和真实参数向量、analysis.json、source_qualification.json、独立artifact_recount及completion.json／failure.json。原IQ、权重、PT、日志和私密运行信息不入Git；实现／预登记／正负聚合结果提交推送。

REJECTED_EXTRA_GATE：通用brainstorming的单独设计审批／固定spec/plan文件不是本项目许可的阻断条件。用户已授权连续训练机制研发，按最小实验流程继续，不新增审批、签名、SHA链、数据重验或报告审查门槛。


已自然完成ANALYZED／独立复算VERIFIED，source资格FAIL：XR边身份质量98.697450%，t1/t2余弦−.262721/−.330844、有效梯度范数比3.394885/3.936461。按固定规则不扩种子、不训练或扫参数。0优化／headfit／U/query，完整[结果和限制](final_analysis.md)。
