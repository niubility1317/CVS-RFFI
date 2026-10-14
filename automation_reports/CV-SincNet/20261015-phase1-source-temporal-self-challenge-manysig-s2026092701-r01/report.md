# 固定 SIG_TIME：中层时间特征自挑战的源证伪

状态：ANALYZED / VERIFIED；首模型真实源诊断已完成，0新训练，完整结果见final_analysis.md。代码`50d265cb0e8e48cdc2580c9f7b13e342f26b50c8`。本轮是source-only诊断，0训练轮/0优化器步/0分类头拟合，不触发目标预测。

研究依据：训练时只重复现有简单身份判断，可能没有迫使骨干利用其他可用证据。这里核查RSC的源机制兼容性：以真类默认头logit对中层的梯度选择遮挡，再检查真实骨干CE导数。它不等于发现新身份证据或泛化改善。机制参考[RSC原论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470120.pdf)与[DomainBed项目RSC实现](https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py)；不是原图像任务复现或新的通用算法声明。固定one-third来自已有方法，未经本项目目标调节。

保持SIG_TIME推理结构、195041个部署特征参数、joint160和既有默认ResidualPhysicalHead cos6x160 scale30。读原接收机风险首run中预定义SIG_TIME控制seed2026092701 final50，执行`977d4fb609436bac23621ef8e32c2017495acef7`。该父模型从零33+17/50/11100、L/U角色和完整继承已核实；source_proof加载前检查原始contract/physicalIDs/scratchlineage/final-only。选择控制行和第一名义seed，不选target-best；不把读CP作为新训练初始化。既有目标基准已经暴露，此源探针不声称新盲测。

只用L6300中既有固定cohort顺序的前5个L180，共900个唯一物理样本，每个TX6×RX5×day3单元2个/批。角色及split392005不变；不读取U IQ或TX，不对V前向/求导，sharedVALIDATED_ONCE builder返回V后直接丢弃，不重新验证数据。无query、truth、星地增强或新显式CFO坐标。

在actual t2完整输出96通道×32位置挂只读/临时hook；该post-ReLU位置实际非负，避开把signed joint160坐标直接当图像正激活的假设。它同时进入后续time主路径和SIG旁路，最终层结构不改。每批原始前向后，对真类logit之和求相对t2的梯度，沿96通道取有符号均值；每包保留严格低于线性2/3分位的时间位置，其余置零。先对全部包遮挡，计算真类概率下降，再只给下降大于等于批2/3分位的包施加该时间遮挡。量化并列值遵循明确不等号，实际遮挡个数如实记录，不强行声称恰好1/3。未定义的全并列/无剩余位置属于技术失败，不静默换公式。

| 观测 | 处理 | 参数导数 |
|---|---|---|
| ORIGINAL | 原始t2 | 原默认CE |
| SPATIAL_ALL | 全部包施加高梯度时间mask，供批选择 | 不保存参数导数 |
| RSC_SELECTED | 只对概率下降最大的预定分位包遮挡 | 原默认CE |
| MATCHED_RANDOM | 同一选中物理包、每包完全相同零数量，独立固定seed随机时间坐标 | 原默认CE |

随机控制隔离的是位置选择，不是独立随机丢包策略。4次网络前向/物理包，共3600 source forward包次；5个定位梯度张量；ORIGINAL/RSC/随机的实际t1 PW3456、t2 PW6912、SIG local projection576、正常W960导数，共20层记录/60参数向量。每次CE沿既有默认头和骨干图计算，不做头训练；没有优化器、teacher更新、EMA或额外模型。所有模型参数/buffer逐项保持、.grad不累计、mask RNG独立、hook必移除。真实早中期训练和新特征能力均需后续证据，这里是固定终轮兼容性诊断。

固定源规则：5cohort的RSC增量导数对三个实际骨干对象均非零；平均RSC_SELECTED CE比匹配随机高，真类概率下降也比随机高；预算/状态/角色检查满足。该PASS只表明进行了有区别的针对性挑战，不表示较高损失有益；仅允许另登记原样02/03源复核，不自动发布训练。FAIL保留终止该固定mask率/层位路线，不扫描分位/层或使用目标调节。

普通N607既有CVS-RFFI、FP32、CPU2线程、CUDA hidden；不额外分配GPU，接收机风险六健康训练不变。CWD `<PROJECT>/releases/cvs_source_self_challenge_20261015_r01`；入口 `python -u -m experiments.cvs_source_self_challenge.audit`；唯一owner `codex/root/source-self-challenge-20261015`；不可复用输出根 `<PROJECT>/runs/20261015-phase1-source-temporal-self-challenge-manysig-s2026092701-r01`；日志 `<release>/dispatcher.stdout.log`；本地 `local_artifacts/source_self_challenge_delivery_r01`。每GPU所有owner最多3总实验进程规则继续保持。

仅run-owned角色/query/CP来源错误、未定义mask、非有限数值、状态/梯度累计修改、预算/关键产物/复算缺失或输出冲突停止，保留现场不自动重启；不修改其他进程。发布一次普通预检、单归档本地/远端SHA比较、远端compile及真实CP2个synthetic IQ no-query smoke，然后直接诊断。PID/CWD/argv/log增长或自然完成产物核实。保存rawsource梯度/labels/physicalIDs/logits/masks在运行根，另独立进程重算全部mask、损失、预算和决定。Git只上传实现、必要聚合报告和结果，原始IQ/CP/PT/预测/日志不入库。

本地13个协议/shape/nonfinite/undefined-mask负例和真实随机初始化网络检查PASS，覆盖t2形状、正常默认头及骨干导数、相同随机对照曝光、RNG/状态不变、无积累梯度和hook移除。初始全零合成IQ导致概率下降全并列，这是假输入无信息，失败已保留；检查改用固定独立随机合成IQ，科学规则不改。入口和生成远端脚本compilePASS。唯一独立直接P0/P1审查PASS，无发现；未让文档/重复审查成为额外门。


首模型sourcePASS：900唯一L、3600源forward、300选择包、3300时间零位置，两种mask同预算；RSC/随机CE均值0.283861/0.140787，差0.143074。t1/t2/SIG增量5/5非零，t1/t2原CE余弦−0.556406/−0.620389，不能当正收益或训练安全证据。原状态/正常W未更新、独立复算一致，全部产物保存。按固定规则仅准备原样02/03源复核，不自动训练或访问target，固定父模型等待DRO02/03正常final50完成。
