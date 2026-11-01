# 物理后验的源特征指导：同初值单阶段30轮、首微批采样

- run_id：`20261102-phase1-Restore-CovarianceFeature-E30-s2026092701-r01`
- group_id：`restoration-covariance-actual-IQ-feature-guidance`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同物理恢复架构/实际初值/源mu-C/数据顺序draw/LR/30；增加原冻结F0连续160特征在实际后验IQ上的.01约束。每逻辑batch只首微批32或末批24，控制额外后验与F0成本。

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

## 实际后验的源特征指导

固定CovarianceMAP结构和全部scratch初值、原源mu/fullC、R07、数据顺序/draw、LR与30/7410预算，新增源域F0特征损失，部署结构和成本不变。控制为已经完整冻结的bf29554a物理估计行，不加载其G权重初始化，不重训控制。Source四路归因48组已复算：低档城区Fclean盲82.242340%，仅换真实Action87.325905%（净+73/1436），仅换缓存真实噪声方差标量82.660167%（+6/1436）。这支持研究RF敏感的参数估计，不证明Main原因、TX准确率或恢复上限。

全63000记录保持.1×Action矩阵相对MSE+.001×噪声logMSE。每逻辑batch256只对首微批32执行实际FP64后验→冻结原3ff8 Tri-SIG-TIME SSL fused160→.01×(cos+源方差MSE)；末批24全部。均匀源shuffle使采样不依赖TX、ID或质量；每轮7896、30轮236880后验/F0记录，另63000 sourceclean特征缓存。特征项用所选样本均值，不额外乘1/8；物理项按实际逻辑batch数归一化，每步一次clip/step。首微批复用现有G.estimate，其他微批实际不执行后验或F0。步均值、物理记录均值、采样特征均值分开记录，最后小批不隐瞒。

F0只提供连续源特征，参数/状态全冻结；不以logits、TX交叉熵、类别原型或另训头指导G。本版明确不是F-free；F0 E50、原63000 sourcefeaturevariance（f57c54ca）及原sourceIQ统计知识均计入。目标sourcefeatures.detach，梯度通过真实物理后验回到G信道/噪声头，当前clean不作为G输入。固定.01为一次预选工程权重，不称已按新模型梯度校准或最优，不扫权重或按Main选择。

原Ls6300/Us56700、source20draw再重复前10，全部学习参数从零单阶段30，F指导从第一更新联合启用；source featurecache不是G预训练阶段。0TX/V/OE/novel/目标训练、早停或选模，OE只冻结诊断。3新检查和唯一独立P0/P1通过，首步source2实际完整后验/F0默认头有限、联合梯度、F0不变及严格E0重载通过后立即30与完整Main。

同一地面完整Tri-SIG-TIME SSL/defaulthead/Main cross_rx126000+cross_both42000×clean+六景全部预测固定后独立CPU truth-last，raw/DURA只精确复用旧r03同ID预测。固定CovarianceMAP既有预测仅在完成后作配对分析，不成为训练teacher。每景距同raw clean≤3个百分点且两个总体各自保持clean；combined门槛140345/168000。source质量不取消正式测试，既有基准曝光如实披露，不声称新盲测。

GPU4/6按实际UUID/free≥12000MiB/existing compute≤2/新增1总≤3准入，一row一条NN、训练/两预测/CPU评分依次自动串联。其他owner保持。技术失败保存、不自动重试或迁移/复位。一次archive SHA与远端compile、实际PID/CWD/CUDA/GPU/log增长核实。1230341参数与1050624字节源prior buffer不变，静态FP64后验成本不冒充实测时延；采样减少额外分支曝光，实际总训练耗时和正式性能仍需实测。完整命令/环境/路径/矩阵/预算/停止规则/产物见登记。
