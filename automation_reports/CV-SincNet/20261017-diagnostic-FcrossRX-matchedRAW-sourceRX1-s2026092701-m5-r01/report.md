# F＋跨 RX：五个同 seed raw 控制的描述性可比性审计

run_id：20261017-diagnostic-FcrossRX-matchedRAW-sourceRX1-s2026092701-m5-r01。当前本地实现/预登记，尚未收集或加载真实初始化快照。固定五个既有源控制，不启动任何模型训练。此分析非阻断，不改变当前SupCon视图实验的配方、判据或健康进程。

完整目标仍是独立Ag价值、As适应与同模型联合收益；本审计不替代这些指标。先前P15时序和P16 session覆盖理由的失败保持关闭，不据此重新晋级或调整其参数。

## 唯一问题和固定来源

问题是：同名义raw8L/1.25、sourcefold和modelseed控制，在真实初始化、源物理角色、完整L/U顺序、记录的普通增强标量和最终默认头判决上有哪些一致与差异？这不是新随机复制实验，五个历史分数已经暴露；不预设差异来自GPU或某个机制，也不把五控制当五seed。

固定cohort由五个已知完成且名义配方相同的控制组成，不按结果增删：P13 refresh_RAW、P15 timing_RAW、sceneCycle B00、isolatedAWGN B00、boundedPhase B00。逐row来源/路径/seed与用途以experiment.json为准。全部仅复用原checkpoint内默认头的已固定源预测，不重做预测、不载入训练后权重，不训练或替换分类头。

每个原模型为F＋跨RX同TX、三SIG_TIME、统一Sinc、joint160、NoExplicitCFO、原defaultcos30联合训练、E20/4440、13L＋7U/oldgate/ownEMA。完整原base .75raw＋.25ordinary和原ordinary-only浅中层SupCon，再额外.25raw同IQ RDrop pair；8L/1.25，实际全部4440步目标分解复算。只比较这些名义共同条件，unused shadow构造与代码路径差别如实保留。

原fit L5040/U45360/V21600、RX3/4/6/8、day1/2/3、6TX、split392005；held source RX1原L1260仅原FINAL20七视图。当前不读任何IQ、U TX真值或target；U ID只作opaque顺序比较，不解码隐藏TX。所有来源的scratch/init/freeze/无祖先/无target/原默认头joint、预算与有效角色在随机初始化快照读取前核对，五组角色ID必须匹配。仅取回原initial_smoke.pt随机未训练状态作206tensor逐对比较，0训练后CP或数据集传输，私有快照绝不提交Git。

## 分析和边界

全量读取五组各4440 view/native/Sinc-unit步、5994 L/U序列、20epoch与完整stdout/CSV；重算每步目标份额、实际预算、U权限与更新健康，比较全部普通输入MSE标量、首次目标/梯度差异及其幅度。比较全部206初始张量和十对shape/dtype/逐项数值，不用文件hash或额外seal。前向0，不实例化网络：首份合法随机状态的结构/有限性检查后继续其余状态；这不是模型前向兼容smoke。

独立重算5×1260×7＝44100个已冻结源hard判决的accuracy/6类macroF1/CM、十对每景修复/新增错误/判决变化；核对原physical IDs/labels和六场景实际configs/physics。记录各景mean/min/max/range/样本SD，仅为五个既有控制的描述统计，不能当置信区间或跨seed证明。相同存储MSE不证明逐点IQ相同，未重新读取/回放IQ，不能唯一归因CUDA、数值精度、loader、dropout或实现。

没有候选晋级阈值；全部五row和十pair都保留。科学/数据不匹配必须如实标明，不按分数剔除后声称精确重复。已有失败路线不复活，当前健康SupCon-view E20不修改。Ag、正式target、最差未见RX、跨seed及low>60%/suburban接近clean均仍未完成。

## 执行与成本

本轮0新modelE/更新/NN/教师/EMA/头拟合/IQ；5个随机初始化CP读取，0训练后CP读取。历史训练成本100modelE/22200更新/实际NN另列，不冒称新增。收集是普通N607已授权路径的只读文件传输，不在远端发布ML代码或启动worker；因此训练release归档/remote compile/模型NNsmoke不适用于此本地元数据阶段。仍按唯一owner/不可覆盖本地产物、真实来源和Git流程执行，不新增hash链或审批。

本地F Python3.13做stdlib读者；现有SSRPython3.10/Torch2.1仅CPU2读取初始state，不使用NumPy桥、不安装/改环境。13个协议负例、206合成张量和3个状态差异控制PASS，已知NumPy2桥警告保留；0真实IQ/CP/NN。唯一限定P0/P1审查进行中，关注越权、路径、不可覆盖和会实际跑错的分析问题，P2不阻断。

精确命令/CWD/输入输出见experiment.json；CWD=E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014，本地输入E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/inputs、输出E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017/final、stdout/stderr留E:/type10-7/local_artifacts/cvs_FcrossRX_RAWcontrol_audit_20261017。唯一owner codex/root/FcrossRX-RAWcontrol-audit-20261017。缺失或错误来源、非有限/错误预算/非法角色/输出碰撞保留原产物并报告，不重训或选择性重跑。预期全量可比矩阵、逐景/逐pair结果、完整曲线、初始化对比、成本和图表与登记/Git交付；原IQ、权重、逐包ID/原预测日志/机器状态和生成图不入Git。
