# F＋跨 RX 同 TX：六步 L 重排匹配 E20

当前状态：REGISTERED／LOCAL_AND_ONE_P0P1_PASS，尚未发布／训练／取得识别成绩。完整联合目标 ACTIVE。本候选只研究训练组织，不是 IQ A_g，也不训练 A_s。

唯一矩阵维护在 [experiment.json](experiment.json)：两条**新** scratch E20，O0 与 O1。两者共用六步预取、专有 L RandomSampler/DataLoader generator、原 F／SIG_TIME／joint160／默认 cosine 头和正常联合更新；原 raw／ordinary 四对 RDrop、浅中层跨 RX 同 TX SupCon、13L＋7U／oldgate／ownEMA保持。唯一差异是 O1 在相邻六次更新的同一 768 个 L 观察内重排 TX/RX 组成，O0 交付原顺序。两者都计算配额、拼接和 gather，控制预取/RNG/CPU路径；不能把旧 B00 当作匹配控制。

## 来源与假设

[完整来源分析](../../../analysis/cvs_L_batch_order_basis_20261017.md)已逐4440批、568320次L观察及13320个已有对比统计复算。没有正对缺失，但批次比例和正对数波动；固定六步重排保持每块物理多重集及全部曝光次数，正对数最小值4→10。统计改善不证明身份识别受益。本轮唯一假设：在共同缓冲控制下，减小L TX/RX组成波动能提高held-source clean，同时保护practical。

真实冻结后端的原迭代器与make_loader已只读核对；现有runtime wrapper强制222步，原L DataLoader无独立generator。新source observer在原native_scope进入后、data构建后延迟解析Order；没有提前创建输出或U迭代。原U加载器不修改，U从不进入L缓冲或标签解析。

## 输入、模型和预算

复用原split392005／VALIDATED_ONCE数据角色，只在fit RX3/4/6/8、day1/2/3上训练L5040／U45360(hiddenTX)／V21600；源RX1 L1260只在固定final20后做clean＋六practical。该源折已有历史暴露，不声称新独立盲测。不是完整L6300或L_train5580/L_dev720。没有训练父权重；真实CP仅匹配相同4RX物理角色的scratch final20默认头no-query兼容smoke后丢弃，不初始化新行。

每行E20／4440更新，8L前向4546560 NN；完整base raw.75＋ordinary.25保留，额外raw对.25，总pair权重1.25，CE/KL/SupCon及原U公式不改。两条新行合计40 model-epoch／8880更新，每模型仍E20，不追加续训。原U学生/EMA教师、V及held8820另计，预计每行总5823318 NN，dispatcher CP smoke2另计。预取/gather新增CPU时间、最大768包逻辑buffer单独记录；不冒称零计算成本。原教师初始化继承计数偏移在最终独立复算时剔除，真实前向不遗漏。

旧ordinary实际共有PA/CFO/时移/增益/AWGN/multipath等历史开关保持并记录，不称它们已验证标签保持。两次RDrop复用同一已生成ordinary IQ；没有新增IQ扰动、损失、教师、U视图、部署分支或诊断头。NoExplicitCFO及必要内部补偿不删除。源评估六practical传播、噪声、接收质量/补偿与固定SNR原样保留，不更改正式received IQ或访问target。

## 固定识别判据

O1相对同预算O0：held-source clean accuracy至少＋1pp，clean Macro-F1不下降；六practical各场景accuracy最多下降0.5pp，seen V最多下降0.5pp。每行必须完整20轮／4440更新、默认头正常联合训练、隐藏U/no target，final base L objective≤.5。批次计数、低loss或门控选择数不作科学收益。两行完成后独立复算17,640个判决、每个TX/day与配对修复/新错、全部4440步公式/原U/EMA/物理多重集、20轮曲线、真实NN和CPU成本。

报告各suburban绝对accuracy及与自身clean差距，探索性2pp不作为本轮额外晋级阈值；不通过降低clean缩小差距。任一原条件失败，关闭固定6步L重排，不改块长、分配、L私有seed、SupCon/CE、RX、模型seed或预算救跑。PASS只支持另行固定独立源seed确认，不自动target、As训练或宣称IQ A_g／四因子／联合目标完成。

## 命令、环境、输出与停止

代码/config以本次Git实际commit为身份；publisher写release_commit.txt及resolved_config，不要求报告或成员SHA。ordinary N607／CVS-RFFI；CWD为PROJECT/releases/cvs_F_LbatchOrder_sourceRX1_e20_20261017_r01。命令：CVS-RFFI/bin/python -u experiments/cvs_batch_order17/runner.py；worker参数仅允许已登记O0/O1 row，无预算/seed/数据覆盖。owner codex/root/F-LbatchOrder-E20-20261017；每GPU总实验进程≤3、计所有owner、空闲≥12000MiB后启动；dispatcher CPU2/CUDA隐藏，worker CPU2/loader0。准确私有runtime留本地，不入Git。

独占run/output root：PROJECT/runs/20261017-phase1-F-crossRX-LbatchOrder-sourceRX1-e20-s2026092701-m2-r01；每行独占source/及logs/ROW.stdout.log。一次普通账户preflight、一个archive／一次SHA传输比较、remotecompile；首步真实匹配CP no-query smoke PASS即继续两条scratch，不生成smoke许可或等待再次审批。发布后只读PID/CWD/argv/GPU/日志增长。错误角色／初始化／非有限／预算／输出冲突／启动或非法预测等系统技术失败保存产物、取消未启动行、健康worker继续，无自动retry；科学失败不提前停止，完成固定预算及全部源预测后判断。其他owner及健康旧实验不动。

## 已执行检查

元数据与接口合成检查：N607实际CPU环境的222步／37块、真实RandomSampler、O0逐值原IQ、O1每包IQ/y/domain及全部metadata一致，Torch/Python全局RNG不变，原U无buffer；0NN／CP／真实IQ／更新。该验证不是完整F训练。

本地新增26项配置／协议／科学反例PASS；两行新模型与原F全部206状态张量逐值一致，默认头6×160，初始预测逐值相同；新增参数0，合成NN6，真实IQ／checkpoint／优化更新／target0。已知本地Torch2.1/NumPy2桥接警告保留；上一轮RandomSampler本地转换失败，在实际N607环境同代码完整检查PASS，没有安装包或改环境。

当前唯一独立P0/P1审查PASS，未发现直接P0/P1；仅限直接影响真实运行正确性/权限/启动/产物的缺陷。没有新增审批、数据复验或hash链。尚无新训练或成绩，不把实现和合成PASS标为实验完成。

源输入读取口径补充：复用既有full-source cache会反序列化原L6300/V27000等存储，原fold.prepare在模型创建前materialize RX1的L1260为只读评估缓存及144个未用于训练的fit anchor。RX1缓存不进入任何student/teacher训练、源V、校准或额外机制；只有final20冻结后才调用默认头评估。不能把“冻结后评分”写成“此前没有读取过RX1 IQ”。该既有source-only折不是正式target；训练角色仍精确L5040/U45360/V21600。
