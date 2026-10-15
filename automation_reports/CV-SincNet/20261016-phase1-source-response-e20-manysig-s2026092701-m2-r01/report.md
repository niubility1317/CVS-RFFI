# 源响应方向的独立 Ag：当前 E20 训练证伪

状态：REGISTERED_NOT_RUN。两个 scratchE20，EMP 为预定候选，ROT 为等核能量方向控制。只改变原三槽配方中的 G；复用已完成 B00 六前向 raw 控制，先核对实际源角色、native args、预算与来源。控制旧目标分数不参与资格判断，不重训或覆盖原行。

唯一假设：源关联方向在当前固定SIG_TIME＋RDrop/defaulthead下能训练出实际的clean收益，而非仅通过冻结输入安全。旧PHYS/ROT未建立优势；当前有限IQ安全PASS且两session的EMP−ROT正负不一，均不冒充有效。只先测独立Ag，不直接叠As或新loss。

固定fullL6300/U56700/V27000、RX1/3/4/6/8、day1/2/3、IQ256/25MHz/equalized1/nativeRMS、RAW_HZ/NoExplicitCFO/joint160/defaultcos。每模型scratchE20=13L+7U，4440更新；同初始化、默认头正常联合训练，U真值隐藏且原oldgate/ownEMA不改。

每个L物理样本raw/G/raw三槽权重.25/.25/.50，每槽两次RDrop复用IQ；替换nativeL身份目标，不累加重复CE。每行L实际曝光568320、L前向3409920，两行合计40model-epoch/8880更新/6819840个L前向。U/EMA/source/formal另记实际值。当前不训练As，不改原正式七视图。

G用固定L6300导出谱预测生成15个TX平衡方向；33tap、deltaL1<=.06、RMS恢复、reflect，ROT逐对核L2匹配。session按epoch/RX/day/eq/view固定，所有TX共享，跨epoch刷新；参数不由TX定义。记录全部实际epoch×session×TX覆盖及波形功率误差。

每行训练及源panel完成后、自己首次target预测前，冻结source_decision：相对匹配B00，fullV clean至少+.5pp，最差源RX下降不超过1pp，并有完整有限预算/梯度/session健康。EMP失败即停止该方向的联合或种子扩展；ROT不因某个目标分数较高替换为候选。EMP未超过ROT时不宣称源物理方向特殊价值。源V是健康/机制证据，不是未见RX测试。

所有注册行无论源科学PASS/FAIL，冻结后仍自动原clean+6practical，七视图prediction固定后独立CUDA隐藏CPU truth-last评分。当前资格、dispatcher及训练不读取目标成绩。每GPU全部owner合计最多3进程，free>=12000MiB并共用锁；只有本run系统技术错误停止待启动行，其他健康任务不变。精确命令、路径、环境、矩阵和预期artifact见experiment.json。

本地真实CVS合成检查PASS：两个图198个同初始state张量、1536个网络前向、每步六前向/成对同IQ、默认W/SIG梯度和EMA绑定/状态，另10个标签顺序负例、13个协议负例及10包算子对照。第一次检查发现适配遗漏legacy构造器引用，修复后整项通过；无真实IQ/checkpoint/更新。后续真实checkpoint无query smoke是launcher首步，PASS立即继续，不创建额外许可。

尚未发布、启动或取得任何新训练/目标结果。当前low urban>60%、clean正收益及跨seed复现仍未完成。

唯一限定审查发现并修复一个P1：Windows生成的B00参考路径使用反斜杠，Linux完整config比较将失败。改为POSIX序列化并同步修复两行登记；发布前定点复核该问题，不重复全量审查。

一次限定审查及唯一P1定点复核现均完成，剩余P0/P1为0。实际native引导生成的两行完整配置与登记一致；正在固定代码提交，尚未启动。
