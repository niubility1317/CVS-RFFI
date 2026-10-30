# PAMF 融合前四贡献密度：源侧候选

固定run为`20261030-phase3-star-contribution-density-s2026092701-r01`。唯一主`contribution640_density_ratio`，固定控制`fused160_density_ratio`。本轮检验融合前互补信息，不根据上一轮novel真值挑分支/权重/容量/阈值。保留原完整Tri-SIG-TIME SSL F0/default6head和DURA v1.0身份，神经参数与buffers不更新。

控制逐值复用`20261030-phase3-star-density-s2026092701-r01`实际codeb8163a03e8d3d299127c7914272a2d866f92c89a的density_ratio聚合模型、六阈值和两角色预测；不是重新挑控制或重新校准。主仅拟合新的640维分布与自己的V风险阈值，父控制分数不进入主拟合或主阈值。两policy共享原F0默认身份输出与同一物理P1。

## 数据和冻结来源

L6300（旧六类各1050）及OE6134的原F0四项贡献来自protected-cache实际code7dd90671968ddf58aab5a5b302b4d56371c8b9c9；只读原特征、不加载R1权重或构造其adapter。L标签仅旧六类，OE仅outside-six成员与公共RX/day，真实OE TX/K隐藏。源known V27000（旧六类各4500）只冻结后风险校准，source novel3150（19-15/19-16/7-1各1050）仅预测后独立truth-last；原Us56700和target U/Q不读。

源RX索引1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23。每L/novel设备×RX×day70条，每V设备×RX×day300条。sourceRX/日期已见、novel有历史开发曝光，本轮不是未见RX、新未知设备盲测或未来匿名注册。四source角色物理互斥共42584个已有记录，不增加物理样本。

所有source新消费IQ沿已有G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/no additionalEQ、referenceSNR30dB/reflect；郊区城区各0.5及六分量面积概率，receiver2027/scene2026103001/channel2026104001，信道前一次unitRMS、输出不逐样本RMS。本轮只消费原缓存P1，不生成新received、不重抽信道、不读clean/P0实值、不加载/运行G。P0仍指星地received。原F0 E50/11100/232993、source split392005及原L/Us/V祖先保持。

## 同次特征与主模型

在原F0 reader直接挂临时native hooks，捕获同pass时频identity160、原gain加权PA160、原response_gain加权reference160及signature160。参考29坐标27必须仍为0。default logits仍调用原reader.classify_features(fused160)，不安装unused openparts、不训练新身份头；finally无论成功失败清除hooks，原状态保持。

每条记录的四贡献分别用FP64单位化；合法零分支精确保0，不补随机或均值向量。拼接640后全局单位化，转FP32存储；全四分支为0明确报错。fit/query均调用同函数，统计核心再按相同FP64全局归一化处理。没有学习分支权重或使用未知角色分配权重；单位化取消各分支原幅度，保留其方向与零状态。主feature不是原fused160的可逆重标，可能保留互补信息也可能增加RX/质量干扰，效果由既定评价决定。

主采用上一轮相同族的Gaussian mixture：仅L拟合90个类×RX×day mode和pooled格内共享协方差，10%各向同性收缩；Cholesky白化。主D=640、mode质量1/90、径向方差按70/(70+640)向L共同方差收缩。OE在同空间按公共15域各1/15质量加权，固定32容量/seed2026103019/25次weighted Lloyd；mode方差按n_eff/(n_eff+640)收缩，floor1e-8。likelihood包含mode质量、尺度/体积、距离和白化Jacobian，主score=logp_L−logp_OE。

控制沿已冻结D160模型、原D相关收缩和原阈值。两者改变了表示维度、统计自由度与实际计算/存储；不能把任何差异全归因于融合损失，也不声称同容量/同成本。维度扩展仅新增明确640 schema，原160 schema和调用保持，禁止通过临时修改全局DIMENSION运行主模型。

## 执行与评价

首步实际原F0 checkpoint两条合成G10无query smoke，同pass覆盖四贡献/默认身份/参考mask/hooks清理，PASS即继续cached L/OE拟合。主模型冻结后V27000逐样本P1 forward并导出同次贡献640和默认logits；与固定控制的opaqueID/公共metadata/默认身份核对。V仅既定42组known风险/common-tail六阈值校准，不改变任何模型统计。主τ冻结后才novel3150逐样本forward，并连接固定控制novel预测。神经/optimizer更新0、真实新增F30150＋合成2、G0，不把多表示当新物理样本。

两policy全预测完成后，独立CUDAempty/BLAS1 CPU scorer才唯一连接novel truth；从两聚合模型和保存特征重建raw/margins，主报宏A_K/R_U/H及C_K/原F0损失、AUROC/OSCR、逐真TX/RX/day/scene与配对。控制不能替主；source晋级仅known42＋整体source R_U≥90%的早筛，其他source切片完整诊断，并非最终跨未见RX/逐主场景验收。失败保留、不按novel调整任何参数/分支或重新抽样。

部署只保存聚合o/W/mode均值/尺度/质量/六τ，不保存逐source物理行；本轮诊断特征仍为私有实验产物。原模型/缓存历史成本保留，新维度的stat字节/实际F次数/全25拟合轨迹/运行时间报告，不将源拟合耗时当B1部署延迟。

一个GPU预测worker，启动actualfree≥4096MiB、每卡新增本任务≤2，其他用户不停止/修改/重启；随后独立CPU scorer。一次实际代码commit、聚焦协议负测与实际checkpoint首步smoke、唯一直接P0/P1审查、最小登记/单preflight/archiveSHA/compile、启动PID/CWD/argv/GPU/log与终态读回；不增加hash/receipt/审批链。非法来源/角色、非有限、碰撞或技术失败保留、不自动重试；UNKNOWN先核原PID/root。预期configs/logs、smoke、聚合models/25步fit、τ/校准、两policy known/novel预测、独立score及观察。

本轮通过也仅支持下一次另预登记的跨RX星地开发验证。用户最新v3的C_K/R_U90、A_K距同输入原F0闭集≤2pp，以及进一步高拒识、无标签适应/发现和未来A_N80/REG旧损≤2pp目标保持ACTIVE；source/同池结构不替代完整目标，AN/HOW本轮N/A。
