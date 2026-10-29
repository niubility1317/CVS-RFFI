# U 池相互近邻适应与两阶段匿名注册

Run `20261030-phase3-star-reciprocal-adapter-s2026092701-r01`。两臂为普通高置信 self-PL 控制与 reciprocal 主方法，各执行DA0/DA1×REG0/REG1四态；唯一主为reciprocal::DA1_REG1。完整参数见对应JSON。目标仍是跨RX星地CK/RU≥90%、相对原同输入F0闭集AK损失≤2pp、无标签适应有旧类收益、未来匿名AN≥80%且保留旧类及后续拒识，不把组件完成改成总目标完成。

上一轮源端uniform开放暴露在source有效，却在RX24-1降低闭集身份能力。因此本轮回到原Phase1 F0，保持F/default6头/G及source paired拒识器/6τ冻结，仅用合法无标签U学习小残差特征适配器。不是从上轮失败增量继承，也不是据Q选更好的control。相互近邻提供目标结构的依据可参见[NRC原论文](https://arxiv.org/abs/2110.04202)；冻结原头利用目标数据可参见[SHOT原论文](https://proceedings.mlr.press/v119/liang20a.html)。本方法采用项目特定开放门控、残差网络和注册规则，不声称复现这两篇论文，图像结果也不转称RFFI证据。

## 数据、监督与冻结源状态

直接复用已VALIDATED_ONCE的RX24-1 G10 U384（2021-03-08）/futureQ864（2021-03-15），物理互斥，source RX仍1/3/4/6/8、三源日。P0是原单次received，P1是同received经冻结DURA，不重抽信道或读取clean参考。G10沿600km/25MHz/2.45GHz、receiver2027、六景面积概率/scene2026103001/channel2026104001、residual/post_sync/noEQ、输入一次RMS/输出不RMS。capsule/split由配置绑定，不重新验证数据或改变物理角色。

目标算法只接收U公有IQ及opaque IDs，无TX、未知类数、配额或Q标签。真实U旧288/到达新96、Q旧576/到达192/后续96的组成仅builder/scorer保管，不进入predictor配置或pseudo函数。已暴露的RX/设备及历史Q保持开发身份，正式RX24-18未接入，不当新盲测。Q不能回收训练，状态冻结后逐样本前向，不计算Q批次统计。

原Tri-SIG-TIME SSL E50/11100/default6head/232993和DURA R69/266951直接复用，Phase1源L6300/Us56700/V27000及10%标签祖先不改。冻结paired源状态来自neighborhood源bundle：八分支P0/P1聚合geometry、normalizers、四3233参数快照及source六条件阈值，选定paired_mlp16_step1000。它有合法source L/OE成员监督祖先，不能称整个系统零源标签；本轮source IQ/逐样本feature读取、source F/head/readout/阈值新更新均0。90组source anchors若由旧loader加载仍是合法聚合，不作为新目标匹配bank或再拟合源数据。

## U-only 适应和匹配控制

原F0/G先实际checkpoint无query合成G10 smoke；验证原head能对输入raw fused160传梯度、head参数和F/G不更新后，才打开U。P0/P1两次原F0前向得到完整分支、P1 logits与unit160，source paired margin提供初始已知门控。

适配器160→32 ReLU→160共10432参数，CPU局部RNG seed2026092701、最后层全0。对每条raw fused先取单位向量，网络残差逐条L2限幅0.1，再单位化并恢复原raw范数；保留尺度以兼容冻结source geometry。该界限约束单条扰动，不宣称保证2pp身份保护。原六类头不重训、不替换，分类和拒识使用适配后P1 fused和同原头logits，其他P1分支及P0不变，source6τ不再校准。

两臂初始trusted均为source旧类margin≥0且P1 MSP≥.95。控制只给trusted自身argmax硬伪标签。主在同一trusted基础上，对初始接收但非trusted的样本，若有至少3个trusted相互10近邻、最高标签票占至少.8，则扩展硬伪标签；允许与其原argmax不同。图用unit160的FP64余弦，排自己，opaque ID稳定解ties。所有伪标签训练前固定，不用真实K，也不把未选U强行归入旧六类。

每臂full U固定200步，adapter-only fresh AdamW lr.001/wd.0001/betas.9,.999/eps1e-8/foreachFalse/clip5，无增强、动态采样或早停。损失为选中伪标签CE＋all-U非对角unit Gram MSE权重1，教师为原U特征detach；未选样本只参与结构保持。无伪标签则明确SKIPPED/0更新/identity，仍完成四态预测和评分，不擅自放宽门控。loss/梯度/参数变化及实际曝光逐步记录，训练loss不替代独立未来收益。

## 发现、注册及未来判决

DA0和每臂DA1分别在U拟合注册状态，K可以不同或0。候选必须原margin和当前margin都负；在候选诱导的当前unit160 mutual10NN图上取连通分量，不扫描K或给真实类数。分量至少16条，与初始trusted存在相互邻边的记录比例≤.05，按opaque ID交替两半的中心距≤本分量LOO距离95%半径才注册。中心为FP64均值160，半径floor1e-6。该纯U规则只表达稳定候选假设，不保证设备纯度或跨日泛化。

判决分两阶段：先以当前默认头和冻结paired阈值判断旧类；old margin≥0时保留该旧身份，匿名原型不覆盖它。仅旧类拒识后才检查匿名距离是否在半径内，否则继续拒识。于是同DA下REG不会改变任何已接收旧label，也不会给原被拒的样本新增旧label；后续未知误接仍需真实未来Q验证，不能由这一性质推定开放性。

所有adapter/伪标签/注册状态及阈值均在Q首次读取前冻结。Q只逐批原F0/G及两adapter前向，状态不更新。全部2×4预测固定后，独立CPU scorer先核公有完成、ID互斥、joint/gate、任意K及冻结状态，再读取U truth为每arm/DA独立匿名映射，全部映射冻结后才Q truth。映射不回算法；未匹配匿名是接收错误，不算拒绝。AN分母含全部真实U到达类，漏发现记0；REG0 AN/HOW为N/A，缺必需Q类覆盖时指标为N/A。

主报逐TX宏AK、状态对应RU和H，附CK/原F0损失/candidate闭集/AN/laterRL/REG保留/HOW、六scene、RX/day/TX/CI、DA/REG两类配对得失、伪标签正确性和未知污染、注册组成与纯度。U诊断不能冒充未来AN。只有DA1_REG0满足绝对v3七组及positive旧AK适应、固定主DA1_REG1满足全部跨RX绝对v3约束和未来AN80/旧保留/后续拒识，才允许下一独立正式确认；control不替主，source90非前置，不自动扩seed或改写旧实验。DA0未适应对照的绝对指标完整报告，但不额外要求其先达标才检验适应方法；不把这项开发候选资格当完整Phase3完成。实际高景缺类仍N/A，不为凑门槛重抽样。

## 运行和交付

N607单GPU训练/预测worker共享一次U/Q F0/G提取，启动前free≥4096MB，每GPU至多新增2本任务进程，不因foreign存在禁整卡，也不停止/修改/重启其他用户。adapter两臂顺序独立同init/同200步，不继承控制状态。独立CPU scorer随后自动执行；一个owner、一个不可覆盖run/release、代码配置Git commit、聚焦协议负测与一次直接P0/P1、一次preflight/archiveSHA/compile、实际PID/CWD/argv/GPU/log增长及truth-last。技术失败保留partial；科学低分/无seed/K0不重启或调参。

完整source bundle实际存四个MLP快照，推理若沿原score_readouts会计算四个再取主，按实际存储/调用计数，不报成只一份3233。两adapter同时留存的参数及每条分类头/adapter计算另计，不能以原F0不更新称新增计算为0。训练/状态拟合、原F/G、source祖先和批量耗时分别报告，不把批量时间当B1时延。weights/IQ/逐行预测/原始日志private，公开方法、登记、汇总报告提交推送核验。
