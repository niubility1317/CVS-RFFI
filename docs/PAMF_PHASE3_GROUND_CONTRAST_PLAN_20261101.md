# 地面原生特征的已知／未知对比训练

Run `20261101-phase3-ground-contrast-s2026092701-r01`，owner `codex-phase3-ground-contrast-20261101`。上一轮地面中心化训练有源域局部收益，但正式拒识与跨RX已知保护未达标。本轮继续用户指定地面训练优先；完整跨RX星地拒识、零目标TX标签适应、发现及未来匿名注册目标保持，不缩减为源域成绩。固定DURA与历史产物、其他owner健康任务均不改变。

## 固定假设与两臂

仅改变logits风险可能不足以形成跨接收机可用的开放特征。本轮直接约束原默认分类头实际使用的160维坐标，比较仅已知对比KCON与另把OE纳入负项的OECON；这是待验证假设，不预言拒识、泛化或未知内部可分性一定提高。

两学生分别从完整原F0 Tri-SIG-TIME SSL E50/11100复制权重，重置相同AdamW微调，不从上轮CPAIR或任何目标适配模型继承。默认六类头与骨干联合更新，232993参数/232945可训练/头960；Sinc48与所有buffers固定。原F0、DURA R69v1.0、CPAIR及历史模型保持，无新投影、分类头、EMA、PL、目标U学习或增强。

坐标u=F.normalize(raw160,eps=1e−4)，与原生cosine头相同，不改变输出logits或再次单位化。已知anchor均为合法L。正对必须同真实TX、公共RX不同且day不同；其他同TX对从分母排除，自身不作正负对，其他已知TX为负项。每anchor须有正对。公共domain0..14按rx=domain//3、day=domain%3解析，不能由OE TX或未知K决定配对。

温度固定T=.1。记s_ij=u_i·u_j/T，P_i为上述正对，A_i=P_i∪已知其他TX。KCON每anchor损失为logsumexp(s_ij,j∈A_i)−mean(s_ip,p∈P_i)。OECON仅把全部当前OE64的相似度加入同一分母，其余不变。OE只有outside-six成员监督，不构造OE正对、不假定各OE异类、不读取OE TX；该损失也不保证OE之间不塌缩或未来匿名注册有效。

两臂共同目标为原CPAIR的CE＋T2教师KL×4＋公共15域等权centered pair-risk，再加0.1×对应原生对比损失。centered evidence c(z)=LSE(z)−mean(z)−log6，pair为同RX/day的softplus(cOE−cL)；原F0同G(clean)缓存L教师。每步共享L192/OE64、同SourceSampler种子2026103017/18，逐步与旧CPAIR全部2220步索引核对；新两臂重置相同采样器和原F0初值。AdamW lr/wd1e−4、eps1e−8、clip5，FP32/TF32off，每臂固定2220更新/568320前向，最终步冻结，既不追逐CAL/NV也不依据Q挑checkpoint。

参考[SupCon原论文](https://arxiv.org/abs/2004.11362)的监督对比思想。本轮是公共RX/day正对掩码、OE负项和原生分类坐标的项目变体，不声称完整论文复现；既有64维开放投影／VICReg／冻结特征后MetricNet对照不等价本轮完整原生学生，不继承其分数。

## 对照与推理

参考run `20261101-phase3-ground-centered-s2026092701-r01` / code `1ae125733a0f25eb9c887351eb63f11f730d8cea`，其CPAIR只作固定对照与源采样轨迹参考。四新读出为KCON_energy/KCON_centered/OECON_energy/OECON_centered；旧CPAIR_energy/CPAIR_centered完整cache及对应阈值复用。固定六列顺序为两旧、两KCON、两OECON，唯一主方法OECON_centered（index5）。不因某raw energy次要分数较高而改主。

同模型energy/centered共用同一默认头logits和closed，无额外推理参数。新CPAIR参考的col0是DMEAN，不能当作原F0或CPAIR；本轮旧控制来自其col2/3，canonical原F0只能来自独立baseline_logits/closed。两个旧方法各四overall共8项在独立评分后必须与旧JSON完全一致。

## 原数据权限与校准

只复用D7 ground r03 `20261101-phase3-ground-open-s2026092701-r03-source-clean-v1` / `sourceV-half-y-rx-day-s2026110101` 的VALIDATED_ONCE G(clean)。新训练L6300有六类y、OE6134仅成员监督和公共RX/day；不读Us/Us IDs或TX、目标U/UID、CAL/TEST/NV/query训练值。原F0等源Us祖先保留，源有标签和CAL依赖保留，不称全系统无标签。

六已知14-10/14-7/20-15/20-19/6-15/8-20；源RX index1/3/4/6/8=1-19/18-2/19-2/2-19/3-19，2021-03-08/15/23、split392005/eq1/256点。原V27000按每真实类×RX×day格300固定分为CAL150/TEST150，总各13500、物理互斥；knownTEST每类2250。构建者合法读V已知y分层并把TEST真值留私有侧，不把这次源TEST称新盲测。

源Q每条件16650=knownTEST13500+NV3150；未知19-15/19-16/7-1各1050，public角色隐藏。跨RX开发接收机3-2/index34、未来03-15/23 Q10238（5525/4713），known4508六类1491/1182/150/406/1186/93，unknown5730为5-5/10-7/2-19/7-11各1454/1439/1427/1410。03-23缺20-19/8-20，完整已知指标N/A，不以present-only认证。未注册不证明从未在OE或历史出现，不新增OE真TX来审查类互斥。

四条件source_clean_DURA/source_star_DURA/cross_clean_DURA/cross_star_DURA全部固定DURA v1.0 FP32/TF32off/reuse+graph/nativeoff。源G(clean)/G10P1复用，跨RX同序物理记录逐条经G；G10MIX01/600km/25MHz/2.45GHz/post_sync/noEQ/refSNR30dB/reflect/一次输入RMS/输出无RMS、scene3001/channel4001/future3133/3134/receiver2027不变。目标clean只冻结测试，不用clean参考恢复star，不重抽观测。

网络固定后仅用CAL13500按既有known24组规则拟合四新读出的24个阈值，另继承旧12值，共36个FP64/288B；主六值48B。所有阈值先于query冻结，零尾无法保护时记录失败并继续合法预测，不做测试修复。6方法×4条件24点完整后独立CPU接sourceknownTEST/sourceNV/crossclean/crossstar四真值各一次；预测不接TX、真实K、配额或query批次统计。

正式要求仍是macroRU95、CK95、原同DURA F0闭集−AK≤2pp且名单完整；源clean全部24组，跨clean总体+clean，源/跨star总体+六主景。主方法与全部五控制配对，known/unknown净条数、原F0与学生闭集、分RX/day/TX/scene/day×scene、AUROC/OSCR/CI分别报告。ground_source/cross_clean/ground_stage/frozen4与完整Phase3分开；未通过NO_PROMOTION，不按Q扩seed。即使冻结拒识过关，目标UDA/发现/未来注册仍未完成；AN/H_OW=N/A、完整ACTIVE、正式RX24-18未访问。历史NV/Q曝光与全局物理历史UNKNOWN保持。

## 成本与执行

两新学生训练共1136640前向、预测共134552，合计1271192；旧CPAIR真实新前向0、原F0真实新前向0，G20476。旧source data G99284/F036450、原F0/G/CPAIR训练与cache成本继承。训练合成smoke G2/F02/学生14（初值2+两臂各L4OE2），预测G2/F02/两新学生各2，共4；r03/PCD真实与合成0。主部署仍499944神经参数+六阈值48B，无新bank或投影，阶段墙钟不是B1时延。

trace完整记录每臂2220步CE/KD/centeredpair、KCON/OECON同模型诊断和选中SupCon、权重/温度、正负对计数及相似度、OE批内几何描述、源indices/曝光/unique、head浅中深梯度/Sinc0/clip和15域风险。对比同一模型时含OE分母的损失应不低于仅known，容忍FP32报告误差但记录原差值；不同学生绝对损失不作为测试性能排名。

默认N607，实际free≥4096MiB、每卡至多新增两个本任务worker，不停止/改动他人任务。单owner不可覆盖root；首真实CP无query smoke PASS即训练两臂→固定冻结→预测首smoke→CAL→四Q→CPUscore→全日志分析/报告/Git。只保留最小流程一次P0/P1、聚焦负测、单预检/发布archive一次SHA比较/compile和PID/CWD/argv/GPU/log增长。停止只限协议/祖先/输出冲突、非有限或不能合法预测，不按低分重跑。具体commit/命令/环境CWD/路径/GPU/预期产物写同run登记，不新增审批或哈希链。
