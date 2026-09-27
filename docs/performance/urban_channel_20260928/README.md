# Urban星地增强瓶颈与训练加速

核对日期：2026-09-28。分析依据为完整历史日志、当前source训练快照、实际执行代码以及本轮有界合成诊断。**urban瓶颈不能归结为困难场景比例不够：当前A1/RC4无标签路径实际上没有训练urban身份不变性；同时低仰角城市模型含有大量遮挡、低新增信噪比及接收失锁样本。**前者是已证实的接线覆盖事实；后者需要分清模拟质量与真实TX可辨识性，不能仅凭低准确率宣称信息完全消失。

本轮将保持训练语义的执行加速与新的学习方法分开。已实现独立RC4教师身份前向，准备了只改变执行开关的source-only加速配置；不改现有运行任务、信道强度、数据角色或目标评分规则。新增urban学习目标仍需source侧证据及完整训练验证，没有把代码实现或合成基准写成urban准确率已经提高。

## 1.核查范围

- 读取所引用的“详解DAOT与FastTrust-RC4”对话，并从实验总索引找到9月18日四组、9月20日六比例及9月27日五seed原生CVS记录。
- 完整解析历史1852条epoch：六比例1200条、旧四组652条。后者包含用户停止的full三组，不把partial模型当作同预算E200对照。
- 2026-09-28 00:06:34（UTC+8）只读采集当前五seed全部601条已完成epoch及全部26539行stdout。进度分别为392005的E115，其余四seed的E123/E120/E121/E122，均未生成final checkpoint。这是时间点证据，不是稍后实时状态。
- 旧七个E200模型的8232000条预测复计沿用9月21日既有证据；本轮没有再次访问这些target预测或truth。历史target指标仅描述已发生的问题，不用于新配方排序、阈值或选择性重跑。
- [信道与监督逐项源码考证](channel_findings.md)、[历史完整曲线分阶段摘要](evidence/historical_log_analysis.json)、[当前source完整日志快照](evidence/current_source_logs.json.gz)。本轮重新解析历史结构化记录；历史stdout沿用原全文件扫描结果，没有冒称再次从远端重读旧stdout。

## 2.先纠正两个关键的执行事实

### 2.1 配置列出urban，不代表DAOT用了urban

A1入口强制`daot_teacher_view_count=2`，L/U两个训练入口只有该值≥3才构造hard视图。当前实际为：

```text
DAOT teacher：clean + practical_high
DAOT student：practical_high
hard配置中的mid/low_urban：未执行
```

旧六比例报告中“teacher mid/low_urban保持不变”的表述不准确，应纠正为“未执行的hard配置字段保持不变”。这也改变了故障解释：不能说当前DAOT把受损urban强行对齐导致失败，因为urban并未进入该DAOT路径。

### 2.2 RC4的urban无标签卫星监督也未开启

`rc4_satellite_hard_only=false`使卫星mask为空，相关卫星hard CE不执行。残留`rc4_lambda_satellite=.10`不是启用证据。因此当前困难urban的直接身份监督主要来自E80后的**有标签拼接卫星CE**。增加L拼接分支的urban比例，不会自动扩大DAOT/RC4的U域学习范围。

六比例只改变L拼接分支的场景组合及条件比例；E1–40/E41–90/E91起的应用概率仍为0.3/0.6/0.8。困难比例75%是已应用星地样本中的条件比例，不是所有训练输入或所有U样本的75%。

## 3.为什么low_urban会难很多

这里high/mid/low指仰角，不是轨道高度；轨道高度默认600km。urban的high/mid/low仰角分别45°至80°、20°至45°、10°至30°。按当前代码先验积分，三个urban场景平均BLOCKED概率约9.81%、38.38%、56.38%；low_suburban约5.94%。这是模拟器先验，不是从目标准确率反推。

`residual_noeq`仍保留传播多径H。它只把短记录内的时变传播近似为冻结等效信道；noeq使G=1，不能解释为“已均衡后只剩轻微残差”。低仰角增大距离损耗，BLOCKED没有直达分量，噪声底却固定在参考功率下。AGC同时放大信号和噪声，不能恢复信噪比；失锁又改变残余频偏和相位跟踪。这是相互关联的困难因素，不是简单再加一种噪声。

256点、25Msps仅为10.24µs。多径零点、有限窗、滤波边界与新增接收机效应可能遮蔽某些TX特征，但物理TX标签并没有因此变成错误标签。需要测的是观测可辨识性，不能将“标签相同”视为“每个受损视图都应具有同样的可学习置信度”。

当前模型的仰角、LOS概率、阴影和多径处理属于工程化代理。3GPP TR38.811分别讨论仰角相关LOS、大尺度损耗和快衰落，但本实现的具体状态概率函数、误差门限及残差近似没有因此自动成为标准标定参数，也不是实测卫星链路。[3GPP报告公开文本](https://atisorg.s3.amazonaws.com/archive/3gpp-documents/Rel15/ATIS.3GPP.38.811.V1530.pdf)

新增质量诊断使用合成单位功率IQ，只量化模拟器产生的遮挡、接收状态和相对于新增噪声的信号功率。真实ManySig输入已有地面噪声，因此这里的SNR不是真实接收总SNR，更不是TX识别率。具体数值及边界见[本轮测量](MEASUREMENTS.md)。

## 4.历史曲线能证明什么

六组固定E200的low_urban为52.42%至53.71%，clean为79.14%至80.69%。提高困难比例并未产生各类一致增益：例如历史mid50∶50相对旧配方总体增加0.5083个百分点，但TX3召回下降约8.79个百分点。单个seed、同一模型的多个epoch及大量相关视图都不能提供多次独立训练证据。

完整曲线中，DAOT从E21启用，L卫星CE从E80启用，E91应用概率升至0.8；E80附近总loss上升含有目标构成变化，不能直接称发散。六组source clean V最终98.61%至98.66%，不能据此认为urban已经学会。日志中的DAOT教师准确率和U consensus测的是实际clean/high视图，也不能充当urban教师可靠性。

各组仍有21至24次非有限梯度安全跳步，非有限loss跳步为0。训练完成不等于数值完全洁净；本次加速保留梯度检查、裁剪、AMP与跳步处理。旧日志NaN关键词主要在未启用诊断字段，不等于相同次数的训练失败。

## 5.方法优化应解决覆盖与监督质量

**优先设计是保持clean＋high教师，单独扩大学生对urban的覆盖。**这不是把`daot_student_scenario`字符串改成urban就完成了：当前teacher第二视图与student共用同一张量，直接改字符串会一起改变教师。需要在L/U两个入口显式分离teacher和student视图，保证教师可靠参考与新增学生扰动可单独归因。

建议按以下source-only证据顺序推进，避免直接进行大比例网格搜索：

1. 使用合法source L与单一只读V，按LOS/PARTIAL/BLOCKED、lock、质量及仰角分层，核对真实教师正确率、错误一致率、类间margin和身份特征距离。U隐藏TX不得用于正确率核验。当前尚未完成这项真实source困难视图推理，因此不预填阈值或宣称哪项损失必然提升。
2. 若source证据支持，保留原教师，加入独立urban学生策略。低质量样本仍保留，分别考察监督权重、身份一致性覆盖率和类别下限，而不是删掉失败样本或调弱固定评估信道。
3. 保持同一数据契约、E200固定末轮、同一执行优化与源域选模规则，比较原方法和单项改动。新的完整训练预算尚未确定，本轮没有自动启动这一对照。

现成可靠性函数不能直接套用：Practical没有它所依赖的K_db字段；A1 mean路径又将reliability/recoverability覆盖为1；原recoverability描述教师而非新的urban学生。且加权均值会抵消整批一致降权，EMA loss归一化也可能抵消尺度变化。需要显式区分样本相对可靠性与整体有效监督覆盖，不能用一个看似较小的权重证明门控已生效。

不优先同时开启full扩展、EQ、更多教师视图和新损失，因为它们改变多个因素并增加计算。模拟器EQ使用真实模拟H加估计误差，不等于已实现实际接收端导频或盲均衡；不能把这条捷径当成可直接部署的urban修复。

## 6.加速落地与使用边界

9月20日已经实现常量/FIR缓存、buffer拷贝、同batch双进程、验证复用、增量日志及固定评估缓存。六比例完整训练由旧约34.30小时降至19.17至19.46小时，纯训练batch部分由25.42小时降至17.06至17.53小时。总耗时下降43%至44%是已有历史结果；它包含不同评估工作量与资源争用，不能作为本轮新增优化的成绩。

9月28日00:06快照显示，正在运行的五seed原始配置中这些执行开关多数仍关闭。保留原实验的版本一致性，本轮没有热修改或重新启动它们；加速配置供后续已登记的独立run使用。

本轮新增独立`a1_rc4_teacher_identity_only`：仅为eval EMA教师计算RC4实际消费的`tx_logits/z_id`，跳过未使用的域/辅助分支，不启用捆绑的`a1_runtime_fast`。完整前向与身份前向的CPU FP32、CUDA FP32及FP16输出、参数、buffer和RNG已逐元素核对。student的BN/Dropout、路由、loss与EMA频率保持原实现。

命名RNG延迟创建也进行了精确等价与配对速度测试。减少初始化并不保证加速，因为Mapping访问增加开销；最终启用决策以[测量结果](MEASUREMENTS.md)为准，不把通过保真测试当成通过性能测试。

[新加速配置](../../../experiments/adv3b02_xuc/configs/rc4_practical_residual_noeq_fast_20260928.json)来自9月27日source-only原配方，仅改变[11个执行开关](execution_recipe_diff.json)，继续使用两教师视图、high学生、原CE/RC4参数及E200末轮规则。因此它是**原方法的执行加速版**，不是已经训练验证的urban新方法。使用现有`train_rc4_matched.py`入口；完整run仍需按数据契约登记独占输出并核实资源。固定评估缓存目录需由相应评估run明确设置，当前训练recipe未伪造一个公共缓存路径。

本轮不引入跨训练batch预取：普通增强与模型共享全局Torch随机序列，直接提前下一批会改变轨迹。也不将训练student直接拼批以省前向，因为那会改变BN/Dropout。性能测量有预热、交替顺序及GPU同步；同步只用于有界测量，不放进正常训练循环。[PyTorch Profiler官方说明](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

## 7.交付与仍待验证的事项

源码、默认关闭的新教师开关、聚焦测试、两个可复跑benchmark、source-only加速配置和本报告已纳入同一Git分支。基准使用本机NumPy2.2.6、Torch2.10.0和RTX5070Ti；没有在N607的Torch2.1/RTX3090上重新跑完整训练，不承诺新E200耗时。

仍未证明：新的urban方法能提升准确率、真实source低质量视图的类别可辨识性、N607整步/整轮增益，以及新的独立确认数据泛化。完整训练预算问题保持待定。已有目标结果不回流训练选择；不同模型seed不能消除同一目标物理数据已经暴露的事实。

- [测量与验证明细](MEASUREMENTS.md)
- [本轮诊断登记](../../../automation_reports/CV-SincNet/20260928-phase1-urban-execution-synthetic-s000041-r01/experiment.json)
- [当前source运行证据摘要](evidence/current_source_summary.json)
- [教师前向微基准](../../../experiments/adv3b02_xuc/tools/benchmark_rc4_teacher.py)
- [信道微基准与质量诊断](../../../experiments/adv3b02_xuc/tools/benchmark_urban_channel.py)
