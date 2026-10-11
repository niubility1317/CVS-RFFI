# 从头20轮：早期身份几何与半监督固定对照

状态LOCAL_VERIFIED，未发布。目标仍为clean三种子均值84%。用户允许从头20轮并重新设计日程；E20追加20轮是另一条路线，不混入本批预算。

## 研究问题

源U高置信可能使概率KL对部分分类间隔变化不敏感，但尚未证明实际特征存在可修复缺陷。设弱视图logits为a、强视图为b，q=softmax(a)。KL对b的梯度是softmax(b)−q，局部Hessian为diag(q)−qq^T。q接近单类概率1时，部分间隔变化的概率响应很小。本批检验从早期加入表示目标是否有帮助，不把这项数学可能性称为已证实瓶颈。

保留身份CE和旧门控/参考响应/cosine分类头。同一U物理IQ的弱视图采用当前学生eval且停止梯度，强视图采用当前学生train。无EMA副本，无U身份真值。

- KL：mean(1[max(q)>=.95] KL(q||softmax(b)))，固定批大小分母，权重0.25。
- feature：mean(||normalize(z_strong)−stopgrad(normalize(z_weak))||²)，权重0.25，全U参与。
- margin：0.5*mean((center(b)−stopgrad(center(a)))²)，权重0.01，全U参与；center去掉逐样本logits公共平移。

几何目标不硬分配低置信U类别，但可能保留错误身份结构、过度压缩特征或缩小类间间隔。记录单位特征标准差、损失及骨干分量梯度。损失单位不同，权重不是梯度匹配或最优值声明；不宣称一致性公式本身为新发明，也不将本批称为所有物理因素已可辨识解耦。

## 六方案×三种子

|方案|L监督|U概率KL|接收响应动作|U特征一致性|U间隔一致性|
|---|---|---|---|---|---|
|CE|clean CE|关|关|关|关|
|SSL|clean CE|开|关|关|关|
|PHYS|clean/action CE|开|开|关|关|
|FEATURE|clean/action CE|开|开|开|关|
|MARGIN|clean/action CE|开|开|关|开|
|BOTH|clean/action CE|开|开|开|开|

主比较BOTH−PHYS，两个单项分别与PHYS比较，组合再与各单项比较。三seed为2026092701/02/03。同seed初始权重完全一致；P0复用已有E20预测仅评分，不加载权重。18row固定执行，不按局部测试成绩改队列。

全部从头20epoch×222=4440次更新，L batch128。前2epoch（444步）仅L，随后4epoch（888步）渐入U和动作，之后保持。研究变体有3996个U批、511488次样本使用，校验56700条U全部覆盖。CE不读取U cache，不创建U iterator，不做弱/强前向。AdamW全身份参数峰值2e-4、weight_decay1e-4、20epoch cosine至1e-6。没有学习型辅助网络或其预热状态；本批隔离身份学习目标变化。

物理组沿用L-only接收响应库：4个其他RX×半/全强度八候选均匀采样、33tap有限线性卷积、幅度限制±3dB。L按留TX/日统计；U只用RX/day与L总体统计。L动作CE占比渐入至0.5；U强视图叠加std0.015噪声。四个物理组共享输入、动作和噪声。该有限响应是源接收差异代理，不是星地增强，也不保证所有动作严格保留身份。

## 数据和实现边界

L6300/U56700/V27000及物理ID不变。旧source_cache只取原始x、合法L/V的y、rx/day/ids；所有旧特征、margin、辅助参数丢弃。不会加载E200/E20后重置头伪称scratch。已测试在禁止torch.load时仍能构建原生模型。

新循环采用明确RX/day完整轮转，和历史native训练器不承诺逐位一致。因此新CE是同执行路径的必要对照，历史E20是用户指定参考。E20真实来源另经三seed检查，独立scorer再次核验。此前epoch预算基准已暴露，如实报告，不让测试结果回流选择或调参。

源V只在E5/10/15/20检查健康，不选epoch/方案。无半监督的基线和含U变体均采用相同合法源角色。固定final20后自动clean168000条预测，另进程truth-last评分，14个overall/RX/TX与4个日期指标逐row完整输出。

## 检查和执行

本地检查通过：同seed六方案初始state逐项相同；禁用所有checkpoint读取仍能scratch构建；warmup与普通CE完全相等；CE/SSL/物理组实际身份前向数1/3/4；U fixture无y；两种几何目标分别向encoder传递有限非零梯度；停止梯度、logit平移不变、相同视图零损失；权重保存重载预测一致；配置篡改和不完整评分被拒绝。

远端预检检查三seed真实源角色和scratch反向，并检查E20参考来源，不访问target。一次独立P0/P1审查后，固定Git版本发布。共享八卡，每卡最多3个总进程，至少12GB空闲，保留其他owner任务。每row一个owner从训练串联预测与评分；失败保留产物，健康行继续，不自动重跑。完整step压缩日志、epoch JSONL/CSV、实际参数、梯度、位移、耗时和显存保存。所有18row训练和测试完成才汇总最终结果。


2026-10-12：状态 LOCAL_ONLY_SUPERSEDED。用户要求浅／中层 MixStyle 和 Fishr，扩大为 m30 新登记；本 m18 未发布、未启动。此前静态检查记录保留。
