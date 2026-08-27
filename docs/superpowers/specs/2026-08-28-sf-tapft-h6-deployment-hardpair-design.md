# SF-TAPFT H6部署化与HardPair增强设计

## 1.目标与边界

本轮在ADV3B02 CORE90 checkpoint和既有`p2_min_v1/VALIDATED_ONCE`目标域support capsule上，把H6 Fast-Strong V3从研究选择流程改造成固定策略部署入口，并以同row实验验证M02性能档和support-derived HardPair。模型冻结前只读取60条K=10 support；query不参与训练、选择、校准或停止判断，prediction完整后才由独立scorer连接最大Q180 truth。

本轮不注册新类，全部结果使用`DA0_REG0/DA1_REG0`。不加入Adapter、完整`t3`、frequency/domain更新、CVaR或vector scaling，不扩大receiver/scene/K/seed矩阵；泛化确认留到本轮候选通过后。

## 2.实现设计

### 2.1 deployment-only入口

新增固定full-support入口，直接消费冻结配置和全部support，只训练一次，不执行4-fold CV、逐步validation或query访问。H6固定更新target head和`t3.norm(weight+bias)`，日程为300步S15轨迹、150步低学习率尾段和70步cached-head精修。M02-DeployReplay固定更新target head和all-time norm，训练长度从历史selection artifact恢复并冻结，不在本轮重新选择。

### 2.2冻结前缀缓存

缓存边界必须位于全部可训练`t3.norm`参数之前，并保留从该边界到target head的可微后缀。实现先以FP32缓存验证：

- 同一模型、support和状态下，完整路径与缓存路径的最大logit差`<1e-5`；
- 对全部许可参数，完整路径与缓存路径的最大梯度差`<1e-5`；
- prediction完全一致。

FP16缓存作为独立资源候选。由于半精度舍入不保证`<1e-5`，其验收采用有限数值、prediction闭合及同row性能非劣，不把FP16舍入误判为实现错误。若真实ADV3B02图无法在`t3.norm`前形成封闭可重放后缀，则R0记录`PREFIX_CACHE_UNSUPPORTED_FOR_GRAPH`，保留deployment-only固定训练，不以伪缓存替代。

### 2.3严格delta

复用现有delta-only loader/exporter，只保存target head和许可norm的FP16 delta、temperature、class IDs和基础checkpoint绑定。H6上限10KB、M02上限10KB；不得保存完整模型state作为部署状态。

### 2.4 Support-derived HardPair

每个训练步只从当前support logits和support labels确定每类最难竞争类：

$$j_c^*=\arg\max_{j\ne c}\frac1{n_c}\sum_{i:y_i=c}l_{ij}.$$

损失为：

$$\mathcal L_{hard}=\frac1C\sum_c\frac1{n_c}\sum_{i:y_i=c}[m+l_{i,j_c^*}-l_{i,c}]_+.$$

本轮只比较`lambda_hard=0、0.03、0.05`；margin、训练步数、Norm范围、rho、temperature和anchor完全相同。困难类别编号不得写入配置或代码，query侧不执行HardPair。

## 3.实验矩阵

|row|候选|唯一变化|用途|
|---|---|---|---|
|R0A|H6 full deploy|固定full-support、无CV|部署基准|
|R0B|H6 cached FP32/FP16|仅改变执行与缓存精度|等价性和资源|
|R1|M02-DeployReplay|all-time norm和历史固定步数|性能档公平复跑|
|R2A|H6 HardPair-0.03|`lambda_hard=0.03`|困难边界保护|
|R2B|H6 HardPair-0.05|`lambda_hard=0.05`|强度消融|

R0先闭合。R1、R2A、R2B随后可按每张GPU最多两项并行发布，但每个run ID只有一个launch owner。低性能不停止任务。

## 4.晋级规则

以同一最大Q180上的H6为锚：BA不低于83.3333%，class floor不低于56.6667%，任一类别准确率下降不超过5pp，NLL不高于0.521858。资源上可训练元素不超过1584、FP16 delta不超过10KB、无新增推理分支、backbone训练forward不超过450次，或缓存后实际训练MACs低于原H6。

同时满足性能和资源门槛的最小候选才可晋级。本轮即使通过，也只形成单receiver、clear-weak、K=10、单seed候选；多receiver、三场景、K=10/5/2和三seed是后续推广确认，不在本轮提前宣称。

## 5.测试与交付

实现按RED→GREEN：先增加deployment-only不触发CV、前缀logit/梯度等价、FP16有限性、HardPair类别置换不变、HardPair禁用零差异和query不可见测试，再实现最小代码。聚焦测试后执行真实checkpoint无query smoke和一次独立P0/P1正确性审查。正式结果写入run报告，镜像到Git承载面，精确stage、commit、push并核对远端OID。
