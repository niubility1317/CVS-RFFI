# SF-TAPFT H6 P0原位适配与轻量部署设计

## 1.目标与边界

本轮在不改变H6目标函数、许可参数集合和训练日程的前提下，消除部署训练对完整checkpoint模型及完整初始状态的重复持有，建立“基础checkpoint+紧凑delta”的真实可加载闭环，并用隔离资源基准量化额外内存、时间与缓存成本。

训练只读取匹配`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`的旧类K=10 support。query不参与训练、安全复核、fallback、停止或候选选择；prediction完整后才由独立scorer连接truth。本轮不注册新类，状态名使用`DA0_REG0`和`DA1_REG0`。

## 2.原位适配所有权

保留既有研究入口的复制语义，新增显式部署原位入口。调用方把一个本轮独占、可修改的模型实例交给适配器；适配器不得再复制完整模型。开始训练前只保存：

- 许可更新模型参数的CPU锚点；
- target head的CPU锚点；
- 需要证明冻结的buffer值；
- 构建紧凑delta所需的dtype、shape和参数名。

优化器参数组只能包含许可参数。非许可参数的安全证据由“优化器可达性+`requires_grad`集合+冻结buffer等值”组成，不新增逐成员hash或签名。训练异常或FP16安全复核失败时，只从上述锚点恢复许可参数并执行预登记fallback；不得回读query或source。

## 3.稳定前缀/后缀接口

把现有hook型缓存能力整理为两个稳定接口：

- `encode_h6_prefix(model,x,precision)`：在所有许可`t3.norm`参数之前生成不可变cache；
- `forward_h6_suffix(model,head,cache)`：从cache重放可微后缀并输出logits。

`H6SuffixTrainer`只持有原模型、head和cache的引用，不注册第二份模型参数。缓存必须记录总字节数、tensor数量、dtype和batch维度；释放训练对象后不得保留无用的全模型副本。

## 4.紧凑delta闭环

delta导出不再依赖“未变化的base model对象”，而是直接用训练前许可参数锚点计算：

$$\Delta\theta_i=\theta_i^{adapted}-\theta_i^{base}.$$

部署包至少包含基础checkpoint绑定、类ID、温度、参数名、shape、dtype和delta tensor。materializer必须从冻结checkpoint加载模型，严格核对参数名/shape/class IDs后应用delta。P0部署行默认不生成完整clean-single bundle；兼容模式可显式开启，但不能把它计作最小部署状态。

## 5.FP16支持集安全复核

FP16缓存训练完成后，使用适配后模型执行一次FP32完整路径support前向，并与缓存后缀输出核对：

- logits全部有限；
- support argmax零不一致；
- 真实类别margin符号不恶化为负；
- 每类support recall零下降；
- 记录最大logit绝对差和最小margin。

任一硬条件失败时，在query prediction前恢复许可参数锚点，并以相同seed、相同日程、相同目标函数执行FP32缓存fallback。若fallback仍失败，则标为系统技术失败且不产出性能结论。

## 6.实验矩阵与资源测量

|row|执行模式|cache|输出|用途|
|---|---|---|---|---|
|P0A|现有复制模式|FP16|兼容bundle+delta|同row功能与资源锚点|
|P0B|原位模式|FP16+FP32安全复核|delta-only|目标部署候选|
|P0C|原位模式|FP32|delta-only|FP16 fallback与数值对照|

资源测量在单GPU、单进程、无并行实验干扰条件下执行。每行预热3次、正式10次，每次CUDA同步后计时；报告wall-clock median/P90/max、CPU RSS峰值、CUDA allocated/reserved峰值、cache字节、delta字节、训练forward次数、可训练/实际变化元素。常驻推理模型内存与适配额外峰值分开报告。

## 7.验收与晋级

功能验收：P0B与P0A在相同seed/support上许可参数最终值、delta materialize后logits和prediction一致；非许可参数及buffer无变化；delta-only可独立形成prediction artifact。

性能验收沿用当前H6 Q180锚点：BA不低于83.3333%，class floor不低于56.6667%，任一类别准确率下降不超过5pp，NLL不高于0.521858。工程晋级要求P0B适配额外峰值显著低于P0A，delta不超过10KB，且训练wall-clock不劣于P0A的10%。若P0B未满足，保留P0A为科学实现并分析瓶颈，不以工程失败改写性能结论。

P0完成后才进入新未暴露capsule上的P1 D0–D4；P2多receiver/scene/K/seed确认只对P1晋级结构执行。
