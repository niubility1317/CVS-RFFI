# CVS Stage2-C极轻型快速适应目标模式

日期：2026-07-14
协议源：`E:\type10-7\项目.md`第10.3.1节
状态：Git承载面的协议增量镜像；科研语义仍以根目录`项目.md`为准

## 成功门槛

|指标|独立确认门槛|
|---|---:|
|target-old总体准确率|`old_acc>=0.88`|
|旧类逐类下限|`min_old_class_acc>=0.85`|
|5个真实target-new TX|`seen_new_acc>=0.90`|
|10个真实target-new TX|`seen_new_acc>=0.88`|
|20个真实target-new TX|`seen_new_acc>=0.83`|

确认矩阵固定覆盖5个target receiver、至少5个独立确认seed和3个正式`leo_weak`场景。5/10/20类使用按数据覆盖预先确定的嵌套真实TX集合。开发seed只用于选择统一K-shot工作点和超参数；锁定后不得跨K、跨seed或跨新类规模拼接结果。

## 部署与资源边界

- 逐样本推理；禁止query真实old/new角色、全批类别数量、每类quota、query标签和query集合图结构。
- query不得参与adapter拟合、阈值拟合、模型选择或早停。
- 默认冻结`ADV3B02` backbone，不执行backbone梯度更新。
- 极轻型首选档：1-view、adapter可训练参数不超过50,000、适配不超过20epoch、无dense query图、持久化适配状态不超过128KB。
- 允许对K个互不重复的物理support样本在同一`leo_weak`族内生成至多3个预注册增强view，只用于一次性support-only enrollment；增强view不得重复计入K，query仍按1-view逐样本推理且不得参与拟合。必须报告support增强清单、每个物理support的backbone/FFT前向次数和一次性enrollment计算量；该结果不能外推为跨场景或真实在轨泛化。
- 可以把冻结ADV3B02自带source classifier对同一物理样本产生的6维logits作为逐样本特征输入；source classifier必须保持冻结，logits不能经过query-batch归一化、角色门控或类别配额重排，也不能使用query标签选择融合权重。该机制属于冻结source classifier bank的复用，不是old/new query角色Oracle。
- 可以用冻结source prototype bank对历史已注册类执行小权重prototype shrinkage；source bank只能来自`R_s`和source checkpoint既有类别，不能包含`R_t` query、target-new或unknown样本，推理时仍必须在全部注册类上执行同一个逐样本argmax。
- 报告逐类、逐receiver、同row old/new/H、完整loss trace或闭式求解诊断，以及MAC、时延、峰值显存、状态大小和相对identity-only单qKNN的Pareto变化。

## 声明边界

开发集达标、平均指标达标但旧类最低类不达标、Oracle角色/配额结果、query参与适配结果、或只覆盖单seed/单receiver/单场景的结果，均不得声明目标完成。
