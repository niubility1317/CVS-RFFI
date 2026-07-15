# CVS Phase2 Stage2-B/Stage2-C极轻型快速适应目标模式

日期：2026-07-14
协议源：`E:\type10-7\项目.md`第10.3.1节
状态：Git承载面的协议增量镜像；科研语义仍以根目录`项目.md`为准

## 成功门槛

|指标|独立确认门槛|
|---|---:|
|target-old总体准确率|`old_acc>=0.92`|
|旧类逐类下限|`min_old_class_acc>=0.88`|
|5个真实target-new TX|`seen_new_acc>=0.92`|
|10个真实target-new TX|`seen_new_acc>=0.90`|
|20个真实target-new TX|`seen_new_acc>=0.86`|

以上绝对门槛默认在统一`K=10`达到。开发阶段仅允许在K10选择一套candidate与超参数；锁定后以相同candidate评估嵌套K5。K5复用K10前5个物理support和完全相同query，matched row的`old_acc`、`min_old_class_acc`、`seen_new_acc`和`H_old_new`相对K10均不得下降超过3个百分点。

`old_acc>=0.92`与`min_old_class_acc>=0.88`同时用于Stage2-B旧类校准和Stage2-C同row旧类适应。Stage2-B仅包含合法target-old support/query，不得报告或替代Stage2-C的seen-new准确率与`H_old_new`门槛；Stage2-C仍须同时满足5/10/20个真实target-new TX的门槛。

K1自2026-07-15起作为极少support下旧类适应的强制压力门槛。K1复用K10第1个物理support和完全相同query，不得用K1 query重新选择candidate、域适应方法、超参数、epoch或门限。每个K1 row必须同时报告`old_acc_before_increment`、加入全部注册类后的`old_acc`和`old_adaptation_gain=old_acc-old_acc_before_increment`；独立确认矩阵总体聚合及每个target receiver聚合均要求`old_adaptation_gain>=0`。未满足时只能声明K1负迁移诊断，不能声明极少shot适应有效；K1正增益也不能替代K10绝对性能门槛。

K1允许使用source validation和注册support标签进行角色对称的全局对角CORAL、低参数FiLM、support增强view一致性、稳健原型及support原型Gram去混淆。所有已注册类必须使用同一规则；禁止依据old/new身份选择不同adapter、shrinkage、阈值或决策分支，也禁止只对source旧类使用prototype bank后声称无角色Oracle。增强view仍按同一物理sample ID计为1-shot，并须单列一次性enrollment前向与状态开销。

分K旧类遗忘与绝对准确率并列为正式目标，固定使用嵌套`K∈{1,5,10,20}`。每个matched row必须报告`old_acc_before_increment(K)`、`old_acc_after_increment(K)`、`average_forgetting(K)=old_acc_before_increment(K)-old_acc_after_increment(K)`和`old_adaptation_gain(K)=-average_forgetting(K)`；同一K前后必须复用相同旧类query和相同View策略。K1继续要求总体及逐receiver的`average_forgetting<=0`；K5/K10/K20要求独立确认集平均遗忘不高于对应identity-only单qKNN基线，并同时保留各K准确率、最低类和H约束。汇总同时报告`worst_K_forgetting`与`mean_positive_forgetting`，不得通过牺牲新类性能、跨row拼接或挑选单个最好K/epoch降低表面遗忘。

K1/K5/K20 query只用于锁定后的压力评估和独立确认，不能重新选择candidate、adapter、epoch、域适应超参数或门限。跨K抗遗忘机制只能由source validation、注册support稳定性和K10开发row确定；某个K失败后不得在同一query上继续逐K调参。

K1适应后的target-old准确率还必须显著超过严格直接ADV3B02旧类分类头。使用相同旧类query sample ID和相同基础LEO View定义`delta_vs_direct_ADV3B02_K1=old_acc_after_increment(K1)-direct_ADV3B02_old_acc`；独立确认总体要求至少`+0.02`且matched row配对95%置信区间下界`>0`，每个receiver聚合要求不低于0。直接基线严格加载同一base checkpoint，不使用target support、query拟合、FFT、TTA或adapter；它没有target-new输出头，因此不能用于比较新类准确率或H，也不能替代其他K1/K10门槛。

确认矩阵固定覆盖5个target receiver、至少5个独立确认seed、3个正式`leo_weak`场景以及matched K10/K5。5/10/20类使用按数据覆盖预先确定的嵌套真实TX集合。锁定后不得跨K、跨seed或跨新类规模拼接结果。

## 部署与资源边界

- 逐样本推理；禁止query真实old/new/unknown角色、全批类别数量、每类quota、query标签、query排序/分块和query集合图结构。历史role/quota Oracle结果统一标记为`PROTOCOL_INVALID_FOR_DEPLOYMENT`，不得生成新候选或进入本目标的开发、确认、排名和完成判定。
- query不得参与adapter拟合、阈值拟合、模型选择或早停。
- 默认冻结`ADV3B02` backbone，不执行backbone梯度更新。
- 极轻型首选档：1-view或逐样本1→3→5-view门控、adapter可训练参数不超过50,000、适配不超过20epoch、无dense query图、持久化适配状态不超过256KB。
- 用户授权的`performance-relaxed`档允许把首选参数、适配轮数/步数、持久状态或平均View计算提高50%–100%，绝对上限为100,000参数、40epoch、512KB和5次backbone前向。角色/类别配额Oracle、query拟合、dense query图和跨query决策仍禁止；报告必须给出实际增幅和同row Pareto对照。
- 允许对K个互不重复的物理support样本在同一`leo_weak`族内生成至多3个预注册增强view，只用于一次性support-only enrollment；增强view不得重复计入K，query仍按1-view逐样本推理且不得参与拟合。必须报告support增强清单、每个物理support的backbone/FFT前向次数和一次性enrollment计算量；该结果不能外推为跨场景或真实在轨泛化。
- 正式target-old/target-new support和query必须逐行来自对应`leo_weak`场景，clean waveform/feature/prototype不得进入support、query、候选选择或promotion。新训练的`effective_feature` ground adapter及source receiver holdout也固定为`leo_weak-only`reference/prototype；历史ADV3B02基础checkpoint的既有训练来源单独声明，不得把它与本次新增适配输入权限混写。cache缺失场景证明、含clean channel view或row场景不一致时必须fail closed。
- 可以把冻结ADV3B02自带source classifier对同一物理样本产生的6维logits作为逐样本特征输入；source classifier必须保持冻结，logits不能经过query-batch归一化、角色门控或类别配额重排，也不能使用query标签选择融合权重。该机制属于冻结source classifier bank的复用，不是old/new query角色Oracle。
- 可以用冻结source prototype bank对历史已注册类执行小权重prototype shrinkage；source bank只能来自`R_s`和source checkpoint既有类别，不能包含`R_t` query、target-new或unknown样本，推理时仍必须在全部注册类上执行同一个逐样本argmax。
- 报告逐类、逐receiver、同row old/new/H、完整loss trace或闭式求解诊断，以及MAC、时延、峰值显存、状态大小和相对identity-only单qKNN的Pareto变化。

## 声明边界

开发集达标、平均指标达标但旧类最低类不达标、Oracle角色/配额结果、query参与适配结果、或只覆盖单seed/单receiver/单场景的结果，均不得声明目标完成。
