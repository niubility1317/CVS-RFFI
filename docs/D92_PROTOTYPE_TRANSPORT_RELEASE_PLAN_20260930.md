# 原型条件化LocalRidge联合适配：运行计划

状态：IMPLEMENTED_VALIDATED_PREPARED_NOT_LAUNCHED。run为`20260930-phase2-d92-prototype-transport-support-m2-r01`，release为`d92_prototype_transport_support_20260930_r01`。root是唯一launch owner。本文记录本次唯一完整support试验，尚无真实性能或资源收益。

LocalRidge仍为最终分类器；合法目标support类原型确定局部竞争关系，监督学习10个存储参数、9个约束自由度，形成保范切向旋转。固定保留0.5原interaction距离，另一半使用适配距离。B仅旧类训练；C_seq真继承B参数、锚点和最终旧原型，统一注册全部类；C_reset从零参数及零锚点重新适配，作为继承策略对照。不额外训练自由logits、encoder或BN，不引入源样本/逐样本源特征。

矩阵固定160parent：2个既有source-only模型seed×rx3/rx1两个cohort×每row40任务。旧6类，新增0/2/5/10/20，K1/5/10/20；support seed2026092711，实际场景practical_high和practical_low_urban。practical residual、residual/post_sync/noeq、25MHz。只复用合法support特征缓存，不读query；数据身份不变，不重建/重验数据。trueK1仅数值检查，proxy trainK1不微调、精确R0。A及B−A缺失记N/A；B0不充当地面A。

使用四次有界归一化投影梯度，每次最多三次Armijo试探，初始步长0.125、回溯0.5、Armijo系数0.0001。数值容差明确固定在frozen config。当前接受态唯一，不选最好外层表现、不扫描步长或预算。零梯度、零位移、试探预算耗尽保留合法当前态并记录原因，不伪装成收敛；失败试探仍计成本。

输入availability证据为当前run的`evidence/prototype_transport_preflight_1790763166365017600.json`：四个现有cache binding VERIFIED，三个新output/release路径均不存在，CPU两lane、BLAS每lane2线程、GPU不用。此前一次草稿错误沿用了通道run编号，注册被existing-path guard拒绝且preflight UNAVAILABLE；没有新启动或覆盖旧run。错误配置及旧路径availability证据均保留，修正后的编号和全部输出路径已加入编排测试。

相关行为验证共54项：核心27项；入口13项（修复后按受影响版本验证）；编排14项。核心实施审查无P0/未关闭P1。真实修复及原失败证据保留：原型加权GEMM/GEMV批量归约差异、独立summary遗漏冻结eta零和容差因子、outer trial覆盖Armijo trial字段，以及草稿run编号错误。未放宽严格映射检查、投影KKT或日志逐项一致性。合成数值通过不代表方法性能改善。

最大信息阶段936，最大接受更新3744、目标前向12168、内头36504、额外最终头936、总头40608、伴随三角求解22464。这些是固定矩阵上界，非实际计数；更新阶段计数只统计实际接受更新，零更新阶段的初始前向/反向依然计费。前向alpha三角求解不包含在伴随计数内。proto构造/距离/变形前向、接受和拒绝试探、反向、评分调用/物理样本数均独立累计；共享准备仅计一次。

仅80B的参数数值不能表示完整模型。完整状态必须计原/适配support、两组原型、alpha、中心化与继承绑定；记录CPU硬件、线程、实际训练/评分耗时、RSS和状态bytes。部署包、总星地传输、星载硬件时延和GPU显存未测量记N/A；新增地面数据/统计payload为0。本方法原型在卫星侧由合法support计算，不能把原型RAM写成地面传输量。

训练日志保留CVS文本、full JSONL、紧凑JSONL/CSV，含实际配置、各试探损失分项/步长/梯度/参数、solver状态与耗时；source validation为N/A/SOURCE_ACCESS_FORBIDDEN。完整矩阵结束后由独立summary验证全部binding、cost与solver events，再重算外层support指标，保留完整K×新增类数、模型/cohort、RX/场景分层和配对变化。不能把内部训练loss/准确率或support诊断冒充query验证。

发布只在代码/配置/记录显式提交、推送及远端OID核对后执行；不热修改、停止或重启健康任务。低性能不会触发停机或选择性重跑。query及新增独立数据验证均不在本次启动范围。

- [结构设计](D92_JOINT_NEXT_AFTER_CHANNEL_20260930.md)
- [核心实现说明](D92_PROTOTYPE_TRANSPORT_CORE_20260930.md)
- [入口与独立汇总](D92_PROTOTYPE_TRANSPORT_ENTRY_20260930.md)
- [核心实施审查](D92_PROTOTYPE_TRANSPORT_IMPLEMENTATION_REVIEW_20260930.md)
