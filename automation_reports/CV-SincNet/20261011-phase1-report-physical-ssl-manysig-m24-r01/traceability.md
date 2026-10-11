# 新报告实现追踪

源附件：`E:\codex\home\attachments\37f52a1c-3585-4f9c-adc4-11f2426c5f11\已粘贴的文本.txt`；完整快照 `design_source.txt`。末尾“明确”之后的CFO撤回与DL/DQ/DR方案覆盖前文CFO校准及T频偏任务。

当前48行健康S实验保持不变。此前未发布F21候选保留代码和记录，标为DEFERRED_REPORT_ALIGNMENT，不启动旧特征机制。新增报告版按下表逐项实施，不以框架/单元检查宣称实验完成。

| ID | 来源章节 | 要求 | 目标文件 | 状态 | 验证/说明 |
|---|---|---|---|---|---|
| R01 | 末尾修订/九 | 同seed原CE E200，40epoch/8880，三seed；强LR服从最新用户要求 | `experiments/cvs_report_ssl/design.py/source.py` | verified | 完整960 epoch/213120 step及24row冻结、预测、独立评分证据已核实；不等于性能收益 |
| R02 | 末尾一 | 29维参考响应取消显式CFO，审查去旋转；维度兼容保留槽位0，L适应后再教师 | `experiments/cvs_report_ssl/reference.py/model.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R03 | 末尾一 | 40维状态末两维相位步进替换相关幅值，所有状态/参考缓存及旧作用模型重新适配 | `experiments/cvs_report_ssl/physical.py/source.py` | verified | 完整960 epoch/213120 step及24row冻结、预测、独立评分证据已核实；不等于性能收益 |
| R04 | 一/八 | L6300/U56700/V27000角色不变；U只IQ/RX/day/opaqueUID，禁止TX派生采样和跨RX同sig配对 | `experiments/cvs_report_ssl/source.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R05 | 一 | 连续sig索引划分如实记录，不称随机，不推断绝对时间漂移 | `experiments/cvs_report_ssl/design_source.txt/report.md` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R06 | 八 | 15个RX×日期轮转、域内UID随机完整遍历；L采样和匹配CE控制一致 | `experiments/cvs_report_ssl/sampling.py/source.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R07 | 三/九 | 固定BU分母软KL，q/weight停止梯度，tau=.95；无强制onehot和U熵压缩 | `experiments/cvs_report_ssl/objectives.py/source.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R08 | 末尾四/九 | 前888次L适应；完成step888后克隆EMA=.99，step889开始U，随后888步lambdaU由0到1 | `experiments/cvs_report_ssl/source.py` | verified | N60721个SSL row已越过L888；实际teacher_updates=identity_updates−888，U_identity_packets=128×teacher_updates |
| R09 | 三 | 身份可靠性与物理任务可靠性分别定义；低TX置信U仍可参与物理任务 | `experiments/cvs_report_ssl/objectives.py/auxiliary.py/source.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R10 | 末尾二/四 | DL已知小幅复滤波，同UID原/作用视图，中段和动作头优化；不使用U margin真标签 | `experiments/cvs_report_ssl/auxiliary.py/model.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R11 | 四 | 动作监督实际复tap分量与明确范围；确定前向保留梯度 | `experiments/cvs_report_ssl/auxiliary.py/source.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R12 | 四 | 动作零动作/零读出/线性读出/解析估计/UID留出和边界痕迹诊断 | `experiments/cvs_report_ssl/source.py:physical_validation` | verified | 完整960 epoch/213120 step及24row冻结、预测、独立评分证据已核实；不等于性能收益 |
| R13 | 末尾二/三 | DQ四周期20点重复残差；仅unit-modulus对齐，参数不进入身份/条件教师 | `experiments/cvs_report_ssl/physical.py/auxiliary.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R14 | 五 | 留出周期/位置残差预测与误差可靠性；不称严格blindspot或Noise2Noise保证 | `experiments/cvs_report_ssl/auxiliary.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R15 | 五/六 | 四周期协方差rank<=3；70个L/格完整协方差rank<=69；对角收缩，不求高维裸逆 | `experiments/cvs_report_ssl/conditional.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R16 | 末尾三 | 相对包络、多延迟相关幅值、局部x[n+1]x[n-1]conj(x[n])²；低幅值可靠度加权 | `experiments/cvs_report_ssl/physical.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R17 | 末尾二 | 非线性/记忆特征归身份端，保留原学习分支；新增零初始融合支路单独消融 | `experiments/cvs_report_ssl/model.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R18 | 六 | DR共享TX锚点＋RX/日期偏移＋收缩条件尺度；固定L先验；R标量margin撤回 | `experiments/cvs_report_ssl/conditional.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R19 | 六 | L与U软均值分开，U统计权重不机械复用身份mask，alpha上限.2 | `experiments/cvs_report_ssl/conditional.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R20 | 六/十 | 条件教师与EMA在独立V同覆盖比较错误率；V只诊断/预定源选择，不拟合身份模型 | `experiments/cvs_report_ssl/source.py:teacher_validation` | verified | 完整960 epoch/213120 step及24row冻结、预测、独立评分证据已核实；不等于性能收益 |
| R21 | 七 | L锚定软相似关系/图传播，诊断域聚集、孤岛和跨RX关系 | `experiments/cvs_report_ssl/relation` | deferred | 报告可选独立对照；首批先C1/C2/C3/C4，保留后续计划 |
| R22 | 四/九 | DLT交互仅单因素效果成立后加入 | `experiments/cvs_report_ssl/interaction` | deferred | 报告明确后续增量，本轮不混入 |
| R23 | 八 | Fishr先稳定L+U，第一增量只用L真实标签 | `experiments/cvs_report_ssl/fishr` | deferred | 报告明确后续，不把所有U纳入梯度方差 |
| R24 | 九 | C0匹配L-only，C1软SSL，C2DL，C3DR，C4DQ，C5源侧有效模块组合 | `experiments/cvs_report_ssl/design.py/source_selection.py` | implemented | C0/C1/C2/C3/C4已完整验证；C5源选择C1+C2已冻结，独立新run待执行 |
| R25 | 十 | parent fix/break按TX/RX/day，macro、最差域、NLL；NLL需新logprob prediction固定后评分 | `experiments/cvs_report_ssl/evaluate.py` | verified | 完整960 epoch/213120 step及24row冻结、预测、独立评分证据已核实；不等于性能收益 |
| R26 | 十 | U覆盖/分布/翻转/教师分歧/有效统计量；伪标签错误率仅V模拟，连同覆盖报告 | `experiments/cvs_report_ssl/source.py/conditional.py` | verified | 完整960 epoch/213120 step及24row冻结、预测、独立评分证据已核实；不等于性能收益 |
| R27 | 项目/用户 | 逐行自动clean预测＋独立truth-last；每seed及三seed均值84验收；max3总进程/GPU | `experiments/cvs_report_ssl/dispatch.py/evaluate.py` | verified | 完整960 epoch/213120 step及24row冻结、预测、独立评分证据已核实；不等于性能收益 |
| R28 | 用户最新调优 | 普通权重强LR与Sinc频带unit optimizer可独立增量 | `experiments/cvs_report_ssl/../cvs_stage2_ssl/frequency_optim.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R29 | 末尾撤回 | 不新增CFO校准/估计/预测/条件教师身份特征；此前T频偏主任务撤回 | `experiments/cvs_report_ssl/reference.py/design.py` | verified | 本地行为检查＋完整N607训练/评分核实；见full_source_analysis与clean_results |
| R30 | 八/项目 | 成本实测：学生/教师/物理前向、aux参数、FP64状态、训练推理显存耗时 | `experiments/cvs_report_ssl/source.py/evaluate.py` | verified | 完整960 epoch/213120 step及24row冻结、预测、独立评分证据已核实；不等于性能收益 |
| R31 | 末尾四 | 小幅随机频移可选，仅抑制依赖，无预测任务 | `experiments/cvs_report_ssl/augmentation` | deferred | 首批线性/重复主假设先验证，暂不增加此可选扰动 |
| R32 | 十 | 目标U、补充TX标注、多包判决不扩展当前权限 | `experiments/cvs_report_ssl/permissions` | rejected | 当前契约不授权；未来独立授权才能研究 |
| R33 | 六 | 受限低秩TX×RX交互与更复杂条件协方差 | `experiments/cvs_report_ssl/conditional` | deferred | 先验证共享/加性/对角条件教师再做独立增量 |

当前26项verified、1项implemented（R24的C5独立run未执行）、5项deferred、1项rejected、0项blocked。行为verified不等于性能提升；84%未达到。
