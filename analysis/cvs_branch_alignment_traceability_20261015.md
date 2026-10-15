# CVS 同分支 clean→urban 对齐设计追踪

当前状态：全部八组单seed小试及冻结后机制已ANALYZED／VERIFIED；当前verified17／pending4（B14/B15/B18/B21）／deferred1／rejected0／blocked0。源家族推进条件未通过，三个模型seed与条件稀疏组合未运行，不宣布最佳位置或full design parity。目标low urban60%未达，目标分数不参与推进。
来源：用户 2026-10-15 附件 `E:/codex/home/attachments/b46410e4-fab4-4f9c-9dc7-8d0dc6f90e52/已粘贴的文本.txt`。下一轮主线是统一损失的位置消融，不预设 T、S 或其组合最优。当前 E50 四行及 KL/joint160 两行继续其固定 release，不热修改、不复用其目标成绩选择新配置。前一附件的同信道中间视图、局部 SIG 分组教学继续保留，放在位置定位后的独立阶段，不混入本轮八组。

完整要求保留 A0/A-T/A-F/A-P/A-R/A-S/A-J/A-ALL 与三个预定义模型种子2026092701/02/03，E50。按既有最小协议先做单 seed 八组小试，正式三 seed 及三种稀疏组合的推进规则必须由合法源证据预先固定，不能依据 target。单 seed 不能宣布最佳分支。现有 JOINT_DIR 从 epoch1启用、校准口径不同，不能直接冒充新矩阵的匹配 A-J。

本轮固定的统一启用：前5轮相同配对CE/原旧门控流程预热，epoch6至50启用各自唯一方向目标；33+17阶段、11100更新不变。A0按相同路径提供 clean ownEMA 目标但无对齐项。一次新源强度校准使用 A0 预热第5轮及其 ownEMA，其他候选仍从零训练，绝不加载按目标成绩选出的模型；不复用当前成熟 E50 的旧系数。

源开发用途：从原 L6300 的 TX×源RX×day 平衡单元各70条中固定留8条，即720条L_dev，候选训练只用其余5580条L_train；L_dev不进入有标签、无标签或EMA更新。U56700真值继续隐藏，V27000只健康诊断，外层允许源物理集合/目标集合不变。训练、校准和开发的实际物理ID及来源必须明确记录；新矩阵全部使用同一内部折，旧全L模型仅作操作性冻结诊断，不能作为该内部折的初始化/匹配基线。内部分折已先在PROJECT_PROTOCOL明示并实际实施；外层源／目标权限与正式IQ保持。历史全L模型曾接触这些L_dev，不能把该折宣传成全历史未见的独立盲测。

| ID | 来源 | 验收要求 | 目标 | 状态 | 证据/限制 |
|---|---|---|---|---|---|
| B01 | 一/二 | 只对齐同分支跨视图，全部网络与默认头照常CE联合训练 | branches.py/training | verified | 不冻结未选分支，不要求不同分支相等 |
| B02 | 二T | 捕获进入实际fuse前的t_emb，保持计算图 | branches.py | verified | 不能假定t_proj后无drop/mask |
| B03 | 二F | 捕获完整f_emb，含启用的freq_stats_proj增量 | branches.py | verified | 以实际fuse输入验证 |
| B04 | 二P | 捕获pa_norm实际输出，含pa_local/pa_delta处理 | branches.py | verified | 不用pa_proj近似 |
| B05 | 二R | 捕获response_projection，绝不直接匹配固定raw29 | branches.py | verified | 需可学习投影梯度 |
| B06 | 二S | 捕获output_norm gain前160维，梯度到local_projection/t2 | branches.py/checks.py | verified | 首轮不对齐错位112维 |
| B07 | 三 | 同一方向损失、eps、ownEMA clean eval/stopgrad和共用L资格 | objective.py | verified | 无urban学生置信要求，不改原U监督 |
| B08 | 三.4/八 | 各分支gain前；J用原数值/stopgrad gain；aux W=0/CE W正常 | objective.py/checks.py | verified | 分支本身不能detach |
| B09 | 四 | 一次同预热checkpoint全特征参数全局梯度校准，W排除、未接收梯度按0 | calibrate.py | verified | median范数的比，非先前五组比值中位数 |
| B10 | 四 | 小梯度/非有限拒绝，不无限放大；记录每层梯度和实际更新/裁剪 | calibrate.py/observer.py | verified | 有界静态lambda，不在线均衡 |
| B11 | 四.3 | ALL先按同规则分支定标再校准组合总强度，omega和为1 | objective.py/calibrate.py | verified | 不直接相加五个单项系数 |
| B12 | 五 | 同一冻结模型clean替换单分支，真实融合/后处理/default头重算 | replace.py | verified | 操作性诊断，不是训练上界/可得clean输入 |
| B13 | 五 | T只换fuse输入T；S仍urban；F换完整F，rho/其他分支保留 | replace.py/checks.py | verified | 用实际其他张量/最终输出断言 |
| B14 | 六 | 完整8配置统一E50/预热/EMA/数据/启用/教师前向；无对齐A0 | design.py/source.py | pending | 全部8配置×首seed已实际E50／11100；正式3seed未运行，源家族未达预登记推进条件 |
| B15 | 六 | 3预定义seed逐row自动clean+6实测、独立truth-last、保留负差 | dispatch.py/evaluate.py/report.py | pending | 首seed已完成56视图／9408000预测／1008指标及独立复算；3seed与复现仍未完成 |
| B16 | 七 | L内部开发折明确、V只健康；源选择规则/阈值先固定 | protocol/inner_source.py/design | verified | 不改外层目标/不解封U标签 |
| B17 | 七 | b1+b2、b1+J、b1+b2+J条件增益，按源规则选并重新总强度校准 | 后续组合 | deferred | 等完整单项源证据，非先验T+S |
| B18 | 七 | clean保持/多seedlow/urban最差RX/少约束成本，源选择与目标报告分离 | selection.py/report.py | pending | 源family_expandable=false，无合格首轮信号；未按target放宽／挑组，b1/b2三seed选择尚未发生 |
| B19 | 八 | 教师detach、学生训练缓存保图、日志副本detach | branches.py/checks.py | verified | 不复用旧detach诊断observer作loss |
| B20 | 八 | 共享T/S上游作用、梯度夹角、分支方差/gain/贡献，区分放弃与鲁棒 | observer.py/mechanism.py | verified | 实际8冻结模型／ownEMA、固定A0 clean正确703/720；46800前向／131.674秒／0更新；13个共享参数张量的T/S角独立报告 |
| B21 | 交付 | 实际命令/commit、更新、父成本、全前向/墙钟、完整曲线与分层图 | registration/report | pending | 首seed完整曲线／8分层图／70776514总前向／全部预算与墙钟已交付；正式多seed交付仍待科学推进 |
| B22 | 全局 | 固定SIG_TIME网络/默认头/正式IQ，合法来源和运行保护 | design/tests | verified | 固定结构／默认头／正式IQ；完整11366400增强记录仅5580合法L_train IDs，15 session含6TX，90单元支持，源映射无U真值读取 |


实际训练release为152d8921c903fc6d5f782c199b075d0837840ece。全部scratch50（33+17）／11100成功更新，八组88800更新；原CE/旧U监督正常训练五组身份特征、默认W和三gain。全量step／epoch／stdout均已解析，无非有限跳步／U真值暴露／技术错误；aux W/gain边界0，max_grad_norm5。一次源epoch5全局校准3600前向／6.332秒／0更新，theta195041／71、W排除、CPU/CUDA RNG恢复，ALL先单项定标再独立总强度定标，lambda静态有界无裁切。FULL_FP32_POLICY.cudnn_deterministic=false，首5轮不能宣称bitwise同权重轨迹。

真实分支接口与训练11项定点检查、冻结机制4项检查、真实checkpoint no-query smoke、一次训练P0/P1与一次新源机制P0/P1已通过。recipe部分写入竞态已用同目录完整临时写入后rename修复，仅定点核实；未重复审查未变代码或报告。后分析release0b5cc8b0451cfad03d298af3ba39102a665947af，PID3006209完成后独立产物核实VERIFIED，实际46800前向／131.674秒／0optimizer和head-fit更新，student/ownEMA state不变，V前向0、无U/query读取。

完整信道trace另行遍历11366400个增强输入。原日志sample_id为source:<base_index>，登记为opaque physical ID；审计初次直接比较这两个namespace而失败，未修改、停止或重训任何实验。使用原152d发布的有标签L_s原生索引作字段映射（6300条，0数据集__getitem__／U标签张量／模型前向调用），再核对实际每组恰为5580个L_train、0个L_dev／其他ID。原生builder仍加载既有ManySig容器，仅遍历L_s metadata、不构造query loader；不把样本使用隔离表述成容器字节完全未读。15个RX/day/eq session各含6TX，90个TX×RX×状态单元支持；8组记录的物理／场景／损伤字段相对A0完全匹配。此为同一已有数据的索引解释，不是额外permission／allowlist／hash／receipt链或发布门槛。

源指标只在固定L_dev720的14个epoch观察，final50用于已冻结推进条件。任何单项均未满足low+1pp、clean>=−.5pp、urban>=0、最差RX>=−1pp：ALL的low+0.8333pp同时urban−0.1852pp，其他信号也不足。family_expandable=false，保持全部负对照，不挑目标最高组扩seed。B17的三种组合继续deferred，不预设T+S。不能将未扩seed写成收益跨seed重现。

正式指标：匹配内部折A0 low53.2714%／urban64.1504%／clean79.8060%；A-S为53.6387%／64.4821%／79.7030%，A-J为53.5726%／64.5524%／80.5048%，ALL为53.2482%／63.9597%／79.1923%。这是首seed结果记录，不作为位置筛选或新阈值。旧全L训练的B0、C3、KL／方向不是该内部折的严格对照。

机制负结果：T/F配对cos约.978/.993、R约.783、S约.618，目标确实进入真实分支；但最近错误TX同样更近、若干方差下降，默认头判别收益很小。固定A0 clean正确703集合上，A-S修复16／新增11、ALL13／8，A-J12／12；J的low源margin由A0 .095764降至.081752。J中R幅度／joint比约.110（A0 .237），表明融合分配改变也参与；不把gain下降或对齐数值提高宣称为鲁棒成功。T/S共享13张量的源eval梯度cos中位数约.043至.093，未显示普遍强烈冲突，不能据此堆叠梯度手术或通用模块。

完整成本：各父0／新增50／累计50；各6190200主训练前向、70560开发面板、1404000原V、scratch/final smoke4及1176000测试。八行合计另加共享smoke2、定标3600、后分析46800，共70776514新增前向样本，不是FLOPs。训练墙钟、clip次数及所有逐row／逐视图／RX/day/TX指标保存在同run报告；预算结束不等于收敛。实际图形已生成并目视核实，绘图Matplotlib的labels接口变化改为显式set_xticks，仅重绘、无模型重跑。

最终证据：E:/type10-7/automation_reports/CV-SincNet/20261015-phase1-sigtime-branch-align-e50-manysig-s2026092701-m8-r01/report.md；local_artifacts/cvs_branch_alignment_20261015/final/{training_audit,channel_trace_audit,source_family_decision}.json、source_branch_results.json、source_gradient_aggregate.json及source_delivery_r01/readback.json。实际绘图代码experiments/cvs_urban_branch50/figures.py、全trace读取log_audit.py已纳入Git；IQ、权重、预测、原始日志与生成图不入Git。

当前最高风险：源上方向更近仍缺少足够身份判别收益，跨seed与目标60%均未证明。后续同信道高质量中间视图／局部SIG分组教学等原授权路线保留，但先需合法源物理与教学证据；不借本次target结果挑分支、改物理测试或放宽本家族推进规则。
