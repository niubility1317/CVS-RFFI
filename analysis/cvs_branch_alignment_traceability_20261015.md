# CVS 同分支 clean→urban 对齐设计追踪

来源：用户 2026-10-15 附件 `E:/codex/home/attachments/b46410e4-fab4-4f9c-9dc7-8d0dc6f90e52/已粘贴的文本.txt`。下一轮主线是统一损失的位置消融，不预设 T、S 或其组合最优。当前 E50 四行及 KL/joint160 两行继续其固定 release，不热修改、不复用其目标成绩选择新配置。前一附件的同信道中间视图、局部 SIG 分组教学继续保留，放在位置定位后的独立阶段，不混入本轮八组。

完整要求保留 A0/A-T/A-F/A-P/A-R/A-S/A-J/A-ALL 与三个预定义模型种子2026092701/02/03，E50。按既有最小协议先做单 seed 八组小试，正式三 seed 及三种稀疏组合的推进规则必须由合法源证据预先固定，不能依据 target。单 seed 不能宣布最佳分支。现有 JOINT_DIR 从 epoch1启用、校准口径不同，不能直接冒充新矩阵的匹配 A-J。

拟固定的统一启用：前5轮相同配对CE/原旧门控流程预热，epoch6至50启用各自唯一方向目标；33+17阶段、11100更新不变。A0按相同路径提供 clean ownEMA 目标但无对齐项。一次新源强度校准使用 A0 预热第5轮及其 ownEMA，其他候选仍从零训练，绝不加载按目标成绩选出的模型；不复用当前成熟 E50 的旧系数。

源开发用途：从原 L6300 的 TX×源RX×day 平衡单元各70条中固定留8条，即720条L_dev，候选训练只用其余5580条L_train；L_dev不进入有标签、无标签或EMA更新。U56700真值继续隐藏，V27000只健康诊断，外层允许源物理集合/目标集合不变。训练、校准和开发的实际物理ID及来源必须明确记录；新矩阵全部使用同一内部折，旧全L模型仅作操作性冻结诊断，不能作为该内部折的初始化/匹配基线。此内部分折尚未实现，须先在科学协议中明示；不得静默改数据契约。历史全L模型曾接触这些L_dev，不能把该折宣传成全历史未见的独立盲测。

| ID | 来源 | 验收要求 | 目标 | 状态 | 证据/限制 |
|---|---|---|---|---|---|
| B01 | 一/二 | 只对齐同分支跨视图，全部网络与默认头照常CE联合训练 | branches.py/training | pending | 不冻结未选分支，不要求不同分支相等 |
| B02 | 二T | 捕获进入实际fuse前的t_emb，保持计算图 | branches.py | verified | 不能假定t_proj后无drop/mask |
| B03 | 二F | 捕获完整f_emb，含启用的freq_stats_proj增量 | branches.py | verified | 以实际fuse输入验证 |
| B04 | 二P | 捕获pa_norm实际输出，含pa_local/pa_delta处理 | branches.py | verified | 不用pa_proj近似 |
| B05 | 二R | 捕获response_projection，绝不直接匹配固定raw29 | branches.py | verified | 需可学习投影梯度 |
| B06 | 二S | 捕获output_norm gain前160维，梯度到local_projection/t2 | branches.py/checks.py | verified | 首轮不对齐错位112维 |
| B07 | 三 | 同一方向损失、eps、ownEMA clean eval/stopgrad和共用L资格 | objective.py | pending | 无urban学生置信要求，不改原U监督 |
| B08 | 三.4/八 | 各分支gain前；J用原数值/stopgrad gain；aux W=0/CE W正常 | objective.py/checks.py | verified | 分支本身不能detach |
| B09 | 四 | 一次同预热checkpoint全特征参数全局梯度校准，W排除、未接收梯度按0 | calibrate.py | pending | median范数的比，非先前五组比值中位数 |
| B10 | 四 | 小梯度/非有限拒绝，不无限放大；记录每层梯度和实际更新/裁剪 | calibrate.py/observer.py | pending | 有界静态lambda，不在线均衡 |
| B11 | 四.3 | ALL先按同规则分支定标再校准组合总强度，omega和为1 | objective.py/calibrate.py | pending | 不直接相加五个单项系数 |
| B12 | 五 | 同一冻结模型clean替换单分支，真实融合/后处理/default头重算 | replace.py | verified | 操作性诊断，不是训练上界/可得clean输入 |
| B13 | 五 | T只换fuse输入T；S仍urban；F换完整F，rho/其他分支保留 | replace.py/checks.py | verified | 用实际其他张量/最终输出断言 |
| B14 | 六 | 完整8配置统一E50/预热/EMA/数据/启用/教师前向；无对齐A0 | design.py/source.py | pending | 当前8组×3seed要求不缩成T/S/J三组 |
| B15 | 六 | 3预定义seed逐row自动clean+6实测、独立truth-last、保留负差 | dispatch.py/evaluate.py/report.py | pending | 先单seed小试不声明稳定/最优 |
| B16 | 七 | L内部开发折明确、V只健康；源选择规则/阈值先固定 | protocol/inner_source.py/design | pending | 不改外层目标/不解封U标签 |
| B17 | 七 | b1+b2、b1+J、b1+b2+J条件增益，按源规则选并重新总强度校准 | 后续组合 | deferred | 等完整单项源证据，非先验T+S |
| B18 | 七 | clean保持/多seedlow/urban最差RX/少约束成本，源选择与目标报告分离 | selection.py/report.py | pending | 不用target挑最好首seed |
| B19 | 八 | 教师detach、学生训练缓存保图、日志副本detach | branches.py/checks.py | verified | 不复用旧detach诊断observer作loss |
| B20 | 八 | 共享T/S上游作用、梯度夹角、分支方差/gain/贡献，区分放弃与鲁棒 | observer.py/mechanism.py | pending | loss变小不等于成功 |
| B21 | 交付 | 实际命令/commit、更新、父成本、全前向/墙钟、完整曲线与分层图 | registration/report | pending | 不能伪造路径或性能 |
| B22 | 全局 | 固定SIG_TIME网络/默认头/正式IQ，合法来源和运行保护 | design/tests | pending | 不新增模型/独立头/测试期clean或元数据 |

实施顺序：先真实分支捕获/替换负例与源冻结诊断；明确内部源开发折/匹配预热和预算；再实现/登记/审查/发布完整八组小试；按预登记source-only条件扩三seed，保留所有有效负对照；最后稀疏组合与中间视图等独立研究。此记录不是额外审批或签名层。

本地证据：实际core路径experiments/adv3b02_xuc/code/model.py；freq_stats_proj=true；实际joint祖先参数195041/71 tensors，含三gain、不含默认W。`experiments.cvs_urban_branch50.checks` 5项PASS，包含实际六位置反传、F统计/P完整输入、T替换保留urban S和rho等。B06/B08这里只证明真实单次forward/梯度边界，完整native训练接线仍在B14/B20。B12/B13有本地替换验证，远端真实冻结checkpoint诊断尚未发布。

2026-10-15进展：真实B0源替换run已ANALYZED，26640前向/0更新/20.838秒，模型state逐值不变、其余分支/rho保持；本地11项与compile PASS，8配置单seed小试已登记/一次P0-P1 PASS，尚未发布。L内部分折科学边界已先更新PROJECT_PROTOCOL，实际ID/全局定标和完整E50训练证据须待新run，不把本地接线当真实八组结果。当前verified9/pending12/deferred1，完整三seed/组合未完成。

八组小试已以commit152d8921c903fc6d5f782c199b075d0837840ece发布并RUNNING；A0首epoch/实际inner物理ID5580/720、默认模型参数863730和aux零边界已VERIFIED。其余位置仍待一次源定标，B09/B11和完整B14/B15仍pending，不提前标已完成。
