# 接收机切向平坦化：固定剩余种子确认

首种子原预登记源健康判据已在读取目标汇总前通过，最新用户要求继续此前正收益路线。本次 ZERO、RAW_MATCHED、IDSAFE 三项原样确认剩余 seed02/03，不依据目标分数挑选 RAW，不调整层位、强度、投影、梯度预算或训练配方。原 IDSAFE 主项保持，完整三种子对照同时报告 RAW。

代码、物理划分、原默认联合 cosine 分类头、NoExplicitCFO、旧门控＋ownEMA、33L+17U、50 epoch/11100 更新均保持。每个新模型仍有2498次L60三前向辅助调用，额外L前向449640包。每4步辅助、前1110步预热、1110步ramp，t1/t2交替，同TX/day跨RX方向，RAW与安全投影方向逐包等能量；R作用真实joint160，类别W无R梯度，fθ辅助梯度最多为主LCE的20%。无换头、无目标反馈。

| model seed | 复用历史默认预测 | 新从头训练 |
|---|---|---|
| 2026092702 | BASE_REF | ZERO、RAW_MATCHED、IDSAFE，各50epoch |
| 2026092703 | BASE_REF | ZERO、RAW_MATCHED、IDSAFE，各50epoch |

首种子四行已完成且保留。当前新增6模型、300modelepoch/66600更新、8行clean默认头预测评分。与首轮合并后为9个新模型、450modelepoch；3个历史baseline固定50epochs各11100更新只复用，不冒充零历史成本。结果按3预定modelseed统计均值、样本SD、seed配对差和最差RX/day。原单seed RAW81.4405%、IDSAFE81.0298%只作为已暴露结果，不用于方法修改、候选重排或选择性重跑。

第一seed02新源训练必须全部达到原finalLCE<=.5、源V平均下降<=1pp/最差RX下降<=2pp，才推进seed03；此决定只读源信息，不读目标结果。源科学失败保留结果并停止pending；技术失败错契约/输出冲突/nonfinite/预测或评分失败停止pending，不重启健康active，保留输出。

每row冻结后立即自动clean预测、fresh CPU独立truth-last评分，其他行结果不反馈训练。全部结束合并原首seed的已完整评分，独立重算216个CM记录；全矩阵前不宣称稳定增益或84%。仍明确是已暴露benchmark，非新的独立盲测。

唯一owner `codex/root/identity-tangent-confirm-20261014`，普通N607账户，max3总进程/GPU含otherowners，至少12000MB空闲，不停其他健康任务。命令 `python -u -m experiments.cvs_identity_tangent_confirm3.dispatch`；环境/CWD/绝对输入输出/逐row命令详见experiment.json，执行Gitcommit以release_commit.txt记录。真实同契约checkpoint无query smoke是launcher首步，通过后继续。既有机制一次审查/数学负测及真实训练证据复用，不重复审查未变候选。新增种子接线只核对同seed初始权重与完整统计，不作为新方法效果证明。

当前：LOCAL_VERIFIED，发布后独立PID/CWD/argv/GPU/日志回读。新主线“跨RX关系＋证据响应”另行消融，本次确认不叠加该机制。


R01 独立 FAILED：归档路径遗漏新确认模块，模块入口不能导入；尚未提交或启动训练，未创建run根。旧归档/失败preflight保留。R02 仅补正确归档路径，新输出根；模型机制/预算/种子未变。


## 独立发布回读

执行Gitcommit `882c89f783377947c81739745c8a1e1ea96e8f3e`已push且remoteOID一致；归档运输一次SHA比对、remote compile、真实checkpoint noquery smoke PASS，PID/CWD/argv/GPU/log独立回读VERIFIED，GPU最大总实验进程3含其他owner，未停止其他任务。当前row状态：BASE_REF-s2026092702 epoch50, ZERO-s2026092702 epoch35, RAW_MATCHED-s2026092702 epoch33, IDSAFE-s2026092702 epoch32。此为局部源训练启动证据，headfit0，不代表识别提升或84%。详见launch_evidence.json。

## 完整三种子收尾

当前状态：ANALYZED / VERIFIED。确认run的8行全部默认头预测/独立评分闭合，与原首seed4行合并成固定3modelseed、2016000决定、216个CM记录。各CM/三seed均值与样本SD再次独立重算一致。原健康任务无重启，未用target反馈修改方法。

| 条件 | clean均值 | 样本SD(pp) | Macro-F1 | 每seed最差RX均值 | 每seed最差day均值 |
|---|---:|---:|---:|---:|---:|
| BASE_REF | 80.5268% | 2.3776 | 0.800024 | 67.9708% | 76.0556% |
| ZERO | 80.6038% | 2.3131 | 0.800706 | 68.0653% | 76.2865% |
| RAW_MATCHED | 80.5579% | 1.6481 | 0.800379 | 67.0653% | 76.9524% |
| IDSAFE | 80.3077% | 1.7081 | 0.797296 | 66.9042% | 76.4286% |

RAW−ZERO三seed差：＋0.7006、＋0.4310、−1.2690pp，均值−0.0458pp；IDSAFE−ZERO：＋0.2899、＋0.1315、−1.3095pp，均值−0.2960pp。IDSAFE−RAW三seed均负，均值−0.2502pp。首seed RAW小正差未变成稳定3seed提升，不能继续把它列为已确认正收益路线。RAW/IDSAFE最差RX均值比ZERO分别低1.0000/1.1611pp；最差day均值RAW较高，取舍如实保留。均未达84%。

已解析9个新增模型全部50epoch/11100step、每模型2498辅助记录和完整stdout，共450新增modelepoch/99900更新/22482辅助调用；每层1249次、额外L前向4046760。ZERO所有R/辅助梯度0，RAW/IDSAFE实际梯度比始终≤0.2、全期均值约0.1889，sourceL-only/U真值0/0headfit，无nonfinite跳步或fatal标记。各seed源健康标准通过不等于target性能通过。3历史baseline训练150epoch/33300更新另计，不冒充零成本。

实际源LCE和V表明新约束确实执行而非空分支，但没有默认头稳定收益；线性TX均值统计投影不能代表全部身份信息保护。本轮不从已暴露target反调epsilon、cap、层位或选择性重跑。已确认COMPACT_LINEAR/LONG_SILU/Sinc路线的独立证据仍保留，新关系/响应矩阵继续其冻结计划。

完整source曲线/全部训练扫描本地保存于local_artifacts/cvs_identity_tangent_confirm3_20261014_r02/complete_source_analysis.json；compact结果与权限/成本证据见result_evidence.json，源与目标逐row原产物在immutable remote roots。
