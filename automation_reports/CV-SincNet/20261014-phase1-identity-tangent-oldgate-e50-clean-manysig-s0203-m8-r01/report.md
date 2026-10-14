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
