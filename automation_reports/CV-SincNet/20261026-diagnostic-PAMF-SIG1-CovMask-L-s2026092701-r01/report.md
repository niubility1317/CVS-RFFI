# 完整训练 L 的选择性相关掩码拟合

状态：ANALYZED_SOURCE_ONLY。完整L拟合与冻结检查已完成，原固定10%掩码数量规则未通过；无训练性能结论。唯一矩阵、命令、代码／环境/CWD、数据句柄、GPU、输出与停止规则在 experiment.json。此run仅冻结实际单视图E20进行source-only统计，零F/G训练和PL/Sup对照重训。

实际来源沿既有load_prefix核实m201 seed01的SIG_TIME单视图ordinaryPL E20/4440、13L7LU、scratch无祖先/目标污染/隐藏U真值；不把父模型称纯监督。当前只打开L6300训练IQ，用raw及三固定practical residual同包As，捕获time_down与t2.pool的实际相关矩阵。source_channel保留原channel/hardware设置，namespace固定PAMF/selective_covariance/L/frozen_E20。U/V/target IQ不打开，G权重不加载，source契约复用既有合法验证。

两层均用相对ridge1e-3相关矩阵和严格上三角。以5RX扰动敏感度的中位数取最高25%，以RX/day条件TX类间/总方差取最低50%，在交集中按sensitivity/(TXassociation+1e-6)排序，固定取全部边的10%。各层随机同尺寸mask固定seed2026102623，full包含所有上三角。低单变量TX关联不保证没有非线性身份信息；选中项不叫纯接收机效应。无白化、求逆、另训头或新推理支路。

不足固定数量时记录该层科学负结果，保留完整统计，不自动放宽quantile/fraction；此时不输出可供正式训练的masks.pt。不完整数据、错权重、非有限或前向异常是技术失败，保留退出且不盲目重启。统计/掩码.pt属于私有源派生产物，不提交Git。

首步实际sourceL2/defaulthead/两层feature smoke通过即完整50批L6300（末批28），25200条拟合F视图＋2条smoke；50条结构化记录及全部stdout完整保留。拟合结束检查F参数和所有注册buffers无变化。仅全局CUDA+pending均空、free>=10000MiB启动一个worker，不停止其他owner。唯一一次P0/P1审查和聚焦边界测试，不添加审批／hash／重复数据gate。

训练候选尚未发布，掩码可行也不证明clean、六景、DURA读出或原三seed等联合目标达成。后续计划见[当前选择性相关方案](../../../docs/goals/PAMF_SIG1_SELECTIVE_COVARIANCE_20261026.md)。

本地9项聚焦检查和3模块compile通过；唯一独立P0/P1审查PASS，未发现剩余问题。真实E20/no-query smoke仍由launcher第一步执行并立即继续；无额外审查或许可artifact。

## 实际完整结果

执行commit `2e70800a486f425ad9301cd107e44827b60e1b68`，单archive传输比较VERIFIED、远端compile PASS。实际sourceL2/defaulthead smoke PASS后立即完整读取50批L，末批28；6,300个物理L及18,900同包As视图完整，90个TX×RX×day格各70条。完整50条结构化记录及1,595字节stdout已解析，无致命错误，F参数/注册buffers全部未变。

实际process.json中的PID/CWD/argv/GPU与launch记录匹配；约33.71秒完成，第一次独立post-state时worker已自然退出，未取得存活时快照。终态pipeline、smoke、完整50条记录和result均已读回，没有为了截图重跑，也未操作其他owner进程。

| 层位 | C×T | 全部上三角项 | 原固定10%要求 | 实际合格项 | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| time_down | 48×128 | 1128 | 112 | 94 | 原数量规则FAIL |
| t2.pool | 96×32 | 4560 | 456 | 294 | 原数量规则FAIL |

合格项严格满足原敏感度最高25%／条件TX关联最低50%及正敏感度，没有为了凑足数量回填。实际集合分别占8.333333%与6.447368%；固定10%是本轮自行预设的稀疏度，不是用户要求的物理必要条件。该结果仅否决原“必须选满10%”规则，不能称协方差路线性能负结果。masks.pt没有输出；完整statistics.pt已保存在原独立run，用于审计／合法源分析，不提交Git。

下一独立候选改为“同样选择条件下，自然支持集至多10%”，即保留94／294项，随机控制按真实数量匹配，full控制不变。这是在读取新训练/目标结果前由完整L证据提出的显式规则修订，另登记新analysis run，不覆盖或改称本run通过，不重跑F/As或调整敏感度/TX阈值。后续源统计差别和训练性能仍须独立检验。原半监督目标保持active。
