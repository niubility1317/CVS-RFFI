# 完整训练 L 的选择性相关掩码拟合

状态：PLANNED。唯一矩阵、命令、代码／环境/CWD、数据句柄、GPU、输出与停止规则在 experiment.json。此run仅冻结实际单视图E20进行source-only统计，零F/G训练和PL/Sup对照重训。

实际来源沿既有load_prefix核实m201 seed01的SIG_TIME单视图ordinaryPL E20/4440、13L7LU、scratch无祖先/目标污染/隐藏U真值；不把父模型称纯监督。当前只打开L6300训练IQ，用raw及三固定practical residual同包As，捕获time_down与t2.pool的实际相关矩阵。source_channel保留原channel/hardware设置，namespace固定PAMF/selective_covariance/L/frozen_E20。U/V/target IQ不打开，G权重不加载，source契约复用既有合法验证。

两层均用相对ridge1e-3相关矩阵和严格上三角。以5RX扰动敏感度的中位数取最高25%，以RX/day条件TX类间/总方差取最低50%，在交集中按sensitivity/(TXassociation+1e-6)排序，固定取全部边的10%。各层随机同尺寸mask固定seed2026102623，full包含所有上三角。低单变量TX关联不保证没有非线性身份信息；选中项不叫纯接收机效应。无白化、求逆、另训头或新推理支路。

不足固定数量时记录该层科学负结果，保留完整统计，不自动放宽quantile/fraction；此时不输出可供正式训练的masks.pt。不完整数据、错权重、非有限或前向异常是技术失败，保留退出且不盲目重启。统计/掩码.pt属于私有源派生产物，不提交Git。

首步实际sourceL2/defaulthead/两层feature smoke通过即完整50批L6300（末批28），25200条拟合F视图＋2条smoke；50条结构化记录及全部stdout完整保留。拟合结束检查F参数和所有注册buffers无变化。仅全局CUDA+pending均空、free>=10000MiB启动一个worker，不停止其他owner。唯一一次P0/P1审查和聚焦边界测试，不添加审批／hash／重复数据gate。

训练候选尚未发布，掩码可行也不证明clean、六景、DURA读出或原三seed等联合目标达成。后续计划见[当前选择性相关方案](../../../docs/goals/PAMF_SIG1_SELECTIVE_COVARIANCE_20261026.md)。

本地9项聚焦检查和3模块compile通过；唯一独立P0/P1审查PASS，未发现剩余问题。真实E20/no-query smoke仍由launcher第一步执行并立即继续；无额外审查或许可artifact。
