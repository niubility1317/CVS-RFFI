# E50 预算收益：合法源域分层核对

仅复用 seed2026092701 的原完整 As E30／E50 冻结源域汇总；未读取 formal 文件、IQ、checkpoint 或标签原件，新增 NN／更新／预测／评分均为0。600条指标的正确数与准确率、120次 RX/day/TX 分区合并、draw计数和修复净数一致，导出150条 E50−E30 同层变化。

|源域视图|As E30准确率%|As E50准确率%|变化（百分点）|修复／新增错误|
|---|---:|---:|---:|---:|
|clean panel|98.333333|98.444444|+0.111111|4／2|
|low suburban|88.666667|90.111111|+1.444444|78／26|
|mid urban|72.805556|74.777778|+1.972222|123／52|
|low urban|59.083333|60.722222|+1.638889|136／77|

|源域视图|draw0变化（百分点）|draw1变化（百分点）|
|---|---:|---:|
|low suburban|+1.611111|+1.277778|
|mid urban|+1.944444|+2.000000|
|low urban|+1.277778|+2.000000|

合并两draw后，mid／low的全部RX、day、TX单维层都提升。但这不代表逐包或逐draw全部改善：low urban的draw0/TX0和draw1/TX4各下降0.333333个百分点，low suburban的draw1/RX1下降0.555556个百分点。clean panel的RX1、day3、TX0也分别下降0.277778／0.333333／0.333333个百分点；这些是重叠分层，不能将下降包数相加。

**现有E50汇总没有LOS/PARTIAL/BLOCKED、锁定状态、质量代理、CFO、噪声着色或AGC维度。** 因此不能说追加预算特别缓解了某种物理损伤，也不能将另一条连续E30路线的状态结果移植为本E50的机制证据。当前唯一继续检验的假设仍是固定原As配方的预算收益可跨原定模型seed复现，不据本汇总另加损失、参数或晋级门槛。

面板只有1800个独立物理样本；两个信道draw合并计数3600，并非两个模型seed。完整V27000原生clean为98.255556%→98.374074%（+0.118519个百分点），与panel分开。源域low urban的60.722222%不能替代正式low urban的52.387500%，也不构成正式超过60%或跨seed重现的证据。

实际输入为引用owner已有的source_summary.json、source_metrics.csv、source_complete.json。输入目录：[原始源域汇总](<E:/type10-7/code/worktrees/cvs_iq_timing_20261016/local_artifacts/F_As_budget_E50_20261018/collection-final-r01/aggregates>)。独立核对与全部逐层变化：[verification.json](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/source_E50_budget_slices_r01/verification.json>)、[150条源域配对变化CSV](<E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_recovery_E30_20261018_r01/source_E50_budget_slices_r01/source_E30_E50_slices.csv>)。本报告不改变任何健康实验、固定测试IQ或信道。
