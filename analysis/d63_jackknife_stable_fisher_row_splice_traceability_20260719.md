# D63跨折稳定Fisher行拼接预注册与追溯

## 1.问题与单一假设

D62在匿名类别行上使用support inner-held的总体TP/FP Pareto门，取得当前最高聚合开发结果，但low-elev的forgetting恶化3.33pp、rain的before下降1.67pp。这说明总体计数可以让某一held fold的伤害被其他fold的收益抵消。D63只验证一个机制假设：若候选行除了总体Pareto安全，还必须在每个leave-one-inner-fold证据子集上均不降低任一类TP且不增加任一类FP，则可以剔除这种跨fold不稳定行，并保留真正稳定的D61 Fisher残差收益。

## 2.冻结方法

- 基础分数、D61 Fisher残差分数、full/block权重、K个类平衡inner-held划分和outer affine编译与D62完全相同。
- 单行总体门：替换匿名类别`c`后，`TP_c`不降、`FP_c`不升，且至少一项严格改善。
- 单行稳定门：对每个leave-one-inner-fold证据子集，替换该行后所有类别的TP均不降且FP均不升；不要求每个子集严格改善。
- 联合原子门：所有通过的行同时替换后，总体以及每个leave-one-inner-fold子集都必须对所有类别TP不降、FP不升；否则整组精确回退D46。
- 最终仅编译一个固定FP32 affine并按正式路径量化为int8；query侧无额外图、优化或状态更新。

固定公式：`accept_c=aggregate_coordinate_Pareto_strict AND all_leave_one_fold_coordinate_Pareto_nondegrading; joint=aggregate_and_all_leave_one_fold_Pareto_nondegrading; row_c=D61_else_D46`。

## 3.禁止项与停止规则

- 不使用old/new角色、class ID、receiver、scene、outer fold、query真值、query分布、类别配额或全局重分配。
- 不扫描阈值、alpha、rank、gain、温度、门限或场景mask；D63超参数计数为0。
- 只运行receiver20-1、seed713101、K10/new5、clear/low-elev/rain×5fold的既定开发单元，复用`VALIDATED_ONCE/p2_min_v1`数据。
- 若任一场景before、after、new、H或forgetting相对D46发生伤害，或量化门失败，状态记为诊断阴性且不扩第二seed/125。
- 若三场景无伤害且聚合after/H/forgetting或joint至少一项严格改善，才允许讨论第二seed；本轮不得直接启动125。

## 4.必须输出的性能证据

完成后必须记录105行闭包、7候选同row总体表、3场景表、11类准确率、15fold表、三向混淆、D46/D61/D62/D63差值、总体与leave-one门接受/拒绝原因、量化生命周期、训练轨迹、资源上界、artifact哈希和最终判定。不得仅报告缺陷。

## 5.执行面

- 计划实现：`code/scripts/probe_d63_jackknife_stable_fisher_row_splice.py`。
- 计划测试：`tests/test_probe_d63_jackknife_stable_fisher_row_splice.py`。
- 输出：`automation_reports/CV-SincNet/d63_jackknife_stable_fisher_row_splice_probe_20260719/jackknife_stable_fisher_row_splice`。
- 本地`ssr-gpu`串行验证并从detached clean worktree执行；本轮不访问N607。
