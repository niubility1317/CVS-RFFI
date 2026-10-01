# Margin单样本prior复用的本机软件测量

实测结论：在本次固定合成评分状态和本机环境中，同时输出B/C的复用路径耗时中位数降低8.09%至11.65%；仅输出C时增加5.76%至31.01%。全部8行、24组配对计时的完整float64分数逐位相等。该接口适合已有单独B输出的配对计算；单独部署C仍使用原默认路径。本次没有改变实际模型、训练方法或当前远端基准，不声明准确率或星载收益。

## 完整固定矩阵

旧类6、新类20，完整注册类26；K=1/5/10/20。下表时间均为连续8个单样本API调用的3次固定重复中位数，单位ms，已包括同一套本地audit汇总。每行原路径/复用路径交替先执行，全部重复保留，未挑最好重复。变化=(复用/原路径−1)×100%，负数表示更快；不同K使用预先固定的合成support生成规则。

| K | 旧类support数 | 注册总support数 | B/C配对原路径（ms） | B/C配对复用（ms） | 配对变化 | 仅C原路径（ms） | 仅C复用（ms） | 仅C变化 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | 26 | 32.578 | 28.783 | -11.65% | 23.848 | 25.221 | +5.76% |
| 5 | 30 | 130 | 60.686 | 53.989 | -11.04% | 34.936 | 45.770 | +31.01% |
| 10 | 60 | 260 | 84.403 | 77.578 | -8.09% | 68.979 | 74.310 | +7.73% |
| 20 | 120 | 520 | 143.117 | 130.822 | -8.59% | 120.660 | 131.395 | +8.90% |

仅C的K5计时差距较大，但固定3次重复不能证明统计显著性或稳定的普遍比例。完整配对样本和路径次序见[summary JSON](../automation_reports/CV-SincNet/20261001-d92-margin-single-query-cost-synthetic-r01/results/measurements/summary.json)，AI紧凑[JSONL](../automation_reports/CV-SincNet/20261001-d92-margin-single-query-cost-synthetic-r01/results/measurements/measurements.jsonl)与[CSV](../automation_reports/CV-SincNet/20261001-d92-margin-single-query-cost-synthetic-r01/results/measurements/measurements.csv)均独立核实为24行。

## 数学与工作量

固定关系为 `f_C(x)=P f_B(x)+r_C(x)`。PAIR原路径为一次B加上C residual与重算B prior，复用路径为一次B加一次C residual。每8个样本，residual求值由24次减少到16次；kernel pairs由304K减少到256K，raw distance pairs由608K减少到512K。reference distance是raw的子集，不再相加。

C_ONLY原路径已经只有必需B prior和C residual两次求值。复用路径仍须先计算一次B，再计算C residual；两条路径每8样本都是16次residual求值、256K kernel pairs和512K raw distance pairs。optional接口新增输入捕获、状态/视图核对等开销，故不能把PAIR减少的重复工作套用于仅C部署。默认C评分没有改动。

真实计数按已执行audit汇总，B历史工作不重复计入C。每次packet准备与cache核对的SUM/MAX、实际调用数量分别保存在完整结果；峰值不当成总量相加。完整分数比较在计时区间外，不用argmax、容差或四舍五入替代位级等价。

## 硬件、资源与范围

- Windows x64，处理器标识 `AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD`，12个逻辑CPU；NumPy2.2.6、SciPy1.15.3、threadpoolctl3.6.0。实际解释器、DLL和线程池见结果hardware，BLAS/OMP/环境均核实2线程。
- 实际测量进程全生命周期峰值working set为112877568 B（107.65 MiB），包括导入、合成状态、预热和全部cases；不分别归因于原/复用路径，不与Linux训练RSS或星载峰值直接比较。
- 实际测量总墙钟5.504 s。固定合成U有736×8个数值坐标，未更新任何模型参数；没有训练、QP拟合、梯度、checkpoint或地面模型载入。
- 新增地面数据/摘要载荷0 B；单样本packet逐对象常驻字节、实际代码传输字节、设备能耗、星载耗时与显存N/A。本机CPU测量不能确认设备收益。
- 合成评分展开不是训练或KKT解。A旧/B旧/C旧/C新准确率、H、适应提升、注册遗忘、新旧差距均N/A；正在进行的完整真实三阶段基准另行评分。

## 执行与证据

run `20261001-d92-margin-single-query-cost-synthetic-r01`，唯一launch owner=root；实际runtime `479f6389d6a6f8ea1899fca017eeb8a6d1dc5069`已先push/OID核实。8项限定软件测试通过后，唯一实际测量全部8行完成。独立[结果读回](../automation_reports/CV-SincNet/20261001-d92-margin-single-query-cost-synthetic-r01/evidence/result_readback_20261001.json)交叉检查实际PID、argv/CWD/python、运行commit、全部row、终态、24行AI表和工作量。原始输出保留于本地local_artifacts/d92_margin_single_query_cost/20261001-d92-margin-single-query-cost-synthetic-r01，Git只镜像本次小型结果和记录。

源码[测量CLI](../tools/measure_d92_margin_single_query_reuse.py)、[测试](../tests/test_measure_d92_margin_single_query_reuse.py)与[validation JSON](D92_MARGIN_SINGLE_QUERY_COST_VALIDATION_20261001.json)不读取真实query、truth、权重或实验评分。原完整query run仍为20261001-phase2-d92-margin-joint-repeat-m2-r01、runtime d86edc323；全部四行预测固定后才执行独立评分，不根据本次成本或中途结果改训练参数。
