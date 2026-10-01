# Margin单样本prior复用的本机软件测量

固定测量CLI已通过8项限定软件测试，尚未实际运行完整成本测量。唯一测量run为20261001-d92-margin-single-query-cost-synthetic-r01；预登记[报告](../automation_reports/CV-SincNet/20261001-d92-margin-single-query-cost-synthetic-r01/report.md)和配置记录保存完整8行。实际模型准确率基准继续使用原runtime，本测量与它独立。

输入是固定随机生成的完整评分展开，旧类6、新类20，K=1/5/10/20，8个单样本输入、1次预热、3次计时重复，BLAS/OMP各2线程。分别比较PAIR和C_ONLY；每次完整float64分数逐位相等是直接正确性要求，不只比较最终类别。原路径和复用路径交替先执行，保留全部配对计时，不筛最好重复。

实际执行时记录硬件、解释器、库版本、线程池、逐row真实work、packet准备和核对时间；reference distance是raw的子集，不相加。PeakWorkingSet是全测量进程生命周期峰值，包含导入、合成状态和预热，不能当成每row或真实星载峰值。额外地面数据载荷0 B；代码传输、设备能耗与星载收益N/A。合成评分系数不是训练、QP解或真实模型，不从这些数值推断精度。

源码[measure_d92_margin_single_query_reuse.py](../tools/measure_d92_margin_single_query_reuse.py)和[测试](../tests/test_measure_d92_margin_single_query_reuse.py)不读取真实数据、权重、query、truth或实验评分。正确性证据见[validation JSON](D92_MARGIN_SINGLE_QUERY_COST_VALIDATION_20261001.json)。相关数学条件见[复用实现](D92_MARGIN_SINGLE_QUERY_REUSE_IMPLEMENTATION_20261001.md)。仅部署C仍必需一次B prior，不能套用PAIR消除重复B求值的节省。
