# Source coefficient association

固定 E200 的全部源 V：8 个模型，各 27000 包，6 TX × 5 RX × 3 day，每单元 300 包。系数排列为 real4、imaginary4。按每模型完整平衡析因分解，再计算四 seed 均值与样本 SD；四 seed 共用数据，不能当作四份独立数据集。

七个主效应/交互项为单元均值的正交分解，within-cell 为单元内 population trace variance 的平均，不除以 300。总量是 8 个系数分量方差之和。逐 seed CSV 与 JSON 保留全均值、各项原始值及重构误差。

|结构|因素|trace variance 均值 ± seed SD|占比均值 ± seed SD|有效占比 seed|
|---|---|---:|---:|---:|
|frontfilter_static|TX|0 ± 0|N/A|0/4|
|frontfilter_static|RX|0 ± 0|N/A|0/4|
|frontfilter_static|day|0 ± 0|N/A|0/4|
|frontfilter_static|TX_RX|0 ± 0|N/A|0/4|
|frontfilter_static|TX_day|0 ± 0|N/A|0/4|
|frontfilter_static|RX_day|0 ± 0|N/A|0/4|
|frontfilter_static|TX_RX_day|0 ± 0|N/A|0/4|
|frontfilter_static|within_cell|3.469447e-16 ± 4.012577e-16|N/A|0/4|
|frontfilter_static|total|3.469447e-16 ± 4.012577e-16|N/A|0/4|
|frontfilter_dynamic|TX|1.4514847e-06 ± 8.1421233e-07|0.014438 ± 0.010703|4/4|
|frontfilter_dynamic|RX|1.748415e-05 ± 5.3003816e-06|0.165663 ± 0.100429|4/4|
|frontfilter_dynamic|day|2.4272938e-07 ± 1.2979926e-07|0.001808 ± 0.000091|4/4|
|frontfilter_dynamic|TX_RX|9.5047899e-06 ± 2.030149e-06|0.091370 ± 0.052217|4/4|
|frontfilter_dynamic|TX_day|8.3843743e-07 ± 2.9150065e-07|0.007214 ± 0.002997|4/4|
|frontfilter_dynamic|RX_day|7.8000654e-07 ± 1.5155675e-07|0.007253 ± 0.003693|4/4|
|frontfilter_dynamic|TX_RX_day|3.9302811e-06 ± 1.2543141e-06|0.036958 ± 0.020964|4/4|
|frontfilter_dynamic|within_cell|0.00010227116 ± 7.2908443e-05|0.675294 ± 0.180393|4/4|
|frontfilter_dynamic|total|0.00013650304 ± 7.6145872e-05|N/A|0/4|

总方差不超过 1e-10 时占比记 N/A，避免将静态系数的浮点消减微差放大成因素占比；实际方差原值保留，不静默归零。此容差只影响数值展示，不是性能或晋级门槛。不同 seed 的滤波基各自学习，系数坐标不保证跨模型对齐；这里汇总各模型内部方差，不平均不同模型的系数向量来推断物理因素。

这些是源系数与 TX/RX/day 标签的观测关联，非因果归因，不能证明信道恢复、TX/RX 解耦或泛化。within-cell 同时包含包间变化和未建模因素，不等同于噪声。该诊断禁止用于选模、重选 epoch、调参或选择性重跑。未访问 target，不改变已有源冻结。
