# SAM的配对增量与交互差

状态 PARTIAL。只读取已完成的独立评分；每项仅使用全部组成方案均已完成的同seed配对，不混用不完整种子。差值单位为百分点。

|比较|种子1|种子2|种子3|配对均值|配对SD|种子数|
|---|---:|---:|---:|---:|---:|---:|
|SAM_SSL-SSL|N/A|-0.7577|N/A|-0.7577|N/A|1/3|
|SAM_PHYS-PHYS|N/A|N/A|N/A|N/A|N/A|0/3|
|SAM_RANK-RANK|N/A|N/A|N/A|N/A|N/A|0/3|
|PHYS-SSL|+0.0851|+0.4304|-0.1524|+0.1210|0.2930|3/3|
|RANK-PHYS|+0.1798|+0.4375|+0.2345|+0.2839|0.1358|3/3|
|SAM_PHYS-SAM_SSL|N/A|N/A|N/A|N/A|N/A|0/3|
|SAM_RANK-SAM_PHYS|N/A|N/A|N/A|N/A|N/A|0/3|
|SSL-P0|+1.1095|-0.5488|-0.3440|+0.0722|0.9041|3/3|
|SAM_SSL-P0|N/A|-1.3065|N/A|-1.3065|N/A|1/3|
|PHYS-P0|+1.1946|-0.1185|-0.4964|+0.1933|0.8876|3/3|
|SAM_PHYS-P0|N/A|N/A|N/A|N/A|N/A|0/3|
|RANK-P0|+1.3744|+0.3190|-0.2619|+0.4772|0.8295|3/3|
|SAM_RANK-P0|N/A|N/A|N/A|N/A|N/A|0/3|
|SAM_by_physical|N/A|N/A|N/A|N/A|N/A|0/3|
|SAM_by_risk|N/A|N/A|N/A|N/A|N/A|0/3|

SAM_by_physical = (SAM_PHYS−PHYS)−(SAM_SSL−SSL)。SAM_by_risk = (SAM_RANK−RANK)−(SAM_PHYS−PHYS)。正交互只说明本批中SAM增量随该配方改变，不能单独证明整体超过基座或达到84%，也不等同统计显著。

完整RX、TX、日期分层配对与缺失种子保存在 evidence/paired_interactions.json。主比较SAM_RANK−RANK，所有固定配方均报告；本分析不选择性重跑，不回流设计或训练。
