# SAM的配对增量与交互差

状态 PARTIAL。只读取已完成的独立评分；每项仅使用全部组成方案均已完成的同seed配对，不混用不完整种子。差值单位为百分点。

|比较|种子1|种子2|种子3|配对均值|配对SD|种子数|
|---|---:|---:|---:|---:|---:|---:|

SAM_by_physical = (SAM_PHYS−PHYS)−(SAM_SSL−SSL)。SAM_by_risk = (SAM_RANK−RANK)−(SAM_PHYS−PHYS)。正交互只说明本批中SAM增量随该配方改变，不能单独证明整体超过基座或达到84%，也不等同统计显著。

完整RX、TX、日期分层配对与缺失种子保存在 evidence/paired_interactions.json。主比较SAM_RANK−RANK，所有固定配方均报告；本分析不选择性重跑，不回流设计或训练。
