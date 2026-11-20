# SatIQ PAMF-SUP 共享卫星权重第 5 轮配对复核

[配置](experiment.json) · [事件](events.jsonl) · [首轮配对](../20261120-diagnostic-satiq-shared-satellite-paired-source-e1-s2026111907-r01/report.md) · [第 5 轮训练](../20261120-diagnostic-satiq-shared-satellite-head-source-e5-s2026092701-r01/report.md)

比较同采样、同等 141,125 步的两组原生 PAMF。两组沿各自首轮检查点及随机状态继续训练，解析配置只在共享卫星权重开关和续训来源上有差异。已存在的 GRID256 对照命中 363 条，主 CE 为 7.0429。固定两份检查点后，使用相同 49,814 条已知 V 和全部 3,103 候选逐包复核。按卫星分组核对命中净差，并用时间戳事后切片。第二项把主 CE 拆成卫星边缘 CE 与同星内波束 CE，真卫星仅用于离线诊断。独立 T 留出。
