# 星地对比：LoRa 与 Orthogonal 技术接续

r01 两个 POSTER→LoRa 任务查询预测阶段因 Python 闭包变量重用而将 POSTER 特征按普通网络接口解析，报 `IndexError`；两处输出目录没有预测文件，原始日志和目录保留。修复后按真实 source method 固定特征提取接口，并在仅支持数据上覆盖注册后的特征提取；这不是 MoPC 数值故障。

本 run 从原 MoPC 队列保存的相同 B 状态继续；r01 已完成的 8 组预测先经完整性校验并复用，立即独立 truth-last 评分；两处零预测技术失败在新独立输出目录重试，其他未开始任务按原矩阵执行。任何无法分类的部分预测阻止自动接续。原始输出、日志及数值隔离保持。

五种主干、MIX01–03、三模型种子、三池、七接收机，K=5/10、N=5/10；LoRa 平衡样本库和 Orthogonal support-prototypes 桥各 945 组。Tweak M1 单条 query。Orthogonal 桥可能改变 B 旧类入口判决，不称原作者完整网络，也不声称性能已验证。

N607、GPU 0–7、本任务每卡最多 6 个进程、余量至少 6000 MiB，不干预其他 owner；每个任务冻结后立即进入独立 CPU 评分队列。系统性错误停止派发并排空，记录原始错误和未评分项。原数据协议、查询隔离和 truth-last 不变。

唯一输出 `runs/20261117-phase2-satellite-mix3-savedb-c-m3-r02`，配置 `preparations/20261117-phase2-satellite-mix3-savedb-c-m3-r02/scheduler.json`，代码版本为本报告所在提交，N607 CVS-RFFI Python 入口 `experiments.pamf_phase2_practical.satellite_saved_b_c`，发布及输出根以该配置为准。
