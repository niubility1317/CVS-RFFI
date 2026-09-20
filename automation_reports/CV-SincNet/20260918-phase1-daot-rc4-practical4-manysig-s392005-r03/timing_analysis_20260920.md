# practical训练耗时分析：2026-09-20

范围：完整解析当前可用metrics_epoch.jsonl与完整stdout：full_noeq E1–142、ZF E1–129、MMSE E1–152、residual E1–200，及原LEO E1–200。这是活动日志截至本次快照的耗时分析，不是全训练完成声明。所有结构化记录解析成功；扫描完整stdout未发现Traceback、CUDA OOM或SIGSEGV标记。

|组|完整epoch|累计epoch耗时h|训练批次h|额外源域验证h|周期目标测试h|单次目标测试中位分钟|
|---|---:|---:|---:|---:|---:|---:|
|full_mmse|152|46.57|34.50|6.98|4.85|48.46|
|residual_noeq|200|34.30|25.42|4.42|4.16|22.50|
|full_zf|129|46.53|37.69|6.24|2.37|47.60|
|full_noeq|142|46.62|37.43|5.69|3.25|38.99|
|original_leo_control|200|19.03|17.29|1.01|0.41|1.77|

主要耗时机制：practical_adapter.apply_practical将GPU张量复制到CPU float64，通过tolist转NumPy，逐记录调用CPU信道模拟，再经tolist构造GPU张量。这是Torch2.1/NumPy2.2 ABI兼容路径。leo_practical/batch.py逐记录循环，full包含逐样本跟踪和多径处理。没有GPU向量化信道核。DAOT E21启用后新增L/U视图与教师/学生处理；MMSE非测试轮中位耗时从E1–20的325秒到E21–40的951秒，residual从198秒到541秒。断点与模块启用一致，但没有独立算子profile，不能把全部增量归给单一算子。

周期测试每次覆盖168000条IQ×4场景=672000条预测，其中504000次星地增强。full noeq一次约39分钟，ZF约48分钟，MMSE约48分钟，residual约22.5分钟，原LEO约1.8分钟。周期测试在训练进程内同步执行，暂停下一epoch。11次测试预计累计full约7–9小时，residual已实测目标测试累计约4.16小时（含最终弱信道参考）；不是全部慢速原因。

重源域验证每epoch重复：MMSE通常约162秒、residual约80秒，原LEO约12秒；200轮累计约9h/4.4h/0.7h的量级，中位数外推并非真实完工时间。训练批次占现有总耗时约74%–81%，所以减少目标测试也无法消除主体开销。

full_noeq/ZF有明显时间波动：E91后非测试epoch中位约1655/1805秒，但最新epoch仅820/1073秒；无法用固定算法复杂度解释全部波动。共享GPU、CPU与线程竞争是待profile核实的候选因素。快照GPU利用率属于整卡，不能当作本实验利用率；训练主进程累计CPU约112%–114%，符合串行CPU阶段较重，但不能单凭该值量化瓶颈比例。原LEO也有阶段性慢速，跨run耗时比不是同硬件负载严格benchmark。

优化顺序：优先复用固定测试/验证增强视图缓存（验证按ID、配置、seed一致）；剖析CPU信道与拷贝耗时后对逐记录路径做保真批量化；隔离资源后再比较吞吐。训练增强含动态epoch/role/seed，不能直接冻结缓存替换。减少DAOT视图、降低测试频率或改变增强概率会改变实验配方，不在本次诊断中修改活动实验。

证据：evidence/timing_audit_20260920.json（完整epoch计时与测试scope、资源快照）、evidence/timing_summary_20260920.json（分阶段统计）。没有算子级profile与历史资源连续记录，不能给出CPU信道/DAOT/GPU竞争各自的精确因果百分比。
