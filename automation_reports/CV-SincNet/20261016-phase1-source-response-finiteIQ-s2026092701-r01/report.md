# 源谱方向：当前 E20 默认头的有限 IQ 验证

状态：ANALYZED，执行与终点核实 VERIFIED，必要源安全条件 PASS。实测方向在 E1/E20 固定 session 下相对 RAW 的正确数净变化为 +2／−14，最差源 RX 下降 0.055556／0.203704 pp。没有训练新模型，也没有建立实测方向稳定优于旋转方向的证据，不能称为 clean 或星地泛化突破。

## 固定配置与执行

执行代码：4657eec3f23ef2b9c16cb8c058338526224a4cc7。固定原 scratch E20_RDROP seed2026092701 final 默认头；严格复用原来源核查与 fullL6300/U56700/V27000 契约，非 E50/E200，非目标分数选模。源 V 全部 RX 曾参与原源训练；这里是观测变化安全诊断，不能称为未见 RX 识别提升。

源银行只用 20261011 既存 L6300 的跨 TX/day 预测数组，按 excluded day/RX 平均全部 6 个 excluded TX，固定 15 方向。DCT1 至 8 形状经 IFFT、居中 33 抽头 Hann 截断并移除零延迟标量增益；固定正交旋转为控制。每对核 L2 相等，共同缩放使 delta 核 L1≤.06，内部幅度界 0.94 至 1.06。不反演原录音硬件，也不增加噪声、PA、CFO、剪裁等损伤。

session seed20261016023，rotation seed20261016011，按 epoch/RX/day/eq/view 固定，所有 TX 共享。原 IQ256、25 MHz、equalized1、native RMS；变换后恢复包 RMS。七条件为 RAW、EMP/ROT 的 E1/E20 reflect 及 E20 edge。EMP reflect 两条件预定 overall 损失≤1 pp、各源 RX 损失≤2 pp；ROT/EDGE 不作替代候选。没有根据本结果调整幅度、方向或阈值。

## 真实源默认头结果

|条件|正确数|源准确率 %|相对 RAW pp|修复／新增错误|默认头间隔/30|
|---|---:|---:|---:|---:|---:|
|RAW|26425/27000|97.870370|+0.000000|0／0|0.216999769|
|EMP_E1_REFLECT|26427/27000|97.877778|+0.007407|12／10|0.216508240|
|ROT_E1_REFLECT|26376/27000|97.688889|-0.181481|16／65|0.214146957|
|EMP_E20_REFLECT|26411/27000|97.818519|-0.051852|4／18|0.216098830|
|ROT_E20_REFLECT|26435/27000|97.907407|+0.037037|19／9|0.215979174|
|EMP_E20_EDGE|26413/27000|97.825926|-0.044444|5／17|0.216095924|
|ROT_E20_EDGE|26435/27000|97.907407|+0.037037|19／9|0.215997726|

EMP−ROT 在 E1 为 +0.188889 pp，在 E20 为 −0.088889 pp；两条件方向相反。EMP E20 的平均默认头间隔也低于 RAW，未证明更强身份分离。不同 session 是输入随机流控制，不是独立模型 seed，不把它们当作跨 seed 复现。

各核的能量成对匹配，但实际输入上的波形 MSE 不完全相同：EMP/ROT 的 E1 分别为 0.000093886／0.000097628，E20 为 0.000058066／0.000062613。不能声称每个波形承受了完全相同的损伤。最大相对功率误差 3.576280e-7；抽查批次未经 RMS 缩放的 reflect/edge 内部 [16,240) 最大差为 0。参考 [80,160) 在有效区内，但端部仍可通过整包 RMS 标量和网络全局处理影响决定。EMP E20 的 edge 比 reflect 多对 2 条，不据此换边界或晋级。

## 验证、成本与产物

真实 checkpoint 无 query smoke2 PASS 后立即继续。一次归档传输比较及远端编译 PASS，普通账户 CPU2／CUDA 隐藏；独立两次读取确认 PID/CWD/argv 匹配及日志、源进度增长。终点进程自然退出、complete 为 ANALYZED、failure 不存在。原模型参数、buffer 和 Torch RNG 均未改变，优化器／EMA／头拟合更新均 0，无 target 访问或新预测。

本地数学/算子检查通过，含 9 个算子负例、3 个协议负例、3 个安全判定例；一次限定独立 P0/P1 审查 PASS。取回后完整独立重数 189000 条源决定，核对各条件同一 27000 物理 ID、标签、混淆、纠错和间隔；预定安全结论重算一致。六个变换条件的 RX×day×TX 540 个单元均为 300 条。

新增前向 189002，其中正式源诊断 189000、smoke2；墙钟 120.415114 秒，0 新模型／0 新 epoch／0 优化更新。原只读父模型成本单列 E20／4440 更新／923.560061 秒，不作为本轮训练继承或新增预算。没有训练曲线，因为本轮不训练。正式 clean 及六 practical 未重测或修改。

实际发布命令（工作树 E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014）：

```
F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_source_response20.publish --output E:/type10-7/local_artifacts/cvs_source_response_finiteIQ_20261016/release
F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_source_response20.harvest --output E:/type10-7/local_artifacts/cvs_source_response_finiteIQ_20261016/final
```

第一次取回调用在运行中返回 RUNNING，未创建部分 final 目录；完成后才取回全量。没有重新提交、重启、热改或选择性重跑。

已生成并查看[真实源结果与 RX 分层图](E:/type10-7/local_artifacts/cvs_source_response_finiteIQ_20261016/final/frozen_source_response.png)；[逐条件 CSV](E:/type10-7/local_artifacts/cvs_source_response_finiteIQ_20261016/final/source_metrics.csv)、[独立核算汇总](E:/type10-7/local_artifacts/cvs_source_response_finiteIQ_20261016/final/analysis_summary.json)、[完整源决定](E:/type10-7/local_artifacts/cvs_source_response_finiteIQ_20261016/final/sample_decisions.jsonl.gz)、[resolved config](E:/type10-7/local_artifacts/cvs_source_response_finiteIQ_20261016/final/resolved_config.json)、[终点核查](E:/type10-7/local_artifacts/cvs_source_response_finiteIQ_20261016/terminal.json)均存在。具体输入、命令、停止规则及预登记矩阵以 experiment.json 为准。

## 结论与未完成项

当前受限三抽头理想方向覆盖不足的结论保留；新有限响应通过必要安全检查，但物理关联方向相对旋转控制尚无一致优势。旧 33 抽头 PHYS/ROT 的负结果仍有效，不把本次安全 PASS 当作对其推翻。

下一最小训练检验是固定当前 E20 配方下 EMP 与 ROT 的独立 A_g，不先增加组合损失或级联视图。必须同数据、初始化、loss 及前向控制，并在新目标结果前固定源资格；源 clean 不改善就结束该方向，只有独立价值建立后才检验与原 A_s 联合。当前没有新训练或正式性能结果，low urban>60%、clean 正收益及跨 seed 复现仍未完成，整体 goal active。
