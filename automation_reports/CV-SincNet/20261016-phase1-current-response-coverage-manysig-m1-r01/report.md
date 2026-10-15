# 当前三抽头增强的源谱形方向覆盖诊断

状态：ANALYZED。本地执行及终点产物核实 VERIFIED。预登记结论为 IDEAL_MAGNITUDE_DIRECTION_GAP：当前三抽头族在固定理想幅度代理中的加权方向覆盖仅 9.3023%，分辨率控制为 9.2692%，相差 0.00033097，未触发 .01 的分辨率不足规则。这支持该受限族的方向覆盖不足，不能把扩大同一扰动幅度当成已获支持的修复。

这里的覆盖率是每个交叉拟合源方向与单个可选响应之间的非负投影能量比，再按预测方向能量汇总；不是所有响应的线性张成空间，也不是识别准确率或可恢复样本比例。放松的标量不受幅度界限制，仅为诊断，不可部署为 IQ 操作。

## 固定输入与方法

复用 20261011-phase1-rx-spectral-crossfit-manysig-m1-r01 已导出的合法 L6300 聚合数组，无新 IQ、U、V、目标数据或模型读取。原统计 6 TX×3 day×5 RX×70，18 折同时排除当前 TX 与 day，固定主观测 rectangular/DCT1to8。独立重算原完整交叉拟合预测的相对 MSE 为 0.614141006955，与旧产物一致。不事后挑更有利的 Hann 观测。

当前族 H(w)=1+a*cos(w)、abs(a)<=.06，半径来自已有增强配置。使用 FFT256、32 个池化频带、中心化 log power 及正交 DCT1 至 8；主网格 17 半径×2048 角度，共 32769 原子；分辨率控制 9×1024，共 8193。参数只按被排除 TX/day 以外的数据所给预测方向选择，被排除单元只计算误差。

|源方向重构|相对被排除单元的 MSE，零对比=1|
|---|---:|
|完整已有源交叉拟合预测|0.614141|
|有界当前三抽头族|1.035205|
|放松幅度后的当前形状族|1.033042|

有界族对预测向量的残差能量比为 0.944581。放松幅度后，被排除单元重构仍未优于零对比；这表明当前形状很难复现这些源关联，但不表示使用该增强训练必然有害。

|源 RX|加权方向覆盖 %|有界族被排除 MSE|完整交叉拟合 MSE|
|---|---:|---:|---:|
|1|6.5109|1.103513|0.753807|
|3|11.3990|1.032841|0.448666|
|4|20.5237|0.964525|0.593191|
|6|4.0547|1.107998|0.775601|
|8|5.1501|0.984866|0.602910|

## 实际执行与验证

执行代码 commit：f3ee86ca9f00723474274646189a7ea8b1032f7b。本地 Python：F:/App/miniconda3/python.exe，NumPy CPU2，CWD：E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014。

```
F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_response_coverage.run
F:/App/miniconda3/python.exe -X utf8 -m experiments.cvs_response_coverage.analyze
```

主计算耗时 0.375108 秒；新增 epoch、优化更新、默认头拟合、模型前向均为 0。先前 6300 条源谱统计的计算成本不重复计入。0 新 seed，不构成跨 seed 性能复现。

本地 5 个数学例、9 个源角色负例、1 个配置负例和 4 个判定例 PASS；一次限定独立 P0/P1 审查 PASS。源数组执行后完整结果、完成记录和逐折数据已核实。纯统计阶段不消费 checkpoint 或 N607，真实 checkpoint smoke、远端发布及 GPU preflight 不适用，不创建替代许可。

实际生成并查看了[源方向覆盖与重构图](E:/type10-7/local_artifacts/cvs_response_coverage_20261016/run/analysis/source_response_capacity.png)；[逐折/RX CSV](E:/type10-7/local_artifacts/cvs_response_coverage_20261016/run/analysis/per_fold_rx.csv)、[汇总](E:/type10-7/local_artifacts/cvs_response_coverage_20261016/run/analysis/summary.json)、[原始计算结果](E:/type10-7/local_artifacts/cvs_response_coverage_20261016/run/results.json)和[完成记录](E:/type10-7/local_artifacts/cvs_response_coverage_20261016/run/complete.json)均存在。输出根只执行一次，无覆盖或选择性重跑。

## 解释边界与下一步

RX 关联混合了硬件、传播、采集及交互，不是已辨识的纯 RX 硬件传递函数。当前计算是平坦带内输入、无限信号的理想幅度代理；忽略有限窗、带内权重、log floor、人工边界及相位。因此不能据此宣称某个新 FIR 身份保持、clean 变好或 low urban 超过 60%。

已找到并复用历史 18 折统计；同时确认旧 33 抽头 PHYS/ROT 训练未建立实测方向优于旋转方向的优势，不重复宣布该方向是新方法。下一步唯一候选问题是：在当前 SIG_TIME＋RDrop 的固定默认头下，源关联的较丰富响应方向能否在同样有界强度和有限 IQ 实现中安全覆盖该缺口。需要保留旋转方向及边界控制；未获该证据前不自动启动混合训练、扩 seed 或调整幅度。模型与原七视图不变，整体联合性能目标尚未完成。
