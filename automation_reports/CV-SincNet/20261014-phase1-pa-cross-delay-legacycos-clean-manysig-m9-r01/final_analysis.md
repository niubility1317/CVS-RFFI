# PA 跨延迟特征实验结果

状态 VERIFIED / ANALYZED。run_id：20261014-phase1-pa-cross-delay-legacycos-clean-manysig-m9-r01；实际执行代码 1f8814c046a8c68eba8a334e66a0aceb76e491e0。旧门控＋EMA＋cos，32 label＋18 pseudo，固定 final50；默认头随骨干联合训练，独立头更新为 0。

|结构|clean 均值 ± 样本 SD|Macro-F1|最差 RX 均值|
|---|---|---|---|
|LEGACY50|80.6454% ± 2.0918 pp|80.1381%|67.8611%|
|PA_CONV|80.7738% ± 2.1914 pp|80.2796%|68.2375%|
|PA_CROSS_DELAY|80.8133% ± 1.9364 pp|80.3279%|68.5681%|

|对比|平均配对差|三个 seed 配对差（pp）|正向 seed|
|---|---|---|---|
|PA_CONV − LEGACY50|0.1284 pp|0.0137 / 0.0845 / 0.2869|3/3|
|PA_CROSS_DELAY − LEGACY50|0.1679 pp|0.1429 / 0.3351 / 0.0256|3/3|
|PA_CROSS_DELAY − PA_CONV|0.0395 pp|0.1292 / 0.2506 / -0.2613|2/3|

完整逐 seed、RX/day/TX、混淆矩阵、Macro-F1、纠错与新增错误、50 轮源曲线见 results.json。CROSS 相对 LEGACY 三 seed 均为正，净纠错为 240/563/43 包；相对同容量 CONV 为 217/421/−439 包。三阶结构相对容量控制仅 +0.0395 pp，2/3 正，不能据此确认机制收益。最差日期不一致改善，84% 未达到。

6 个新模型共 300 model-epochs / 66600 optimizer 与 EMA 更新；3 个历史基线150 model-epochs 单列复用，无新增训练。CONV 与 CROSS 各新增 576 特征参数，五组实际参数全部有非零最终变动。各模型源 V 97.83% 至 97.98%，L CE 0.2445 至 0.2782；这些健康指标不用于选模。各新模型原 epoch 时间合计约 2083 至 2411 秒，GPU 竞争不同，不把它作为受控速度结论。

独立核验完整读取 66600 新 step 记录及450 epoch 记录，11100 更新/模型、非有限跳过 0、EMA .999、U 真值隐藏、32＋18 伪标签预算一致；严格重建全部9个真实默认 checkpoint 并做零 IQ smoke。9 行各168000唯一物理ID，1512000固定默认决定、162条指标逐项从原预测重算；基线504000决定与历史 NPZ 完全一致。所有原 worker 自然退出，queue/failure 为空，预测后才连接独立 truth，模型在推理期间冻结。

所有结果保留，不按目标成绩重跑或调整下一轮。下一轮 TIME 2×2 来自用户预先给定的结构归因设计，与本 PA 目标结果无调参反馈关系。无单独 head fit、support 适配、新 TX、Phase2 或 LEO 部署声明；本 exposed benchmark 的旧6 TX clean 结果不等于独立盲测或真实部署性能。
