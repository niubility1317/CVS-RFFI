# 投稿图表进度与统计口径

采用项目既有 Python 制图工作流，仅读取已独立核验的冻结评分，不读取 IQ、checkpoint 或 truth，不执行模型、训练或选模。正式训练代码仍为 cb9e19fbbbfee31704064be0ad4f6ec96d29b966，制图脚本独立保存。

## 图表与论证

| 图 | 论证问题 | 数据与展示 | 当前状态 |
|---|---|---|---|
| 图 1：源 RX 数 | 多样性与训练标签量能否区分？ | 1/3/5 源 RX，自然/固定标签预算，三链分面、三种子点及 SD | **已完成并验证** |
| 图 2：标签效率 | 少标签表现是否稳定、是否伴随不同计算量？ | 1%/5%/10%，标出标签数及实际更新范围；ResNet18、IB-RFF CE/full | **已完成并验证** |
| 图 3：RX×日期 | 机制的增减来自哪些接收机和日期？ | F0/Chrono 每 RX×日期 full−CE 的配对均值，共用对称色标，标出未见日期 | **已完成并验证** |
| 图 4：压力结果 | clean 与模拟星地压力是否存在不同方向的收益？ | 四方法各自 CE/full × 监督/PL；上排绝对性能，下排同种子相对 clean 下降 | **已完成并验证** |

各图沿用既定矩阵，不根据结果改换方法、场景、源组合或种子。更强 CE 与负结果保留。图 3 的未来端点为目标 RX 的 D3 子集，不将 Chrono 的全日期 overall 当作未来结果。ManyRx 的不平衡主指标在表格中采用 RX 等权准确率。

## 图 4 的证据与交付

48 个 F0 模型、每行 7 个视图，共 336 个种子点。112 组均值和样本 SD 逐项与四方法既有完整报告核对；相对 clean 的下降直接在同一种子内相减，再计算均值及 SD。未将两个独立 SD 相加代替配对差的实际 SD。

图中同时保留绝对性能与下降量，避免把较低 clean 基础上的小下降单独解释为更强识别性能。每个模型的七视图来自相同物理包，不增加独立重复数。n=3 指模型种子，误差线为样本 SD，不是置信区间；没有显著性标记或推断检验。

交付采用 183 mm 宽画布、统一无衬线字体、可编辑文本 SVG、文本可提取 PDF、600 dpi PNG 及预览图。已检查画布边界、PDF 尺寸、SVG 文本、PNG 分辨率，并完成预览目视检查。使用评分子集重绘成功，证明交付不依赖私有训练环境或模型文件。

- [图 4 PDF](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig04_pressure_r03/pdf/fig04_pressure.pdf)
- [图 4 可编辑 SVG](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig04_pressure_r03/svg/fig04_pressure.svg)
- [图 4 600 dpi PNG](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig04_pressure_r03/png/fig04_pressure.png)
- [逐种子源数据 CSV](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig04_pressure_r03/source_data/fig04_pressure.csv)
- [重绘说明及配套脚本](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig04_pressure_r03/README.md)
- [项目制图代码](../../../tools/figures/plot_phase1_external_dg.py)

图形及派生评分数据在本机交付，不提交原始实验输出到 Git；代码和分析说明使用 Git。四幅图均已完成数值核对、目视检查及评分子集独立重绘。整体 219＋96 实验目标的完成状态另见最终汇总与成本证据。

## 图 3：接收机与日期分层完成

48 个监督模型（F0/Chrono × 四方法 × full/CE × 三种子），672 个 RX×日期逐种子配对点，全部 224 个显示单元采用三种子配对均值。每单元每种子 6000 包；四方法 D3 单元平均均与各自主要未来结果一致。共用对称色标、不截断数据；黑框标出 F0 的较早未见 D0 和 Chrono 的未来 D3，不能混用两个时间含义。源数据保留各 seed，不将单元数或日期数视为独立训练重复。

已调整底部说明间距并目视检查；从独立评分子集重绘成功，数据 CSV 与初版相同。

- [图 3 PDF](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig03_receiver_date_r02/pdf/fig03_receiver_date.pdf)
- [图 3 可编辑 SVG](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig03_receiver_date_r02/svg/fig03_receiver_date.svg)
- [图 3 源数据](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig03_receiver_date_r02/source_data/fig03_receiver_date.csv)
- [重绘说明](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig03_receiver_date_r02/README.md)

## 图 1：源接收机数量曲线完成

33 个去重模型、三条源 RX 链、自然与固定标签预算，显示 54 个种子位置；共享的 m=1 预算点和 m=5 链端点均明确标注。所有点逐项与 33 行完整报告核对，保持 10000 更新、batch180 的实际预算。已完成目视与矢量/分辨率检查，并用交付评分子集独立重绘；两次导出的 54 行数值一致。

- [图 1 PDF](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig01_source_count_r02/pdf/fig01_source_count.pdf)
- [图 1 可编辑 SVG](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig01_source_count_r02/svg/fig01_source_count.svg)
- [图 1 源数据](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig01_source_count_r02/source_data/fig01_source_count.csv)
- [重绘说明](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig01_source_count_r02/README.md)

## 图 2：训练标签效率完成

27 个去重模型、三个训练标签比例和三条模型路线，共 27 个逐种子点。1%/5%/10% 分别对应 630/3150/6300 个 L_s 标签，另有固定 27000 个 V 标签；此图全部为增强监督，不使用 PL。ResNet18 的 10% 点复用同配方 E200 强 CE，不使用 5000 更新通用 ERM 替代。图中标出实际更新范围，保留全部非单调结果，不能将不同标签量的固定 epoch 训练称为等计算比较。

所有点已与标签效率报告核对；183 mm 画布、可编辑 SVG、PDF 文本与 PNG 600 dpi 检查通过，预览无边界裁切。使用评分子集独立重绘后，27 行 CSV 逐字节一致。r01 的边界检查失败版本及 r02 保留为历史；最终交付为 r03。

- [图 2 PDF](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig02_label_efficiency_r03/pdf/fig02_label_efficiency.pdf)
- [图 2 可编辑 SVG](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig02_label_efficiency_r03/svg/fig02_label_efficiency.svg)
- [图 2 源数据](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig02_label_efficiency_r03/source_data/fig02_label_efficiency.csv)
- [重绘说明及配套脚本](../../../local_artifacts/phase1_external_dg_aug_pl_gpu23_r3_20261019/publication_figures/fig02_label_efficiency_r03/README.md)
