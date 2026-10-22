# PAMF 与 Full / Full＋PL 对比图：独立图优化

日期：2026-10-22。状态：`VERIFIED`，仅使用已完成评分结果重绘。

## 范围与来源

按用户要求，每张图片只保留一个数据图表，拆除原三联图，并同步优化七场景图。
只绘制 RIEI、DRIFT、IB-RFF、SDG 的 Full 与 Full＋普通 PL，以及 PAMF（T1＋DURA）。
不包含四个方法的 CE 版本、ResNet CE 或 raw-T1 曲线。

- 外部方法：`20261019-phase1-external-dg-concat-pl-m219-r03` 的完整 F0 已评分行，
  4 个方法 × 2 种训练方式 × 3 个模型种子，共 24 行。
- PAMF：`20261021-phase1-T1-DURA-comparison-m153-r01` 的 F0 `DURA-LU` 读出，
  3 个 T1-E50 识别器种子共用一个固定 DURA v1.0。
- 种子：2026092701、2026092702、2026092703。
- 7 个视图：clean 与 6 种预登记 practical；每视图 168000 个物理包，7 个目标 RX 各 24000。
- 共 9 个系列、27 个模型行、189 条种子／场景评分记录。此处完成不表示其他对比矩阵全部完成。

不加载 IQ、checkpoint、预测数组或测试真值，不训练、推理、选模或重测，不更改默认模型。
源数据复用既有已审计分数，重绘前后逐条一致。

## 交付

项目根目录下的交付目录为：
`exports/phase1_external_dg/20261022-pamf-full-pl-single-charts-r01/`，同名 ZIP 为完整下载包。

| 图 | 文件前缀 | 唯一数据图表 | 尺寸 |
| --- | --- | --- | --- |
| 1 | `Fig1_SixScenario_Accuracy` | 六景平均准确率与三种子散点、样本标准差 | 183 × 125 mm |
| 2 | `Fig2_Clean_Channel_Balance` | clean 准确率与六景均值的关系 | 183 × 130 mm |
| 3 | `Fig3_PlainPL_Increment` | Full＋PL 减 Full 的同种子配对差 | 183 × 100 mm |
| 4 | `Fig4_SevenScene_Accuracy` | 七场景均值与样本标准差热力图 | 183 × 126 mm |

每图分别提供可编辑 SVG、嵌入 TrueType 字体的 PDF、600 dpi PNG 和无损 LZW TIFF。
`preview/` 包含彩色及黑白预览。热力图色条仅为图例，不是第二个数据图表。
包内另附 `source_data.csv`、`summary.csv`、`paired_pl_changes.csv`、图注、说明、复现脚本与验证结果。
旧组合图及两个草稿目录均保留，不覆盖旧文件。

## 视觉编码与统计

RIEI 使用蓝色，DRIFT 使用青色，IB-RFF 使用紫色，SDG 使用金赭色，PAMF 使用朱红。
方法颜色跨图保持一致；Full 用空心圆，Full＋PL 用实心方形，PAMF 用菱形。
热力图所有单元格共用 0% 至 90% 的顺序色阶，不逐行归一化。
采用直接标签、独立数值列和留白；取消 a/b/c 面板，分离原本拥挤的说明。
热力图数值文字根据背景亮度选择深浅色，保持可读性。

六景均值先在每个种子内计算，再跨 3 个种子求均值与样本标准差（ddof=1）。
PL 增益先在同种子内求 Full＋PL 减 Full，再汇总；单位为百分点。
误差线不是置信区间，不使用显著性标记。七个视图相关，不能当作独立重复。
PAMF 标准差以固定 G 为条件，不代表三个独立恢复器的方差。
各方法训练配方和预算不同，不能把系统间差值解释为等算力或单组件因果贡献。

## 验证

- 8 项针对性测试通过，覆盖完整矩阵、异常记录、种子配对、先六景聚合后求样本标准差，以及每图恰好一个数据坐标区。
- 标准库 `statistics` 独立重算全部 72 项均值／标准差和 12 项配对差，与导出结果一致。
- 最终包的 189 条源记录与旧稿逐条一致，复现脚本与实际制图实现一致。
- 全部 SVG 保留文本与矢量数据标记，未嵌入位图数据图；PDF 单页宽度为 183 mm 且文本可提取。
- PNG/TIFF 的像素尺寸、600 dpi 及 TIFF 无损压缩通过读回检查；文字未越出画布。
- 四张彩色预览及黑白预览完成视觉检查；标题、图例和场景分组无重叠。
- ZIP 含 31 个文件，读回无损坏。生成图、原始评分数据及本机环境不纳入 Git。

复现命令（在解压后的目录内执行，输出目录必须是新目录）：

```text
python reproduce.py --source-csv source_data.csv --output NEW_FIGURES
```

依赖：Python、matplotlib、numpy、Pillow、pypdf。未安装新软件。
本次仅记录制图实现、针对性测试和本说明，不修改实验登记、其他 owner 的进程或实验配置。

## 追加交付：七场景独立分组柱状图

用户随后明确要求把七个场景各画成一张分组柱状图。本次追加交付为
`exports/phase1_external_dg/20261022-pamf-seven-scene-grouped-bars-r01/` 及同名 ZIP。
前述四张摘要图及其下载包保持原样。

- 每图包含四组外部方法的 Full／Full＋PL 并排柱，以及一个 PAMF 朱红单柱，共 9 根柱。
  Full 位于左侧，用浅色斜线；Full＋普通 PL 位于右侧，用同方法颜色的实心填充。
  不虚构 PAMF＋PL，也不增添 CE-only 对照。
- 方法顺序为 RIEI、DRIFT、IB-RFF、SDG、PAMF，沿用既有蓝、青、紫、金赭、朱红配色。
  七图均为 183 × 116 mm，纵轴统一从 0% 至 100%，不截断柱高。
- 文件前缀依次为 `Fig1_Clean`、`Fig2_Suburban_High`、`Fig3_Suburban_Mid`、
  `Fig4_Suburban_Low`、`Fig5_Urban_High`、`Fig6_Urban_Mid`、`Fig7_Urban_Low`。
  每图仍有可编辑 SVG/PDF、600 dpi PNG/TIFF、彩色及黑白预览。
- 柱高与柱顶数字为三个模型种子的准确率均值，误差线为样本标准差（ddof=1）。
  固定 G、相关视图、配方与预算不等的解释边界沿用前述约定。
- 10 项针对性测试通过，包括七图的实际场景列、四对并排柱、唯一 PAMF 柱、统一零起点纵轴。
  63 根场景柱的均值和标准差用标准库独立核算一致；189 条源记录、72 项汇总和 12 项配对差
  与前版逐条一致。仅重绘，不重新训练、预测或评分。
- 七张彩色预览已逐张检查，另抽查 low urban 黑白预览；斜线／实心编码、数值和图例可辨。
  各格式读回、矢量文本、画布边界和分辨率检查通过；ZIP 共 49 个文件，读回无损坏。

七张柱状图的复现命令：

```text
python reproduce.py --layout scene-bars --source-csv source_data.csv --output NEW_FIGURES
```
