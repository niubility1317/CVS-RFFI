# 十三种网络纯CE E200追加四种子：CVS为残差融合版本

- run_id：`20261012-phase1-standard-ce-seed-extension-manysig-m52-r01`
- group_id：`phase1-standard-fullwidth-ce-baselines`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（本地完整分析；远端合并失败保留）（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Matched historical E200 seed extension; 12 common architectures and residual-fusion CVS; native CVS excluded by user.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 固定发布计划与验证

用户于 2026-10-12 确认其余 13 种网络各追加 4 个模型种子，取消原 CVS。CVS 默认为 residual_fusion。新增模型种子为 2026092705/06/07/08，原 01 至 04 的已完成分数仅用于最终统计合并，不用于模型选择。

每行从零训练 E200，共 10000 次更新，固定末轮权重；数据、架构、优化器和其余配置与原 E200 计划相同。固定 split seed392005，L_s/U_s/V=6300/56700/27000；仅 L_s 身份 CE，无额外损失、训练增强、U_s 或教师。沿用原 clean 测试 capsule 和物理包，不重新划分。每行完成即预测 168000 个相同包，预测完整后独立 truth-last 评分。当前已暴露历史代理基准，不声明新的独立盲测。

唯一 launch owner 为 experiment.json 中的 owner；精确发布命令、环境、CWD、run/log/output 路径与逐行矩阵见该记录。GPU 沿用最多 2 个总 worker、至少 12 GB 空闲的保守队列，计入其他 owner。新旧输出不覆盖。技术停止规则：任意两行技术失败后暂停尚未启动的源训练，已健康运行的行继续完成并评分；低分不触发停止，不自动重跑。

本候选唯一独立 P0/P1 审查发现合并前过早 ANALYZED 的问题，已修复为 MERGING_EIGHT_SEEDS → 成功 ANALYZED／失败 FAILED_COMBINE；针对性复核 PASS。7 项相关协议与回归测试通过。launcher 首步执行真实 checkpoint 的 synthetic no-query smoke，PASS 后直接启动。N607 普通账户只读 preflight 于本次任务通过；发布使用一份 release archive、一次本地/远端 SHA 对比及远端 compile。

代码/config 身份为本次 Git 提交（remote release_commit.txt），后续启动证据记录实际 OID。预期新增 52 行各有 source、prediction、scores 产物。仅在新矩阵完整评分且旧产物匹配时生成 combined_eight_seed.json/csv、combined_per_seed.csv、combined_paired.csv；包含完整 8 种子均值、样本标准差、最差 RX 和 CVS 配对差，保留负结果与退化种子。未完成时不得报告 8 种子结果。

## 实际启动核验

VERIFIED。实际执行提交为 `627449a354b0755061a6cc7f4d494752d9340d38`，本地与远端分支 OID 一致；release archive 单次传输 SHA 对比、remote compile 和真实 checkpoint no-query smoke 均通过。

独立两次读回确认调度进程 CWD/argv 属于本次 release，52 行固定矩阵不含 native。启动快照为 7 行 TRAINING、45 行 QUEUED；已观察 epoch 与 7 行日志增长，CUDA_VISIBLE_DEVICES 与实际派发 GPU 一致。健康既有任务保留。完整证据保存在 `local_artifacts/standard_ce_seed_extension_20261012_r01/`，大体积及私有 runtime 证据不提交 Git。

启动时尚无新增 clean 测试结果；以下完整 clean 结果段落记录最终状态。队列会继续逐行 E200 → freeze → clean prediction → 独立 scoring。全部新行评分后自动合并原 4 种子，合并失败会明确记录 FAILED_COMBINE；恢复先检查原进程及产物，不重复发布。

## 完整 clean 结果与交付（2026-10-12）

VERIFIED。2026-10-12 14:32:38 HKT 只读快照确认新增 52/52 行均为 SCORED；原批剔除原 CVS 后为 52 行，合计 104 行。13 种网络各有原 4 个和新增 4 个模型种子（2026092701 至 2026092708），缺项为 0。每行从零训练 E200/10000 次更新，固定末轮权重，再完成 168000 个 clean 包的预测与独立 truth-last 评分。52 行实际 resolved config 与对应旧配置仅允许的种子及输出/运行环境字段不同，无训练配方差异。

远端 dispatcher 保留 FAILED_COMBINE，原因是汇总脚本对 score 中已有的 variant/model_seed 重复传参。全部预测与评分已完成，该错误未影响行级结果。本地修复字段复制并加入身份匹配检查；4 项相关回归检查通过。仅用已评分产物回放修复后的完整合并，得到 182 个网络/分层组；与重新计算的 13 网络总体表一致。所有 1456 个分层评分从混淆矩阵复算一致，RX/TX 分层矩阵之和与 overall 一致。最终分析在本地完成，不修改远端历史状态，不重跑训练、预测或评分。

CVS 指 residual_fusion。新交付的总体/分层/配对表、图和 Excel 均移除原 CVS，历史 run 产物保留。下表是 8 个模型种子的均值 ± 样本标准差（ddof=1），不是置信区间，也不是独立数据划分的重复实验。模型种子控制初始化、shuffle 与架构内 dropout；split seed392005 与物理测试包固定。纯 CE 训练范围为 L_s，无额外损失、输入增强、U_s、教师或预训练。

| 网络 | 完成种子数 | 准确率（%） | Macro-F1（%） |
|---|---:|---:|---:|
| CVS | 8 | 78.96 ± 1.01 | 78.78 ± 1.18 |
| ResNet-50-1D | 8 | 74.99 ± 1.06 | 74.78 ± 0.88 |
| DenseNet-121-1D | 8 | 73.09 ± 1.74 | 72.36 ± 1.98 |
| Complex ResNet-18-1D | 8 | 72.97 ± 1.95 | 73.03 ± 1.83 |
| ResNet-18-1D | 8 | 72.74 ± 1.27 | 72.68 ± 1.34 |
| ResNet-34-1D | 8 | 72.53 ± 2.05 | 72.35 ± 2.26 |
| VGG-16-BN-1D | 8 | 70.13 ± 2.06 | 70.66 ± 1.50 |
| BiLSTM (3 layers) | 8 | 67.38 ± 1.14 | 66.96 ± 1.33 |
| ConvNeXt-Tiny-1D | 8 | 65.08 ± 0.80 | 63.98 ± 0.96 |
| ResNet-50-2D | 8 | 64.87 ± 2.15 | 64.16 ± 2.45 |
| ResNet-18-2D | 8 | 64.81 ± 2.85 | 64.52 ± 3.25 |
| Complex ResNet-18-2D | 8 | 64.50 ± 3.14 | 64.34 ± 4.03 |
| Transformer (6 layers) | 8 | 52.83 ± 22.43 | 49.77 ± 27.88 |

CVS 准确率较最高的常见网络 ResNet-50-1D 高 3.97 个百分点；本矩阵未做显著性检验。Transformer 的 2026092703 和 2026092708 均退化为单类预测（准确率 16.6667%，Macro-F1 4.7619%），两个负结果均保留。此前已暴露 clean 代理基准的背景保留，不据此声称新的独立盲测或星地鲁棒性。

最终文件：

- Excel：`E:/type10-7/outputs/01a0fd26-2621-7662-a9bd-9ff19509e812/standard_ce_clean_results.xlsx`；11 个工作表，104 行种子结果、1456 行分层指标、3744 行总体混淆矩阵记录；6 张高分辨率论文图和 2 张可编辑图表；总体公式与 CSV 核对一致，13 行 n=8、缺项=0，无公式错误。
- 论文图：`E:/type10-7/exports/paper_figures/20261012-standard-ce-clean-eight-seed-r01/`；6 类图延续总体表现、RX 热图、参数量/准确率、架构配对差、CVS 分层表现和混淆矩阵，更新至完整 8 种子；PDF/SVG/600 dpi PNG、英文图注和对应 source_data。
- 数据：`E:/type10-7/exports/paper_figures/20261012-standard-ce-clean-eight-seed-r01-source/`，所有当前表均标记实际完成种子数。
- 验证证据：`local_artifacts/standard_ce_seed_extension_20261012_r01/progress_20261012_final_r01/readback.json` 与 `final_analysis_verification.json`。实验原执行提交仍为 627449a354b0755061a6cc7f4d494752d9340d38；报告工具修复另以本次 Git 提交记录。

## 参数量图可读性修订（历史版本，已由下一节替代）

fig03 改为全范围概览＋密集区域放大：概览采用对数参数轴、0 至 100% 准确率，放大图采用线性参数轴，显示 2 至 42 M 参数、59 至 80% 准确率区域。编号用引导线连接原始数据点，完整图例移至下方三列。Transformer 完整标准差保留在概览，13 个模型与全部 8 个种子均未剔除。

VERIFIED：数值标签不重叠、文本不出画布；所有源 CSV 与其余 5 张图的 PNG 均保持不变。最新图包为 E:\type10-7\exports\paper_figures\20261012-standard-ce-clean-eight-seed-r02（PDF/SVG/600 dpi PNG）。Excel 原文件被占用，更新版另存为 E:\type10-7\outputs\01a0fd26-2621-7662-a9bd-9ff19509e812\standard_ce_clean_results_updated.xlsx。文件由 artifact tool 生成新图片与绘图尺寸，再保留原工作簿包装；仅图 3 图片及其高度改变，所有其他 XLSX 成员保持一致，包括数据、公式、格式和可编辑图表。图片与下一张图之间有足够间距；11 个工作表、6 张论文图、2 张可编辑图表、n=8 和无公式错误核验通过。

## 单图参数轴、配色与资源指标最终更新（2026-10-12）

用户明确拒绝第二幅放大图，指定在一张图内压缩 CVS 参数区域，主体横轴为 3 至 50 M 线性刻度。最终 fig03 按此实现：CVS 位于最左端，点旁注明实际参数量 164225，即 0.164 M；不是 0.01 M。压缩区在横轴明确标记，全部模型均值与完整 SD 保留。13 个模型采用固定、明显不同的颜色，参数图和资源图同时采用不同点形；跨图颜色一致。此前 r02 双图形式保留为历史产物，不再是当前交付。

新增 fig07 推理时延/吞吐率、fig08 参数/卷积矩阵计算量/状态/推理显存；数据来自已完成的 104 行资源测量，无新训练、目标预测或评分。计算量复核发现旧 profiler 对 BiLSTM 只计分类头、对 Transformer 漏计融合注意力。现从实际注册架构的权重形状和固定 64-token 输入补齐循环矩阵及注意力 projection/QK/AV：BiLSTM 为 941627392 MACs，Transformer 为 1233390592 MACs；原部分计数仍保留在资源测量 P 列，未改写历史原始测量。

卷积/矩阵 FLOPs 按 2 × MACs 计算，不包含 FFT、归一化、激活、池化、bias 与物理/逐元素操作，因此不声称完整网络 FLOPs。CVS 每包有 2 次 FFT，另列展示。时延是既有 RTX3090/FP32、IQ 2×256 的 5 次预热与 20 次 CUDA 同步前向均值；采集时是历史共享 GPU 环境，不作为独占硬件性能基准。吞吐率先按每个种子的 128000/B128毫秒计算，再求 8 个种子的均值与样本 SD。CUDA allocated 峰值不是进程总显存；MiB=1048576 bytes。clean 全量流程耗时还包括输入传输、CPU 回读与预测文件序列化。

| 指标（8 个种子均值） | CVS | ResNet-50-1D |
|---|---:|---:|
| clean 准确率（%） | 78.96 | 74.99 |
| 参数量（M） | 0.164225 | 15.966982 |
| 卷积/矩阵 FLOPs（M/包） | 19.418 | 399.761 |
| 单包前向时延（ms） | 4.843 | 5.452 |
| batch128 前向时延（ms） | 4.587 | 15.190 |
| batch128 吞吐率（千包/s） | 28.107 | 8.854 |
| 常驻模型状态（MiB） | 0.627 | 61.112 |
| clean 推理 allocated 峰值（MiB） | 46.169 | 102.523 |

CVS 在本矩阵中准确率最高且参数量最低，但单包时延和推理显存并非全模型最优；更快基线及所有负结果完整保留。所有判断只限本固定 clean 对比，不引申为独立盲测或真实星地部署优势。

最终交付路径：

- 图包：`E:/type10-7/exports/paper_figures/20261012-standard-ce-clean-metrics-r01/`，8 类图的 PDF/SVG/600 dpi PNG、图注、CSV、完整 PDF 和 ZIP。
- 效率汇总源：`E:/type10-7/exports/paper_figures/20261012-standard-ce-clean-efficiency-r01-source/`，104 行逐种子数据、13 行均值与 SD、计算口径及 workbook 输入。
- Excel：`E:/type10-7/outputs/01a0fd26-2621-7662-a9bd-9ff19509e812/standard_ce_clean_results_metrics.xlsx`；12 个工作表、8 张论文图、4 张可编辑图表。新效率表和原资源表的单位换算、FLOPs、吞吐率均用 Excel 公式；原有图表保留，当前表不含原 CVS。
- 验证：`local_artifacts/standard_ce_seed_extension_20261012_r01/metrics_delivery_verification.json`。

VERIFIED：104 个种子行、1456 个分层指标、3744 个混淆矩阵记录，13 行实际 n=8、缺项=0。全部总体与效率公式和 CSV 核对一致，无公式错误。51042 个已有数据、公式和格式单元格保持一致（论文图页、说明及新增资源字段属于明确修改范围）；8 张嵌入图片与最终 PNG 一致且不重叠。图表数值文本不出画布，fig03 编号不重叠，完整 8 页 PDF 和 ZIP 已核验。
