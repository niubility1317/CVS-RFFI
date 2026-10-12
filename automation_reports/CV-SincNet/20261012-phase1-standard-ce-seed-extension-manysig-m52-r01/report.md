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

## 参数量图可读性修订

fig03 改为全范围概览＋密集区域放大：概览采用对数参数轴、0 至 100% 准确率，放大图采用线性参数轴，显示 2 至 42 M 参数、59 至 80% 准确率区域。编号用引导线连接原始数据点，完整图例移至下方三列。Transformer 完整标准差保留在概览，13 个模型与全部 8 个种子均未剔除。

VERIFIED：数值标签不重叠、文本不出画布；所有源 CSV 与其余 5 张图的 PNG 均保持不变。最新图包为 E:\type10-7\exports\paper_figures\20261012-standard-ce-clean-eight-seed-r02（PDF/SVG/600 dpi PNG）。Excel 原文件被占用，更新版另存为 E:\type10-7\outputs\01a0fd26-2621-7662-a9bd-9ff19509e812\standard_ce_clean_results_updated.xlsx。文件由 artifact tool 生成新图片与绘图尺寸，再保留原工作簿包装；仅图 3 图片及其高度改变，所有其他 XLSX 成员保持一致，包括数据、公式、格式和可编辑图表。图片与下一张图之间有足够间距；11 个工作表、6 张论文图、2 张可编辑图表、n=8 和无公式错误核验通过。
