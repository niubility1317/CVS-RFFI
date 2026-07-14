# qKNNV42技术汇报构建说明

## 报告主线

报告采用“结论→任务边界→输入输出→方法机制→正式效果→证据审计→局限与下一步”的技术汇报结构。当前主结论只使用2026-07-13正式Stage2-C矩阵；2026-07-07单行高分仅作为legacy diagnostic对照。

## 图表选择说明

| 图表 | 回答的问题 | 选择形式 | 原因 | 数据源 |
|---|---|---|---|---|
| K-shot效果图 | K增加后old_acc、seen_new_acc与H如何变化 | 分组柱状图 | K只有5个离散档位，柱状图强调组间比较，避免暗示连续趋势或外推 | `local_artifacts/cvs_publication_stage2_summary_20260713/method_k_summary.csv` |
| 方法H排名图 | 同一paired matrix下谁最能兼顾旧类与新类 | 水平条形图 | 只有4个方法，目标是排序；水平布局便于显示长方法名 | `local_artifacts/cvs_publication_stage2_summary_20260713/per_run_results.csv` |

没有使用折线图，因为5个K点不足以支撑连续趋势外推。没有使用饼图，因为这里不存在有意义的整体构成关系。receiver、场景、paired差异和极值行以表格呈现，便于保留精确值和同一行上下文。

## 可复现构建

在仓库根目录执行：

```powershell
conda activate ssr-gpu
python docs/qknnv42_report_20260714/build_report_artifact.py
node E:/codex/home/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs --input docs/qknnv42_report_20260714/artifact.json --output docs/qknnv42_report_20260714/report.html
```

`artifact.json`是canonical report artifact；`report.html`是自包含可交付报告。构建脚本会检查当前summary输入、重新聚合正式结果，并逐目录审计125个qKNNV42运行的结构化artifact。

## 口径边界

- 当前正式矩阵：5个receiver×5个seed×5档K-shot=125个qKNNV42运行，每个运行包含3种LEO弱场景。
- 当前正式已注册类：6个old TX+2个seen-new TX。
- 历史高分行：6个old TX+20个new TX、K=5、单seed、不同切分与query口径，不与正式矩阵聚合。
- 当前summary的95%CI来自1.96×标准误正态近似，不是比较协议文字中的bootstrap CI。
- 本报告不把简化LEO物理增强表述为真实卫星链路验证或部署成功。

## 变更与验证记录

本次新增文件及用途：

- `build_report_artifact.py`：从正式summary和逐运行目录重算指标、检查artifact并生成canonical report artifact。
- `artifact.json`：报告的结构化数据、图表、表格、来源与叙事定义。
- `report.html`：由portable artifact交付脚本生成的自包含HTML技术汇报。
- `source_notes.md`：记录报告主线、图表决策、复现命令和口径边界。

验证结果：

- 在`ssr-gpu`环境运行构建脚本成功；125个qKNNV42运行、375个场景结果和1000个必需文件槽位通过逐目录结构化检查。
- portable artifact校验与HTML打包均为`passed`，识别31个block、2张chart、2组metric card和10张table。
- 打包工具因缺少兼容的Chromium headless-shell只返回`structural_only`验证；随后使用本机Chrome以1440×8000视口加载自包含HTML并人工检查整页截图，标题、正文、两张图、指标卡和表格均正常呈现，没有空白图或明显溢出遮挡。
- `python -m py_compile docs/qknnv42_report_20260714/build_report_artifact.py`和artifact一致性断言通过。

编辑前Git状态为当前分支相对远端`ahead 1026`，并已有2个不相关Markdown修改和多组未跟踪`local_artifacts`。本次只处理`docs/qknnv42_report_20260714/`，没有覆盖或清理既有工作树内容。
