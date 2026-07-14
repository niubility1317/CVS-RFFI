# qKNNV42技术汇报构建说明

## 报告主线

报告采用“结论→任务边界→输入输出→方法机制→正式效果→证据审计→历史差距根因→局限与下一步”的技术汇报结构。当前主结论只使用2026-07-13正式Stage2-C矩阵；2026-07-07单行高分仅作为legacy diagnostic对照。根因章节使用同一历史support/query重放、反事实消融、120-seed分布、40-seed feature bridge和当前375个场景错误流，不把oracle诊断混入正式排名。

技术受众规范映射如下：标题与技术摘要对应Required Structure 1–2；正式效果、receiver/场景差异和根因消融对应3；问题定义、输入输出和指标口径对应4；方法机制、重放与消融设计对应5；证据完整性、局限和稳健性边界对应6；建议步骤对应7；进一步问题对应8。根因章节放在证据完整性之后，是因为读者需要先确认125个正式运行artifact完整，再解释历史与正式结果为何不可比。

## 图表选择说明

| 图表 | 回答的问题 | 选择形式 | 原因 | 数据源 |
|---|---|---|---|---|
| K-shot效果图 | K增加后old_acc、seen_new_acc与H如何变化 | 分组柱状图 | K只有5个离散档位，柱状图强调组间比较，避免暗示连续趋势或外推 | `local_artifacts/cvs_publication_stage2_summary_20260713/method_k_summary.csv` |
| 方法H排名图 | 同一paired matrix下谁最能兼顾旧类与新类 | 水平条形图 | 只有4个方法，目标是排序；水平布局便于显示长方法名 | `local_artifacts/cvs_publication_stage2_summary_20260713/per_run_results.csv` |

没有使用折线图，因为5个K点不足以支撑连续趋势外推。没有使用饼图，因为这里不存在有意义的整体构成关系。receiver、场景、paired差异和极值行以表格呈现，便于保留精确值和同一行上下文。

根因章节没有新增waterfall或贡献率图。role/quota、scenario mask、FFT辅助和label propagation存在强交互，消融ΔH不可相加；waterfall会错误暗示可加因果归因。该章节使用同split消融表、错误流表、候选类缩减表和根因排序表保留精确口径。原有两张正式矩阵图保持不变。

## 可复现构建

在仓库根目录执行：

```powershell
conda activate ssr-gpu
python docs/qknnv42_report_20260714/root_cause_audit.py
python docs/qknnv42_report_20260714/build_report_artifact.py
node E:/codex/home/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599/skills/build-report/scripts/deliver_portable_artifact.mjs --input docs/qknnv42_report_20260714/artifact.json --output docs/qknnv42_report_20260714/report.html
```

`artifact.json`是canonical report artifact；`report.html`是自包含可交付报告。构建脚本会检查当前summary输入、重新聚合正式结果，并逐目录审计125个qKNNV42运行的结构化artifact。

## 口径边界

- 当前正式矩阵：5个receiver×5个seed×5档K-shot=125个qKNNV42运行，每个运行包含3种LEO弱场景。
- 当前正式已注册类：6个old TX+2个seen-new TX。
- 历史高分行：6个old TX+20个new TX、K=5、单seed、不同切分与query口径，不与正式矩阵聚合。
- 历史行使用old/new角色分区、每类等额quota、场景硬筛、60 epoch adapter、5-view TTA和96维FFT辅助；这些机制只能作为legacy诊断，不属于当前正式逐样本argmax协议。
- 根因消融保持历史support/query哈希不变，但组件存在交互，ΔH不能相加为严格贡献分解。
- feature bridge只近似对齐receiver、K、clear场景和2个new类，query样本与生成链仍不完全相同。
- 当前summary的95%CI来自1.96×标准误正态近似，不是比较协议文字中的bootstrap CI。
- 本报告不把简化LEO物理增强表述为真实卫星链路验证或部署成功。

## 变更与验证记录

本报告文件及用途：

- `build_report_artifact.py`：从正式summary和逐运行目录重算指标、检查artifact并生成canonical report artifact。
- `root_cause_audit.py`：在`ssr-gpu`中精确重放历史split，执行7组同输入消融、40-seed feature bridge并解析当前125次运行错误流。
- `root_cause_audit.json`：保存根因审计数据、输入哈希、复现检查、协议对照、消融、seed分布和局限。
- `artifact.json`：报告的结构化数据、图表、表格、来源与叙事定义。
- `report.html`：由portable artifact交付脚本生成的自包含HTML技术汇报。
- `source_notes.md`：记录报告主线、图表决策、复现命令和口径边界。

验证结果：

- 在`ssr-gpu`环境执行`root_cause_audit.py`成功；历史old/new指标、support哈希和query哈希三项重放检查全部通过，完成7组同split消融、40-seed feature bridge、120-seed分布统计及125个正式运行/375个场景行错误流解析。
- 在`ssr-gpu`环境运行构建脚本成功；125个qKNNV42运行、375个场景结果和1000个必需文件槽位通过逐目录结构化检查。
- portable artifact校验与HTML打包均为`passed`，识别46个block、2张chart、3组metric card和19张table。
- 打包工具因缺少兼容的Chromium headless-shell返回`structural_only`；canonical payload一致性、runtime根、reader根和semantic fallback结构已验证，增强reader的视口、source dialog和图表SVG浏览器检查未执行。本报告保留语义化图表数据表，不把该结果写成完整浏览器视觉认证。
- `python -m py_compile docs/qknnv42_report_20260714/root_cause_audit.py docs/qknnv42_report_20260714/build_report_artifact.py`和artifact一致性断言通过。

编辑前Git状态为当前分支相对远端`ahead 1031`，并已有2个不相关Markdown修改和多组未跟踪`local_artifacts`。本次只处理`docs/qknnv42_report_20260714/`，没有覆盖或清理既有工作树内容。
