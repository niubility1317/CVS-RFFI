# ChatGPT Pro网页GPT审查提示

快照时间：`20260626_132424`

你正在审查CVS-RFFI/CV-SincNet项目。请先读取GitHub仓库中的以下文件，不要只看README：

1. `README.md`
2. `docs/source_controls/AGENTS.full.md`
3. `docs/source_controls/PROJECT_PROTOCOL.full.md`
4. `docs/PROJECT_PROTOCOL.md`
5. `experiment_records/CV-SincNet/LATEST_SNAPSHOT.md`
6. `experiment_records/CV-SincNet/metrics_inventory.csv`
7. `experiment_records/CV-SincNet/current/stage2_optimizer_state.json`
8. `experiment_records/CV-SincNet/current/current_state_view_latest_for_automation.json`（如果存在）
9. `experiment_records/CV-SincNet/latest/`下最新报告的`report.md`、`metrics.json`、`score_table.csv`和`manifest.json`。

请用中文输出，并严格遵守这些边界：

- 区分startup PASS、landed submit、artifact-complete、runner completion、negative diagnostic和deployment success。
- 不要把Stage2-A/B的unknown rejection写成seen-new identity accuracy。
- 不要把clean view成功写成satellite/LEO deployment success。
- 不要用孤立最大值/最小值拼结论；指标必须来自同一candidate/run row，或明确标为marginal statistics。
- 如果仓库缺少某个文件或指标，写成缺口，不要猜测。

输出结构必须包含：

## 证据边界
## 当前主要成果
## 主矛盾
## 次要矛盾
## 必须解决的问题
## 修改建议
## 文件级落地建议
## 下一轮实验矩阵建议
## 不能写入论文/报告的声明

修改建议必须落到具体文件或模块，例如`code/cvsrffi/spaceborne_fewshot.py`、`tools/spaceborne_fewshot_da_matrix.py`、`tools/optimizer_validate_matrix.py`、`paper_reproduction/cvs_aligned/`、`docs/PROJECT_PROTOCOL.md`或具体报告路径。

审查完成后，把结果保存为`docs/ai_review/<timestamp>/chatgpt_pro_review.md`，或将正文交给Codex写入该路径并提交。
