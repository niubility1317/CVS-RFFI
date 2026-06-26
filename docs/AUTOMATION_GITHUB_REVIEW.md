# 十二小时GitHub整理与AI审查自动化

本仓库通过Codex定时任务每十二小时整理一次`E:\type10-7`中的CVS项目状态，并推送到GitHub远端`origin`。自动化目标是让网页端GPT或其他审查模型读取同一份GitHub证据，避免因为本地文件缺失、指标散落或报告未上传而误判。

## 自动化链路

1. 读取`E:\type10-7\AGENTS.md`和`E:\type10-7\项目.md`，以本地协议为准。
2. 执行`scripts/run_cvs_snapshot_cycle.ps1`。
3. 将核心代码、论文复现扩展、测试、工具、launcher、协议控制文件和最近实验证据同步到发布仓库。
4. 写入`experiment_records/CV-SincNet/LATEST_SNAPSHOT.md`、`metrics_inventory.csv`和`snapshot_manifest_latest.json`。
5. 生成`docs/analysis_requests/latest_chatgpt_pro_prompt.md`，作为网页端ChatGPT Pro GPT的审查输入。
6. Codex定时任务基于同一快照写`docs/ai_review/<timestamp>/codex_review.md`并提交推送。
7. 如果配置了网页端GPT URL且运行环境可用，才尝试网页端调用；若被登录、UI或运行时阻断，写`WEB_GPT_UI_RUNTIME_BLOCKED`，不得伪造网页GPT输出。

## 上传范围

会上传：

- `code/`、`baselines/`、`paper_reproduction/`、`tests/`。
- `tools/`中允许公开的`.py`、`.md`、`.ps1`。
- `scripts/launchers/run_*.sh`。
- `docs/source_controls/AGENTS.full.md`和`docs/source_controls/PROJECT_PROTOCOL.full.md`。
- 最近若干`automation_reports/CV-SincNet/*/report.md`以及小体积metrics、score table、manifest、matrix、validation、summary和separability文件。

不会上传：

- WiSig/ManySig数据、`.pkl`、`.npy`、`.npz`。
- 模型权重、checkpoint、`.pth`、`.pt`、`.ckpt`。
- 本地密钥、SSH配置、`.env`、`.local/`。
- 原始大日志、N607远端完整备份、PPT、DOCX、PDF和压缩包。

`optimizer_execution_registry.jsonl`默认只上传tail和fingerprint，避免十二小时提交让Git仓库持续膨胀。

## 网页端ChatGPT Pro边界

网页端ChatGPT Pro没有稳定的无人值守API。自动化不会把网页端调用失败包装成成功。可靠做法是：

- GitHub快照和审查提示始终生成并推送。
- 如果需要人工网页审查，打开`docs/analysis_requests/latest_chatgpt_pro_prompt.md`并让网页GPT读取当前GitHub仓库。
- 如果未来要让Codex尝试网页端调用，在本机非Git目录或环境变量中配置GPT URL，例如设置`CVS_CHATGPT_PRO_GPT_URL`，并保持浏览器已登录。失败时必须写阻断记录。

## 手动运行

```powershell
cd E:\type10-7\github_publish\CVS-RFFI-repo
powershell -ExecutionPolicy Bypass -File scripts\run_cvs_snapshot_cycle.ps1
```

只验证不提交：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cvs_snapshot_cycle.ps1 -NoCommit -NoPush
```

## 审查输出要求

AI审查必须至少包含：

- 证据边界。
- 当前主要成果。
- 主矛盾和次要矛盾。
- 必须解决的问题。
- 文件级修改建议。
- 下一轮实验矩阵建议。
- 不能写入论文或报告的声明。

指标解释必须绑定同一candidate/run row，不能拼接不同row的单项最优值。Stage2-A/B不能声明seen-new identity accuracy，clean view不能作为deployment success。
