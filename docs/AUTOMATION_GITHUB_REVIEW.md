# CVS-only GitHub整理自动化

本仓库通过整理脚本从`E:\type10-7`同步CVS相关发布文件，并推送到GitHub远端`origin`。自动化目标是维护一个干净、可审计、只包含CVS相关源码和资料的发布仓库；本地实验记录、运行产物、AI审查草稿和网页GPT提示不属于GitHub上传范围。

## 自动化链路

1. 读取`E:\type10-7\AGENTS.md`和`E:\type10-7\项目.md`，以本地协议为准。
2. 首次使用或工作树恢复后执行`scripts/install_auto_push_hook.sh`。
3. 执行`scripts/run_cvs_snapshot_cycle.ps1`或等价的Python入口。
4. 将核心代码、CVS对照baseline源码、论文复现CVS扩展、测试、工具、launcher和协议控制文件同步到发布仓库。
5. 写入`docs/RELEASE_SNAPSHOT.md`、`docs/release_manifest_<timestamp>.json`和`docs/release_manifest_latest.json`。
6. 提交后由`post-commit`钩子自动推送并复核远端分支；显式`--push`仍会执行同样的上游设置。
7. 若后续需要AI审查，审查结果应进入PR/Issue评论或本地报告，不写入本仓库的发布文件树。

## 上传范围

会上传：

- `code/`、`baselines/`、`paper_reproduction/`、`tests/`中与CVS协议、CVS对照baseline或CVS复现扩展有关的源码和测试。
- `tools/`中允许公开的`.py`、`.md`、`.ps1`。
- `scripts/launchers/run_*.sh`。
- `docs/source_controls/AGENTS.full.md`和`docs/source_controls/PROJECT_PROTOCOL.full.md`。
- `docs/PROJECT_PROTOCOL.md`、`docs/GROUND_TRAINING.md`、`docs/DEPLOYMENT_PHASES.md`、`docs/PUBLISH_SCOPE.md`、`docs/RELEASE_SNAPSHOT.md`和发布manifest。

不会上传：

- WiSig/ManySig数据、`.pkl`、`.npy`、`.npz`。
- 模型权重、checkpoint、`.pth`、`.pt`、`.ckpt`。
- 本地密钥、SSH配置、`.env`、`.local/`。
- 原始大日志、N607远端完整备份、PPT、DOCX、PDF和压缩包。
- `experiment_records/`、`automation_reports/`、服务器日志、远端备份、本地snapshot。
- `docs/source_notes/`、`docs/source_workspace_docs/`、`docs/analysis_requests/`、`docs/ai_review/`。
- `baselines/baseline_runs/`和其他baseline历史运行产物。
- paper-only复现队列、paper parity测试、Fedbase paper训练器、`paper_resnet/`、`code/SYNC_MANIFEST*`和非CVS paper reproduction配置。

自动化不再上传`optimizer_execution_registry.jsonl`的tail、最近实验报告副本或网页端ChatGPT Pro提示。需要复盘实验时，以本地`E:\type10-7\automation_reports\CV-SincNet\...`和N607实际输出为准。

## 手动运行

推荐使用Git Bash入口，避免在Windows命令行中丢失UTF-8或上游分支信息：

```bash
cd /e/type10-7/github_publish/CVS-RFFI-repo
bash scripts/install_auto_push_hook.sh
conda run --no-capture-output -n ssr-gpu python scripts/sync_cvs_release_snapshot.py --commit --push
```

```powershell
cd E:\type10-7\github_publish\CVS-RFFI-repo
powershell -ExecutionPolicy Bypass -File scripts\run_cvs_snapshot_cycle.ps1
```

只验证不提交：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cvs_snapshot_cycle.ps1 -NoCommit -NoPush
```

## 审查输出边界

若针对本仓库做AI或人工审查，结论必须遵守以下边界：

- 审查只能引用GitHub发布仓库中存在的文件；实验结果解释应回到本地报告或N607产物核验。
- 指标解释必须绑定同一candidate/run row，不能拼接不同row的单项最优值。
- Stage2-A/B不能声明seen-new identity accuracy。
- clean view只能作为control/reference，不能作为deployment success。
- 不能因为发布manifest存在就声称实验已完成、部署已成功或ChatGPT Pro已审查。
