# CVS-RFFIGit推送与分支管理

本仓库是`E:\type10-7`项目的Git承载面。`E:\type10-7`根目录不是Git仓库；需要版本化的代码、脚本、配置和文档应进入`github_publish/CVS-RFFI-repo`或其工作树。

## 默认推送行为

仓库安装`post-commit`钩子后，每次正常`git commit`都会自动执行：

1. 读取当前分支和`origin`远端。
2. 没有上游时执行`git push --set-upstream origin HEAD`；已有上游时推送当前分支。
3. 读取tracking ref和GitHub远端分支，确认两者都等于本地`HEAD`。
4. 推送或远端复核失败时保留本地提交，并明确报告`FAILED`或`UNKNOWN`，不使用强制推送。

首次克隆、重新创建工作树或钩子被清理后，在仓库根目录执行：

```bash
bash scripts/install_auto_push_hook.sh
```

钩子默认启用。仅用于隔离的本地测试时，才允许临时设置`CVS_AUTO_PUSH_DISABLE=1`；正常项目提交不得关闭自动推送。

## 分支生命周期

长期可见分支只保留以下四类：

| 分支模式 | 用途 | 规则 |
| --- | --- | --- |
| `main` | 稳定发布基线 | 不改写历史；合并前完成必要验证，提交后自动推送。 |
| `work/cvs-active` | 当前唯一主开发线 | 汇聚当前CVS开发工作；不为每个实验长期创建分支。 |
| `release/cvs-latest` | 最近一次可发布快照 | 只提交可公开发布内容，发布说明与代码同分支。 |
| `ops/git-auto-push` | Git与自动化治理 | 只承载推送钩子、安装器、回归测试和管理文档。 |

临时任务可以使用`task/<topic>`或`hotfix/<topic>`，但任务结束后必须归档。实验候选、运行结果和阶段报告由提交、报告和标签共同表达，不把每个实验永久保留为远端分支。

归档流程：

1. 确认分支没有工作树占用、没有未推送提交和未提交改动。
2. 在分支尖端创建`archive/YYYY-MM-DD/<topic>`标签并推送标签。
3. 独立读取GitHub标签OID与本地标签OID，确认归档已落地。
4. 使用普通远端删除，不使用强制推送；本地分支是否删除取决于是否仍被工作树占用。

带工作树的分支不自动删除。项目外部创建的工作树不由日常分支整理强制移除；只有确认其中没有未提交或未跟踪内容后，才可以单独执行清理。

本次规范化映射为：`codex/full-ablation-20260728`→`work/cvs-active`、`codex/git-auto-push-20260819`→`ops/git-auto-push`、`codex/cvs-rffi-release-20260626`→`release/cvs-latest`。旧名称的提交尖端保留为归档标签。

当前治理分支：`ops/git-auto-push`。

## 提交纪律

- 修改前后都检查`git status -sb`；有并发或用户未拥有的改动时，只显式stage本次文件。
- 不在脏工作树中使用`git add -A`；快照脚本只能stage自己生成的发布路径。
- 每个任务使用独立分支和清晰提交；不要把多个实验、报告或临时文件混在一个管理提交中。
- 本地分支列表按最近提交时间倒序显示，远端获取开启自动prune；分支清理只处理已归档且无工作树占用的分支。
- `E:\type10-7`中的实验运行产物、checkpoint、数据集和本地临时目录不进入发布仓库。
- 远端分支发生非快进冲突时停止自动推送并报告，保留本地提交，等待明确的合并/变基决定；不通过`--force`掩盖冲突。

## 发布脚本

`scripts/sync_cvs_release_snapshot.py --commit --push`仍支持显式快照发布，并在没有上游时使用`--set-upstream origin HEAD`。日常代码提交由post-commit钩子自动推送；两条路径都必须以远端分支复核为闭环。
