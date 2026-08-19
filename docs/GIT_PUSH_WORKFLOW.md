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

## 分支职责

| 分支模式 | 用途 | 规则 |
| --- | --- | --- |
| `main` | 稳定发布基线 | 不改写历史；合并前完成必要验证，提交后自动推送。 |
| `codex/<topic>-<YYYYMMDD>` | 一个对话或一个独立任务 | 默认工作分支；每个提交自动推送到同名远端分支。 |
| `release/<YYYYMMDD>` | CVS-only快照或阶段发布 | 只提交可公开发布内容；提交后自动推送，发布说明与代码同分支。 |
| `hotfix/<topic>-<YYYYMMDD>` | 紧急修复 | 只处理单一故障；禁止强推，修复后合并回需要的长期分支。 |

当前管理分支：`codex/git-auto-push-20260819`。它只承载自动推送规则、安装器、回归测试和管理文档，不承载实验结果或运行产物。

## 提交纪律

- 修改前后都检查`git status -sb`；有并发或用户未拥有的改动时，只显式stage本次文件。
- 不在脏工作树中使用`git add -A`；快照脚本只能stage自己生成的发布路径。
- 每个任务使用独立分支和清晰提交；不要把多个实验、报告或临时文件混在一个管理提交中。
- `E:\type10-7`中的实验运行产物、checkpoint、数据集和本地临时目录不进入发布仓库。
- 远端分支发生非快进冲突时停止自动推送并报告，保留本地提交，等待明确的合并/变基决定；不通过`--force`掩盖冲突。

## 发布脚本

`scripts/sync_cvs_release_snapshot.py --commit --push`仍支持显式快照发布，并在没有上游时使用`--set-upstream origin HEAD`。日常代码提交由post-commit钩子自动推送；两条路径都必须以远端分支复核为闭环。
