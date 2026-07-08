# 训练日志隔离整理报告

时间：2026-07-08 10:36 Asia/Hong_Kong  
操作者：Codex  
目标：将CVS训练日志与复现/对比方法日志分区存放，并约束后续复现/对比实验写入专门日志目录。

## 读取的控制文件

- `E:\type10-7\AGENTS.md`
- `E:\type10-7\项目.md`
- `E:\codex\home\skills\cv-sincnet-n607-automation\SKILL.md`

## 本地整理结果

| 项目 | 结果 |
|---|---:|
| 本地dry-run计划移动 | 109 |
| 本地实际移动 | 109 |
| CVS日志 | 71 |
| 复现/对比日志 | 38 |

本地CVS日志根：`E:\type10-7\logs\cvs\`。  
本地复现/对比日志根：`E:\type10-7\paper_reproduction\logs\`。  
本地迁移清单：`E:\type10-7\analysis\training_log_reorg_20260708\executed_manifest.csv`。

## N607整理结果

N607预检通过：直连`N607`可用，项目根`/home/szu2070436088/2510044040/CV-SincNet`可见，GPU可见。整理前发现活跃训练run：`phase1_dgleo_v2full32_main8_20260708`，因此跳过其远端日志目录，不移动当前运行输出。

| 项目 | 结果 |
|---|---:|
| 远端dry-run计划移动 | 625 |
| 远端实际移动 | 625 |
| CVS日志 | 568 |
| 复现/对比日志 | 57 |

远端CVS日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/cvs/current/`。  
远端复现/对比日志根：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/current/`。  
远端执行清单：`/home/szu2070436088/2510044040/CV-SincNet/logs/_organization_manifests/20260708_log_reorg_executed.csv`。  
本地备份清单：`E:\type10-7\analysis\training_log_reorg_20260708\remote_executed_manifest.csv`。

远端整理后`logs/`顶层只保留`cvs`、`old_logs`、`_organization_manifests`、当前活跃run日志目录，以及两个异常目录名（空白、`;`）。异常目录未移动，避免把非训练日志误归档。

## 代码和文档改动

| 文件 | 用途 |
|---|---|
| `tools/reorganize_training_logs.py` | 新增本地日志迁移工具，按CVS与复现/对比分桶并写manifest |
| `tools/training_log_organizer.py` | 将`logs/cvs`与`paper_reproduction/logs`加入默认扫描根，修复`paper_reproduction`误过滤 |
| `tests/test_training_log_organizer.py` | 增加新根和`paper_reproduction/logs`发现测试 |
| `code/scripts/launch_cvs_safd_vs_existing_riei_drift_20260611_011500.sh` | 将复现/对比launcher默认`LOG_ROOT`改到`paper_reproduction/logs`，`RUN_ROOT`改到`paper_reproduction/runs` |
| `paper_reproduction/README.md` | 记录复现/对比日志隔离规则 |

## 验证

| 命令 | 结果 |
|---|---|
| `conda run --no-capture-output -n ssr-gpu python -m pytest tests/test_training_log_organizer.py` | 根目录与Git承载面均通过，10 passed |
| `python tools/training_log_organizer.py --project-root . --out-dir analysis/training_log_catalog_post_reorg` | 重新编目成功，3968条记录 |
| N607 SSH/SCP后本地`ssh.exe`与到`172.31.111.215:22`、`172.31.105.18:22`连接检查 | 无残留 |

第一次`conda run`因GBK输出编码触发包装错误，设置`PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`、`CONDA_NO_PLUGINS=true`并使用`--no-capture-output`后验证通过。该错误未作为项目测试失败处理。

## 版本状态

`E:\type10-7`根目录不是Git仓库。本次代码/文档改动已镜像到Git承载面`E:\type10-7\github_publish\CVS-RFFI-repo`，日志移动本身通过manifest留痕，不作为Git内容提交。
