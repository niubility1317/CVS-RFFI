# DADDA目录改名记录

日期：2026-07-13

## 目标

将DADDA论文复现实现目录从`paper_reproduction/dadda_cross_receiver/`统一改名为`paper_reproduction/DADDA/`，使目录名与论文方法名称一致。

## 改动范围

- Python包导入改为`paper_reproduction.DADDA`。
- 文档、论文逐项对照矩阵、历史实施计划和单元测试中的实现路径改为`paper_reproduction/DADDA/`。
- 新运行的默认checkpoint目录改为`paper_reproduction/runs/DADDA/`。
- 保留`method_id=dadda_cross_receiver`、配置文件名和测试文件名，作为既有实验结果的稳定标识。
- 不移动既有N607历史运行目录`paper_reproduction/runs/dadda_cross_receiver/`，避免破坏已有checkpoint和报告的可追溯性。

## 验证与同步

- 本地验证：`conda activate ssr-gpu; python -m pytest tests/test_dadda_cross_receiver.py -q`。
- 本地CLI验证：`python -m paper_reproduction.DADDA.train --help`。
- N607同步目标：`/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/DADDA/`。
- N607验证：`python -m paper_reproduction.DADDA.train --help`。
