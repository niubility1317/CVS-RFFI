# Phase2 clean 已就绪方法全量接续

对原固定的 1932 个 capsule/seed、MIX 的 16 组 K/N 及其他任务的 K5/N20，发布 6 个原生域适应、原生 LoRa、ISSL 长预算和 PAMF SCALR 共 9 个方法。首轮三个 run 的 75 个完整作业以 capsule/seed 精确清点，五对已有 pilot 的 capsule/seed 暂整体排除，避免重算 K5/N20；这五对在 MIX 的其余 K/N 待行级接续。其余 1927 对 × 9 为 17343 个作业。原生 Orthogonal、MoPC 和所有 B→C 交叉组未纳入此队列，不称全方法已发布。

N607 物理 GPU 1–3，本任务每卡最多 5，启动预留 6000 MiB。现有 Orthogonal 源训练和其他 owner 进程均视作已有进程并避让。运行根 `runs/20261114-phase2-clean-ready-full-m4-r01`，准备目录 `preparations/20261114-phase2-clean-ready-full-m4-r01`，代码以本报告所属提交为准；启动命令为 `python -m experiments.pamf_phase2_practical.clean_scheduler dispatch --config <preparation>/scheduler.json`。每个任务先做真实源 checkpoint smoke，失败停止新增派发并排空，已有完成和部分产物保留。

本队列只处理已就绪方法子集，完成标记不代表全量矩阵完成。既定全对照评分屏障保持关闭，跨历史运行的完整行索引、MoPC/Orthogonal 及 pilot 剩余点固定后才可自动评分。
