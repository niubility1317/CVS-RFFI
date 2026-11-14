# Orthogonal 修复版源权重

从头训练 4 个模型种子 2026092701–04，使用 `group1_source_orthogonal_tempered_PL_v2`，每种子 200 epoch，源 L/U/V，复用源物理 ID、轮次与信道配方均一致的 200 轮源增强缓存；不读取目标支持、query 或真值。N607 GPU 1 单 worker 串行，独立输出 `runs/20261113-phase2-orthogonal-fixed-source-m4-r01`。使用 `releases/20261113-phase2-clean-preparation-r04` 的 commit 743a48997460945f8a15c156a06afa2e85052e00 和 `preparations/20261113-phase2-clean-pilot-r03/orthogonal-source.json`。

命令为 `python -m experiments.pamf_phase2_practical.clean_orthogonal_source train --config <config> --seed <seed>`，四次独立运行。每次进入训练前检查独占资源和输入，训练器从实际随机初始化 checkpoint 执行源样本 smoke；源训练失败或非有限即停止后续，不覆盖已有输出。完成后保存 source.pt、训练日志和完成标记。
