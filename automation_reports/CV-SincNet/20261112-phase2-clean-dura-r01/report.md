# PAMF clean DURA 缓存

以冻结 DURA v1 对 clean bank 的 554823 条物理记录各恢复一次，输出 `runs/20261112-phase2-clean-dura-r01`，复用每个任务原物理记录映射。不使用身份模型、query 标签、拟合或评分。使用 `releases/20261113-phase2-clean-preparation-r04`，N607 GPU 3 单进程，数据来自 `runs/20261112-phase2-cleanpaired-data-r01`。命令 `python -m experiments.pamf_phase2_practical.clean_restore --runtime <preparations/20261113-phase2-clean-pilot-r03/runtime/pamf-00.json> --output <runs/20261112-phase2-clean-dura-r01>`。

源样本的不同 batch 一致性和恢复器参数不变在处理前后检查，发生任何异常停止并保留部分目录，只有全部成功才写完成标记。
