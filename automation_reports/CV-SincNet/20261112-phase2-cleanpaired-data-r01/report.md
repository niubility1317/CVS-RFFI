# Phase2 三池 clean 配对数据准备

该数据作业只生成归一化地面 IQ 和对应的原任务行映射，不启动模型训练、预测或评分。输入为 Full 地面 pool 和原三池 r10 的 capsule、partition、score-side 物理 ID。独立不可覆盖输出为 `runs/20261112-phase2-cleanpaired-data-r01`。逐 RX 复用一份 clean bank；每个 pool/episode/RX 建立记录位置映射，保留原 split、support 标签和 query 补集，模型侧不输出物理 ID 或 query 标签。检查 3×23×7 个 capsule 及各 20 个 K/N split、物理池、旧类 B/C support 和 K 前缀。

目标为 `/home/szu2070436088/2510044040/CV-SincNet` 下的两个 20261029 原始数据根；数据 builder 使用 Python 3、NumPy、CPU，GPU 不参与，单进程且不修改现有任务。代码以本文件所属 Git 提交为准，单独归档传输并在服务器编译后以 `python3 -m experiments.pamf_phase2_practical.clean_pairing --raw-root .../20261029-phase2-fullpool-episodes-s2026102901-r01/builder_only/pool --task-root .../20261029-phase2-threepool-episodes-s2026102901-r01 --output-root .../20261112-phase2-cleanpaired-data-r01` 执行。任何来源/ID/split/数值错误立即停止，保留部分输出，完成标记只在全部 483 capsule、9660 split 通过时写入。数据构建不授权启动实验。
