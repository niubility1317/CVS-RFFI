# Orthogonal 修复版源权重接续

r01 在进入训练前被资源检查拦截，启动器与子进程均携带 GPU1 环境变量，启动器被当作占用者；r01 无源模型输出，日志和失败标记保留。r02 使用新运行目录，启动器不持有 CUDA_VISIBLE_DEVICES，仅训练子进程设 GPU1。源数据、四种子、200 epoch、缓存和数值配方同 r01。代码使用 release 20261114-phase2-remaining-prep-r01（commit 2d7cec36d74de6b445b35b6fba82ba4e4087efe2），无目标、query 或评分。逐 seed 串行，失败停止后续，独立输出各 seed、运行日志及总完成标记。
