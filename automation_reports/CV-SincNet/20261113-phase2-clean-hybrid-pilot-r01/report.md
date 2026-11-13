# Phase2 clean 交叉对照首轮

从 clean-pilot-r03 计划接续五个固定 capsule、种子 2026092701、K5/N20。六种域适应 B 各接 Orthogonal 和 LoRa，30 个作业共 60 个 B→C 方法组合。每次 B 的两个 C 从同一个实际 B 状态分叉。首轮原生预测已经固定或在途，未保存可载入的 B 参数，本轮按同一源权重、support 和 seed 重新拟合 B，不重算或覆盖原生预测。

N607 使用物理 GPU 1、2、3，每卡本任务最多 5 个进程，至少 6000 MiB 启动预留，避让已有进程。原生调度器已无待派发任务，在途 ISSL 自然完成；新调度器把原生 worker 识别为已有进程并避让其卡，不超过合计上限。输出 `runs/20261113-phase2-clean-hybrid-pilot-r01`，配置 `preparations/20261113-phase2-clean-hybrid-pilot-r01/scheduler.json`，代码为此报告所属提交。

发布归档使用同名 release，执行 `python -u -m experiments.pamf_phase2_practical.clean_scheduler dispatch --config <project>/preparations/20261113-phase2-clean-hybrid-pilot-r01/scheduler.json`。每个作业先执行真实源 checkpoint 无 query smoke。系统错误停止新增派发并排空，保留全部输出；只固定本子集预测，完整矩阵和评分屏障保持关闭。
