# PAMF SCALR clean 首轮

对 clean-pilot-r03 预定的五个 capsule、模型种子 2026092701、K5/N20，运行 PAMF Tri-SIG-TIME SSL 与固定 SCALR R42，分别使用 raw/raw、DURA 恢复/恢复。DURA clean bank 已完成 554823 条，复用相同物理记录与 support/query；不复用星地预测。每个作业独立输出，源权重和恢复器源样本 smoke 通过后执行 A/B/C 查询预测。

N607 在物理 GPU 1、2、3 上调度，现有任务和其他 owner 计入限制并避让，每卡最多 5 个 worker、启动预留至少 6000 MiB。运行 `runs/20261114-phase2-clean-pamf-pilot-r01`，使用 `releases/20261114-phase2-remaining-prep-r01`，code commit 2d7cec36d74de6b445b35b6fba82ba4e4087efe2。当前预登记的全部方法尚不齐全，按既定比较屏障保留五项固定预测；待全部方法预测齐备自动独立评分，至多 4 个评分进程。失败时停止新增派发并保留输出。
