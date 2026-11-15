# DADDA→MoPC 上限 1000 的支持诊断

与上限 10 的并发诊断使用同一真实 B-K1、同一 clean K1/N20 支持、seed 和 80 更新；将全局梯度裁剪上限改为 1000，测量非有限是否消失以及每次触发次数。裁剪改变优化更新，结果只用于数值诊断，不能认定性能无影响，不能冒充仓库原版。每进程重复 3 次，共 21 个受控诊断任务；无 query、truth 或预测。

N607 root `runs/20261115-phase2-dadda-mopc-clip1000-support-m21-r01`；运行 `clean_mopc_concurrent_diagnostic`，输入 affine-r01 的 runtime/0 和 `0086-hybrid-dadda_sda` 原 job，指定 `--k 1 --n 20 --variant clip1000_uninstrumented`。物理 GPU5/6/7，每卡本任务最多 7 个进程，启动前空余至少 6000 MiB，保留每 worker 3000 MiB，并避开其他 owner。发生非有限或系统性异常停止新增、自然排空并保留诊断。禁止覆盖历史输出或读取 query/truth。

结果：21 个进程、63 次重复均完成，每次 80 更新，未观察非有限。每次有 1 至 33 次更新触发裁剪，中位数 5 次；存在干预较多的重复，性能影响未知。单一支持任务的复现不构成全矩阵修复或性能等价证据。
