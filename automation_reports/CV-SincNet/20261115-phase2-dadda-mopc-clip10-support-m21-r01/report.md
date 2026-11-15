# DADDA→MoPC 裁剪候选并发支持诊断

原公式关闭逐层观测时，已启动的 10 次同 B、同 clean K1/N20 支持诊断均出现真实非有限。此次单独验证全局梯度裁剪上限 10 的数值候选：沿用同 B、seed、支持、预算和 public_code 损失，裁剪会改变超过上限时的 SGD 更新，必须标为诊断候选，不能称原仓库方法或声称性能等价。每进程重复 3 次并记录触发裁剪的更新数，不读取 query/truth，不输出预测，不修改原实验。

N607 新独占 root `runs/20261115-phase2-dadda-mopc-clip10-support-m21-r01`；用 `clean_mopc_concurrent_diagnostic --runtime <affine-r01 runtime/0.json> --job <affine-r01 jobs/0086-hybrid-dadda_sda.json> --output <该 root> --k 1 --n 20 --variant clip10_uninstrumented`，其子进程使用 `clean_mopc_savedb_diagnostic`。允许 GPU5/6/7、各最多 7 个本任务进程，派发前剩余显存至少 6000 MiB、每个新 worker 保留 3000 MiB，避开其他 owner。出现任何非有限或资源异常即停止新增并排空，保留全部诊断证据。完成数值诊断不构成对真实性能无影响的证据，也不自动放行正式比较。

结果：21 个进程、63 次重复均完成，每次 80 更新；每次有 33 至 36 次更新触发裁剪，中位数 35 次。未观察非有限，但更新改变频繁，性能影响未知，不能作为仓库等价修复或直接投入正式比较。未访问 query/truth 或输出预测。
