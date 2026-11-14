# Clean MoPC 故障并发支持诊断

已知 `0061-hybrid-protonet_cda` 在 K10/N20 注册时真实出现非有限；先前三张空闲 GPU 上串行同 B 支持诊断 30 次未复现。正式故障发生于多进程并发，现使用失败任务原 saved B-K10、原 clean 支持与原 C seed，公开仓库公式、无裁剪、无逐层 hook，最多 21 个独立诊断进程，每进程重复 3 次。结果只用于排查同 B 同支持的执行稳定性，不读取 query、truth，不产生预测；不同 GPU 与服务器负载仍不能等同原故障现场。

使用 GPU 5、6、7，每卡最多 7 个进程，启动前保留至少 6000 MiB，按在途进程预留 3000 MiB 并逐个派发；已有其他 owner 进程的 GPU 不使用。每条任务记录 JSON，首次非有限或资源/子进程错误停止派发并自然排空，保留全部记录和原任务 partial。代码入口 `python -m experiments.pamf_phase2_practical.clean_mopc_concurrent_diagnostic --runtime <r01 hybrid runtime> --job <r01 failed job> --output <project>/runs/20261115-phase2-mopc-clean-concurrent-support-m21-r01`。不得依据完成或未复现宣称正式故障已修好。

2026-11-15：实际 21 进程、每卡 7 进程均完成，各重复 3 次，共 63 次原公式、同 B、同支持集检查未出现非有限。调度器结束、无残留、无 query/truth/预测，不能据此证明根因消除。代码 `c220a804c38cdac4dd68855228163c6994b75f18`，单归档 SHA 及远端编译通过。
