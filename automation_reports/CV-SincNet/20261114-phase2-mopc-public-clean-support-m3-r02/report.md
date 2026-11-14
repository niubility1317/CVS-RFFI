# MoPC 仓库版 clean support 诊断及身份核实

继 r01 的三组 source-only/clean-support 数值检查，在独立不可变 r02 重复三组检查，并在完成事件记录实际 capsule、源 checkpoint、seed、K/N、仓库语义、阶段预算与裁剪状态，供正式 pilot 自动校验。任务、显卡、源模型和预算与 [r01](../20261114-phase2-mopc-public-clean-support-m3-r01/report.md) 一致：GPU 5、6 的 K10/N20 各 320 更新，GPU 7 的 K1/N20 为 80 更新；只读源和新类 support，不读取 query/truth、不生成预测。GPU 5–7 启动前确认无其他 owner 且空余至少 6000 MiB。loss/梯度非有限按原 guard 停止并保留日志。旧 r15 论文语义故障仍不能据此声称修复。

本报告所属提交固定运行代码，命令与 r01 相同，输出放 `runs/20261114-phase2-mopc-public-clean-support-m3-r02`。独立 pilot 仅在三份任务身份和来源均匹配且检查成功后启动。

VERIFIED：GPU 5、6、7 均通过 source-only smoke 并结束，分别完成 320、320、80 更新；记录的最大 loss 分别约 406.29、78.16、16.11，未触发非有限保护。诊断无 query、truth 和预测。独立原生 MoPC pilot 已通过身份绑定检查启动。对旧历史 r15 论文语义与实际 B→C 数值故障不作已修复结论；运行代码 commit `8c744f4535e436dc9e173456162840cc2a1f966f`。
