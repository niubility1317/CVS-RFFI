# Phase 2 clean MoPC 稳定版接续

针对原队列真实非有限，使用公开仓库 MoPC 目标和原训练预算，注册时采用 FP64 全局梯度范数及阈值 1000 的梯度裁剪，显式标识 `clip1000_v1`；小于阈值时梯度不变，触发时的 SGD 更新改变，不能称与仓库原版性能等价。ProtoNet 保留 `proto_affine_v1` 桥，Tweak 保持 M1 与单条 query 判决。每个成功任务记录裁剪次数，自动评分单独选择该版本，不混入原版 C。

原生 MoPC 与六类 B→MoPC 各覆盖既定三池、接收机、四模型 seed、每任务实际 K/N，合计 13524 个任务。从 `20261114-phase2-mopc-public-clean-full-m4-r01` 和 `20261115-phase2-mopc-clean-affine-resume-m21-r01` 不可变队列继承身份与数据契约；可用时读取每 K 的实际保存 B anchor，不重新拟合 B；原生 MoPC 已固定 A0 预测直接复用。既有完整 C 是原优化版本，新版本需单独计算 C 以形成一致比较；原结果、partial 和所有祖先 claim 原样保留。旧故障只在本新版本和新 root 中处理，不覆盖旧目录。

代码与 release 以本次 commit 的归档为准；N607 命令 `clean_mopc_stable_resume --parent-config <原队列 scheduler.json> --successor-config <affine 队列 scheduler.json> --plan-root preparations/mopc-stable-20261115-r01 --run-root runs/20261115-phase2-mopc-stable-full-m35-r01 --release-root <本次 release> --score-output runs/20261115-phase2-mopc-stable-auto-score-m4-r01`，实际路径均为项目绝对路径。GPU0/1/5/6/7，每卡本任务最多 7 个训练／预测进程，派发前剩余显存至少 6000 MiB，每个 worker 预留 3000 MiB，自动避开其他 owner。数值故障记录并隔离该 MoPC 任务，余下任务继续；无效数据、程序或资源系统故障仍保护性停止。实际 PID、显卡和日志、固定预测与隔离状态需独立读回。

稳定版自动评分 watcher 与预测队列一起启动，仅在全部保留对比组及全部 K/N 预测固定并通过逐行核验后独立连接 truth，最多 4 个 CPU 进程；有隔离或缺项则等待，不把部分结果当完整比较。112/112 个真实源／B 状态 K/N 支持诊断及最终状态检查通过；原故障 B-K1 的 63 次重复通过。诊断不证明目标准确率无变化，也不保证所有后续任务永不故障。
