# MoPC 稳定版全 K/N 支持诊断

对固定 `clip1000_v1` 候选执行无 query、无 truth、无预测的实际状态诊断：原生 MoPC 源 checkpoint 1 组，以及 ProtoNet、MRIOR、DADDA、TwoStage、Poster、Tweak 的实际 B 状态各 1 组；每组 K1/5/10/20 × N2/5/10/20，共 112 个支持注册任务。ProtoNet 使用原已完成 affine B，其他 B 使用原已保存 anchor；原方法目标、seed 和训练预算不变，裁剪改变超过阈值时的优化更新，记录次数。原生为 A→C，混合方法为 B→C。

输入为 `20261115-phase2-mopc-clean-affine-resume-m21-r01` 中已完成 `0000-hybrid-protonet_cda`、`0061-native_cil-mopc_hr`、`0062-hybrid-{mrior_sda,dadda_sda,twostage,poster_ft,tweak_m1}` 的原 job/runtime 与 B 状态。独占 root `runs/20261115-phase2-mopc-stable-support-matrix-m21-r01`，配置保存在对应 preparation。使用 `clean_mopc_matrix_diagnostic` 调度，GPU0/1/5/6/7、每卡本任务最多 7、启动前空余至少 6000 MiB、每 worker 保留 3000 MiB，避开其他 owner；启动前 GPU5/6/7 已被其他 owner 占用，自动跳过。单项异常留档后继续其余诊断，显卡资源查询或进程启动等系统性故障停止新增并排空。诊断成功仅证明所测状态和支持集有限，不能证明目标性能等价或其他任务永不故障。
