# MoPC 仿射接续同 B 无 query smoke

使用 r01 故障任务同一次 saved B-K10、原 clean 支持和 seed，在 GPU5 从原 B 重建并验证 `proto_affine_v1`，保存/加载新桥 B，再使用仓库 MoPC 目标和 SGD 完成 K10/N20 注册。只读取合法源和支持，不读 query/truth、不生成预测；真实权重来源及支持身份沿用原 run，独占输出 `runs/20261115-phase2-mopc-affine-reuse-smoke-r01`。此检查验证新后继真实 B 复用路径，不证明整个矩阵不会出现数值故障。
