# ProtoNet→MoPC 仿射桥正式接口无 query 检查

正式代码新增显式 opt-in `mopc_bridge=proto_affine_v1`，仅允许 ProtoNet B→MoPC C；原默认实现和 Orthogonal 桥保持不变。旧支持逐条判决需完全一致，中心化 logits 与 softmax 需在容差内一致；C 使用仓库 MoPC 损失及 SGD、无裁剪。C 参数化改变，输出行名带 `mopc_hr_affine_v1`；B 状态读写校验桥接版本，历史结果不覆盖、不静默合并。

在 GPU5 使用真实 source checkpoint、原 clean old/new 支持、P01/MIX03/14-7、seed2026092702、K10/N20 执行 `clean_mopc_hybrid_diagnostic --mopc-bridge proto_affine_v1`；比较真实 B 原决策和生产桥，并检查 MoPC 注册，禁止 query/truth/预测。与先前同 B 候选诊断不同，这次单独适应 B 仅作 smoke，不声称与历史 B 为同一状态。原失败任务和其他 owner 保留。
