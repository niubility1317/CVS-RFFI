# FastTrust-RC4 LogitQ E200实验报告

## 当前状态

- run_id：phase1_adv3b02_fasttrust_rc4_logitq_e200_s392002_20260823
- 状态：LOCAL_VERIFIED
- 目标：修复RC4非有限损失，降低低质量P/N伪监督的有效权重占比，分离H/P/N可靠度语义并提高U_s有效利用率。
- 正式训练预算：200epoch；seed=392002；U batch=256。
- 初始化与数据：同ADV3B02 Core90 checkpoint、同Phase1 split、同训练步数；U_s TX真值隐藏。
- 不变项：保留原始Core90 LEO_WEAK调度与拼接增强，不把星地增强作为本轮主体。

## 预登记矩阵

| GPU | candidate | 机制 |
|---:|---|---|
| 0 | E200_P0_NO_U_ID | 无U身份控制 |
| 1 | E200_P3_DUAL_H | 严格H控制 |
| 2 | E200_P4_H_PSET_B10 | H+P集合质量，P预算0.10 |
| 3 | E200_P5_H_PSETCOND_B10 | H+P集合质量+候选条件蒸馏，P预算0.10 |
| 4 | E200_P6_H_PSETCOND_N_B10_CAP | H+完整P+N集合质量+class×receiver cap |
| 5 | E200_P7_P6_HSAT | P6+严格H星地强视图 |

## 技术停止规则

仅在协议/安全违规、输出碰撞、错误checkout、无法产生完整prediction、launcher级故障或至少两行出现相同确定性训练前异常时停止该run拥有的精确进程树。不得因中间性能低而停止，不得干预其他run。

## 预期产物

每行必须保留status.txt、完整metrics_epoch.jsonl、完整train.log、final_ssdg.pth、clean评测及leo_clear_weak、leo_low_elev_weak、leo_rain_weak三个独立场景指标与日志。训练结束不等于实验完成。

## 设计追踪

详细追踪表见docs/superpowers/plans/2026-08-23-fasttrust-rc4-logit-quality-e200.md。9项已本地验证，矩阵项已本地验证并等待远端编译。

## 本地验证

- RED：新增测试首先因缺少rc4_tail_transition_scale在收集阶段失败，exit=2，确认测试先于生产实现。
- GREEN：RC4聚焦测试12项通过，覆盖float32、float16、bfloat16饱和logit、singleton集合等价CE、全类集合零损失、空候选fail-closed、空路由graph-safe和P有效权重预算。
- 最终回归：MUSE训练集成、Core90调度/星地增强、FastTrust协议与加速相关共92项通过；Python编译、两个shell语法、JSON结构和diff检查通过。
- 真实Core90 checkpoint无query smoke：严格重建成功，missing=0、unexpected=0；真实模型前向后H/P/N集合损失0.02057417，有限梯度63组，query_access=false。
- launcher：Python编译、两个shell语法和6行E200矩阵dry-run通过；逐行读回P0/P3/P4/P5/P6/P7与GPU0–5。

## P0/P1正确性审查

审查范围只覆盖会使下一次真实实验跑错、越权、覆盖输出、误杀进程、不能启动或不能产生合法prediction的问题。未发现P0/P1：U_s标签未进入训练；Core90原调度未改；run root拒绝覆盖；launcher每GPU槽位上限为2；训练结束后仍要求clean和三个LEO场景闭合。V_select盲评分器与0.05/0.15预算敏感性属于NONBLOCKING后续诊断，未扩展本次矩阵。
