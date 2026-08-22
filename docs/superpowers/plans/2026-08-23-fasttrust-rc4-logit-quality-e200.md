# FastTrust-RC4 LogitQ E200实施计划

## 目标与边界

本轮仅优化Phase1中U_s伪标签的有效利用，保留ADV3B02 Core90既有LEO_WEAK拼接增强和200epoch原始调度。U_s的TX真值始终隐藏；V_cal只用于源域校准，V_select只用于源域选择，不读取target truth，不根据target结果回调候选。

正式预算固定为200epoch、seed=392002、U batch=256、同split、同Core90初始化和同训练步数。ChatGPT会话与提交c111899b的分析只作为设计资料；协议和执行规则以当前AGENTS.md和项目.md为准。

## 设计追踪

| ID | 设计要求 | 代码位置 | 验证方式 | 状态 | 说明 |
|---|---|---|---|---|---|
| RC4-01 | P/N集合损失改为纯logit空间，消除float32/AMP的非有限损失 | code/cvsrffi/muse_ssdg.py::rc4_identity_losses | 饱和logit、fp16/bf16、空路由、反向梯度测试 | VERIFIED | set mass=logsumexp(all)-logsumexp(set) |
| RC4-02 | P拆分为集合质量与候选集条件蒸馏，支持单因素关闭 | code/cvsrffi/muse_ssdg.py；code/SSDG/train_ssdg.py | singleton=CE、全类集合损失=0、P-set/P-cond开关测试 | VERIFIED | P3→P4→P5同row分解 |
| RC4-03 | P/N不再沿用top1正确率分数 | RC4Calibration、RC4Route与build/route_fasttrust_rc4 | 校准包字段、路由分数范围与分流测试 | VERIFIED | 分别命名p_correct、p_set_safe、p_exclusion_safe |
| RC4-04 | P使用有效加权质量预算，阻止约99%原始P/N路由 | route_fasttrust_rc4；训练CLI | P权重质量不超过0.10×B_U的测试与遥测 | VERIFIED | 预算是权重质量上限，不是固定样本配额 |
| RC4-05 | H/P/N均使用class×receiver有效权重平衡 | route_fasttrust_rc4 | 不均衡fixture与权重有限性测试 | VERIFIED | 不扩充身份样本 |
| RC4-06 | 交叉拟合按source receiver/domain分组 | build_rc4_calibration | 分组折字段和校准回归测试 | VERIFIED | 避免同接收机泄漏到held fold |
| RC4-07 | E91后保持Core90 p=0.8不变，但对RC4 all-U domain/adv分支平滑重启 | train_ssdg.py | E90/E91/中点/终点调度单测与日志字段 | VERIFIED | 只稳定优化器冲击，不改星地增强主体 |
| RC4-08 | 记录各状态有效权重coverage、三种概率及有限性 | train_ssdg.py遥测 | 训练字段回归和无query smoke | VERIFIED | 用于E200尾段诊断 |
| RC4-09 | 保留Core90 LEO_WEAK原始200epoch调度 | adv3b02_core90_u_satellite_policy与launcher | 既有调度测试、dry-run命令核对 | VERIFIED | 不做50/100epoch边界重映射 |
| RC4-10 | 发布最小同row E200矩阵 | 新config与launcher | schema测试、dry-run、远端编译 | LOCAL_VERIFIED | 6行：P0、P3、P4、P5、P6、P7；待远端编译 |

## 最小E200矩阵

| GPU | 候选 | H | P-set | P-cond | N | class×receiver cap | H satellite |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | E200_P0_NO_U_ID | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | E200_P3_DUAL_H | 1 | 0 | 0 | 0 | 1 | 0 |
| 2 | E200_P4_H_PSET_B10 | 1 | 1 | 0 | 0 | 1 | 0 |
| 3 | E200_P5_H_PSETCOND_B10 | 1 | 1 | 1 | 0 | 1 | 0 |
| 4 | E200_P6_H_PSETCOND_N_B10_CAP | 1 | 1 | 1 | 1 | 1 | 0 |
| 5 | E200_P7_P6_HSAT | 1 | 1 | 1 | 1 | 1 | 1 |

P有效权重预算固定为0.10。0.05/0.15属于后续低成本预算敏感性检查，不扩入本次用户指定的正式E200最小矩阵。

## 验证顺序

1. 先添加聚焦行为测试并观察RED。
2. 最小生产实现后观察GREEN。
3. 运行相邻FastTrust/Core90回归、静态编译和launcher dry-run。
4. 使用真实Core90 checkpoint执行无query smoke。
5. 完成一次仅针对会导致真实实验跑错、越权、覆盖输出、误杀进程、不能启动或不能产生合法prediction的P0/P1审查。
6. 只stage本轮代码、测试、config、launcher、计划和报告；提交、push并核对远端OID。
7. N607只读preflight后制作一次release归档，只做一次本地/远端SHA比较，远端编译后启动；启动后只做一次PID/CWD/cmdline/GPU/log增长读回。

## 有意不扩展的设计项

V_select-as-U独立盲评分器和0.05/0.15预算敏感性行属于后续source-only诊断，不是本次E200启动前gate。当前实现的P/N事件语义已经分离；由于排除集定义为候选集补集，P containment与N false-exclusion在本版具有相同真值事件，但保留独立校准参数和遥测，便于后续改变排除集构造。
