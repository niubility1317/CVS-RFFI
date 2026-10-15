# SIG_TIME 跨接收机梯度方向共识：最小源域诊断

状态：PREREGISTERED。代码提交 d0272fab83dea14716f335e3f8346be2d8f8d555；目标数据与分数不参与本轮设计。

核心假设：源 RX 风险的大小和方差不能直接标识共享更新方向。固定 SIG_TIME，以 AND-mask 启发的梯度符号交集筛选实际 f_theta 的 CE 梯度；比较原梯度及逐参数保留数量完全相同的随机过滤。此包只做冻结诊断，尚未实现训练更新。

矩阵：2026092701 首先执行；若源规则 FAIL，02/03 跳过。若 01 PASS，2026092702 和 2026092703 都按原计划执行，不因 02 失败而跳过 03。每模型固定 900 个合法 L 样本，5 组各 180，覆盖 6 TX × 5 RX × 3 天 × 2 样本；3 模型使用同一有序物理样本。900 次样本前向、30 次梯度调用、20 条选定层记录、45 个全量向量；0 次优化更新、0 次分类头拟合。原始向量与样本 ID 仅保存在私有运行目录。

参数边界：实际部署 joint160 的全部 195041 个 f_theta 参数，仅排除默认分类器的 960 个 class_W 参数，保留 gain、归一化及参考响应。原默认头只读；不使用 U 真值，不做 U/V/query 前向。共享数据构建器返回的 V 被丢弃。

固定源规则：5 RX 符号平均绝对值≥0.5；不缩放梯度。聚合后的最小 RX 余弦≥0，且严格优于逐参数数量匹配随机过滤；梯度范数保留率≥0.1，每组有效坐标保留率在 0.01 至 0.99；t1/t2/SIG 每组梯度非零；参数、buffer、默认 W 不变。全部 3 模型通过只支持另行设计训练实验，不能证明泛化或 84%。

父权重：各自预定种子原 SIG_TIME 对照 final50，scratch50＝33L＋17U／11100 次更新，旧门控＋ownEMA＋cos，默认头原联合训练。首模型执行提交 977d4fb609436bac23621ef8e32c2017495acef7；另两模型 a283291279cb01b939325cf5ccba1d0a671b28c4。加载前复用现有 source_proof 核对实际角色和完整 scratch 来源。没有按目标成绩选父模型。

执行：普通 N607 账户，FP32／CPU 2 线程，CUDA_VISIBLE_DEVICES 为空。CWD：PROJECT/releases/cvs_source_rx_consensus_20261015_r01。命令：python -u -m experiments.cvs_source_rx_consensus.audit。运行根：PROJECT/runs/20261015-phase1-source-rx-consensus-manysig-m3-r01，单 owner codex/root/source-rx-consensus-20261015。

发布入口：python experiments/cvs_source_rx_consensus/publish.py 的模块入口（本地通过受验证 runner 执行），一次 archive SHA 比较和远端编译，真实固定 checkpoint 无 query smoke 通过后立即诊断。读取 submit/PID/CWD/命令/CUDA/日志增长及完成工件确认状态；超时不重复发布。

停止：实际输入/来源不符、缺固定 cohort、非有限梯度、状态变化、保存复算失败或输出冲突属于技术失败，保留产物，不自动重跑、不停止其他任务。科学 FAIL 只按条件矩阵停止扩种子，不改阈值、分组、父权重或随机掩码。

预期工件：preflight、submit；各行 resolved_config、私有 actual_source_gradients、analysis、独立保存复算、completion；全局 analysis、source_qualification、completion，技术失败时 failure。独立复算只读取保存向量，不新增前向或梯度。

验证：13 项协议负例、真实网络合成输入前向和 6 次导数、完整 f_theta 边界、class_W 清零不改变 joint160、状态/RNG 不变、逐参数随机数量相同均 PASS。一次独立 P0/P1 审查 PASS；不重复未变化检查。

参考：[AND-mask 原论文](https://arxiv.org/abs/2009.00329)。本方案为源 RX 符号筛选改编，不声称论文原设置复现。

结果：源诊断完成且独立复算 VERIFIED；首模型固定源判据 FAIL，另外两模型按计划跳过。详见 final_analysis.md 和 results.json。未进入训练或目标测试。
