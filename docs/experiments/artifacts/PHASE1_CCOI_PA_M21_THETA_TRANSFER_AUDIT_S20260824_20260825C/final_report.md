# PA-M2.1独立theta迁移审计最终报告

## 结论

阶段A状态：`A_FAIL`；阶段B状态：`NOT_RUN_A_GATE`；最终路线：`STOP_CURRENT_PA_THETA_TRANSFER`。

本实验冻结Core90和旧C4 challenge encoder，从同一随机模板独立训练C1′与C4′。V_select按TX×RX×day×eq×capture block拆成权重选择子集、回溯审计子集和guard block；审计覆盖4个raw-disjoint fold，并只从独立support bank构造F2–F6。F7因没有已验证的跨receiver同步物理事件ID而保持`UNAVAILABLE`。

阶段B只有在`A_PASS`后才允许拟合。gate不读取true TX、receiver、day或审计标签，只使用预登记的部署时可得特征，并通过有界残差修正保护Core90。

## 证据边界

`V_audit_retro`只对本轮C1′/C4′新权重独立，不是研究历史完全未见集。当前q仍是已知受TX/RX/day/位置捷径污染的received-waveform excitation proxy；即使阶段A通过，也只能进入连续challenge重设计，不能直接晋级当前q。

全部正式JSON均为聚合结果；没有保存样本级q、theta、embedding、IQ或逐样本prediction stream。`target_or_query_access=false`。
