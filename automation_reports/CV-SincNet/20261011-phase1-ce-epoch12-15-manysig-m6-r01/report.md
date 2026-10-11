# CE＋旧门控＋cos：E15/E12各三种子压缩

run_id：`20261011-phase1-ce-epoch12-15-manysig-m6-r01`。状态 LOCAL_VERIFIED，尚未启动；实时状态以独立读回及 events.jsonl 为准。

## 本次配置与范围

用户明确要求发布 E15 和 E12，各三行，共六行。模型 seed 为 2026092701、2026092702、2026092703。每行从零训练，保留原身份网络内部门控及 cosine 分类头。E15 每行 15×222＝3330 次更新；E12 每行 12×222＝2664 次更新。batch128、AdamW、峰值 LR2e-4、最低 LR1e-6、weight decay1e-4；完整 cosine 日程压缩到对应最终轮，源 V 不选择轮次。

只在合法 L_s 的 clean IQ 上计算 CE，label_smoothing=0；实际关闭 U 迭代、半监督、EMA 教师、星地增强、L/T/R 与其他辅助分支。L/U/V 物理角色、原数据契约及固定 clean query 保持不变。全行 scratch，未加载旧 checkpoint；E200 仅复用固定预测作对照。

沿用本系列已明确的 clean-only 范围：每行最终轮冻结后自动预测 168000 条相同 clean query，再由独立 truth-last scorer 评分，保留 RX/TX/day 指标与 Macro-F1。三 seed 齐全后报告实际均值、SD、同 seed E200 配对差；均值下降不超过 0.3 个百分点为描述性容差，不声称独立盲测或统计非劣效。该用户授权开发基准只用于 epoch 比较，不扩展到机制设计或选模。

## 本地验证与发布

六行配置、输出唯一性、两种预算及 cosine 端点、实际 scratch 架构初始 state 一致、纯 CE 前后向和 222 次每轮迭代检查均 PASS；目标访问、resume、预算/测试视图变化负测拒绝。证据：[local_checks.json](evidence/local_checks.json)。一次独立窄范围 P0/P1 审查未发现阻断问题，证据：[independent_review.json](evidence/independent_review.json)。启动读回随后追加。

每 GPU 最多两个总 CUDA/预留进程，使用八卡剩余容量，空闲显存至少 12000MB。已有健康任务不干预。一个 release、一个 launch owner；新输出禁止覆盖；技术失败保留 partial 产物、停止 pending，健康行继续，不自动重试。

逐行配置及路径唯一来源见 [experiment.json](experiment.json)，状态变化见 [events.jsonl](events.jsonl)。远端 run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261011-phase1-ce-epoch12-15-manysig-m6-r01`；日志 `logs/<row_id>.log`。

## 结果与交接

训练、预测、评分尚未完成，暂无性能结果。本次授权仅这六行，不自动启动后续预算。恢复先核实原 run 进程和产物，不重复发布。
