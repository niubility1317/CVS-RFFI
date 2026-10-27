# PAMF 原版 REPL＋固定时间打乱：三视图优先，再跑单视图

状态：LOCAL_VERIFIED／待发布，无新增性能结果。

完整科学设置、数据、配对种子与边界见 `docs/PAMF_REPL_SHUFFLED_VIEWS_20261027.md`；唯一16行矩阵见 `experiment.json.matrix`。16个全新从零E50，20＋10＋20，单模型11100更新；总177600更新。原版REPL、时间排列、Main split、默认头及原生优化设置固定，仅独立签名视图数量3／1不同。

GPU0–7动态检查，跳过其他账号或归属未知的整卡；同账号其他owner健康任务保留，计入每卡总数2。先覆盖每张可用卡，再第二槽；free≥12000MiB，CPU scorer≤1。首个真实checkpoint/source backward smoke PASS 后直接训练。

三视图8模型的全部clean＋六practical、cross_rx126000／cross_both42000决策固定并独立评分、发布三视图整组报告后，才开始单视图8模型。后续自动完成匹配测试与配对统计。DURA新训练／缓存／读出均0。低分不重跑、不选种子、不调参。

源checkpoint只用作无更新诊断smoke，不进入新训练初始化；新模型只继承自身同seed／同结构／同契约的前段。原普通PL run 已用户暂停且独立读回无存活，不自动恢复。

唯一执行owner codex-pamf-repl-view-shuffle-20261027；N607普通账户，现有Phase1解释器，FP32／TF32off，CWD=PROJECT/releases/20261027-phase1-pamf-repl-shuffled-views-m16-r01-release；output=PROJECT/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r01，logs子目录。精确私有环境／路径在local_artifacts/pamf_repl_view_shuffle_20261027及实际runtime；执行命令 PYTHON -u -m experiments.pamf_repl_view_shuffle.runner dispatch --config RELEASE/runtime.json。首次协议／输入／输出／非有限／无法产生合法预测故障停新派发，健康worker保留。实际commit／启动证据在发布后追加。

## 本地验证

5项聚焦测试与编译PASS；唯一独立P0/P1审查PASS。发现新增种子参数模板可能让骨干／默认头仍使用01初始化，已令native runtime配置和args均传入实际seed，完成一次定点复核。真实checkpoint与四组源L/U反向检查为launcher第一步，通过后直接开始矩阵。
