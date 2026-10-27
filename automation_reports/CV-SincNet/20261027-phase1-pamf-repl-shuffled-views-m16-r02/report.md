# PAMF 原版 REPL＋固定时间打乱：三视图优先，再跑单视图

状态：ANALYZED／16个模型、七景预测及独立评分共48阶段全部完成，已核实。

完整科学设置、数据、配对种子与边界见 `docs/PAMF_REPL_SHUFFLED_VIEWS_20261027.md`；唯一16行矩阵见 `experiment.json.matrix`。16个全新从零E50，20＋10＋20，单模型11100更新；总177600更新。原版REPL、时间排列、Main split、默认头及原生优化设置固定，仅独立签名视图数量3／1不同。

GPU0–7动态检查，跳过其他账号或归属未知的整卡；同账号其他owner健康任务保留，计入每卡总数2。先覆盖每张可用卡，再第二槽；free≥12000MiB，CPU scorer≤1。首个真实checkpoint/source backward smoke PASS 后直接训练。

三视图8模型的全部clean＋六practical、cross_rx126000／cross_both42000决策固定并独立评分、发布三视图整组报告后，才开始单视图8模型。后续自动完成匹配测试与配对统计。DURA新训练／缓存／读出均0。低分不重跑、不选种子、不调参。

源checkpoint只用作无更新诊断smoke，不进入新训练初始化；新模型只继承自身同seed／同结构／同契约的前段。原普通PL run 已用户暂停且独立读回无存活，不自动恢复。

唯一执行owner codex-pamf-repl-view-shuffle-20261027；N607普通账户，现有Phase1解释器，FP32／TF32off，CWD=PROJECT/releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release；output=PROJECT/runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02，logs子目录。精确私有环境／路径在local_artifacts/pamf_repl_view_shuffle_20261027及实际runtime；执行命令 PYTHON -u -m experiments.pamf_repl_view_shuffle.runner dispatch --config RELEASE/runtime.json。首次协议／输入／输出／非有限／无法产生合法预测故障停新派发，健康worker保留。实际commit／启动证据在发布后追加。

## 本地验证

5项聚焦测试与编译PASS；唯一独立P0/P1审查PASS。发现新增种子参数模板可能让骨干／默认头仍使用01初始化，已令native runtime配置和args均传入实际seed，完成一次定点复核。真实checkpoint与四组源L/U反向检查为launcher第一步，通过后直接开始矩阵。


## 技术恢复范围

r01真实source smoke在任何新训练／target访问前终止，失败记录保留。当前只删除smoke中手动置累计步数的赋值，以原本段local1配合step_offset4440。实际T1 hook本机复现并修复验证，6项聚焦测试及该问题唯一独立定点复核PASS。科学矩阵与三视图优先屏障不变；首步真实checkpoint smoke通过后自动继续。

## 实际运行 VERIFIED

代码 commit `3ff8f4b7e85b5608f5b45350aa18081499a7ef1d` 已推送并独立核对远端OID，单归档传输比较VERIFIED、远端编译PASS。真实历史同契约checkpoint无query前向，以及首／末seed各自三／单视图共4组真实源L/U完整REPL反向检查PASS，正式优化更新0、target/U真值读取0。

当前8个三视图训练worker在GPU[0, 1, 3, 4, 5, 6, 7]运行；GPU[2]因其他账号占用而排除。全部7张当前可用卡均已使用，同卡总数不超过2，其他owner修改0。8个同PID日志跨61.0秒增长，8个实际CUDA PID/card对应已独立核对；单视图启动0。

三视图全部训练／七景预测／独立评分结束后，先生成analysis_three_view，再自动启动单视图同8种子。原普通PL队列继续暂停且不自动恢复。当前无新模型最终性能成绩，不以运行或smoke代表实验完成。失败r01的检查日志及全部产物保留，具体见其failure_evidence.json；r02只修复smoke本段计数，不改变科学配方。

## 最终结果 VERIFIED

完整结果、网络逐层结构与普通PL比较见 [results_analysis.md](results_analysis.md)。三视图8seed raw clean 86.6307±1.0082%，六景70.8577±0.4990%；单视图8seed raw clean86.3488±0.7215%，六景70.3578±0.4650%。均为168000/scene，未使用DURA。

普通PL现有三视图自然顺序对照3seed clean84.9817±0.1893%，六景70.3594±0.9576%。共同三seed的本轮三视图REPL＋打乱为86.9188±0.7617%／70.5758±0.4566%，配对差+1.9371pp／+0.2164pp；缺普通PL＋同固定打乱的严格方法归因对照。原普通PL论文队列继续用户暂停，未启动新实验或调参。

## 当前方法命名与采用（2026-10-27）

用户指定本run三视图8行作为当前完整PAMF半监督方法，命名Tri-SIG-TIME SSL（三视图SIG-TIME时序重排半监督学习）。只增加有范围的展示别名，原执行矩阵、权重、单视图对照及全部结果保持原身份。方法与四类训练条件见 `docs/PAMF_METHOD_DEFINITION_20261027.md`；本次新增训练／预测／评分／DURA均0，其他实验补充待用户指定。
