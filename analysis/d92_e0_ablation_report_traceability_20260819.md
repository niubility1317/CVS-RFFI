# D92 E0模块消融结果写入追踪

## 目标

把已完成的Stage2-C T1消融矩阵写入D92_METHOD_COMPLETE_REPORT_20260727.md，明确六个功能分组的实验设置、同row指标、配对效应和证据边界。不使用推测数值，不把screening结果升级为fresh confirmation。

## 证据来源

- E:/type10-7/automation_reports/CV-SincNet/cvs_full_ablation_completed_matrix_analysis_20260731/report.md
- paper/ieee_transactions_draft_20260727/experiments/CVS_FULL_ABLATION_DESIGN_PHASE1_PHASE2_20260728.md
- code/cvsrffi/full_ablation_spec.py
- code/cvsrffi/stage2_ablation_factory.py
- code/cvsrffi/stage2_ablation_executors.py
- tests/test_stage2_ablation_factory_catalog.py
- tests/test_stage2_ablation_executors.py
- tests/test_stage2_ablation_release.py

## 追踪表

|ID|要求|目标位置|状态|验证或证据|
|---|---|---|---|---|
|D92-ABL-01|说明A–F与D92 E0六个功能模块的对应关系|报告§21.4|verified|设计文档§7.3–§7.8和full_ablation_spec.py一致|
|D92-ABL-02|加入P2-FULL及P2-A0、B0、C3、D0/D1/D2、E0、F0–F3绝对结果|报告§21.5|verified|225/225场景单位闭合，数值来自已完成矩阵报告|
|D92-ABL-03|加入FULL−ablation同row配对效应与分层bootstrap区间|报告§21.6|verified|每项使用225/225配对单位；bootstrap=10000|
|D92-ABL-04|披露D组非严格单因素、F3物理别名和量化证据边界|报告§21.4、§21.7|verified|执行器映射和完成矩阵结论均明确记录|
|D92-ABL-05|若无真实结果才启动实验；若已有结果则不重复启动|报告§21.7|verified|已存在正式闭合结果；本次没有新增N607运行|
|D92-ABL-06|将FFT96-only与RF32-only登记为P2-A1/P2-A2，并保持相对P2-FULL仅改变feature_profile|full_ablation_spec.py、stage2_ablation_factory.py|implemented|工厂目录测试验证P2-A1/P2-A2的单一配置差异|
|D92-ABL-07|从完整288维D92特征按固定β_aux=4和两级归一化生成FFT-only/RF-only特征，保留无query拟合边界|stage2_ablation_executors.py|verified|feature projection单元测试及Stage2工厂/执行器/row executor聚焦回归通过|

## 结果口径

- Stage2-C screening：5个target receiver、5个K/新类切片、3个development seed、3个leo_*_weak场景。
- 共75个identity、225个场景单位；P2-F3是P2-FULL的物理别名。
- 当前结果只有3个development seed和1个new-class draw，bootstrap区间是条件于该draw的screening区间。
- fresh confirmation仍需要至少5个fresh seed、至少3个new-class draw，以及D组严格单因素拆解。
- 本次追加的P2-A1/P2-A2尚无性能结果；在真实运行完成前只能标记为已实现、待screening证据。

## 变更后的最小验证

1.检查报告UTF-8可读、无替换字符、消融臂覆盖完整且章节顺序不变。
2.检查新增表中关键数值与完成矩阵报告逐项一致。
3.检查git diff --check，只提交报告和本追踪文件，不提交工作区其他未归属artifact。

## 当前最高风险

报告新增内容本身已由已有screening证据支持；最高风险仍是把条件性screening解释成最终确认，或把D组功能对照解释为严格Fisher因果。后续若启动fresh confirmation，必须使用新的run ID并单独记录结果。
