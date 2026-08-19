# D92 E0模块消融结果写入追踪

## 目标

把已完成的Stage2-C T1消融矩阵以及D92 E0的FFT96-only/RF32-only正式筛选结果写入D92_METHOD_COMPLETE_REPORT_20260727.md，明确实验设置、同row指标、参照关系和证据边界。不使用推测数值，不把screening结果升级为fresh confirmation。

## 证据来源

- E:/type10-7/automation_reports/CV-SincNet/cvs_full_ablation_completed_matrix_analysis_20260731/report.md
- paper/ieee_transactions_draft_20260727/experiments/CVS_FULL_ABLATION_DESIGN_PHASE1_PHASE2_20260728.md
- code/cvsrffi/full_ablation_spec.py
- code/cvsrffi/stage2_ablation_factory.py
- code/cvsrffi/stage2_ablation_executors.py
- tests/test_stage2_ablation_factory_catalog.py
- tests/test_stage2_ablation_executors.py
- tests/test_stage2_ablation_release.py
- E:/type10-7/automation_reports/CV-SincNet/d92_e0_fft96_rf32_ablation_screen_20260819_v1/report.md
- analysis/d92_e0_fft96_rf32_ablation_screen_20260819.md
- automation_reports/.../retrieved/runner_summary.json（本地独立取回并验证）

## 追踪表

|ID|要求|目标位置|状态|验证或证据|
|---|---|---|---|---|
|D92-ABL-01|说明A–F与D92 E0六个功能模块的对应关系|报告§21.4|verified|设计文档§7.3–§7.8和full_ablation_spec.py一致|
|D92-ABL-02|加入P2-FULL及P2-A0、B0、C3、D0/D1/D2、E0、F0–F3绝对结果|报告§21.5|verified|225/225场景单位闭合，数值来自已完成矩阵报告|
|D92-ABL-03|加入FULL−ablation同row配对效应与分层bootstrap区间|报告§21.6|verified|每项使用225/225配对单位；bootstrap=10000|
|D92-ABL-04|披露D组非严格单因素、F3物理别名和量化证据边界|报告§21.4、§21.7|verified|执行器映射和完成矩阵结论均明确记录|
|D92-ABL-05|若无真实结果才启动实验；若已有结果则不重复启动|报告§21.7、独立实验报告|verified|正式run已完成150/150 logical rows；闭合后未重复启动同一run|
|D92-ABL-06|将FFT96-only与RF32-only登记为P2-A1/P2-A2，并保持相对P2-FULL仅改变feature_profile|full_ablation_spec.py、stage2_ablation_factory.py、正式结果报告|verified|工厂目录测试验证单一配置差异；正式同row结果显示A1优于A2|
|D92-ABL-07|从完整288维D92特征按固定β_aux=4和两级归一化生成FFT-only/RF-only特征，保留无query拟合边界|stage2_ablation_executors.py、正式结果报告|verified|feature projection及Stage2聚焦回归通过；150行prediction/score闭合且query truth后接|
|D92-ABL-08|完成D92 E0 FFT96/RF32正式筛选并记录场景、K/新类、receiver和资源分层结果|正式实验报告、D92主报告§21.5.1|verified|225/225 scenario unit/arm闭合；A1−A2的H差值为+0.265428，资源和量化检查全部闭合|

## 结果口径

- Stage2-C screening：5个target receiver、5个K/新类切片、3个development seed、3个`leo_*_weak`场景；每臂75个logical identity row和225个scenario unit。
- 本次正式run为150个logical row、450个scenario unit，physical prediction、logical score、completion和status均为150/150，失败数为0。
- 当前结果只有3个development seed和1个new-class draw，仍属于screening；既有P2-FULL只作为历史描述性参照，未在本run中同步重跑。
- fresh confirmation仍需要独立确认设计，以及RF32在完整288维中的严格条件边际消融；本报告不把A1−A2升级为完整模型的单因素因果效应。
- P2-A1/P2-A2性能结果已完成：A1的H为0.565038±0.186408/0.581880，A2的H为0.299610±0.121481/0.277206；直接同row差值为+26.54pp。

## 变更后的最小验证

1.检查报告UTF-8可读、无替换字符、消融臂覆盖完整且章节顺序不变。
2.检查新增表中关键数值与N607 runner summary及正式结果报告逐项一致。
3.检查`git diff --check`，只提交报告、结果镜像、主D92报告和本追踪文件，不提交工作区其他未归属artifact。

## 当前最高风险

FFT96/RF32新增内容已由正式run的150/150闭合证据支持；最高风险仍是把条件性screening解释成最终确认，把历史P2-FULL跨run参照解释成严格配对效应，或把A1−A2解释为RF32在完整288维中的单因素因果贡献。后续若启动fresh confirmation，必须使用新的run ID并单独记录结果。
