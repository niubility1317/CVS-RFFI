# E30→E50 三种子源域状态配对复现

**SOURCE_ANALYZED，VERIFIED。** 全部三个原定模型种子的 12 个既有预测快照已完成分析，96 项源汇总核对一致，保留 2178 个汇总分层和空／小样本组。新增训练、模型前向、IQ／checkpoint 读取及正式评分均为 0；CPU 墙钟 18.792 s。正式文件没有进入此分析。结果与实际图表见 [最终分析](<E:/type10-7/automation_reports/CV-SincNet/20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01/final_analysis.md>)；实际提交 `b217a2554ea89d148c4b21ab9ea556381599260f`。

以下保留启动前登记；其中 PLANNED／等待描述是历史状态，已由上面的实际终态替代。


PLANNED，未远端发布/启动。此任务只复用01/02/03各4个已保存源预测快照，0训练/NN/IQ/CP/formal访问；不会重复E50训练/测试或阻塞原owner闭环。

一个CPU2/CUDA隐藏进程，对每seed自己的NoAs30clean正确集合及同一V1800物理panel进行56已核实的full条件分层，保留2draw/pooled和空/小cell。全部3seed齐全才报告逐seed与mean/sampleSD；draw不是modelseed。状态/质量/边界固定bins和默认头scale30与56相同。

01复用P43/E50原run；02/03复用各自P49r02/E50新run，禁止误用01权重或source key。只读已有source_summary核对accuracy/F1，不从qcomplete合成缺失F1，不回退跳seed。当前02/03的As仍健康训练，先准备实现，等source汇总完成后再启动；正式分数不输入本任务，不据此挑种或改参数。

run=20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01；owner=codex/root/source-state58；CWD=PROJECT/releases/cvs_FcrossRX_source_state_budget_E50_s010203_20261018_r01；command=CVS-RFFI/python -u analysis/cvs_budget50_source_states58_20261018.py --project PROJECT --out PROJECT/runs/20261018-diagnostic-FcrossRX-source-state-budget-e50-s010203-m12-r01/result --code-commit ACTUAL_CODE_COMMIT。代码commit/真实summarykey/最终输出名/检查与一次独立限定P0/P1/一次preflight/releasearchive比较/compile及终态证据在启动前补齐。

stop：源角色/ID/shape/seed/run/sourcekey/条件/96metrics错配或技术失败时保留新输出并停止自己CPU任务；不重算IQ、不推理、不自动重跑、不动健康训练。无效果大小阈值或额外晋级门槛。最终报告所有状态正负效果、三seed波动、局限、CSV/CM/图和真实CPU wall。

实现已冻结：原56文件保持，新增wrapper与6项fixture/compile PASS。新owner summary的精确key为NOAS30-sSEED/AS30-sSEED/NOAS50-sSEED/AS50-sSEED；source status=SOURCE_ANALYZED、target_scores_read=false，映射全12快照并核对96个accuracy/F1值。缺源产物时在创建输出前返回未就绪，不跳seed或合成F1。产物名及每seed目录以experiment.json为准。
