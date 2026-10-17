# 固定E30五模型：仅恢复预测与评分

状态：REGISTERED／NOT LAUNCHED。10 个 Python 文件语法与 120 项本地协议负例 PASS，0 IQ／CP／NN／更新；一次独立 P0/P1 审查 PASS，N607 preflight VERIFIED。真实 CP 无查询 smoke 是发布后的第一步，通过后自动继续预测。

原训练run 20261017-phase1-F-crossRX-clock-factorial-e20plus10-s2026092701-m5-r03 保持原样：五行各从固定E20追加10轮，累计30轮／6660更新。原生收尾读取自身final_ssdg.pth时，被只允许父checkpoint的cached_parent误拒绝，返回HELDOUT_EVAL_INCOMPLETE／7；已保存的安全final10权重无需重训。原错误状态、日志和权重均保留；本run不把旧错误标成成功，只补原已授权的正式预测和独立评分。

B00/B10/B01/B11/FULL_AS五行全部保留，最终checkpoint均按源final-only固定，E30预算与父选择在新E20成绩读取前登记；不按任何目标成绩筛选、调幅度、改损失或重跑。新训练轮次0、更新0、教师0、头拟合0；保留原F＋CROSS_RX、三路SIG_TIME、统一Sinc、joint160、默认cos头、NoExplicitCFO与内部必要补偿。全5源RX、L6300/U56700隐藏TX/V27000、物理ID契约保持。

每行预测前核对原source contract与父链、实际2220个连续成功更新、10轮日志／E10进度、U无真值／无非有限跳步、optimizer实际step2220、final-role／state_epoch10／默认头及骨干已联合改变、严格自身EMA初态。只允许已保存安全final、唯一错误原因为Unexpected checkpoint dependency的收尾缺陷；其他数据／来源／训练／权重或导出失败均拒绝。验证读取现有日志／CP，不读源IQ或目标truth。

首次真实CP无查询smoke仅2个合成包／0更新；其余行自身源阶段完成即可独立预测，无需等待其他行。原clean及六practical的168000物理样本／视图保持；每行7视图与logp全部固定后exec独立CUDA为空CPU scorer才连接truth。用checkpoint内学生默认头，无重训头、校准、筛样、query拟合或全局重分配。预测共5880000正式决策，另每行2包自身smoke＋release2包；教师与训练前向均0。

报告同预算E30 clean净差、lowurban、三个suburban绝对值／自身clean差、B11−B01／FULL_AS、逐样本修复与新增错误、A11−A10−A01＋A00交互、宏F1与最差RX／全部分层。E20与E20＋10分别报告，不冒称scratch30；固定2pp仅本轮探索目标，单seed不完成稳定联合目标。当前没有E30正式性能结论。

原训练对象瞬时clock MSE／scene coverage／reference_steps与逐步Sinc-unit数组在收尾异常后未落盘，不补造记录。保留完整原生step／epoch／物理序列、E10严格计数、优化器状态和最终权重；实际能验证的成本与推算项明确区分。该缺项不改变既定模型、数据或合法预测规则。

owner codex/root/F-E30-score-recovery。代码commit发布时写入release_commit／resolved记录；命令CVS-RFFI/bin/python -u experiments/cvs_clock_e30_score_recovery/runner.py；CWD PROJECT/releases/cvs_F_clock_E30_score_recovery_20261017_r01；唯一不可覆盖root PROJECT/runs/20261017-phase1-F-crossRX-clock-e30-score-recovery-s2026092701-m5-r01。N607普通账号，CPU2/worker，共享每GPU最多3总进程（含其他owner）。产物：逐行source_recovery、7视图predictions/logp、独立scores、同row完成状态、成本／交互／错误分析、报告与登记。

停止规则仅数据／角色／来源、非有限或不完整预测、输出冲突及不能合法执行的系统技术故障；保留产物与其他健康任务，无自动重试。科学判定按既有E30预登记，不因结果延长训练、加损失或扩种子。
