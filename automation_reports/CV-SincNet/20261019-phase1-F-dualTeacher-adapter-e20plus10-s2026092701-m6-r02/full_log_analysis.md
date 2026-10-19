# 双教师适配器 r02 完整源训练日志分析

**FULL_SIX_ROW_CONSISTENT_STRUCTURED_LOGS**。本分析是结果证据，不是发布或科学晋级门槛，不用于调参、候选重排或失败救援。

每行预期读取全部 2220 步、10 轮及 L/U 各 2220 个批次。缺文件、截断、解析错误或计数差异均明确保留。训练 CE 为 LS=0.01；这里的动态训练统计不是冻结模型拟合。

| 行 | 完整性 | 学生 L 包 | U 包 | Tc / Ts 包 | fresh As 包 | Sinc 更新 |
|---|---|---:|---:|---|---:|---:|
| NOAS_FT | COMPLETE_LOGS_CONSISTENT | 284160 | 567000 | 284160 / 284160 | 284160 | 2220 |
| NOAS_ADAPTER | COMPLETE_LOGS_CONSISTENT | 284160 | 567000 | 284160 / 284160 | 284160 | 0 |
| S0 | COMPLETE_LOGS_CONSISTENT | 568320 | 567000 | 284160 / 284160 | 284160 | 2220 |
| S1 | COMPLETE_LOGS_CONSISTENT | 568320 | 567000 | 284160 / 284160 | 284160 | 0 |
| S2 | COMPLETE_LOGS_CONSISTENT | 568320 | 567000 | 284160 / 284160 | 284160 | 0 |
| S3 | COMPLETE_LOGS_CONSISTENT | 568320 | 567000 | 284160 / 284160 | 284160 | 0 |

## 逐行证据

### NOAS_FT

源目录：`E:\type10-7\local_artifacts\F_dualTeacher_adapter_E30_20261019\retrieved_r02\rows\NOAS_FT-s2026092701\source`。run：`20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：0。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | clean CE | As CE | Tc correct/n | Ts correct/n | 学生 As accuracy |
|---|---:|---:|---:|---|---|---|
| early（1–3） | 666 | 0.246833 | 未执行第二学生前向 | 81284/85248 | 75012/85248 | NO_AS_STUDENT_FORWARD |
| middle（4–7） | 888 | 0.227421 | 未执行第二学生前向 | 108193/113664 | 99695/113664 | NO_AS_STUDENT_FORWARD |
| late（8–10） | 666 | 0.212200 | 未执行第二学生前向 | 81245/85248 | 74743/85248 | NO_AS_STUDENT_FORWARD |

ordinary 生成/用于学生/影子包：284160 / 284160 / 0。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

### NOAS_ADAPTER

源目录：`E:\type10-7\local_artifacts\F_dualTeacher_adapter_E30_20261019\retrieved_r02\rows\NOAS_ADAPTER-s2026092701\source`。run：`20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：1.49011612e-08。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | clean CE | As CE | Tc correct/n | Ts correct/n | 学生 As accuracy |
|---|---:|---:|---:|---|---|---|
| early（1–3） | 666 | 0.115008 | 未执行第二学生前向 | 84831/85248 | 75012/85248 | NO_AS_STUDENT_FORWARD |
| middle（4–7） | 888 | 0.116692 | 未执行第二学生前向 | 113104/113664 | 99695/113664 | NO_AS_STUDENT_FORWARD |
| late（8–10） | 666 | 0.116960 | 未执行第二学生前向 | 84827/85248 | 74743/85248 | NO_AS_STUDENT_FORWARD |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

### S0

源目录：`E:\type10-7\local_artifacts\F_dualTeacher_adapter_E30_20261019\retrieved_r02\rows\S0-s2026092701\source`。run：`20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：2.98023224e-08。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | clean CE | As CE | Tc correct/n | Ts correct/n | 学生 As accuracy |
|---|---:|---:|---:|---|---|---|
| early（1–3） | 666 | 0.126509 | 0.909886 | 84831/85248 | 75012/85248 | NOT_LOGGED |
| middle（4–7） | 888 | 0.118171 | 0.758920 | 113104/113664 | 99695/113664 | NOT_LOGGED |
| late（8–10） | 666 | 0.112802 | 0.731752 | 84827/85248 | 74743/85248 | NOT_LOGGED |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

### S1

源目录：`E:\type10-7\local_artifacts\F_dualTeacher_adapter_E30_20261019\retrieved_r02\rows\S1-s2026092701\source`。run：`20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：5.96046448e-08。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | clean CE | As CE | Tc correct/n | Ts correct/n | 学生 As accuracy |
|---|---:|---:|---:|---|---|---|
| early（1–3） | 666 | 0.116532 | 1.435760 | 84831/85248 | 75012/85248 | NOT_LOGGED |
| middle（4–7） | 888 | 0.124190 | 1.266303 | 113104/113664 | 99695/113664 | NOT_LOGGED |
| late（8–10） | 666 | 0.124349 | 1.233186 | 84827/85248 | 74743/85248 | NOT_LOGGED |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

### S2

源目录：`E:\type10-7\local_artifacts\F_dualTeacher_adapter_E30_20261019\retrieved_r02\rows\S2-s2026092701\source`。run：`20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：8.94069672e-08。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | clean CE | As CE | Tc correct/n | Ts correct/n | 学生 As accuracy |
|---|---:|---:|---:|---|---|---|
| early（1–3） | 666 | 0.115914 | 1.459511 | 84831/85248 | 75012/85248 | NOT_LOGGED |
| middle（4–7） | 888 | 0.121247 | 1.312997 | 113104/113664 | 99695/113664 | NOT_LOGGED |
| late（8–10） | 666 | 0.121493 | 1.280498 | 84827/85248 | 74743/85248 | NOT_LOGGED |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

### S3

源目录：`E:\type10-7\local_artifacts\F_dualTeacher_adapter_E30_20261019\retrieved_r02\rows\S3-s2026092701\source`。run：`20261019-phase1-F-dualTeacher-adapter-e20plus10-s2026092701-m6-r02`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：3.27825546e-07。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | clean CE | As CE | Tc correct/n | Ts correct/n | 学生 As accuracy |
|---|---:|---:|---:|---|---|---|
| early（1–3） | 666 | 0.123816 | 1.456168 | 84831/85248 | 75012/85248 | NOT_LOGGED |
| middle（4–7） | 888 | 0.140847 | 1.321859 | 113104/113664 | 99695/113664 | NOT_LOGGED |
| late（8–10） | 666 | 0.142972 | 1.294942 | 84827/85248 | 74743/85248 | NOT_LOGGED |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

## L/U 直接配对与边界

`{"availability": "DIRECT_FULL_ID_COMPARISON", "reference": "NOAS_FT-s2026092701", "comparisons": [{"row_id": "NOAS_ADAPTER-s2026092701", "records_compared": 4440, "different_records": 0, "examples": [], "exact_same_ordered_L_U_ID_stream": true, "comparison_complete": true}, {"row_id": "S0-s2026092701", "records_compared": 4440, "different_records": 0, "examples": [], "exact_same_ordered_L_U_ID_stream": true, "comparison_complete": true}, {"row_id": "S1-s2026092701", "records_compared": 4440, "different_records": 0, "examples": [], "exact_same_ordered_L_U_ID_stream": true, "comparison_complete": true}, {"row_id": "S2-s2026092701", "records_compared": 4440, "different_records": 0, "examples": [], "exact_same_ordered_L_U_ID_stream": true, "comparison_complete": true}, {"row_id": "S3-s2026092701", "records_compared": 4440, "different_records": 0, "examples": [], "exact_same_ordered_L_U_ID_stream": true, "comparison_complete": true}], "missing_rows": [], "no_hashes": true, "limit": "Equal physical-ID order is not a claim of identical dropout/augmentation/whole-training RNG trajectories"}`

缺少或无法完整读取的日志文件（不含未设计记录的指标）：`[]`。

L/U 比较直接逐记录、逐 slot 比较实际 physical ID，没有新增 hash。相同采样顺序不等于不同机制具有完全相同的随机计算轨迹。

学生 As accuracy 仅在日志真有 correct/n 时计算；NOT_LOGGED 表示未记录，不能用教师正确率、损失或源 V 值反推。NoAs 的 As CE=0 是未执行分支的占位量。源 V/正式测试与本训练曲线分开，解析器不打开 formal、truth、checkpoint 或 IQ。
