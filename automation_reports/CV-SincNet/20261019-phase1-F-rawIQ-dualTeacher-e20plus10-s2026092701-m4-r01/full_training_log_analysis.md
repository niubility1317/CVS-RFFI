# Raw-IQ 双教师适配器：四行完整源训练日志分析

**FULL_FOUR_ROW_CONSISTENT_STRUCTURED_LOGS**。本分析是结果证据，不是发布或科学晋级门槛，不用于调参、候选重排或失败救援。

每行预期读取全部 2220 步、10 轮及 L/U 各 2220 个批次。缺文件、截断、解析错误或计数差异均明确保留。训练 CE 为 LS=0.01；这里的动态训练统计不是冻结模型拟合。

| 行 | 完整性 | 学生 L 包 | U 包 | Tc / Ts 包 | fresh As 包 | Sinc 更新 |
|---|---|---:|---:|---|---:|---:|
| NOAS_ADAPTER | COMPLETE_LOGS_CONSISTENT | 284160 | 567000 | 284160 / 284160 | 284160 | 0 |
| S1 | COMPLETE_LOGS_CONSISTENT | 568320 | 567000 | 284160 / 284160 | 284160 | 0 |
| S2 | COMPLETE_LOGS_CONSISTENT | 568320 | 567000 | 284160 / 284160 | 284160 | 0 |
| S3 | COMPLETE_LOGS_CONSISTENT | 568320 | 567000 | 284160 / 284160 | 284160 | 0 |

## 逐行证据

### NOAS_ADAPTER

源目录：`E:\type10-7\local_artifacts\F_rawIQ_dualTeacher_E30_20261019\retrieved\rows\NOAS_ADAPTER-s2026092701\source`。run：`20261019-phase1-F-rawIQ-dualTeacher-e20plus10-s2026092701-m4-r01`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：1.49011612e-08。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | raw CE | As CE | 学生 raw correct/n | 学生 As correct/n | Tc correct/n | Ts correct/n |
|---|---:|---:|---:|---|---|---|---|
| early（1–3） | 666 | 0.112502 | 未执行第二学生前向 | 84314/85248=98.904% | NO_AS_STUDENT_FORWARD | 84831/85248 | 75012/85248 |
| middle（4–7） | 888 | 0.112785 | 未执行第二学生前向 | 112443/113664=98.926% | NO_AS_STUDENT_FORWARD | 113104/113664 | 99695/113664 |
| late（8–10） | 666 | 0.112627 | 未执行第二学生前向 | 84368/85248=98.968% | NO_AS_STUDENT_FORWARD | 84827/85248 | 74743/85248 |

逐轮原始学生正确数（均来自当前训练步已有 logits；含 Dropout，不能当冻结模型拟合）：

| 本段轮次 | raw correct/n | raw accuracy | As correct/n | As accuracy | raw CE | As CE |
|---|---|---:|---|---:|---:|---:|
| 1 | 28098/28416 | 98.881% | 0/0 | 无 As 学生前向 | 0.110873 | 未计算 |
| 2 | 28091/28416 | 98.856% | 0/0 | 无 As 学生前向 | 0.113094 | 未计算 |
| 3 | 28125/28416 | 98.976% | 0/0 | 无 As 学生前向 | 0.113538 | 未计算 |
| 4 | 28096/28416 | 98.874% | 0/0 | 无 As 学生前向 | 0.113018 | 未计算 |
| 5 | 28124/28416 | 98.972% | 0/0 | 无 As 学生前向 | 0.111984 | 未计算 |
| 6 | 28100/28416 | 98.888% | 0/0 | 无 As 学生前向 | 0.113822 | 未计算 |
| 7 | 28123/28416 | 98.969% | 0/0 | 无 As 学生前向 | 0.112314 | 未计算 |
| 8 | 28127/28416 | 98.983% | 0/0 | 无 As 学生前向 | 0.112273 | 未计算 |
| 9 | 28114/28416 | 98.937% | 0/0 | 无 As 学生前向 | 0.113829 | 未计算 |
| 10 | 28127/28416 | 98.983% | 0/0 | 无 As 学生前向 | 0.111777 | 未计算 |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

completion 记录耗时：1133.5757398605347 秒（不含教师预载及后续预测／评分）。new cosine10 的逐步 LR、每步 clip 和原 Sinc 零更新均全量读取。

原 F206 保持、原 head 未变与 raw 八个状态键的变化为 observer 已记录证据；未读取 CP 进行第二次权重比较。

| 新状态键 | 是否变化 | 相对初态 L2 | 初态 L2 |
|---|---|---:|---:|
| B1.weight | True | 0.9566904902458191 | 0.0 |
| B2.weight | True | 1.8034822940826416 | 0.0 |
| conv1.weight | True | 0.6563209295272827 | 2.2239837646484375 |
| conv2.weight | True | 1.289258360862732 | 2.310668468475342 |
| gn1.bias | True | 0.3296123147010803 | 0.0 |
| gn1.weight | True | 0.41527482867240906 | 4.0 |
| gn2.bias | True | 0.6211957931518555 | 0.0 |
| gn2.weight | True | 0.780275285243988 | 4.0 |

全部 L 公式最大残差：1.49011612e-08；原生日志交叉残差：`{"native_first_CE_minus_mechanism_raw_CE": {"n": 2220, "mean": 0.0, "min": 0.0, "max": 0.0, "first": 0.0, "last": 0.0}, "native_hook_total_L_minus_mechanism_total_L": {"n": 2220, "mean": 0.0, "min": 0.0, "max": 0.0, "first": 0.0, "last": 0.0}}`。

逐轮时间字段保留原名，详见 JSON 的 timing；不会把不含教师预载的 completion 秒数冒充端到端总耗时。

### S1

源目录：`E:\type10-7\local_artifacts\F_rawIQ_dualTeacher_E30_20261019\retrieved\rows\S1-s2026092701\source`。run：`20261019-phase1-F-rawIQ-dualTeacher-e20plus10-s2026092701-m4-r01`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：5.21540642e-08。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | raw CE | As CE | 学生 raw correct/n | 学生 As correct/n | Tc correct/n | Ts correct/n |
|---|---:|---:|---:|---|---|---|---|
| early（1–3） | 666 | 0.114601 | 1.416484 | 84199/85248=98.769% | 44911/85248=52.683% | 84831/85248 | 75012/85248 |
| middle（4–7） | 888 | 0.117328 | 1.268008 | 112239/113664=98.746% | 66030/113664=58.092% | 113104/113664 | 99695/113664 |
| late（8–10） | 666 | 0.117561 | 1.233259 | 84150/85248=98.712% | 50533/85248=59.278% | 84827/85248 | 74743/85248 |

逐轮原始学生正确数（均来自当前训练步已有 logits；含 Dropout，不能当冻结模型拟合）：

| 本段轮次 | raw correct/n | raw accuracy | As correct/n | As accuracy | raw CE | As CE |
|---|---|---:|---|---:|---:|---:|
| 1 | 28066/28416 | 98.768% | 13905/28416 | 48.934% | 0.113475 | 1.521155 |
| 2 | 28063/28416 | 98.758% | 15250/28416 | 53.667% | 0.114742 | 1.388044 |
| 3 | 28070/28416 | 98.782% | 15756/28416 | 55.448% | 0.115585 | 1.340252 |
| 4 | 28068/28416 | 98.775% | 16163/28416 | 56.880% | 0.116392 | 1.303219 |
| 5 | 28045/28416 | 98.694% | 16466/28416 | 57.946% | 0.117612 | 1.271396 |
| 6 | 28073/28416 | 98.793% | 16617/28416 | 58.478% | 0.117152 | 1.259502 |
| 7 | 28053/28416 | 98.723% | 16784/28416 | 59.065% | 0.118155 | 1.237916 |
| 8 | 28041/28416 | 98.680% | 16830/28416 | 59.227% | 0.117106 | 1.240592 |
| 9 | 28057/28416 | 98.737% | 16893/28416 | 59.449% | 0.117805 | 1.231316 |
| 10 | 28052/28416 | 98.719% | 16810/28416 | 59.157% | 0.117771 | 1.227870 |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

completion 记录耗时：1173.709505558014 秒（不含教师预载及后续预测／评分）。new cosine10 的逐步 LR、每步 clip 和原 Sinc 零更新均全量读取。

原 F206 保持、原 head 未变与 raw 八个状态键的变化为 observer 已记录证据；未读取 CP 进行第二次权重比较。

| 新状态键 | 是否变化 | 相对初态 L2 | 初态 L2 |
|---|---|---:|---:|
| B1.weight | True | 1.2377065420150757 | 0.0 |
| B2.weight | True | 2.009833335876465 | 0.0 |
| conv1.weight | True | 0.6843363642692566 | 2.2239837646484375 |
| conv2.weight | True | 1.2666138410568237 | 2.310668468475342 |
| gn1.bias | True | 0.4072066843509674 | 0.0 |
| gn1.weight | True | 0.5124042630195618 | 4.0 |
| gn2.bias | True | 0.5981957912445068 | 0.0 |
| gn2.weight | True | 0.8193830847740173 | 4.0 |

全部 L 公式最大残差：5.21540642e-08；原生日志交叉残差：`{"native_first_CE_minus_mechanism_raw_CE": {"n": 2220, "mean": 0.0, "min": 0.0, "max": 0.0, "first": 0.0, "last": 0.0}, "native_hook_total_L_minus_mechanism_total_L": {"n": 2220, "mean": 0.0, "min": 0.0, "max": 0.0, "first": 0.0, "last": 0.0}}`。

逐轮时间字段保留原名，详见 JSON 的 timing；不会把不含教师预载的 completion 秒数冒充端到端总耗时。

### S2

源目录：`E:\type10-7\local_artifacts\F_rawIQ_dualTeacher_E30_20261019\retrieved\rows\S2-s2026092701\source`。run：`20261019-phase1-F-rawIQ-dualTeacher-e20plus10-s2026092701-m4-r01`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：8.94069672e-08。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | raw CE | As CE | 学生 raw correct/n | 学生 As correct/n | Tc correct/n | Ts correct/n |
|---|---:|---:|---:|---|---|---|---|
| early（1–3） | 666 | 0.115131 | 1.421141 | 84181/85248=98.748% | 44929/85248=52.704% | 84831/85248 | 75012/85248 |
| middle（4–7） | 888 | 0.118114 | 1.276328 | 112223/113664=98.732% | 65964/113664=58.034% | 113104/113664 | 99695/113664 |
| late（8–10） | 666 | 0.118091 | 1.242543 | 84197/85248=98.767% | 50485/85248=59.221% | 84827/85248 | 74743/85248 |

逐轮原始学生正确数（均来自当前训练步已有 logits；含 Dropout，不能当冻结模型拟合）：

| 本段轮次 | raw correct/n | raw accuracy | As correct/n | As accuracy | raw CE | As CE |
|---|---|---:|---|---:|---:|---:|
| 1 | 28073/28416 | 98.793% | 13879/28416 | 48.842% | 0.112914 | 1.525607 |
| 2 | 28058/28416 | 98.740% | 15276/28416 | 53.758% | 0.115700 | 1.392222 |
| 3 | 28050/28416 | 98.712% | 15774/28416 | 55.511% | 0.116780 | 1.345594 |
| 4 | 28058/28416 | 98.740% | 16187/28416 | 56.964% | 0.117645 | 1.309827 |
| 5 | 28068/28416 | 98.775% | 16485/28416 | 58.013% | 0.118046 | 1.279465 |
| 6 | 28054/28416 | 98.726% | 16594/28416 | 58.397% | 0.118207 | 1.266676 |
| 7 | 28043/28416 | 98.687% | 16698/28416 | 58.763% | 0.118558 | 1.249344 |
| 8 | 28068/28416 | 98.775% | 16805/28416 | 59.139% | 0.117572 | 1.249372 |
| 9 | 28074/28416 | 98.796% | 16866/28416 | 59.354% | 0.118371 | 1.241649 |
| 10 | 28055/28416 | 98.730% | 16814/28416 | 59.171% | 0.118330 | 1.236608 |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

completion 记录耗时：1165.4183630943298 秒（不含教师预载及后续预测／评分）。new cosine10 的逐步 LR、每步 clip 和原 Sinc 零更新均全量读取。

原 F206 保持、原 head 未变与 raw 八个状态键的变化为 observer 已记录证据；未读取 CP 进行第二次权重比较。

| 新状态键 | 是否变化 | 相对初态 L2 | 初态 L2 |
|---|---|---:|---:|
| B1.weight | True | 1.2173599004745483 | 0.0 |
| B2.weight | True | 2.2331299781799316 | 0.0 |
| conv1.weight | True | 0.7657836675643921 | 2.2239837646484375 |
| conv2.weight | True | 1.369167447090149 | 2.310668468475342 |
| gn1.bias | True | 0.32305970788002014 | 0.0 |
| gn1.weight | True | 0.43800419569015503 | 4.0 |
| gn2.bias | True | 0.5929445624351501 | 0.0 |
| gn2.weight | True | 0.8427296280860901 | 4.0 |

全部 L 公式最大残差：8.94069672e-08；原生日志交叉残差：`{"native_first_CE_minus_mechanism_raw_CE": {"n": 2220, "mean": 0.0, "min": 0.0, "max": 0.0, "first": 0.0, "last": 0.0}, "native_hook_total_L_minus_mechanism_total_L": {"n": 2220, "mean": 0.0, "min": 0.0, "max": 0.0, "first": 0.0, "last": 0.0}}`。

逐轮时间字段保留原名，详见 JSON 的 timing；不会把不含教师预载的 completion 秒数冒充端到端总耗时。

### S3

源目录：`E:\type10-7\local_artifacts\F_rawIQ_dualTeacher_E30_20261019\retrieved\rows\S3-s2026092701\source`。run：`20261019-phase1-F-rawIQ-dualTeacher-e20plus10-s2026092701-m4-r01`。

全部流记录数：mechanism_steps.jsonl.gz=2220, ordinary_draw_steps.jsonl.gz=2220, unit_steps.jsonl.gz=2220, training_sequence.jsonl.gz=4440, step_metrics.jsonl.gz=2220, continuation_epochs.json=10, training_result.json=1, completion.json=1, epoch_metrics.jsonl=10。

损失公式最大绝对残差：3.24100256e-07。各 CE/KD 使用记录中的全部 128 包分母；mask 是教师正确集合，不能据此替代学生正确集合。

| 时段（本段轮次） | 步数 | raw CE | As CE | 学生 raw correct/n | 学生 As correct/n | Tc correct/n | Ts correct/n |
|---|---:|---:|---:|---|---|---|---|
| early（1–3） | 666 | 0.121334 | 1.405992 | 84051/85248=98.596% | 46080/85248=54.054% | 84831/85248 | 75012/85248 |
| middle（4–7） | 888 | 0.133853 | 1.263473 | 111736/113664=98.304% | 68663/113664=60.409% | 113104/113664 | 99695/113664 |
| late（8–10） | 666 | 0.136155 | 1.234180 | 83738/85248=98.229% | 52488/85248=61.571% | 84827/85248 | 74743/85248 |

逐轮原始学生正确数（均来自当前训练步已有 logits；含 Dropout，不能当冻结模型拟合）：

| 本段轮次 | raw correct/n | raw accuracy | As correct/n | As accuracy | raw CE | As CE |
|---|---|---:|---|---:|---:|---:|
| 1 | 28059/28416 | 98.744% | 14000/28416 | 49.268% | 0.115563 | 1.526829 |
| 2 | 28009/28416 | 98.568% | 15712/28416 | 55.293% | 0.121827 | 1.370771 |
| 3 | 27983/28416 | 98.476% | 16368/28416 | 57.601% | 0.126610 | 1.320377 |
| 4 | 27952/28416 | 98.367% | 16845/28416 | 59.280% | 0.130407 | 1.287695 |
| 5 | 27949/28416 | 98.357% | 17151/28416 | 60.357% | 0.133375 | 1.266218 |
| 6 | 27928/28416 | 98.283% | 17305/28416 | 60.899% | 0.135761 | 1.252336 |
| 7 | 27907/28416 | 98.209% | 17362/28416 | 61.099% | 0.135870 | 1.247641 |
| 8 | 27946/28416 | 98.346% | 17477/28416 | 61.504% | 0.135118 | 1.242047 |
| 9 | 27902/28416 | 98.191% | 17497/28416 | 61.574% | 0.136868 | 1.230981 |
| 10 | 27890/28416 | 98.149% | 17514/28416 | 61.634% | 0.136480 | 1.229512 |

ordinary 生成/用于学生/影子包：284160 / 0 / 284160。

日志一致性问题：`{}`。完整细节、逐轮曲线、native 所有数值字段和原始计数见同名 JSON。

stdout：FULL_FILES_SCANNED。原 F / head 的冻结结论来自 observer 日志及 changed_parameters，解析器没有独立读取权重字节。

completion 记录耗时：1136.9018957614899 秒（不含教师预载及后续预测／评分）。new cosine10 的逐步 LR、每步 clip 和原 Sinc 零更新均全量读取。

原 F206 保持、原 head 未变与 raw 八个状态键的变化为 observer 已记录证据；未读取 CP 进行第二次权重比较。

| 新状态键 | 是否变化 | 相对初态 L2 | 初态 L2 |
|---|---|---:|---:|
| B1.weight | True | 1.7388445138931274 | 0.0 |
| B2.weight | True | 3.0986409187316895 | 0.0 |
| conv1.weight | True | 0.8926241993904114 | 2.2239837646484375 |
| conv2.weight | True | 1.6117054224014282 | 2.310668468475342 |
| gn1.bias | True | 0.48438018560409546 | 0.0 |
| gn1.weight | True | 0.606865406036377 | 4.0 |
| gn2.bias | True | 0.7508581876754761 | 0.0 |
| gn2.weight | True | 1.004971981048584 | 4.0 |

全部 L 公式最大残差：3.24100256e-07；原生日志交叉残差：`{"native_first_CE_minus_mechanism_raw_CE": {"n": 2220, "mean": 0.0, "min": 0.0, "max": 0.0, "first": 0.0, "last": 0.0}, "native_hook_total_L_minus_mechanism_total_L": {"n": 2220, "mean": 0.0, "min": 0.0, "max": 0.0, "first": 0.0, "last": 0.0}}`。

逐轮时间字段保留原名，详见 JSON 的 timing；不会把不含教师预载的 completion 秒数冒充端到端总耗时。

## L/U 直接配对与边界

`{"availability": "DIRECT_FULL_ID_COMPARISON", "reference": "NOAS_ADAPTER-s2026092701", "comparisons": [{"row_id": "S1-s2026092701", "records_compared": 4440, "different_records": 0, "examples": [], "exact_same_ordered_L_U_ID_stream": true, "comparison_complete": true}, {"row_id": "S2-s2026092701", "records_compared": 4440, "different_records": 0, "examples": [], "exact_same_ordered_L_U_ID_stream": true, "comparison_complete": true}, {"row_id": "S3-s2026092701", "records_compared": 4440, "different_records": 0, "examples": [], "exact_same_ordered_L_U_ID_stream": true, "comparison_complete": true}], "missing_rows": [], "no_hashes": true, "limit": "Equal physical-ID order is not a claim of identical dropout/augmentation/whole-training RNG trajectories"}`

未取回或未记录项：`[]`。

L/U 比较直接逐记录、逐 slot 比较实际 physical ID，没有新增 hash。相同采样顺序不等于不同机制具有完全相同的随机计算轨迹。

学生 As accuracy 仅在日志真有 correct/n 时计算；NOT_LOGGED 表示未记录，不能用教师正确率、损失或源 V 值反推。NoAs 的 As CE=0 是未执行分支的占位量。源 V/正式测试与本训练曲线分开，解析器不打开 formal、truth、checkpoint 或 IQ。
