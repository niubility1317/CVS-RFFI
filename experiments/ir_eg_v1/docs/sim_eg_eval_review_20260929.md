# SIM/EG固定目标评估窄审查（2026-09-29）

结论：**PASS，未发现直接阻断本次启动的P0/P1问题。** 本结论限于当前driver、spec及其引用的固定推理、checkpoint加载和评分路径，不覆盖未发布的续训改动。

本次只评估`20260929-phase1-sim-eg-target-eval-s392005-r01`的SIM、EG固定E200 checkpoint。发布内容为新driver与spec；二者的推理和评分命令均指向`ir_eg_source_screen_20260928_r01`。核对本地`predict_phase1_truth_last.py`、`score_native_joint_predictions.py`、`checkpoints.py`与冻结commit `92f40367d61c45b9d16a9718bcd68df17af507c2`无差异。

| 检查项 | 结论与代码证据 |
| --- | --- |
| 输出防冲突 | driver第11、20行拒绝已有run/log目录；当前spec逐行预测、评分、日志路径分别落入新run及独立SIM/EG目录。predictor第81、143行和scorer第40行继续拒绝覆盖已有预测目录/产物。 |
| checkpoint来源 | driver第29至35行在任何目标预测前校验完成状态、E200/44400、row ID、完整`source_info`相等、`target_contact=False`、`scratch_only`和无resume lineage，并做无目标输入的有限输出检查。原loader第6至15行另校验epoch、scratch参数、无baseline/teacher/game_resume及严格加载模型。当前spec检查的checkpoint与实际预测argv一致。 |
| 输入权限 | 当前两行均使用`--mode predict`，不包含训练或resume命令；predictor读取现有`iq.npy`与opaque ID manifest，未进入prepare分支。固定六类identity logits逐样本argmax；推理命令未传truth路径。 |
| 评估配置 | 两行均batch256、workers0，分别绑定GPU0/1后使用`cuda:0`。旧predictor第149、162至164行固定channel seed392005及四场景的独立生成器；两个方法采用相同场景顺序、样本顺序和批大小。 |
| 完整覆盖与truth-last | driver第52至64行等待两个预测进程各自成功退出，检查每份672000条记录、每场景168000个唯一manifest ID及run/row绑定，然后记录`ALL_PREDICTIONS_FIXED`。第67至75行才启动独立scorer连接truth；scorer第43至53行再次拒绝重复ID/场景及不完整覆盖。评分输出的`overall.n`结构与driver校验一致。 |

协议符合性与直接执行质量均通过本次限定范围的静态审查。报告不将单seed、已接触过的目标基准结果描述为独立盲测，spec已明确限制为描述性比较且禁止目标反馈。

证据边界：本审查未执行远端命令、读取目标truth、加载远端checkpoint或运行评估；远端checkpoint实际完成状态、目录空闲、GPU空闲及immutable release部署情况来自主agent实时审计，仍由启动时已有检查及执行后产物证据确认。本审查未新增数据重验、测试或发布门槛，未修改其他文件。
