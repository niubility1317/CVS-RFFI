# Phase2 B 主干：修正支持身份后从已保存 B 接续 MoPC

- run_id：`20261109-phase2-bsource-savedb-mopc-m4-r13`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Correct bank-position to physical-ID comparison; preserve r12 eight pre-registration failure directories and logs with zero predictions. Resume 1906 MoPC tasks from their actual r11 B anchors; retain nine earlier science quarantines and canceled RadioNet. Nonfinite root cause remains unknown.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## Failed with new nonfinite observations

The support identity repair passed targeted tests and was observed to execute actual MoPC registrations from saved r11 B anchors. The controller drained after two new tasks failed; 14 MoPC-only tasks closed and the two failed tasks retain 10 partial prediction files each. For c0189 the first instrumented nonfinite boundary was support logits; for c0192 it was a batch-normalization running variance. Both subsequently had nonfinite gradient elements. These are first observed instrumented boundaries, not proven first mathematical operations or root causes. Existing r11 predictions/B states, r12 evidence, prior quarantines, and cancellation are retained. The full prediction and scoring barriers remain closed. Evidence: local_artifacts/phase2_group2_bnoradio_20261108/r13_failure_state_verified.json and r13_failure_detail_verified.json.

## 合成定位与性能观察

在 N607 GPU7 上，仅使用随机合成 IQ、随机初始化及原 MoPC 公式，固定确定性内核后，5 个合成 seed 中 seed 1731 出现非有限梯度。逐层观察显示，该例先观察到 encoder.4 的 BN running_var 非有限，其前 encoder.3 输出最大绝对值约 5.15e19；这证明合成例中发生了幅度放大和数值溢出，不证明 r13 真实故障同因。对同一合成例，仅诊断性地将二次范数特征清零后，训练完成；另一些 seed 的最大损失反而上升。此干预改变方法，未用于正式实验。

合成重复测试中，默认内核的两个相同 seed 运行内部状态不完全一致；指定确定性内核的两个重复一致，但该设置下 seed 1731 仍可出现非有限梯度。FP64 在一个合成例完成，FP32 提前失败；不同 dtype 的随机抽样可能不同，不能据此证明可修复或等价。三组成对合成训练中，启用原诊断记录和关闭记录的状态及原型在确定性设置下完全一致；原始耗时分别为 2.787/1.885/2.150 秒，带记录为 2.947/2.940/3.120 秒。该测量仅为单卡合成注册，不代表 N607 整体加速；数值故障未解决，未将优化用于正式运行。标量记录微基准仅约 1.05 倍，不作为整体加速。

原始证据保存在 local_artifacts/phase2_group2_bnoradio_20261108/mopc_quadratic_synthetic_verified.json、mopc_synthetic_first_layer_verified.json、mopc_determinism_synthetic_verified.json、mopc_fp64_synthetic_verified.json 和 mopc_trace_deterministic_bench_verified.json。
