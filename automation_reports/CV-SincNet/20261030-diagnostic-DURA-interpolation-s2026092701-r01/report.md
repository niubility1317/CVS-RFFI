# DURA固定二倍插值的源域等价与GPU速度诊断

- run_id：`20261030-diagnostic-DURA-interpolation-s2026092701-r01`
- group_id：`phase1-dura-training-step-profile`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

SamefastLRsourcefinal/no update;onlyfourCore12of15linear2x ops replaced by halfpixel slices;full gradient agreement and alternatingGPUwall

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

## 实际结果与采用判据

VERIFIED/ANALYZED_SOURCE_ONLY。执行commit 800708ba00fcfd96e8deab03ceafc01e99ae9dbf；N607 preflight、单归档SHA比较和compile通过，一次独立P0/P1审查PASS。CPUcontroller3680147/GPU4 worker3680502的存活、CWD/完整命令/CUDA环境和nvidia-smi PID显存1408MiB均已实际读回；GPU日志增长到320bytes，CUPTI实际四步完成。两Ls源checkpoint noquery smoke PASS；完整原模型损失和全部裁剪后参数梯度在rtol2e-5/atol2e-6内一致，模型全部权重/缓冲区不变。无优化器/TX/F/V/target。

固定源B256、3轮交替×每路径3次，native/slice中位分别221.756580/189.741313ms，减少14.437%。三轮各自收益为14.564%, 14.342%, 14.748%，达到预登记至少10%且每轮正收益的判据。9次中位与轮内结果已从完整记录独立复算；共享GPU负载下仅证明该匹配步骤的证据，不与前一不同时间131ms直接比较，不外推整轮/训练成本或识别性能。

仅改变四个Core.forward的12/15次固定二倍半像素插值；action estimator另3次仍native，参数、共享关系、初始权重、损失、CG/FFT和随机输入保持。通过CPU5个双精度边界/前向/伴随检查。没有生成新权重或预测，不进入target评分。下一独立E20从零训练只加入该执行替换，保留原fastLR的全部目标（包括方差辅助项），不混入正在正式测试的noVarAux，不根据该目标结果决定训练配方。正式冻结推理亦采用新slice实现，明确作为独立执行版本验证完整七景；默认部署不变。
