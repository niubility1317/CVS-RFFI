# 源域多层识别特征与波形梯度诊断

- run_id：`20261109-diagnostic-Restore-MultilevelFGradient-sourceLU-s2026092701-r01`
- group_id：`source-multilevel-F0-input-gradient`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用28条合法源配对和固定源恢复器，比较F0浅层、中层、最终特征及波形损失对恢复IQ的梯度；无训练或目标访问。

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

## 本次诊断的固定口径

取 R07 原始顺序中每个源模拟场景与 L_s/U_s 角色各两条，共 28 条；固定已核验的源 CovarianceDetail final30 和地面 Tri-SIG-TIME SSL F0。对恢复 IQ 分别求波形配对目标、F0 最终 160 维特征损失，以及固定浅层 time_down、中层 t2.pool 的 clean 配对特征损失的梯度。浅层和中层按各自 clean 特征均方归一，计算这些 IQ 输入梯度的范数与两两余弦；不对 G 参数求梯度、不更新模型，也不读取 F 分类 logits、TX 标签、V/OE 或目标。第一和第二条真实源样本构成实际模型 smoke，PASS 后接着处理全部 28 条。

小样本同向或异向只描述局部 IQ 梯度，不等于参数更新方向、设备特征有效性或目标识别收益；没有据此修改在训网络或正式测试。G、F0、原始源配对及完整祖先见 experiment.json。

## 本地核验与发布

代码提交 3af4d5eb10f369f43b5595eee7c203baf48cb4e9；5 项特征捕获、异常清理、梯度和源配对定向检查通过。一次独立 P0/P1 审查未发现阻塞问题。N607 直接连接、目标身份、项目目录与 GPU 只读预检通过。远端每卡总计算进程上限为 3，允许 GPU4 或 6，控制器只处理本 run 子进程；至少留 12000 MiB 显存。

该目录登记此前共享的实验索引存在其他任务的未提交更新，因此本次只追加本 run 的登记文件，不覆盖或重建共享索引。远端尚未发布时，状态为 LOCAL_VERIFIED；实际运行以 PID、CWD、GPU、日志和输出读回为准。
