# LoRa 与 WiSig R17 原始及逐包归一化路径配对诊断

- run_id：`20261128-diagnostic-shared-r17-scale-s2026112953-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；源域配对尺度诊断完成。

## 目的与对照

在已曝光源 V 同包 clean/低城区对比原 R17(y) 与 R17(y/RMS)*RMS，诊断融合训练和原始对照的输入尺度契约差异。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码提交 `e1fd00eb5d33d116b1aade199cf419692730a814`；控制器 PID 3830196，LoRa worker PID 3830224 / GPU 7，WiSig worker PID 3830225 / GPU 1。N607 预检、单一归档传输 SHA、远端编译、PID/CWD/GPU 与日志增长已核实；一次独立 P0/P1 审查发现结果行标签需明确，已修复并定向复核。原始 `R17` 和 `normalized_r17` 分列，不改其他 owner 或历史产物。

## 结果与覆盖

无参数、无训练。真实权重 source-only smoke 后，固定 LoRa 60 包与 WiSig 360 包的 clean 和低城区预测，再独立 truth-last。四组原始 R17 与逐包归一化 R17 预测均逐包一致；LoRa clean 59/60、低城区 15/60，WiSig clean 360/360、低城区 332/360。此为反复曝光源 V 上的识别预测比较，不证明波形逐值相同，也不构成新融合候选晋级或目标测试。原 checkpoint 和历史产物保持。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
