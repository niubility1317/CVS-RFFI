# WiSig 主对比的默认与增强实用星地信道盲恢复

- run_id：`20261125-phase1-wisig-blind-capacity-stress-m2-r02`
- group_id：`wisig-blind-capacity-stress`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

将 LoRa Full 六景领先的扩容 DURA 半量版适配到 WiSig 256 点窗。固定两组从零重训，仅改变模拟信道强度，交换默认和增强测试，对比相同物理包的原始接收与冻结 R69。

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

## 重新发布

沿用 [首次发布的预登记设计](../20261125-phase1-wisig-blind-capacity-stress-m2-r01/report.md)。首次发布在控制器导入时缺少归档依赖，未创建运行根。此版补齐依赖并在启动前实际导入入口。训练和评分矩阵、权重选择、数据和权限均沿用预登记。

两个训练版本均使用扩容 DURA 半量架构、固定 40,000 步和相同源 U 物理采样。增强模拟将参考 SNR 下调 6 dB，残余相位跟踪误差标准差设为默认的 1.5 倍。恢复器从单窗 IQ 估计信道作用与有效噪声。默认和增强模型分别在两个模拟强度上用相同 168,000 个物理包、两条路线、clean 与六种 practical、三个冻结识别器交叉测试。
