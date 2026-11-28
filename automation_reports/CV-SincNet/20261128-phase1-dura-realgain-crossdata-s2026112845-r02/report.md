# R02／R07／R17 与实增益融合：WiSig 训练及 LoRa 冻结权重筛查

- run_id：`20261128-phase1-dura-realgain-crossdata-s2026112845-r02`
- group_id：`phase1-lora-wisig-dura-real-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：源域两景筛查已独立评分，两套候选均未晋级。

## 目的与对照

修复历史模块路径；原运行 LoRa 已训练 2000 步，直接复用冻结权重筛查；WiSig 改用原 R17 加实增益，单独训练匹配实增益对照。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

### 固定源域筛查

LoRa 使用 b200_1 的 60 个固定源 V 包，WiSig 使用源 RX 1 的 360 条固定源 V 记录。每套数据的 clean 与 low urban 全部预测固定后，独立 CPU scorer 才连接源 V 标签。两套源 V 与历史基准均有既往曝光，本次未读取目标数据。

| 数据集与路径 | Clean | Low urban |
|---|---:|---:|
| LoRa 直通 | 60/60 | 30/60 |
| LoRa 原生 R17 | 59/60 | 15/60 |
| LoRa 实增益 | 60/60 | 34/60 |
| LoRa 融合 | 60/60 | 32/60 |
| WiSig 直通 | 360/360 | 200/360 |
| WiSig 原始 R17 | 360/360 | 332/360 |
| WiSig 实增益 | 360/360 | 190/360 |
| WiSig 融合 | 360/360 | 329/360 |

LoRa 融合相对实增益改对 5 包、改错 7 包、净减 2 包，95% 配对区间为 −8 至 +5 包；两个预定半样本各净减 1 包。WiSig 融合相对 R17 改对 4 条、改错 7 条、净减 3 条，95% 配对区间为 −10 至 +3 条；两个半样本分别净增 1 条和净减 4 条。两套均未满足预登记晋级规则，因此未启动完整目标测试。

LoRa gain 复用 r01 已冻结的 2,000 步源 U 训练，新增 11,233 参数、训练 198.34 秒、峰值已分配显存 93,241,856 字节。WiSig 融合和实增益对照分别训练 2,000 步、48,000 次 U 暴露，新增各 11,233 参数，训练分别为 356.81 秒和 84.81 秒；两者记录的峰值已分配显存均为 57,982,976 字节。上述数值不包括各父模型的历史训练成本。

两套筛查的固定 DURA v1 同口径对照和全部七景正式验收仍待完成；LoRa 此处对照为独立训练的原生 R17，WiSig 此处对照为原始 R17。详细冻结记录和独立评分保留在 `local_artifacts/20261128-phase1-dura-realgain-crossdata-s2026112845-r02/readback/{lora,wisig}`，远端原件位于同名 N607 run root。

## 交接

本 run 的 source screen 已完成；两数据集均未晋级。后续分别采用不重叠的源验证样本筛查新候选。此前 r01 的导入故障与保留的 LoRa 训练权重见其登记；r02 未重复训练 LoRa。
