# WiSig 短训练前导双路线工程对照

- run_id：`20261123-phase1-wisig-stf-engineering-source-s2026092701-r02`
- group_id：`wisig-observable-satellite-recovery`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在固定的 1,800 个源 V 物理包上，比较粗细载频校正、正则化均衡、估计残差功率和保留原载频偏移的相对作用。两条路线共享算法与固定规则。

沿用 [r01 的接收量清单、推理规则及相关工作](../20261123-phase1-wisig-stf-engineering-source-s2026092701-r01/report.md)。本次补齐归档中的 R69 历史运行依赖 `tools/r17_activation_factorization.py`。r01 保留在原发布目录，未产生模型预测或评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

**VERIFIED／SOURCE_SCORED。** N607 执行提交 `0919fa6f127f19332d40ecd59fb28749ecbf4839`。归档传输、编译、真实检查点无目标 smoke、双路线 14 组冻结预测和独立 CPU 评分均通过。控制器 PID 1918290，真实 smoke 由 GPU 1 上的 PID 1918305 完成，评估由 GPU 6 上的 PID 1918383 完成。独立评分 PID 1918908。预测同包同序，两个 clean 决策矩阵逐值一致。

| 读出 | Clean / 1,800 | Full 六景 / 10,800 | Full 低城区 / 1,800 | Residual 六景 / 10,800 | Residual 低城区 / 1,800 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 直通 | 1,773 | 5,184 | 663 | 5,185 | 660 |
| 粗 CFO | 1,282 | 4,689 | 616 | 4,687 | 619 |
| 细 CFO | 1,284 | 4,668 | 620 | 4,669 | 621 |
| 细 CFO + 正则逆 | 1,164 | 3,483 | 502 | 3,481 | 506 |
| 细 CFO + Wiener | 1,158 | 3,338 | 488 | 3,335 | 493 |
| Wiener 后加回 CFO | 1,766 | 3,555 | 516 | 3,556 | 517 |
| 原 R69 | 1,773 | 10,112 | 1,515 | 10,109 | 1,513 |

从正则逆加入估计残差功率后，六景分别少判对 145 和 146 包。加回 CFO 将 clean 恢复到 1,766，仍低于直通 7 包，且两路六景均比直通少判对 1,629 包。现有证据对应已均衡 WiSig 的短训练字段及固定 F50。当前短前导上的整体逆滤波会损伤这套冻结识别器依赖的幅相结构，逐接收机、逐景的错变对和对变错均保存在评分文件。

Full 的 20 点相干度中位数从 clean 的 0.995 降至低城区的 0.628，残差功率占比从 0.00567 升至 0.366。低城区细粗 CFO 差的第 10 至第 90 百分位为 −24.60 至 32.44 kHz，表观采样钟斜率为 −5,580 至 5,723 ppm。后者是短窗口的复相位漂移读数，结合 0.455 rad 的拟合误差用于描述失配。Residual 对应的中位数和分位区间接近。两条路线预测墙钟分别为 20.51 和 20.54 秒，含 R69 图启动、识别和数据搬运。工程控制训练参数及优化步数均为 0。本轮不具有逐包独立空闲观测，其估计误差以受控合成验证和真实窗口的拟合残差呈现。

## 交接

N607 的 `predictions/complete.json` 存档各场景估计量分位数与耗时，`source-scores.json` 存档完整评分。本轮是合法源 V 的单模型种子工程证据，正式 Main split 三种子双路对比仍需独立闭合。
