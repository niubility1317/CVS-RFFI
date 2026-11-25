# WiSig 主对比的估计监督与纯 IQ 监督 DURA

- run_id：`20261125-phase1-wisig-dura-estimated-main-m2-r01`
- group_id：`wisig-estimated-vs-blind-dura-main`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同一扩容 DURA 从零训练和同包采样，比较 IQ 估计作用/方差辅助与无辅助监督。默认源信道训练，默认和加重测试。

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

## N607 训练启动

两组训练均为 VERIFIED。源端伪监督从同包清洁及受损 IQ 估计，4 个源视图各 2,048 项，共 8,192 项。两组使用相同源 U 角色、初始化、批次顺序和 40,000 步预算。估计组在每四个源视图循环中的一个循环增加作用和对数方差损失。盲训组仅读取共同采样索引，目标文件由估计组单独读取。

| 组别 | GPU | 启动前利用率中位数 | 启动前显存占用 | 训练进程 | 已核实步数 |
|---|---:|---:|---:|---:|---:|
| 估计监督 DURA | 1 | 35% | 12,446 MiB | 3894751 | 300 |
| 无模拟器监督 DURA | 6 | 55% | 6,113 MiB | 3894752 | 300 |

两组真实源 IQ 冒烟通过，独立核对工作目录、命令与 GPU 实际进程占用。执行代码提交为 `400260cbc1f9562c79f9f2f044dac4e6aa47bafc`。计划在终止步冻结后运行无目标检查、8 个 Main split 预测分片和独立真值评分。启动时尚无正式识别结果。
