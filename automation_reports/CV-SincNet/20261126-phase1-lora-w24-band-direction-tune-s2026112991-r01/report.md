# LoRa W24 半量读出 9–32 kHz 方向监督（Full）

- run_id：`20261126-phase1-lora-w24-band-direction-tune-s2026112991-r01`
- group_id：`phase1-lora-w24-restoration-performance`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（Full 源域续训）

## 目的与对照

Matched continuation from the verified 40k outer-loss source model. Only the half-edit 9-32 kHz direction loss differs.

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


## 预登记对照

原生 W24 的等能量源域诊断显示，替换 9–32 kHz 的复数方向，六景净增 62 次，低城区净增 13 包。两组加载同一个从 LoRa 合法源 U 固定训练 40,000 步的首尾联合损失权重，重置优化器并沿用原末期联合损失权重新增 1,800 步。方向损失权重为 0 和 0.10。新项只用同条源 IQ 的两份独立通用扰动与 clean 配对，在实际半量回写的去 chirp 坐标比较 9–32 kHz 双侧复数方向。缩放恢复到逐包能量尺度，并排除 clean 恒等与自然差异探针。两组保持原损失、双阶段四次 CG 和原生实收边缘 IQ。

冻结后用 W24 E104 跑同一 300 个源 V 包的 Full 七景。候选六景须超过直通、匹配对照、40,000 步通用半量和首尾父版半量，低城区至少持平四组，clean 持平直通。父版源预测须逐包再现。全量输出在源域作为诊断记录。通过即自动跑 Full 的三个正式分片和独立评分。

## 发布核验

执行提交为 `bc95ba36c4b2de43e47f2254cdfd68f15492e2da`。21 项定向检查通过，独立 P0/P1 审查无阻塞。发布归档比对 VERIFIED，远端编译通过。控制器 PID 804788 及训练 PID 805489、805484 的命令、工作目录与登记一致。对照与方向组分别在物理 GPU 7 和 6，各用 3,898 MiB。两组严格加载同一份 40,000 步源父权重后通过真实 IQ 无查询检查，参数量均为 1,271,038，两个先验的梯度均有限且非零。拟合检查记录中 W24、TX、V、目标及仿真轨迹读取为 0。实际训练日志已到第 50 和第 100 步。独立本机收集程序已启动，只有达到源晋级规则且正式预测固定后才评分。
