# LoRa W24 实收增益联合训练

- run_id：`20261126-phase1-lora-w24-trained-readout-s2026112991-r01`
- group_id：`phase1-lora-w24-restoration-performance`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

匹配训练检验实收噪声驱动的中心修正强度，固定 Full 七景。

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

## 预登记方法与对照

固定双独立先验、四次 warm-start CG 和原生边缘。两臂从零训练，使用同一随机种子、20,000 个源 U 拟合包、500 个源 U 留出包、通用扰动和 AdamW 日程，批量 16，固定 3,600 步。实验臂将实收带外噪声估计转换成逐包增益，并接入最终中心修正。已有末级全量辅助损失同时作用于缩放后的终点。对照臂保留原训练终点和固定半量推理。源域全量消融对应各自有效终点。训练及冻结阶段读取 W24、TX、V 和目标均为零。

冻结后在同一 300 包 Full 源 V 面板上独立评分。实验臂的部署读出须在六景优于直通、匹配对照及 40,000 步原半量，低城区至少持平三者，clean 至少持平直通。达标才自动运行 30,000 包 Full 七景，对全部冻结预测作本机独立 truth-last 评分。

9 项定向测试通过。独立审查未发现 P0/P1 问题。

## N607 发布与启动

2026-11-26 12:39 HKT，发布提交 `bc55eef4deb9a60335034de185957dead85438d3`。本地与远端归档单次比对通过，远端编译通过。通过控制器、训练进程树、实际工作目录、命令行、GPU 分配和日志增长独立核实。匹配对照在物理 GPU 0，实收增益组在物理 GPU 3，各占约 3,896 MiB，均已完成源 IQ 冒烟并推进正式训练。两臂各有 1,271,038 个参数，两个先验梯度均为有限非零数。冒烟记录 W24、TX、V、正式 practical、仿真轨迹和目标的实际读取计数均为零，使用 0 个校准包。

当前为源域训练。已启动独立本机收集评分进程，仅在预登记源域规则通过且三个 Full 分片完整冻结后读取正式真值。Run 未创建 Residual 评估，不访问历史 Residual 预测作为本轮输入。
