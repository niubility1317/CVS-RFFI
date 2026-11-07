# 物理步长源域交叉拟合诊断

- run_id：`20261108-phase1-physical-line-crossfit-source-s2026092701-r01`
- group_id：`phase1-physical-line-crossfit-source`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

在两个互不相交源记录折上交叉拟合单一校正系数，测量源 IQ 波形误差；不产生部署模型或目标预测。

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

## 方案与运行

固定已完成的物理步长诊断、DURA 和源训练第 20 轮修正器。576 条受损源记录按每个接收机组／场景各 24 条平分两折；每折的全局单一混合系数由另一折的合法 source clean 波形误差闭式拟合。只做源域交叉拟合分析，不保存或部署模型，不读取 F、TX 标签、U_s 真值、V 或目标。样本曾用于探索，DURA 是否见过未知。

代码 commit `8aeae433a5da8697159958cc97d226b5cc47be27`；两项针对性测试、独立 P0/P1 审查、一次远端归档校验和编译通过。N607 CPU 2 线程，隐藏 GPU，后台 PID 963139；真实源数据首批 8 条检查通过。

- 已完成全部两折源域交叉拟合，[完整结果](analysis.md)。RX8 六景均值均优于原物理步长；RX4/6 四景变差，其中两景均值高于 DURA。本结果不是 TX 准确率，不保存或部署模型。
