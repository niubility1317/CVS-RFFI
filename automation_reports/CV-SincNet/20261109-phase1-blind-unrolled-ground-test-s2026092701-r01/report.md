# 盲信道联合展开恢复七景测试

- run_id：`20261109-phase1-blind-unrolled-ground-test-s2026092701-r01`
- group_id：`phase1-blind-unrolled`；类别：`cvs`；阶段：`Phase1-restoration`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅输入接收 IQ，联合估计 33 抽头宽线性循环信道和四步残差恢复，不以 DURA 为训练或推理基座。

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

## 固定方法与范围

新建 8365004 参数的独立恢复网络，输入只有单条接收 IQ。网络联合预测 33 抽头复数宽线性信道及镜像分量，利用可微前向／伴随算子和四步独立神经更新输出单条 IQ。循环、时不变的短抽头信道是求解近似，不声称实际星地信道满足该模型或盲分解唯一。初始信道为单位响应、更新为零。

固定 32768 条合法 L_s/U_s 物理 ID × 三个源域模拟视图 × 20 轮，batch32，AdamW 0.0003、权重衰减 0.001；目标是源配对 IQ 归一化误差与 0.02 倍预测信道对接收 IQ 的归一化重投影误差。训练不读取 DURA 权重或输出、识别器权重或输出、TX 标签、V、目标 IQ 或目标真值；最终轮直接冻结。独立无查询源 smoke 与正式评测仅为核对比较，额外计算 DURA 基准并运行冻结识别器。

GPU6 每阶段不少于 8192 MiB 空闲、少于 3 个 GPU 进程，本任务同卡最多一 worker。真实源 IQ 训练 smoke 和 112 条源记录冻结基准 smoke 后，自动训练、完成 clean 和六个 practical 场景预测，再独立 truth-last。固定 Tri-SIG-TIME SSL Main split 默认头，每景 168000 条，报告 DURA 配对纠错、TX/RX/day 和 raw clean 相对 3% 与绝对 3 个百分点门槛。历史地面模拟测试已曝光，不作独立盲测或真实卫星声明。

代码 9a27afa61f60a20c66bb40b441402698baf29d82；确切命令、资源、输出和停止规则见 experiment.json。项目 Python 中 6 项聚焦检查通过；独立 P0/P1 静态审查无阻断。真实 GPU 预检和正式识别结果待核实。
