# LoRa 与 WiSig 共用专家互谱相位特征实增益筛查

- run_id：`20261128-phase1-shared-phase-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；WiSig 源域训练与 LoRa 源模拟池生成进行中。

## 目的与对照

共同逐频实增益网络加入实增益与 R17 输出的归一化复数互谱，保留 DURA 物理残差和 5% 修正约束。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 991014f4446dcfb182bac36c3634bff320bd4620；N607 控制器 4004392，LoRa 4004409/GPU 2，WiSig 4004410/GPU 3。两侧前 16 包真实专家及教师梯度 smoke 通过，PID/CWD/GPU 和日志增长已核实；预检、单包 SHA、远端编译和独立 P0/P1 审查通过。

## 结果与覆盖

两数据集共用逐频实增益和 DURA 物理残差网络，新增归一化专家复数互谱的实部与虚部，训练/推理 cap 均 5%；分别训练权重，以已曝光源 V 选定的 LoRa 实增益及 WiSig R17 为锚点。每侧合法源 U 4096 包，512 clean、1792 低城区，其余五景各 358 或 359；源训练独立模拟随机流，不读 U 真 TX、V 或目标训练。冻结 W24 E104/Tri-SIG-TIME SSL F0 仅训练时使用，教师样本前向预定 12008/72032（含 smoke），模拟及专家/教师祖先成本另计。源 V 多轮曝光，筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
