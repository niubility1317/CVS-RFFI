# LoRa 与 WiSig 共用逐包专家对齐实增益恢复筛查

- run_id：`20261128-phase1-shared-aligned-gain-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；WiSig 源域训练与 LoRa 源模拟池生成进行中。

## 目的与对照

以逐包最小二乘复数标量将另一恢复专家对齐到源选择锚点，再对差分应用共享逐频实增益和 DURA 物理残差。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 e31d925df61862d8d860c74e53fdc81a05f2d53b；N607 控制器 4030358，LoRa 4030426/GPU 2，WiSig 4030427/GPU 3。两侧前 16 包真实专家与教师梯度 smoke 通过；PID/CWD/GPU 及日志增长核实。预检、单包 SHA、远端编译和独立 P0/P1 审查通过；对齐仅用于差分的审查问题已修复并定向复核。

## 结果与覆盖

每个数据集采用同一逐包最小二乘复数标量，把另一专家相位和幅度对齐到锚点后形成差分；原始专家功率特征和 DURA 伴随物理残差保持，预测的逐频增益为实数，cap 5%。各自独立权重，LoRa 实增益/WiSig R17 锚点来自已曝光源 V。训练使用同一合法源 U 4096 包低城区加权模拟及用户授权冻结教师，训练和推理对齐规则一致；不读 U 真实 TX、V 或目标参与训练。教师样本前向预定 12008/72032，含 smoke；源模拟与专家/教师祖先成本另计，筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
