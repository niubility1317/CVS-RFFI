# LoRa 与 WiSig 共同恢复的 4096 包源域 practical 模拟训练筛查

- run_id：`20261128-phase1-shared-practical4k-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：RUNNING；WiSig 源域训练与 LoRa 源模拟池生成进行中。

## 目的与对照

将各数据集独立源 U 模拟池扩大到 4096 个物理包，以检验固定训练步数下的数据覆盖；筛查包数与规则不变。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 2ffca0776189225bb213841f07a5624aa5e72d48；N607 控制器 3968854，LoRa 3968927/GPU 7，WiSig 3968928/GPU 1。各侧源池前 16 包真实权重和教师梯度 smoke 通过，PID/CWD/GPU 与日志增长已核实；预检、单包 SHA、远端编译及一次独立 P0/P1 审查通过。

## 结果与覆盖

两侧分别使用合法源 U 中 4096 个物理包，512 clean、3584 包近似等量覆盖六种已登记 practical 模拟配置；源训练独立命名空间与随机种子，不使用目标信号、统计、V 或 U 真实 TX。共同 5% 网络、损失、1000 步及各数据集独立权重，LoRa 实增益/WiSig R17 锚点选择基于已曝光源 V。与 1024 包版相比仅扩大源池，实际选取的源包随无放回抽样规模变化；WiSig 1024 包版为 334/360，但半区 +3/-1 未晋级。预定 LoRa/WiSig 教师样本前向 12008/72032，含 smoke；模拟 CPU 和教师／专家祖先成本另计，筛查待完成。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
