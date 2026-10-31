# 条件生成流与冻结 F0 身份引导恢复网络

- run_id：`20261101-phase3-identity-flow-source-E20-s2026092701-r01`
- group_id：`phase3-identity-flow-g10-source`；类别：`cvs`；阶段：`Phase3-restoration-source`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从零训练注意力条件流，混合合法 R07、G10 L_s 和 clean 源配对；固定两条反向噪声轨迹输出；完成全部七景与 Phase3 冻结评测。

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

## 启动核验：VERIFIED

固定代码 `77b93f1b1713880f1cb1901be34cc6dce474105c` 已发布至 N607。真实 Phase3 F0 与 2 条 G10 源 L_s 样本的 smoke 通过，checkpoint 严格重载；F0 身份梯度范数 0.39553237，流损失梯度范数 3.64697480。未读取 TX 标签、V/OE 或目标 IQ。

控制器 PID 2118363、worker PID 2118460，训练子进程 PID 2118464；GPU 4。已独立读回运行路径、进程命令、GPU 占用及启动产物。训练和全部预登记的评测尚未完成，当前没有新的性能结论。


本 E20 为先导实验，不能替代完整协议要求的从零单阶段 30 轮验收；保留既定训练和全部预登记评测。E30 单独从零登记，不依据 E20 目标真值选模或更改训练。

## 先导训练完成：VERIFIED

从零完成 20 轮、9,860 次更新，实际训练耗时 808.53 秒。F0 保持冻结；TX 标签、V/OE 和目标 IQ 在训练中读取数为 0。已进入七视图预测，尚无评分；E20 不替代 E30 完整预算。
