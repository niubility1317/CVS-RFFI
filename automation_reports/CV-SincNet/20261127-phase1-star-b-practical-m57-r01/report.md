# STAR-RFFI B 与原版 STAR-RFFI 全矩阵配对

- run_id：`20261127-phase1-star-b-practical-m57-r01`
- group_id：`phase1-star-b-practical-matrix`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用 B 单种子及全部 57 个原版对照，补齐 56 个同条件 B 训练与评分行。

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


## 配对矩阵与判定

- 35 个训练设置，共 57 行。11 个主实验设置各 3 个配对种子，共 33 行。补充实验共 24 行，由 3 档总标签预算、16 个等样本量接收机组合和 5 个日期留出或前推设置组成。
- 复用 Main split 种子 2026092701 的 B 组冻结评分。其余 56 行从零训练。57 个原版 STAR-RFFI 行全部按原数据角色、种子、增强视图和测试物理记录配对。
- B 组固定从源域有标签 clean 样本学习同发射机跨接收机对比。权重 0.20，温度 0.15。后期真实标签 CE 系数 0.30，100 轮训练，其余使用匹配原版 STAR-RFFI 日程。复用旧增强缓存和源域匹配的 DURA-LU。
- 主指标为纯跨接收机直接识别六景均值。逐组报告 B 与原版的同种子准确率、宏平均 F1、差值及误差。主实验保留 DURA-LU 对照，补充实验沿用原来的直接识别。日期相关结果按原口径存档。
- 预期产物为 Main split 三种子快照、全矩阵八列对照、B 与原版逐种子对比和统一跨接收机汇总。旧 B 种子为先导实验已观察结果，扩展后的匹配结果按实际数据划分和种子解释。
- N607 已核验全部 57 个原版评分和源域增强缓存。独立审查未发现直接阻断项。发布后先执行旧 B checkpoint 源域冒烟，再启用最多四个 GPU worker，每卡一个本轮任务。原版模型无需重训。
