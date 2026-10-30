# 单阶段30轮强频点相位保真恢复

- run_id：`20261031-phase1-DURA-strongPhase-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-strong-phase`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

noVar四基础项保持，仅sourceclean最强64FFT点的IQ误差相位分量权重1→2；从零E30/7410，图266951不变/优化F-TX-teacher-V-target0，自动全七景。

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


## 最小候选说明

源频点幅相oracle显示noVar最强64点phase-only理想修正，使两份缓存的cleanF选择一致478→488/496→502、移除约30%IQ误差，但一景mid仍−1。这不是TX准确率、部署oracle、旧globalphase资格或目标收益证明。故只做一个独立从零E30：在noVar四项损失上把该精确相位误差分量加一次（权重1→2），没有按目标景/类别定权或扫参。sourceclean最强64点mask仅训练使用；推理图不增加FFT/模块。原IQ/prior.1/STFT.05/action.05保持，varianceaux0；实际优化F/TX/教师/V/target0，研发选择F-informed披露，固定final30全七景，不按源误差提前取消正式测试。8tests/唯一P0P1 PASS，首步真实2Ls梯度/strictreload/F noquery PASS后直接继续；GPU4/6/7原授权容量，旧case已自然退出、他人保持。当前待发布，未宣称收益。
