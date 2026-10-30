# 单阶段30轮强频点相位保真恢复

- run_id：`20261031-phase1-DURA-strongPhase-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-strong-phase`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

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


## 实际启动核实

VERIFIED / RUNNING，代码84c5f08b3f334781cad4cb96f9e22bd33d9000eb。8项聚焦检查/唯一P0P1 PASS，一次N607 preflight、单archive SHA比较、远端compile PASS。真实2Ls源G objective梯度/严格重载/原F默认头no-query smoke PASS后立即继续，没有额外批准步骤。

controller1134232/worker1134322，GPU4；/proc实际CWD/完整argv/commit/CUDA、GPU PID和日志增长独立读回。准入已有1个PID/free22883MiB，当前后态2个，本任务1worker，其他owner保持。当前E1/30、247更新，日志617字节。

实际phase项非零（0.23589571），Q1clean能量占比0.88471944，总loss按登记五项复算一致；F/TX/学习教师输入和祖先加载0，实际新损失已经进入训练。该检查不证明正式收益；固定30后自动source1800×7、target168000×7及独立truth-last。当前没有新目标成绩，部署默认与完整3pp/clean/效率目标保持。


## 完整训练与正式七景结果

VERIFIED / ANALYZED。E30/7410、七景与126组独立复算完成；clean84.436905%、lowurban67.311310%。较同预算physical，mid/low urban+0.347619/+0.604167pp，其余四景及clean下降；六景仍低于默认，3pp/clean未达。训练1002.686秒/全流程1187.862秒。完整曲线、全部分组、参数/实际时延与成本见[详细分析](analysis.md)及results.json。默认保持。
