# 恢复性能主线：多尺度注意网络与匹配TF，原源二十份模拟、从零单阶段30轮

- run_id：`20261101-phase1-Restore-MSAT-M20-F0-E30-s2026092701-r01`
- group_id：`restoration-ground-TriSSL-larger-prior-m20-source`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

以源端已知信道诊断为依据，比较更强波形先验；匹配数据/损失/单阶段30轮，实际地面TriSIGTIME SSL原Main split全七景逐row自动评分。

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

## 性能优先的匹配结构检验

主基准是原地面训练完整 Tri-SIG-TIME SSL、原默认头与原 Main split 全量七景。六景各距 raw clean≤3pp 并保持 clean，性能为主、效率为支线。新 MSAT 4249922 参数，四尺度64/128/192/256、四个8头包内全局注意模块、全率跳连；不受 DURA 底座或小模型约束。匹配 TF_M20 为196898参数原TF结构；两者都原 L_s/U_s 从零联合30轮、无G祖先或IQ教师。相同R07二十份源模拟（E21..30重复1..10）、同shuffle/损失/actualF0指导/LR和预算。源V/OE/novel/目标不训练、选模或早停。

每条实际训练行都自动完成 final30 冻结→正确原 Main split 两总体七景预测→全部预测固定→独立CPU原scorer→同row报告。raw与DURA严格复用r03同IQ/ID的已验证预测；新G重新实际推理。任何源端负结果均不取消测试。R07配方为2.462GHz/六景均衡，不能改标为G10；旧TF至新TF同时改变模拟频率、混合与覆盖，不单独归因于20份覆盖。

## 启动已核实

VERIFIED：实际代码8720a8584803366d42d2879fb99502b8d220ac26；一次归档SHA比对与远端编译完成；两臂唯一P0/P1审查PASS。实际source2 smoke与严格E0重载均PASS。按启动时容量分别在GPU7/4新增一条本任务训练进程，原compute数为1/2、显存余量22883/20810 MiB，符合单卡总≤3。实际PID/CWD/命令/CUDA与日志增长已读回；其他owner保持。当前训练进行中，尚无新模型正式性能结果。
