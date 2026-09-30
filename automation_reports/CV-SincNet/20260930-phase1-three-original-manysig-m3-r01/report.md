# ASKNet、WaveMLP、域不变方法原实现单seed源域复现

- run_id：`20260930-phase1-three-original-manysig-m3-r01`
- group_id：`phase1-three-original-reproduction`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留修复后的作者实现，各一seed，源域训练/验证；不是CVS少标签匹配对比，不自动访问目标测试。

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

## 本次数据和checkpoint结论

从零训练。ASKNet/WaveMLP使用非均衡Day1/2首100包和70/30源域划分；DIFL使用均衡Day1随机200包和80/20源域划分。source为作者顺序RX[3:]，target为RX[:3]，两者不交。DIFL教师由同run相同source划分从零训练，source V选模，学生只加载该教师，禁止回退作者或历史权重。

该首批是原实现复现，不是CVS少标签/LEO匹配实验，不用跨方法最高值排名。目标测试未启动；source完成后状态保持SOURCE_TRAINING_COMPLETE_AWAITING_SOURCE_REVIEW。

## 启动前环境兼容修复

真实source smoke发现共享NumPy2.2.5与PyTorch2.1的ndarray转换失败，正式训练尚未启动。仅在本release隔离目录解包NumPy1.26.4轮子，通过PYTHONPATH供本批进程使用，不修改公共环境。数据导出成功并保持原文件，不重复生成。兼容包来源及SHA见experiment.json。
