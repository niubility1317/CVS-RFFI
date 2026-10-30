# 独立纯恢复30轮重复视图配对对照

- run_id：`20261031-phase1-DURA-repeatViews-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-source-draw-count`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

纯G266951 scratch单阶段E30/原20源抽样后复用01..10；匹配fresh的G初始化/CUDA探针seed2026092701/np shuffle2026103004，补充旧增强分支改变CUDA种子的对照缺项。

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


## 最小独立对照说明

首轮源曲线已不同于旧physical，由N607 Torch2.1纯CPU种子trace确认：G643个初始state完全一致，但旧enhanced factory额外torch.manual_seed2026103107重设CUDA，fresh/compact只设2026092701，旧工厂只还原CPU状态。旧七景分数真实、预算匹配，但不能当严格同探针随机数的单机制配对。旧产物/执行archive保持，历史比较限制另记。

本新纯G对照从零E30/7410/noVar4基础项/B256，model和CUDAprobe2026092701/npshuffle2026103004、LR与fresh路径一致，源20缓存后repeat01..10，只与fresh的21..30额外抽样不同。0新增sourcegeneration/神经teacher/F/TX/V/target优化，没有增强分支。目标访问前登记补齐，不根据fresh目标结果调参。3检查/唯一P0P1 PASS，真实2Ls G梯度/严格重载/原F noquery首步后GPU4准入，own1/max3PID/free12GB/goal2（freshGPU6占另1）。评价前精确等待fresh32a实际worker/scorer退出，保留一CPU scorer；固定final30自动全七景168000/truth-last/profile。默认和完整目标ACTIVE。
