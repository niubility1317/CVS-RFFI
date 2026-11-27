# STAR-RFFI 星地增强下的跨接收机小范围优化

- run_id：`20261127-phase1-star-cross-rx-practical-s2026092701-m2-r01`
- group_id：`phase1-star-cross-rx-practical`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（N607 控制器、独立训练进程与源域冒烟已核验）

## 目的与对照

在同一个 Main split 种子上比较两种 STAR-RFFI 跨接收机身份训练配方。

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


## 预登记优化方案

原版 STAR-RFFI 在三个 Main split 种子上的 300 轮源域日志均已核对。第 76 至 100 轮的无标签接受率分别为 96.08%、94.48%、92.32%。接受率衡量覆盖，源域日志不提供 U 的真实身份正确率。新目标直接约束默认分类头使用的 128 维身份特征。

- 两组均从零训练。同一 Main split、种子 2026092701、19,700 个更新、相同 practical 配对、增强日程及固定 E100 EMA 读出。
- 两组都在有真实标签的源域 clean 上加入跨接收机身份对比。正对来自同一 TX 的不同源接收机，权重 0.20、温度 0.15。原 practical CE 和伪标签路径保留。
- 第一组保留后期真实标签 CE 系数 0.10，第二组使用 0.30。原版 STAR-RFFI 及其余六个已冻结对照使用同配对种子直接复用，无重训。
- 三类测试的 clean 与六个 practical 场景均做冻结直接读出与同一匹配 DURA-LU 读出，按物理 ID 完成独立 truth-last 评分。
- 直接读出的跨接收机与双跨六景均值各须超过该测试划分七个历史方法的同种子最佳值至少 0.50 个百分点，且三个目标人群的等权六景均值最高。相对原版 STAR-RFFI，三类测试的 clean 和跨日期六景均值各最多下降 1.00 个百分点。DURA-LU 结果单列。
- 在共享的已知 Main split 上保留两条候选结果。单种子结果用于筛查，不自动扩种子或替换既有报告中的 STAR-RFFI。

正式运行使用 N607 的普通账户与每卡一个本次 GPU worker。独立 P0/P1 检查发现并已修复 practical 分支的新增损失权限和源域冒烟 NumPy 转换问题。定向验证覆盖跨接收机正对与退化批次的梯度、预测拒读真值、七方法匹配验收。真实归档 checkpoint 的无目标冒烟由发布控制器执行。

## N607 启动核验

2026 年 11 月 27 日 08:17（香港时间）核验。归档校验和远端编译通过，实际运行提交为 `29e66465737e3799a57ab96d71762fdf6076e87c`。旧 STAR checkpoint 只用于源域无 query 冒烟，检查通过。两组分别在 GPU 2 和 7 训练，生效参数分别为 0.20／0.10 和 0.20／0.30（跨接收机损失权重／后期真实标签 CE 系数）。两行均会在固定 E100 后自动展开预测和独立评分。
