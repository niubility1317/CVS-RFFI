# 原R17最终IQ源域教师缓存预制

- run_id：`20261031-source-DURA-R17-nativeIQ-cache-s2026092701-r01`
- group_id：`phase1-dura-original-R17-native-IQ-cache`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅渲染原冻结R17最终第三stage IQ，原R07源20视图各63000，0更新/F/TX/V/target；可复用缓存2.58048GB，不是新网络或正式目标预测。

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


## 最小缓存预制说明

仅源数据准备：原R17最终第三stage IQ，20份原R07固定received各63000，不新训G/F、不读取TX/V/target、不生成正式target预测。旧source evidence native IQ误差.01737/.01643优于noVar .02235/.02112，cleanF选择多10/7条但一景仍−1，只作研发代理，不是实际TXaccuracy或目标保证。原R17在目标访问前按source固定且已获IQteacher授权；不用当前目标选出的新G作教师。

CPU controller首步真实2Ls原G noquery/state检查后，GPU4按现有容量锁准入，和GPU6宽coarse并行总最多2worker。B128渲染1.26m输入+CPU smoke2，2.58048GB数值缓存；历史coarse/action/prior各20轮另列。当前新增训练0，教师只读，最终完整计数和状态不变才写cache complete，CLI无score/target入口；不覆盖旧artifact、不停止他人。7checks/唯一P0P1 PASS，当前待发布，缓存成本实测后报告，未自动启动学生或宣称恢复性能。
