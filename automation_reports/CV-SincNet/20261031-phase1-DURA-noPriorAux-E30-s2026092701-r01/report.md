# DURA 单阶段30轮取消中间prior IQ辅助拟合

- run_id：`20261031-phase1-DURA-noPriorAux-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-prior-IQ-loss-ablation`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

唯一损失变化为priorIQ辅助权重.1→0，两个prior/四CG和图266951保持；从零E30/7410，0教师/F/TX/V/target优化，自动完整七景。

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

旧source阶段诊断显示，默认DURA的求解前priorIQ不如noVar像clean，最终恢复却更好。跨模型/预算比较不能证明该辅助项有害，因此本次做同图同30轮单项消融：仅将priorIQ辅助loss.1设0；最终IQ、STFT.05、sourceaction.05及两个prior/四CG保持，prior误差继续记录。没有删除物理求解，没有教师或G权重继承，优化不读F/TX/V/target；固定final30全部七景，不据源误差增长取消测试。

使用授权GPU6一worker；当前GPU4 coarsehint保持健康。可并行训练，随后精确等待旧coarsehint模块/commit/root/CWD的真实worker与scorer句柄退出，再profile/predict/score以保持本goal CPU评分最多1。等待不依赖状态marker，不停止或热改前序；队列耗时单列。E30physical是优化控制参考，旧整个进程包含F指导head和成本，不冒充独立F-free控制。具体seed/数据/命令/环境/路径/资源/停止规则/产物见登记。当前检查与唯一审查通过，尚未发布或宣称效果。
