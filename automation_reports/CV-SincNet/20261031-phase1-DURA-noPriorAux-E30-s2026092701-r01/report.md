# DURA 单阶段30轮取消中间prior IQ辅助拟合

- run_id：`20261031-phase1-DURA-noPriorAux-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-prior-IQ-loss-ablation`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

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


## 实际启动核实

VERIFIED / RUNNING。代码 eb46cab8ef9c253c193a10db1724ed9835ed2cf9，6 项聚焦检查和唯一 P0/P1 审查通过，N607 preflight、一次 archive 本地远端 SHA 比较、远端 compile 通过。真实 2Ls source 学生梯度/严格重载/原 F 默认头 no-query smoke PASS，之后立即继续。

controller 1104507、worker 1104599，实际 GPU 6；独立核实 /proc CWD、完整 argv/commit、CUDA、GPU PID 及日志增长。准入已有 1 个 compute PID、剩余 19245 MiB；此次后态 2 个 PID，本任务只加 1 个，未修改其他 owner。实际已 E1/30、247 更新，训练 F/TX/教师输入及初始化加载均 0。

固定只去除 priorIQ 辅助拟合、保留 prior 网络与物理求解。当前没有正式成绩；固定 final30 后自动七景，source 误差或 clip 变化不取消测试。前序 coarsehint 已实际结束，后续仍按真实进程检查守住评分并发，耗时单列。
