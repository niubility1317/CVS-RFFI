# 紧凑恢复器从零30轮：受损源样本最终IQ教师指导

- run_id：`20261031-phase1-DURA-nativeIQhint-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-final-IQ-teacher`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

G266951从零单阶段30轮，noVar四基础项仅新增非clean源样本最终R17 IQ MSE权重1；clean教师项0、参考detach、部署不读scene/teacher。

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

紧凑G266951从零单阶段30轮/7410，同noVar四基础项；只新增非clean源样本最终原R17 IQ MSE权重1，clean教师项0，参考与clean能量detach，batchmean，不做active-only放大。配方已在945114bf4、wide正式分数读取前固定。最终stage3缓存20×63000已独立核实IDs/roles/全数组finite/teacher不变；循环972.193秒、全制备976.383秒、1.26m输入+2smoke/2.58048GB、历史60组件轮次另计。学生实时teacher加载/前向0，参考访问1.89m/active实记；不复制teacher权重、不读F/TX/V/target优化，既有sourceF/V选择背景披露。仅received用于部署，source scene mask不进入推理；2prior4CG27FFT不变。

8聚焦检查与唯一P0P1 PASS，当前固定N607 Torch2.1/NumPy2 bool合成兼容已核实。首步真实2Ls完整目标梯度/严格重载/原F默认头noquerysmoke后立即继续GPU6；准入最多3总PID/free12GB/goal2/own1card，其他owner保持。每版固定final30后source1800与target168000×7全预测再独立truth-last；source负结果也不取消。rawclean84.676190，每星地≥81.676190，clean保持/原R17逐景1pp/成本分别报告。默认保持，当前待发布，目标ACTIVE。


## 实际启动核实

VERIFIED / RUNNING，执行728129ece6855743fc4939193ab2bc3b7693b9aa。8聚焦检查/唯一P0P1 PASS，一次N607 preflight/单archive SHA比较/远端compile通过。首步真实2Ls新G最终IQ目标梯度、266951参数严格保存重载、原F默认头noquerysmoke PASS后立即继续。

controller1192588/worker1192680/GPU6，实际CWD、完整argv/commit、CUDA、GPU PID和日志增长独立核实。准入已有1PID/free19245MiB，后态2PID，本任务新增1worker，其他owner保持。旧wide与缓存producer自然完成。

已3/30轮/741更新，L18900/U170100，teacher参考189000/非clean激活160614，五项加权loss独立复算一致。F/TX/实时teacher/初始化checkpoint读取0；cache句柄核对0.011453秒，原1.26m教师制备972.193282秒/2.58048GB/历史60组件轮次另计。source clean mask仅训练，部署图仍received-only。

当前只确认按固定配方训练；完整30轮、七景正式测试与实际成本尚待返回。自动全七景和truth-last保持，默认与逐景3pp/clean/效率目标ACTIVE。
