# reference_response：跨TX共同位移补偿×分支贡献预测，CE拼接四seed消融

- run_id：`20261008-phase1-receiver-residual-manysig-m16-r02`
- group_id：`cvs-cross-tx-displacement-branch-contribution`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：RUNNING（实际状态按events.jsonl及独立证据更新）

## 目的与对照

R2修复CPU头初始化影响CUDA dropout RNG的技术问题；保持固定强骨干、最基础CE+原生拼接、四组四seed从零E200及全部测试；R1保留为含随机混杂的诊断，不热改健康任务。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 修复与验证

R1发布后发现新增头初始化重置CUDA dropout RNG，保留为含随机性混杂的诊断；不停止或热改健康任务。R2使用独立包`experiments/cvs_receiver_residual_v2`、独立release和输出，从零执行相同四组四seed，不继承R1权重。仅修复CPU私有生成器，算法、损失、训练预算均不变，设计在R1任何target读取前固定。

已验证四组CPU与CUDA随机数状态均保留，CUDA全机制smoke通过（初始骨干/logits逐位等价、分支和、非零有限辅助梯度、逐样本推理、checkpoint、原生拼接边界）。新增参数为0/5578/2173/7751。实体配置与登记验证通过，复用同次独立P0/P1核心审查，并针对初始化修复作实测验证。

完整机制与固定系数见[DESIGN.md](../../../experiments/cvs_receiver_residual_v2/DESIGN.md)。每GPU最多2个总任务，R2只使用剩余合法名额；不改R1或其他健康任务。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 远端启动VERIFIED

正式R2执行commit为`8a88b3a2e369644ea1d746c5e02006218ee00df6`，独立读回的dispatcher PID1609832，cwd为`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_receiver_residual_v2_20261008_r02`。远端CPU smoke PASS。首批baseline/displacement/contribution/combined的PID分别为1610259/1610264/1610269/1610275，GPU4/5/6/7，日志增长、源训练epoch及resolved_config均已核实；12行排队。每GPU不超过2个总任务，不停止或热改其他健康任务。

当前尚无冻结后独立测试结果，不能宣称收益。远端单owner自动完成全部16源训练、源冻结、112个完整视图预测、truth-last评分与重算。Windows本地一次性收尾同步器负责回收结果、更新本报告/登记/索引、Git提交push并独立核对远端OID；技术失败保留产物并记录准确状态，不自动重跑训练。

[草稿PR #8](https://github.com/niubility1317/CVS-RFFI/pull/8)仅包含本轮实现与修复，未合并。独立PR读回确认head/base均为已核实开发分支。恢复优先读取`local_artifacts/cvs_receiver_residual_v2_20261008_r02/readback.json`和`completion_sync_owner.json`，不得重复发布或启动第二个owner。

## 每卡4进程调度覆盖

用户新增指示“每张卡4个实验进程”，将本轮R1/R2总任务上限从每卡2个覆盖为每卡4个，包括预CUDA预约。原启动记录和策略保留为历史。新控制release只替换本轮调度父进程，按PID/start_ticks/CWD/argv接管训练子进程；已完成行跳过，失败行保留且不自动重跑。每run单launch owner，两个调度器共享GPU预约锁。训练模型/config及预算继续使用原不可变release。source矩阵全部冻结后再预测，全部预测完成后truth-last独立评分。此处为预登记，实际接管和占用须由独立读回确认。


首次容量发布部分完成：R1已接管并启动12行；R2旧dispatcher精确SIGTERM后已独立absent，但/proc退出瞬间environ权限读取竞态导致发布器中断。训练worker未signal。恢复先核实已完成部分，控制release r02修复退出检查并接管剩余状态，不重发原提交、不重跑训练。


## 容量接管VERIFIED

独立/proc身份与nvidia-smi、完成产物核实：新dispatcher PID1641985，控制commit`42c657c3757b61e7a9fb8c71ccd8d9e8c433c9d7`；模型运行commit仍为`8a88b3a2e369644ea1d746c5e02006218ee00df6`。保留0个活跃训练PID/start_ticks/CWD/argv；首seed4行已自然完成E200，未重跑。目前12行在训、0行排队，无失败。

GPU0至3各3个总实验进程，GPU4至7各4个；其中本轮R1/R2共24个，另4个既有任务。每卡已设最多4个，含预CUDA预约；显存余量约19至21GB。固定矩阵剩余任务已全部启动，因此不补开额外实验。证据：[capacity4_verified.json](evidence/capacity4_verified.json)。

新调度器使用专用handoff锁及全生命周期owner锁；跨run共享GPU预约锁。原训练PID不signal，不热改模型，不改变预算/损失/数据角色。完成16源模型后继续原冻结→112预测→独立truth-last评分。当前未完成目标测试，尚无新增机制收益结论。

本地旧observer PID62900退出已独立核实；唯一新observer PID60084已读回正确的新dispatcher，日志增长且stderr为空。其任务仍为结果回收、报告/登记/索引更新及Git交付，不调参、不重跑。

## 远端技术失败

FAILED：{"status": "FAILED", "error": "KeyError('classes')", "no_retry": true}

产物保留，未自动重跑或干预健康任务；完整训练/测试收尾尚未完成，不能宣称收益。
