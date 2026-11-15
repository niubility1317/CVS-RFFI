# Tri-SIG-TIME 额外消融的显存故障恢复

## 故障与固定恢复范围

原批 `20261115-phase1-pamf-repl-more-clean-m16-r01` 的六源 `weight45` 在累计 E47 的源验证申请 328 MiB 时发生 GPU OOM，当时卡上剩余约 166 MiB。该行未冻结 E50、未预测目标；其自身 E30 已完成 2220 步、源代理和权重。停止新派发但保留其他健康 worker，等待其自然完成。已完成的 6 行 clean 预测和独立评分保留，不重新测试。

恢复其余固定 10 行：5 行复用原批各自合法冻结 E50，只做尚未开始的 clean 预测与独立评分；失败行从其自身完整 E30 续训 E31–E50，失败的 E31–E47 轨迹保留，不读取其目标结果；4 行尚未开始，从原相同契约 E20 正常训练 E21–E50。候选、超参数、seed、数据、L/U 损失、checkpoint 固定规则和测试范围均不变。前一次显存事故意味着失败行会额外花费部分训练计算，不能声称它只进行了 30 个实际新增 epoch。

使用 GPU 0、1、3、4，与原批仍运行的 2、6 分离；本批每卡最多 2 个 GPU worker、每卡全进程最多 5 个，并且预留至少 10000 MiB 及未初始化 worker 的额外空间。旧 owner 必须自然结束且状态与固定恢复矩阵一致后才能派发。新的独立 root、一次真实权重无 query 冒烟、固定 clean 三集合、逐行冻结、独立 truth-last 评分；U 真值仍隐藏，V 置信精度仅是代理。

该恢复仅依据技术状态，不依据目标成绩调整或选取候选。原选种子曾依据已曝光目标，所有分数均为探索证据。

# Tri-SIG-TIME OOM 后未完成矩阵恢复

- run_id：`20261115-phase1-pamf-repl-more-recovery-m10-r02`
- group_id：`phase1-pamf-repl-weight-clean`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

保留原矩阵已完成6行，恢复其余10行：5个冻结E50复用，1个自身E30续训，4个从E20启动。

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

## 发布与恢复核验

**VERIFIED / RUNNING**：执行 commit `32101c4c1ae6505277582316d64740739cce0942`，归档传输校验和远端编译通过。旧 owner 已自然结束并核对唯一 OOM；10 个真实权重、无 query smoke 全部通过，失败行自身 E30 的 2220 步及血缘核验通过。5 个复用 E50 的预测与独立评分已启动，失败行和未开始行已在 GPU 3、4 启动并观察到增长的逐步日志，余下任务自动排队；每卡本批最多 2 个 worker。

## 已复用的 E50 clean 结果

5 行均沿用原批自身 E50，未再次训练；已完成新的 clean 预测及独立 truth-last 评分。主指标合并 cross_rx 与 cross_both。

|候选|clean|相对普通 PL|相对原 Tri|
|---|---:|---:|---:|
|十类 style0|82.6452%|−1.5542 pp|+0.0536 pp|
|十类 kl05|82.5306%|−1.6688 pp|−0.0610 pp|
|六源 weight075|86.0771%|−0.3514 pp|+0.9278 pp|
|六源 shallow0|85.3493%|−1.0792 pp|+0.2000 pp|
|六源 shallow025|85.4257%|−1.0028 pp|+0.2764 pp|

十类 style0 的完整 E21–E50 两段共 6660 次 U 调用，`time_down` 和 `t2.pool` 的风格应用次数均为 0，和其配置一致。所有结论限曾按目标选取的单 seed，不能作为独立盲测或跨种子结果。
