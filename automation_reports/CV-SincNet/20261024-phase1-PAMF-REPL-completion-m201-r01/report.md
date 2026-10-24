# PAMF-REPL 全部当前论文实验与匹配对照

- run_id：`20261024-phase1-PAMF-REPL-completion-m201-r01`
- group_id：`phase1-pamf-repl-complete-20261024`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

以选定 PAMF-REPL 替换当前半监督方法，重做匹配主比较、三核心、四内部机制、相关 L 细项、两层结构及效率。189 新 F、12 精确复用；DURA 不训练，仅补已有 G 的同 F 读出。

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

## 发布实际状态（2026-10-24）

VERIFIED：实际执行版本 `dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f`，修复发布 `20261024-phase1-PAMF-REPL-control-r02`。源码已push并独立核对remoteOID；单发布包传输SHA一致、远端编译和隔离导入PASS。

真实同契约E20 checkpoint源侧smoke PASS：768 L前向、256隐藏标签U、6学生视图、全部REPL机制有效梯度；target读取0/U真值读取0，无正式更新。

独立核查八卡共24个GPU worker，各卡不超过3，计入全机其他进程；16个同PID日志跨137.1秒增长。实际快照见launch_verification.json，不把快照当固定当前数。

所有当前F/读出/统计/时延及论文图表交付仍在执行；不能把发布或smoke当性能完成。DURA新增训练0，IQ恢复缓存新增0。189新F通过84唯一E20前段共享减少重复，目标分数不回流训练、参数、队列或选模。

## CPU 评分入口恢复（2026-10-25）

VERIFIED：恢复版本 `a79bdf385489cb3822b37434a07e3c43e5ce839a`、`20261025-phase1-PAMF-REPL-control-r03`。只有CPU入口和owner更换；GPU科学版本仍为 `dacafc8bfd7d3ef16e6f648cdc4491b65133ae8f`。

两失败CPU score在连接预测/真值前误调用模型初始化；本地复现，4项聚焦测试和一次限定P0/P1问题复核PASS。仅严格匹配完整dispatch argv/CWD的停派owner被替换，所有子进程保持或自然结束。旧失败/日志/预测保留，两同预测新评分均SCORED_COMPLETE。

初查13条健康训练中11条已自然完成、2条仍同PID。恢复后17个同PID日志跨244.6秒增长，本run每卡≤3，未解决故障0。

资源补充：核查期间GPU4在本run三进程之后新增其他账号进程，形成暂时四进程共驻留；非本run第四槽。保持全部健康进程，不使用管理员或结束其他owner；既有每卡/全机容量guard阻止受影响卡新派发。实际分布见verification，不声称全机始终每卡≤3。

历史735阶段含2个被替代评分；有效仍733，228预测/228评分，201逻辑F/189新F/12复用不变，DURA训练与新IQ缓存均0。当前计数是读回快照，不是全组完成声明。见score_entry_recovery_verification.json。
