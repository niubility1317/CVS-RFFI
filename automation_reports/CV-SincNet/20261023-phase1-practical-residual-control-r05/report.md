# Full practical residual IB作者HSIC导入恢复r05

- run_id：`20261023-phase1-practical-residual-control-r05`
- group_id：`phase1-full-practical-residual-factorial`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

独立加载原作者HSIC，恢复两条失败IB F及四阻塞消费者，接管健康G且保留三张卡。

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

## IB-RFF 第二阶段作者导入恢复 r05

一次小时检查发现两条 F0-ibrff-A（seed2026092701/02）在 stage2 首个 HSIC 损失计算前失败，指纹均为 `ModuleNotFoundError: No module named 'utils.hsic'`，已按规则停止新派发。源侧复现确认 G 的常规 `utils` 包遮蔽 IB 作者无 `__init__` 的同名目录；不是数据或数值机制变更。

修复只按已登记作者路径独立加载原 `hsic.py`，不替换全局 G utils、不改 HSIC 实现或0.04权重。Torch1.11真实同seed第一阶段 checkpoint与12条合法L样本检查 loss/gradient 与直接作者函数逐元素一致，target/Utruth/optimizer update均0。16项聚焦测试PASS，限定独立P0/P1恢复审查PASS，无未解决P0/P1。

新不可覆盖 control-r05只替代2失败F和4阻塞预测/评分，共6阶段；2条失败F保持同scratch配方/seed/最大预算从零恢复，旧24/28轮stage1、2600更新及约250.31共享worker秒作为技术失败部分成本保留，不混入正式模型成本/评分。130条未启动IB F只换执行release，其它健康、已完成或未启动非IB均不改。原6个完整F、2恢复IQ缓存、4条健康G和producer原PID/CWD/config保持。

科学矩阵仍600新F+4完整G、627读出、1880有效阶段；历史1892阶段、12技术排除。新旧映射由current jobs.json/technical_recovery.json及previous_recovery指针确定。GPU0–4且每卡总compute<=3，5/6/7继续保留，不占其它owner。发布前不能称已恢复GPU训练；实际结果需独立PID/CWD/argv/GPU/log增长核实。
