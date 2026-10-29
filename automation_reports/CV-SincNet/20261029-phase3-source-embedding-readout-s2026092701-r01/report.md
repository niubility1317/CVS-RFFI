# PAMF第三阶段：固定特征增量与跨分支拒识读出组合

- run_id：`20261029-phase3-source-embedding-readout-s2026092701-r01`
- group_id：`pamf-phase3-source-embedding-readout`；类别：`cvs`；阶段：`Phase3-source-fixed-embedding-plus-open-readout`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定两种真实E51encoder，各重导8分支/重建L几何，以相同原4OE4200训练100D读出；MLP预指定、原F保护、完整两行选择监督encoder低成本优先。

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

实际 code/config commit：`806dec253b7556ee3d4af842bf9a2d8fd714009e`。新候选30项聚焦测试及2个subtests通过，一次独立P0/P1 PASS。真实E51checkpoint L2 smoke作为导出首步，PASS后继续；两编码器各40652次样本前向，编码器/原6类head新增更新0。各自L-only新几何，原4OE4200和L6300仅训练匹配二分类读出，各200步；V只阈值、novel只冻结后预测评分。仅MLP可晋级，完整两行后选择；原F6同输入保护。N607一次preflight、既有数据复用，每卡新增≤2、不停止foreign。

## 实际结果

VERIFIED / ANALYZED_SOURCE_ONLY，NO_PROMOTION。两个MLP未知宏拒识86.3492%／90.7619%；OE行FRR1.5556%、open97.7407%、相对原F总损伤0.7259pp。全部7方法的known保护总体和每RX均通过，两个MLP仅RX1未知达到95%，eligible空/selected=null。四条200步loss共800点、8stage日志、realL2smoke完整；GPU2/1实际owned PID与退出读回已保留，foreignstop/target0。完整结果见 docs/PAMF_PHASE3_SOURCE_EMBEDDING_READOUT_RESULTS_20261029.md。同novel重复source，不是跨RX星地/适应/注册成功；不追改此前MSP或本轮阈值。
