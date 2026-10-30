# 外部对照 Phase1 七场景同样本补测

- run_id：`20261031-phase1-external-matched-nodura-m40-r01`
- group_id：`phase1-external-group1-matched-seven-views`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用10种对照各4个本轮源权重，使用PAMF原同一168000样本及固定七场景，只进行原生冻结直接识别、不接DURA；预测后独立truth-last评分，逐接收机及种子汇总。

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

## 本轮固定矩阵与验证

十个已绑定外部对照、2026092701–04各四模型，复用原 Group1 source.pt，保持各自默认分类/度量判决；PAMF四seed原七景独立评分直接复用。各外部模型使用原PAMF F0 cross_rx 126000及cross_both 42000同一物理索引、同一clean和六practical视图。每方法每seed全部14个预测固定后才单独连接truth；输出整体及按接收机、日期、发射机的accuracy和macro-F1，按混淆矩阵合并，计算四seed均值及样本SD。无DURA、目标适应、重训、目标驱动选择或额外头；历史未绑定的CSIL/Tweak/MoPC源模型不能替代。

模型实际load校验来源、源数据合同、种子、冻结recipe和无目标继承；预测与truth分进程。首步十个真实源模型的CPU no-query smoke，通过后自动运行；出现系统技术失败停止接纳新预测，已启动任务保留。每卡本任务至多1个worker、同卡CUDA及待执行总数小于2、剩余显存至少12000MiB；其他owner任务保留，CPU scorer全局最多1。

独立P0/P1审查发现已修正增量profile检查与现有--job score-*进程识别；两个修复完成定点复核。实际运行状态以远端PID、日志、状态文件及评分产物为准。

## 发布读回（2026-10-31，香港时间）

VERIFIED：代码提交 `c3ae5704c0871f4850a30593a161e42ae9702814` 已push并独立核对远端分支；一次发布归档传输比较及远端编译通过。controller PID 1002746，CWD和命令行独立读回一致。十种实际source checkpoint的CPU source-only无query smoke全部PASS，target/truth读取0。首个队列状态为40预测、40评分排队，GPU worker 0；此前其他实验继续运行，实际等待资源。该状态不是预测或评分完成证据。
