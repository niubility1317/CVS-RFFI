# 窗口尾部误差是否值得优先恢复：等能量源IQ对照

- run_id：`20261031-diagnostic-DURA-source-tail24-utility-s2026092701-r01`
- group_id：`phase1-dura-source-tail24-identity-utility`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定defaultDURA/source512×01/20，尾24点sourceclean替换对比同L2修正能量的全窗口最近clean IQ；原F默认头，0G/TX-V-target。

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


## 固定源位置效用比较

原源inverse诊断中，最后24点单位采样误差约为前232点的3.20倍。这里检验该位置是否值得优先用于身份保持，不把高IQ误差直接当成重要指纹。固定defaultDURA/source512（128 L_s/384 U_s）×缓存01/20，四例为native、仅尾24点替换sourceclean、同修正L2能量的全窗口最近clean投影、allclean。没有改变片段位置或长度。

令e=clean−native，et仅保留尾24点，第二修正为et，全窗口控制修正为(||et||/||e||)e。两者逐条具有相同L2修正能量，后者是该球内距sourceclean最近的IQ；零误差或零尾部误差不添加无关修正。这个source oracle不适用于部署，不是算法性能上限。allclean与自身F一致512只是阳性自校验，不是真实TX100%。

两视图尾部修正均须较native至少多4条与clean F选择一致、较等能量全窗口至少多2条；每源RX和clean不低于全窗口、每非clean景最多低1条，完整F特征误差均值不高于native。通过仅支持另登记边界结构研究，不自动训练；失败不改片段或阈值，也不以描述控制替代主候选。

2项聚焦数学/支持域/等能量/零边界检查和实际模块入口PASS，唯一P0/P1 PASS。首步真实2Ls原F完整特征/原默认头与缓存匹配后立即继续。CPU2/空CUDA；仅两个新IQ例跑F，主完整F2048/原头4096，smoke另F2/原头2；0G权重读取/前向/模型优化/TX/V/target。F50既有50轮/11100源监督与历史G/缓存成本继续披露，F全state及grad检查。一次preflight/归档SHA比较/远端编译/实际PID-CWD-argv-CUDA日志和独立全数组复算；默认与完整3pp/clean/同质量效率目标保持。
