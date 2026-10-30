# 30份源信道抽样的单阶段恢复训练

- run_id：`20261031-phase1-DURA-freshViews-E30-s2026092701-r01`
- group_id：`phase1-dura-scratch-source-draw-count`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

紧凑G266951从零单阶段E30/noVar四基础项；原源缓存01..20只读复用，新增原native namespace21..30，后10轮不再复用01..10；无神经教师。

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


## 最小源抽样对照说明

紧凑G266951从零单阶段30/7410/noVar四基础项；唯一变量是30份源信道抽样，原R07的01..20只读链接，21..30沿用同一原native生成器/种子体系/namespace规则生成，不再重复01..10。L6300/U56700/物理ID/顺序/clean单位/源RX-day与固定V面板均保持。30份抽样不是30个新物理总体，clean分支仍可能相同。新10份630000次模拟访问/4.1958GB数值字节，8CPU workers制备，实际成本完整单列；没有神经教师、F/TX/V/target训练读取，不把原R07模型当初始化。

4聚焦检查/唯一P0P1 PASS。controller首步真实2Ls G梯度/严格重载/原F默认头noquerysmoke，随即CPU制备新10份；全部nativeactionreplay/shape/FP32/finite/scene-order/完整计数通过后才GPU6准入训练，期间不占GPU worker。各模块第一步联合更新，B256/.001warm247cos2e-5/WD1e-4/clip1，model/data/probe RNG不变，每轮实际source_cache_view写日志。固定final30后自动七景168000各景/独立truth-last/profile/源V1800，source负结果也不取消。GPU6新增1/max3总PID/free12GB/goal2，其他owner保持。当前待发布，默认与每景3pp/clean/效率目标ACTIVE。
