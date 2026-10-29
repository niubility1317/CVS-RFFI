# PAMF第三阶段：星地域残差投影源端证伪

- run_id：`20261029-phase3-star-nuisance-s2026092701-r01`
- group_id：`pamf-phase3-star-nuisance-source-falsification`；类别：`diagnostic`；阶段：`Phase3-source-star-open-set`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Source-only fixed control/RX/RXday MD/RMD; L-only protected nuisance projection, known-only V risk calibration, G10 heldout-device score.

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

## 实际发布前固定

代码/config commit fccbde38a407891db5acce776d6ad7af891f1341；固定源端三变体、MD/RMD、P0/P1共12列，唯一主候选rxday::RMD::P1。全部神经0更新，原六类默认头和G冻结；source L拟合，known V仅42组阈值，novel public在冻结后读取，全部预测完成后独立truth评分。新联合指标定义见 docs/PAMF_PHASE3_METRICS_20261029.md。

聚焦组件共34项unique测试：metrics6、data6、core7、runner/score13、launch2，子项另记；source owner最后仅状态名称改动后定向1项通过。data与模型候选各一次独立P0/P1 PASS。数据3150已VALIDATED_ONCE，原ID/物理角色保持；首次真实checkpoint合成星地smoke由source job首步执行，PASS即继续，不新增许可。

一次N607 preflight已通过：原source合同/原F0/G与metadata/raw路径存在，目标句柄0，实际各卡余量已记录；动态选择free至少4096MB，串行一个本任务GPU worker，不停止其他owner。archive单次SHA比对和remote compile在发布时执行；不按source分数调参或改候选，不追改历史G10/clean实验。科学失败完成评分并NO_PROMOTION，无target访问；通过只表示下一冻结目标小矩阵的source eligibility。
