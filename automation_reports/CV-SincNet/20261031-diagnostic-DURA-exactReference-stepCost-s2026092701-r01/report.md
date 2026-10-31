# 精确源域F约束：GPU训练步实际开销

- run_id：`20261031-diagnostic-DURA-exactReference-stepCost-s2026092701-r01`
- group_id：`phase1-dura-source-exact-reference-step-cost`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同GPU/同scratchG/源256和两个缓存，原生产cached5与精确exact10交替3预热+12计时；0优化，记录实际F/G计数、状态、时延与显存。

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


## 固定同卡步成本比较

Source01/20、固定256=128L128U、同scratchG/原F50。cached5使用上版生产J5/offset5；exact10使用已通过非线性源可达性的同一10维参考、一次F reverse注入。每case/view3预热+12计时，顺序交替，无优化/模型更新。两景各自marginal median ratio和paired median ratio均≤1.5才通过成本条件；P95/显存完整描述，不按平均隐藏坏景，不反复重测。

计时包含输入复制/物理forward/guide-Freverse/Gbackward-clip/指标复制，不含optimizer/cache生产/epochshuffle/排队。F两case共同驻留，显存不冒称独立部署占用。GPU G15360/F7680/VJP7680，CPU首步真实2Ls另G2/F4/VJP2；0训练/TX/V/target/保存G。3测试/入口/唯一P0P1通过，原资源授权4/6/7、总≤3/卡新增1、goal≤2，保留他人任务。通过只支持另登记单阶段30轮与七景正式，默认与完整目标保持。
