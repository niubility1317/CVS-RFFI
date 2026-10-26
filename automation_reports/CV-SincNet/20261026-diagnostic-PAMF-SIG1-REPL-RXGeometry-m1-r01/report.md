# 完整PAMF单SIG_TIME：冻结E20的身份保护RX方向源证伪

- run_id：`20261026-diagnostic-PAMF-SIG1-REPL-RXGeometry-m1-r01`
- group_id：`pamf-repl-v2-20261026`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

仅冻结已完成m201单视图E20和原DURA，提取L6300/V27000 raw及12G均值特征，真实L-only RX方向和跨TX/V迁移诊断；0训练，不重复任何对照，不读U或target。

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


## 本地验证

7项聚焦检查与唯一P0/P1审查PASS。实际单视图checkpoint/G源smoke和完整源特征运行尚待执行。参见[预登记](preregistration.md)、[本地检查](local_verification.json)。不训练任何对照或学生，原三项目标未完成。


## 实际启动核实

RUNNING。执行commit `22041cdee1a95cac538675e3c16a360cb9a023cf`，一次archive比较VERIFIED／远端compile PASS。实际单视图E20/defaulthead＋G源L2检查PASS，随后立即全量提取；controller和唯一worker的PID/CWD/argv/GPU以及初次日志增长独立核实。L6300已完成，V先1280后9984/27000，证明推进中；当前没有几何终点。零optimizer、零U、零target、无控制或学生训练。私有缓存/运行细节保留本地，公开证据见[launch_evidence.json](launch_evidence.json)。
