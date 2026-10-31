# Phase3 TF完整30轮权重复用：补齐测试配置的冻结评分恢复

- run_id：`20261101-evaluation-P3Restore-TF-frozen-E30-s2026092701-r02`
- group_id：`phase3-P3Restore-TF-E30-frozen-test-recovery`；类别：`diagnostic`；阶段：`Phase3`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原发布包缺既有G10 synthetic-smoke配置；用相同b338代码加原已跟踪配置的新独占release补全部既定测试。0重训/0调参/旧日志和失败产物保留。

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

## 发布包故障与最小恢复

纯IQ模型已完整30/7410，首次自动Phase3测试在真实输入前的synthetic smoke因缺既有配置文件失败。归档只含三个代码目录，缺少被_synthetic按release根定位的configs JSON；文件本来就在同一b338 Git提交中。新release加上这一个既有文件，执行代码、模型、训练预算和测试矩阵保持原状。原失败root/log不覆盖。F0训练保持健康完成，原owner终止后交接缺失测试；没有重训或根据成绩筛选。

方法沿用唯一P0/P1审查。发布包修复仅一次定点复核。实际新测试的PID/CWD/argv/GPU/日志与所有预测完成后独立truth-last另行记录；旧训练run的原定输出目录尚不存在时才写入其声明的预测产物。

## 恢复完成

VERIFIED：两版全部既定Phase3及旧F50预测、独立评分完成，8个阶段均已退出，0重训。主性能口径由用户明确改回原Main split地面Tri-SIG-TIME SSL，见 20261101-evaluation-P3Restore-TF-TriSIGSSL-main-s2026092701-r03；本run仅辅助归档，不将其弱clean/开放指标移作主结果。
