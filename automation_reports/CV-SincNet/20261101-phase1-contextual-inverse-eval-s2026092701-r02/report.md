# 动态逆变换恢复器：复用已训练权重的评测恢复

- run_id：`20261101-phase1-contextual-inverse-eval-s2026092701-r02`
- group_id：`phase1-contextual-inverse-evaluation-recovery`；类别：`cvs`；阶段：`Phase1-evaluation-recovery`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原E30训练完成但首个预测因inference_mode的Sinc缓存版本计数失败；修复加载上下文，固定原权重，不新增训练。

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


## 历史 F50 七景结果与独立复算

VERIFIED/ANALYZED。原始 30 轮 G 权重逐字节复用，无新增优化；源 1800×7 和目标 168000×7 的预测各自封存后独立评分。原 raw clean 为 84.676190%，六个星地场景逐个均超过 3 个百分点差距，目标未达。依据保存的全部预测和 truth 独立复算 overall/TX/RX/day 共 126 组的准确率及纠错、损伤一致。

| 场景 | 新 G 准确率 | 距 raw clean | 相对 DURA |
|---|---:|---:|---:|
| clean | 84.051190% | 0.625000 pp | -0.626786 pp |
| practical_high | 81.426190% | 3.250000 pp | -1.398810 pp |
| practical_mid | 78.713690% | 5.962500 pp | -2.847024 pp |
| practical_low_suburban | 75.914286% | 8.761905 pp | -4.144048 pp |
| practical_high_urban | 77.177976% | 7.498214 pp | -4.567857 pp |
| practical_mid_urban | 60.292262% | 24.383929 pp | -15.194048 pp |
| practical_low_urban | 47.661905% | 37.014286 pp | -21.227976 pp |

低城区相对 DURA 修正 12621 条、损坏 48284 条，净减少 35663 条正确结果。clean 损伤 0.625 pp，处于 1 pp 上限内；不能以 clean 通过掩盖六景不通过。比较使用历史原 F50、同一已曝光的工程测试，不能当成 Phase3 F0/open-set 成绩或新盲测。

原训练从零 30 轮/7410 更新，训练墙钟 206.276 秒、173570 参数；同卡性能记录中候选为 eager、DURA 为 CUDA Graph，执行方式不同，不能以少参数或单个时延推出同质量加速。结果与完整分组、profile 和复算脚本保存在对应 local_artifacts，Phase3 配对评测按预登记继续；新研究主线按用户要求转向有别于另一聊天时频残差架构的条件生成与身份稳定性。
