# 主路相位全协方差加权，冻结 CPU 推理 Main 七景

- run_id：`20261107-evaluation-Restore-CoherentMainPhase-CPU-main-s2026092701-r01`
- group_id：`original-Main-frozen-coherent-main-phase-cpu`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：experiment.json；状态记录：events.jsonl
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原 BF E30，只新增源诊断中固定全协方差均值估相路径。[计算环境] 两线程 CPU 独立推理，原 Main 双总体七景和冻结 F0。

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

## [计算环境] CPU 正式预测已启动

VERIFIED / RUNNING。CPU 预测进程 3944144 已通过真实 L_s 两条检查，并已处理原 Main 首批。128 条源样本实测 19.20 条/秒，折算全部 117.6 万条模型调用约 17.01 小时。这是源端吞吐外推，实际进度由预测日志和各景产物确认。独立评分在两类总体全部七景固定后才开始。

## 首个星地场景完整预测已核实

VERIFIED：独立读取clean.npy与practical_high.npy，每份126000条int64类别预测且范围0至5，累计固定252000条，完成2/14个原Main“人群×场景”单元。实际进程已进入practical_mid，尚未提交truth评分；新IQ/模型/truth读取均为0。其余场景继续，预测固定不等于已取得识别成绩。

两份固定预测文件证据

## 完整 Main 独立评分与复核

VERIFIED / ANALYZED。原 Main 两总体七景、四条决策流 56 单元全部固定后独立 truth-last；1736 组、1302 raw 配对复核 PASS。六景 1/6 达标；两个 clean 总体相对 raw 净 +300/+145。相对原 CovarianceMAP 合并六景上升，但跨日期总体三景下降；相对 DURA 五景下降。无新训练或目标拟合，默认 DURA 不变。 完整分析
