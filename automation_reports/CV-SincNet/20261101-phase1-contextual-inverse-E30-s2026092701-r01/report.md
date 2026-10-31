# 动态频域逆变换与时域残差独立恢复网络

- run_id：`20261101-phase1-contextual-inverse-E30-s2026092701-r01`
- group_id：`phase1-contextual-inverse`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

从零训练条件化复数频域和共轭镜像校正，加条件化时域残差；只用L_s/U_s配对源信号，固定30轮。

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


## 真实启动状态

VERIFIED/RUNNING。代码 9959956a0572de08fbcac3258fd30a5c0769763a，独立 P0/P1 静态审查无阻断；选择性发布包 SHA-256 ca4ae5f0901e5666aa4bae36cf4ea5323cec598235b153e8fc1fe0d8439d36df，N607 依赖层和 150 个 Python 文件编译通过。真实 2 条 L_s 的检查点重载、冻结历史 F50 默认头 smoke 通过，训练中不读取 TX、F、OE、V 或目标。

控制器 PID 2092089，GPU 7 worker PID 2092179，已独立核验进程、命令、工作目录、GPU 占用及训练日志；观测到至少完成 E2/30。源损失不是目标性能。固定 30 轮后按既定流程进行历史 F50 七景预测与独立评分；Phase3 的三视图 F0、开放集及注册能力另行评测，尚未完成。
