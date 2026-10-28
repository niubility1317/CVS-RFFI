# PAMF Phase2 Full eq：八套独立综合与三种纯场景数据制作

- run_id：`20261029-phase2-full-eq-preparation-s2026102901-r01`
- group_id：`phase2-full-eq-shared-data-r08`；类别：`diagnostic`；阶段：`Phase2-data-builder`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Full实际数据与四compact精确映射，保护历史query，计数选20新类；8mixed×150query/类＋3pure×500，24support池，分阶段actual-K读取；不训练/身份预测/评分。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

本地执行代码：`E:/type10-7/code/worktrees/standard_ce_baselines_20261010`，数据定义见`configs/pamf_phase2_full_eq_data_r08_20261029.json`，分配分析见`docs/PAMF_PHASE2_FULL_EQ_ALLOCATION_R08_20261029.md`。发布器只打包实际已提交文件，实际commit写入远端runtime与submission记录。

N607普通用户项目：`/home/szu2070436088/2510044040/CV-SincNet`。已完成direct-only preflight；磁盘约2.1TB余量。原始数据在`data/raw_full_wisig_eq_20261029`，20ZIP／34,378,471,511字节已逐包远端CRC PASS，不再解压或复制另一份原始数据。

正式执行入口：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase2_practical.full_runner --runtime runtime.json`。CWD为同run的`releases/<run_id>-release`，CPU单进程、CUDA_VISIBLE_DEVICES为空。controller依次执行inventory、select、audit、build；全部输出写入独占`runs/<run_id>`。

预计输出539,448条独立物理记录，support候选48,048、query491,400；385个按RX/domain/N隔离的capsule、3,220个actual-K split。新20类在精确历史排除后只按数量确定，source与原生方法预算不变。A读取器只打开query，B/C只打开当前split实际K的support。

停止规则：首次精确映射／容量／角色／非有限数值／输出碰撞或子进程异常停止，保留partial，不覆盖或自动缩量重跑。builder不消费模型，因此checkpoint smoke不适用；身份训练、身份预测、评分与DURA目标缓存均不在此controller中。

本地6项聚焦检查PASS。唯一一次独立P0/P1审查发现发布归档目录成员不兼容，已修复为仅普通文件重打包并定点确认，未解决P0/P1为0。无附加发布gate。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
