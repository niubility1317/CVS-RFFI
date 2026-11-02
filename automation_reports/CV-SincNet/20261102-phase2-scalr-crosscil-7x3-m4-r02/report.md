# Phase2 第二组：七种域适应与三种注册核心配置修复恢复

- run_id：`20261102-phase2-scalr-crosscil-7x3-m4-r02`
- group_id：`phase2-scalr-group2-da-native-cil-cross`；类别：`comparison`；阶段：`Phase2-B-C`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

r01 source-only smoke failed before target predictions due missing frozen_sources runtime; immutable r02 corrects configuration, retains all old artifacts.

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

## 启动读回

2026-11-02：源代码 commit 411ffe929db5f5527e81748cb9a104c730f83bbf。独立 P0/P1 审查及资源并发修复的定向复核通过；23 项本地针对性测试通过。r01 的配置缺键已修复且保留失败产物。r02 发布包传输校验、远端编译通过；七种方法各取第一个实际源 checkpoint 的 source-only no-query smoke PASS，未读目标或 query。controller 1615082；独立读回 16 个 worker 分布在 8 张 GPU，首个 worker 的预测文件增长。13524 任务、252252 预测行；全部冻结后才独立评分。当前无完成的第二组指标，不宣称全部组合完成。ISSL 保持停止，CSIL 作者版本 N/A；外部 CIL 不接 DURA。
