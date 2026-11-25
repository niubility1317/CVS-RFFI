# 报告缺项：优化 Tri 跨接收机六景与 DURA 配对测试

## 结果范围与运行矩阵

用户于 2026-11-25 授权补齐当前报告空缺。固定六个历史报告设置：接收机划分一、二，1% 与 5% 训练标注，六源和十类；各使用原最新配方的三个既有 E50 种子，共 18 个识别器。只测原 cross_rx 目标集，分别读取冻结的 raw 与匹配 LU DURA 恢复波形；七景为 clean 和六种 practical。原历史 clean raw 预测按完整身份和物理 ID 精确复用，无重新训练。每行七景两路预测全部完成后独立评分；六景先按场景等权，再按三个 F 种子统计均值和样本标准差。每设置共用其原匹配 G，不能称为三个 G 种子独立复现。

模型路径取既有 `20261120-phase1-pamf-tri-setting-profiles-clean-m33-r01/sources.json`；1% 取 `20261121-phase1-pamf-tri-label01-earlyanchor-reach-clean-m2-r01` 的 01 与 `...seedcheck-clean-m2-r01` 的 02/03，均为 E20 校验 0.60 固定配方。每行对 source-only E50、同 seed、source contract、配方、既有完整 clean 预测和匹配 DURA 缓存进行检查；评测不读取未标注 TX 真值，不更新模型。已有目标 clean 开发暴露历史按原记录披露。

单源 Hybrid 三种子 raw 和 DURA 七景已完成，原预测与独立评分经只读核对可复用：跨接收机 DURA 六景均值 55.829475%，种子标准差 0.293232 个百分点；不纳入本次新预测。

N607 用户账户，最多每卡一个本任务预测 worker，启动阈值可用显存 12000 MiB，评分同时最多一个 CPU worker；保留其他用户进程。路径、逐行配置、停止规则和输出见 experiment.json。一次真实 checkpoint 源样本冒烟通过后立即派发；技术异常停止新派发，健康任务与产物保留。

## 本地检查

冻结配置矩阵、禁止预测访问 truth、未完成预测前禁止评分及共享评价配置恢复的四项聚焦测试通过。独立 P0/P1 审查未发现阻断问题。预测、评分与最终数值待远端实际产物完成后更新。

- run_id：`20261125-phase1-pamf-report-gap-sixscene-m18-r01`
- group_id：`phase1-pamf-report-gap-sixscene`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

六个原报告数据设置、各三个既有 E50 优化 Tri；固定同数据契约 DURA-LU 恢复缓存，跨接收机 clean 与六个 practical，raw 和 DURA 配对；无新训练。

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
