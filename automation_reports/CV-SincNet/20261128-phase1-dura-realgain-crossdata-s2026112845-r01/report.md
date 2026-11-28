# LoRa 与 WiSig 物理残差逐频实增益筛查

- run_id：`20261128-phase1-dura-realgain-crossdata-s2026112845-r01`
- group_id：`phase1-lora-wisig-dura-real-gain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：PARTIAL；LoRa 源训练完成，筛查及两条 WiSig 训练因历史模块导入失败而中断。

## 目的与对照

对每包冻结恢复结果计算 DURA 接收作用伴随残差，学习共享结构的逐频点实增益；两套数据分别训练权重，并补 WiSig 实增益对照。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

LoRa 在 N607 GPU 0 完成预定的 2,000 步、8,000 次源 U 暴露，`lora/gain.pt` 和 `lora/fit.json` 已冻结，训练 198.34 秒，新增 11,233 参数。LoRa 首次预测在读取源 V 前因历史 W24 `tri_clean95` 模块路径缺失而失败。

两条 WiSig 训练在更新前均因历史 `refine_runner.select_gpu` 导入路径冲突退出。本 run 未完成任何筛查评分，未读取目标数据。全部失败日志、冻结权重和独立输出 root 保留。

## 交接

替代 run `20261128-phase1-dura-realgain-crossdata-s2026112845-r02` 修复历史依赖加载顺序，复用本 run 的 LoRa 冻结权重，不重复 LoRa 训练；WiSig 按用户最新要求切换至完整原始 R17。替代 run 的源域筛查已完成并记录结果。
