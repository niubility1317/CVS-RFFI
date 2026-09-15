# 实验管理落地与历史整理记录

日期：2026-09-16。任务：让以后每次CVS、对比、联邦、诊断及独立评估都有名称、说明、实际配置和可快速定位的存储记录，并整理历史实验。

## 结果

- 新增[experiment-management skill](../.agents/skills/experiment-management/SKILL.md)和[统一规范](EXPERIMENT_MANAGEMENT.md)，通过AGENTS.md、项目.md、实验流程、N607操作和职责映射接入每次启动及跨对话查找。
- [experiment_registry.py](../tools/experiment_registry.py)提供模板、登记、字段核对、run/row状态事件、历史整理、搜索及有出处的配置/产物定位。工具不启动训练、不读取IQ/truth，也不改变进程。
- 新实验逐row明确model/split/data/augmentation/support/evaluation六类seed、数据契约、checkpoint来源、optimizer/LR/预算、精确命令、输出/log和预期artifact。状态与报告中的证据关联，不凭索引给出科学有效性或当前运行结论。
- Git承载面的旧控制文档与本机现行规则不一致，已同步本机当前规则及科学协议镜像，仅调整克隆相对链接；科学语义以本机项目.md为准。其他任务的代码、配置和暂存内容保留。

## 历史覆盖

|记录类型|数量|解释|
|---|---:|---|
|本地历史证据组|4028|报告目录、配置、分析、产物及Git检出证据；一个组可含多个run/row|
|早期日志目录记录|7778|从两个既有CSV目录导入，保留源行、方法族、seed/比例、命令和原日志路径|
|N607目录入口|3608|2026-09-16香港时间01:55普通账户只读快照，含直属子目录/文件名|

本地枚举509977个文件，保留414842条产物路径、10189900个带源文件和JSON位置/行号的元数据字段。23612个压缩细目文件共约59.9MiB，查询只读取对应记录的细目。

这些数量均不是独立实验数或已完成实验数。同名镜像、日志备份、失败、停止、替代、source-only、仅配置及partial记录保留各自来源，不推断合并。历史RUNNING声明不代表当前进程，N607目录存在也不证明训练已开始或完成。

16个历史JSON为空、截断、混有非JSON文本或含非法控制字符，原始内容和路径均保留，以locator_only登记。详见[覆盖与缺项](../experiment_registry/coverage.json)。未解压归档，未读取权重/IQ/预测/原始日志内容；远端快照仅到surface/group/直属children，其他磁盘、隔离worktree、深层远端内容及仅存在于对话的记录仍按需定位。

## 验证

- ssr-gpu环境下9项聚焦测试通过：seed角色区分、禁止覆盖、重复输出、run/row状态隔离、历史同名、旧RUNNING、补登记无events、UTF-16及无config命名元数据、CSV导入和远端目录状态边界。
- 独立skill试用发现“行状态未展示”“无events的历史补登记被漏掉”两项问题，均已修复并加入回归验证。
- skill-creator格式验证通过；39处文档链接和UTF-8读回检查通过。
- 所有压缩细目已完整解压读取校验，首条JSON与所属record_id匹配；目录ID无重复。
- 本机真实查询CORE90、DAOT+FastTrust、MoPC、RIEI和response约0.2–0.6秒；`SIM_s392005`的model_seed按文件及JSON位置精确命中，约0.23秒。
- [验证记录](../experiment_registry/validation_20260916.json)保存查询示例、覆盖、耗时与缺项。[N607快照](../experiment_registry/snapshots/n607_directories_20260916.json)保存只读目录证据，0项目录读取错误。

## 使用入口

从[实验总索引](../experiment_registry/README.md)按方法打开目录页，或执行`tools/experiment_registry.py search "方法关键词"`。找到ID后用`show`查看报告/配置位置；细目可按field、row或artifact类别查找。新实验用template/new，状态变化用record；常规登记只增量更新，不重新扫描历史全库。

本次仅落地管理工具和元数据目录，没有启动、停止或修改N607实验，也没有为历史结果补造指标或有效性结论。Git交付结果以本次提交和远端OID读回为准。
