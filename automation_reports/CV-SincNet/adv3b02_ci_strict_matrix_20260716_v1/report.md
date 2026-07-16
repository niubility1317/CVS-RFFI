# ADV3B02类增量严格Stage2-C重跑

版本化镜像。主报告位于`E:\type10-7\automation_reports\CV-SincNet\adv3b02_ci_strict_matrix_20260716_v1\report.md`；实现与运行过程中两者保持同步。

## 追踪表

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|TR-01|7.1|Phase2只接触预叠加LEO weak样本，clean样本及派生信号不可达|包构建器、预检、运行审计|pending|待完成|严格阻断|
|TR-02|7.2|逐样本全注册类决策，无角色Oracle、类别配额和batch全局分配|apply-only predictor、测试|pending|待完成|严格阻断|
|TR-03|7.2|不可变prediction artifact与独立scorer|predictor、scorer|pending|待完成|不得回流|
|TR-04|9.3|同target receiver合法old/new support/query|75个密封包|pending|待完成|真实ManyTx TX|
|TR-05|10.3|5 receiver×5 seed×4 K×3新类规模|matrix generator|pending|待完成|每方法300 cell|
|TR-06|10.3|K-shot嵌套与同query|matrix/package audit|pending|待完成|不按query调参|
|TR-07|10.3|冻结严格ADV3B02特征提取器|enrollment runtime|pending|待完成|三方法统一checkpoint|
|TR-08|10.3|CSIL机制与损失|CI head module、测试|pending|待完成|CVS-aligned extension|
|TR-09|10.3|MoPC-HR机制与损失|CI head module、测试|pending|待完成|CVS-aligned extension|
|TR-10|10.3|Orthogonal机制与损失|CI head module、测试|pending|待完成|CVS-aligned extension|
|TR-11|10.3|同row性能、遗忘与分组证据|scorer、summarizer|pending|待完成|禁止拼接极值|
|TR-12|10.3|资源账本|enrollment/apply receipts|pending|待完成|参数/step/时延/显存/状态/前向|
|TR-13|10.3|matched Stage2-C MRIOR-SDA|MRIOR运行链|pending|待完成|历史Stage2-B不可替代|
|TR-14|Experiment Reporting|N607完整运行记录|报告、state/log|pending|待完成|正式启动前补全|
|TR-15|Version Management|本地优先、Git提交、SCP SHA核验|提交与同步清单|pending|待完成|保护并发改动|

## 2026-07-16本地实现状态

- 已实现三种统一冻结ADV3B02 feature-head：CSIL、MoPC-HR、Orthogonal Incremental。
- 已实现support先行、head SHA锁定后才物化query的predictor流程。
- 已实现75个密封包、900个cell、2700个场景行的计划生成器；矩阵按package独占分片，避免多GPU争写同一包。
- 已实现smoke receipt派生正式launch authority。
- `ssr-gpu`下52项focused/cross-module测试通过；`py_compile`与`git diff --check`通过。
- N607 smoke、正式矩阵与matched Stage2-C MRIOR-SDA仍待执行，因此当前不能声明实验完成或性能优于基线。

## 同步证据

- Git提交：`4a1cfe7`。
- N607四个新文件SHA依次为`fca6f70e...d15dd`、`5f0ce7fb...2027f`、`4b3f4501...64aa1`、`6eb3cdea...421e`，与本地一致。
- 2026-07-16 12:06 CST直连预检PASS，8张RTX3090空闲且无活动训练进程。

## Runtime导出attempt1

- PID`2283502`在runtime发布前fail closed：远端旧导出器的graph复跑命中FFT内部dtype漂移。
- 本地恢复固定256行内部batch、动态外层slice，并关闭不稳定graph文本复跑；独立数值parity门禁保持。
- 8项相关测试、`py_compile`和`git diff --check`通过；重试必须使用全新`runtime_artifacts_v2`。
