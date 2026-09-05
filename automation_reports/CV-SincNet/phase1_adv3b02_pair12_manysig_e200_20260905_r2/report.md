# ADV3B02配对改革E11技术故障修复与重新发布

## 范围与原因

用户授权修复并重新发布。原run为`phase1_adv3b02_pair24_manysig_e200_20260905_r1`，独立读回确认12行技术失败、12行健康运行。仅替换失败的B_SAFE、TANGENT、ROUTE、TANGENT_ROUTE各3个seed；其余12行保留原release继续训练，不重跑。旧输出和失败日志完整保留，见`parent_failure_readback.json`。

两个已复现的根因都在E11配对机制启用后触发：

|故障|根因|修复|验证|
|---|---|---|---|
|B_SAFE：Half did not match Float|外层autocast把显式float输入的矩阵乘法输出重新转成Half，随后allclose和Float比较失败|仅固定分类头一致性检查禁用autocast，保留原误差阈值及头匹配检查|修复前CPU bfloat16和真实CUDA fp16均失败；修复后通过|
|方向约束：Namespace无sat_fs_hz|实际parser未提供此字段，旧smoke人工注入字段掩盖缺失|沿用现有物理probe的25MHz默认读取方式，支持既有显式sat_fs_hz调用；本轮STO以sample为单位、PA/IQ增益为无量纲，均不使用频率换算|真实parser参数下切向、路由及组合的CUDA前后向，修复前失败、修复后通过|

本次不增加模块或调参，不关闭AMP或机制以绕过错误。修改仅涉及`code/cvsrffi/pair_reform_runtime.py`的两个故障入口，以及验证/恢复发布工具。

## 验证改进

`smoke_adv3b02_pair_reform.py`改用真实训练parser，支持读取恢复清单，取消人工补入不存在字段。重新发布首步使用真实CORE90 checkpoint、CUDA fp16 autocast和GradScaler；覆盖六种机制的L/U路径、有限梯度、成功优化步、teacher无梯度、memory提交和identity-only前向。synthetic batch=2，方向抽样比例仅在smoke设为1以保证执行；正式训练argv不变。smoke没有数据集或query I/O，其结果只证明执行正确性，不证明性能。

本地先运行失败回归：5项全部失败并匹配上述指纹。修复后运行`tests/test_pair_reform_recovery.py`、`test_pair_reform_checkpoint_smoke.py`、`test_pair_reform_training.py`、`test_pair_reform_runtime.py`共36项通过，包含真实CUDA验证；恢复清单与调度器10项通过。AMP旧接口弃用警告不影响结果，保留该接口以兼容远端PyTorch2.1。

## 恢复矩阵与规则

恢复run：`phase1_adv3b02_pair12_manysig_e200_20260905_r2`。12条完整命令保存在`configs/phase1_adv3b02_pair12_recovery_manifest.json`，由`tools/pair_recovery_manifest.py`生成。

每个恢复行与原行相比只替换run_id、release、输出及环境路径；candidate_id、seed、原始CORE90初始化、E200、130/70阶段、所有损失及11/21启用时间保持一致。原配置final_only未保存E10恢复checkpoint，所以从同一CORE90重新执行完整E200，不假称续接E10状态。

恢复行沿用原assigned_gpu；与原健康12行合并后，GPU0—7各3个本轮任务。恢复队列最大12并发，每卡最多3个新队列任务，保留8192MiB启动显存要求；原有其他实验不干预。

最终24行的证据来源映射固定为：A_POINT、ASYMMETRIC、POINT_MEMORY、MATCHED_ZERO取r1，B_SAFE、TANGENT、ROUTE、TANGENT_ROUTE取r2。r1中失败行属于历史技术失败，不重复计入24行有效候选结果，不用于择优。选择规则沿用原24行报告，源V选择、目标只读及truth-last边界不变。并发时序不同，耗时不作严格独占性能比较。

相同指纹修复后若再次出现，停止盲目重发并报告；不因低性能停机。健康r1任务不停止、不热补丁。本次无目标评分。

## 发布与监控

远端项目：`/home/szu2070436088/2510044040/CV-SincNet`，新代码`releases/<恢复run>`，新产物`runs/<恢复run>`。启动使用远端CVS-RFFI环境Python执行新release的`tools/pair_matrix_start.py`及恢复清单。禁止复用r1或已有输出目录。

发布状态：VERIFIED。运行代码提交`e9b7709da3368e25e68d6b8a84b0cae39098868d`，Git push后独立远端OID读回一致。发布包SHA256本地／远端一致：`a4f5c4faeb7b11400d609c142c7dc73988872538b82cd20e61dcdd845c50ccc1`。远端编译和完整manifest读回通过。聚焦P0/P1审查PASS，无阻断项。

真实N607检查：VERIFIED，CORE90权重epoch194、missing/unexpected均为空；`amp=true`，六种机制各完成L/U两次成功优化步，teacher domain前向0、query/target输入0、memory命中率1。详见`checkpoint_smoke.json`。本次直接覆盖已失败的AMP／真实参数入口。

重新启动：VERIFIED，恢复调度器PID2176400，12行全部RUNNING、待发0，12份日志均已生成。逐一读取/proc核实有效24行的实际argv、父进程和cwd，全部匹配对应固定release与run；合并GPU0—7各3行。恢复PID及启动快照见`startup_readback.json`。旧健康12行PID未变化，没有停止或热补丁。

监控更新：VERIFIED，`luna-adv3b02`保持ACTIVE、每小时一次，已通过应用工具更新并独立读回为r1健康12行＋r2恢复12行。检查将重点确认r2越过E11后无相同指纹复发。

结果边界：当前已证明修复入口的CUDA执行及重新启动，恢复正式训练尚未再次到达E11，不能将启动读回当作E200稳定性或性能结论。技术失败历史仍保留，后续以有效24行的正式训练日志和最终产物报告效果。

每小时Luna监控将同时跟踪r1健康12行与r2替代12行；历史r1故障不反复报警。仅当有效24行全部终态后暂停监控，运行中不声称最终效果已验证。
