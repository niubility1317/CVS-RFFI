# D92 E0 true256 numerical recovery: all five fixed seeds

- run_id：`20260928-phase2-cvs-d92-practical-manytx-m5-r02`
- group_id：`phase2-cvs-d92-true256-practical-matched`；类别：`cvs`；阶段：`Phase2`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Preserve final200 source checkpoints and all successful prediction prefixes; support-only diagnosed guarded FP64 common-affine centering before FP32. Complete the original fixed matrix without target-feedback tuning.

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

## 固定恢复方案

替代原r01的Phase2未完成部分；Phase1已完成独立目标评分。本run保留全部5个模型seed，不重新训练或选择模型，继续使用原始DAOT A1＋FastTrust-RC4第200轮权重及其source L生成的v2 rank3 ground。D92版本固定P2-256-FULL（E0去RF32，identity160＋FFT96），接收信号、support/query、K、所有随机角色保持原合同。

两处故障经support-only诊断确认是FP32系数舍入改变近似平局的类别。修复commit4c4c1eac9b4491762c7255653d1b30347204d14e：仅当原部署检查失败时，在FP64消去所有类别共同的仿射项，再转FP32，仍严格核对sklearn预测。原成功路径数值bitwise不变；新增audit字段仅说明是否触发修复。

在新输出目录逐行复用原成功预测前缀，严格核对seed、feature/ground/capsule路径、checkpoint摘要、方法配置及每条split、类别、query ID、scores/argmax。3个完整seed原样复用；392005从830条之后、2026092703从1968条之后继续。不覆盖旧run，不按评分选择重跑；失败种子尚未读取query truth。所有5行均达到2100 split和2121条后，独立scorer才读取truth并输出scored_results.json。

唯一launch owner：codex/root/cvs-d92-recovery-20260928。CPU五lane，无GPU训练。详细命令由tools/cvs_d92_recovery.py生成并写入state.json、各row日志及d92_startup.json；发布器tools/publish_cvs_d92_recovery.py只发布已提交并远端OID一致的版本。发布后先对两个原失败split完整support-only fit核查FIT_PASSED，再启动恢复。

验证：D42相关31项测试、新3项数值回归、两份实际失败inner-support回放通过；matched 7项测试通过，含完整前缀不重拟合、注册记录与DG之间截断恢复、来源seed错误和record身份/argmax拒绝。数学修复与恢复流程各完成独立P0/P1定点审查PASS。启动前状态为LOCAL_VERIFIED，尚不代表远端恢复完成。

## 已启动与独立核实

VERIFIED：release cvs_d92_recovery_20260928_r02来自已push并独立核实OID的commit c0e538d60c2e52a50112ee9b43337b4277d463d3。远端合成smoke通过，两个原失败split（6533d11a97e532c6b8fe719b、ef8921d662af2517554b00a8）完整support拟合均FIT_PASSED、active_feature_dim=256、query_truth_read=false。dispatcher PID2061061的/proc argv与cwd正确。

evidence/readback_1790568134.json：3个完整seed已逐条验证并复用至2121/2121；392005已从830推进至850，2026092703从1968推进至1990，两条新增预测路径均正常运行。整体RUNNING，尚无scored_results.json。此前错误已越过，但完整矩阵仍须等剩余划分完成，不将启动成功写成实验全部结束。

继续核查使用tools/read_cvs_d92_recovery.py，只读进程与状态；状态变化后用tools/record_cvs_d92_recovery.py更新原登记。若所有行成功，dispatcher自动调用独立scorer并生成completion.json；若技术异常，保留新partial并停止评分，不自动重试。
