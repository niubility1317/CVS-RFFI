# 同一源域受损IQ：CSIL、MoPC-HR与当前PAMF的冻结互补检验

- run_id：`20261019-diagnostic-FcrossRX-sameIQ-experts-s2026092701-m2-r02`
- group_id：`phase1-same-observation-external-teacher`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用P79缓存和AS30默认头预测；固定历史seed01/final200外部模型读取相同IQ，先证伪教师资格。0训练、0新增信道、0目标访问。

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
# r02 技术恢复预登记

模型、源物理 ID、IQ 字节、场景/draw、256 点输入、类别映射、两 draw 的 +1pp/+.5pp 教师资格和 25204 前向预算均保持 r01 原预登记。r01 仅真实 CP smoke 完成、数组转换失败，无性能结果，原输出保留。原方案见 ../20261019-diagnostic-FcrossRX-sameIQ-experts-s2026092701-m2-r01/report.md。

唯一修复是 Torch2.1/NumPy2 C 数组桥：标签和预测采用 tolist，float32 IQ 按 C 顺序通过 frombuffer 传入；不经过 Tensor.numpy 或 torch.from_numpy。原环境无 IQ 最小复现确认旧桥类型不兼容，替代路由通过；本地 5 项检查和唯一问题定点复审 PASS。新 run/release 以 r02 结尾，不覆盖 r01；命令、GPU0/CPU2、路径、停止规则与精确矩阵见 experiment.json。


## 已完成的同观测源结果

状态 VERIFIED / ANALYZED_SOURCE_ONLY：两份真实CP/defaulthead smoke通过；两模型共25204样本前向、0更新、0新增IQ/target/query，权重和BN状态保持。实际GPU读出在2.263s完成；进程在首次独立状态检查前已经退出，故没有存活PID/CWD/GPU采样证据，完成状态由独立completion、smoke、完整日志和比较产物确认。

|模型|clean|mid urban|low urban|
|---|---:|---:|---:|
|csil|98.7222%|86.6667%|78.1944%|
|mopc_hr|98.4444%|86.4444%|79.8056%|
|PAMF AS30|98.3333%|72.8056%|59.0833%|

两教师均在两个draw上通过预先条件。CSIL是附件预先首选/固定首个迁移对象，MoPC-HR仅作同观测证据，不根据本轮更高低urban分数选择教师。两模型E200与PAMF E30预算不同，只证明同一观测中存在可迁移判断；不宣称结构的匹配训练优势或部署提升。完整状态/锁定/新增噪声比/RX/day/TX配对分层保留在远端各method/comparison.json，本地raw readback不进Git。两个draw不是两个模型seed。

已知故障r01保留；r02桥接修复经本地回归、唯一问题复审和真实运行验证。下一步为同父、同计算的CE vs CSIL知识迁移，两条新增E10另立run。
