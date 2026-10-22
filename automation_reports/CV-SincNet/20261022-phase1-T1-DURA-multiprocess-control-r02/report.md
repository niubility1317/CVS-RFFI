# 最终 T1＋DURA 每卡三进程资源接管

- run_id：`20261022-phase1-T1-DURA-multiprocess-control-r02`
- group_id：`phase1-cric-dura-complete-system`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Resource-only r01 to r02 controller handover. Increase own capacity from2 to3/card; preserve all769 original jobs, immutable science3e880690 commands/configs, healthy worker PIDs and completed artifacts.

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

## 资源调整预登记（2026-10-22）

用户明确要求增加每卡并发。本次仅替换资源控制器，原科学 worker 固定 `3e880690` release/CWD/配置，原 769 个任务和三种子预算保持；五源不补跑，完成预测/评分直接复用。

GPU0 至7每卡从最多2条增至3条本 PAMF worker，同时总计算进程仍最多3条，启动前空闲至少12000 MiB，同卡最多1条 G/G_student。正式时延仍独占并排除GPU2/3，不为获得计时卡停止健康训练。

前置只读资源证据：8卡空闲13360至20916 MiB、系统可用内存393574432 kB；原16条任务运行。9项聚焦测试和一次独立P0/P1审查通过。科学 no-query 实际权重 smoke 复用原PASS；本次没有模型或输入变化。

新 root 不可覆盖；只在旧 controller 的 PID/CWD/完整argv匹配后撤下旧 controller，并接管全部健康科学子进程及scorer。两次相同技术故障停止新提交、保留运行任务；唯一旧统一计时导入失败维持既有r02替代。发布后独立核对新owner、原PID保留、原科学命令/CWD、GPU实际并发和日志增长。

此前按16并发实际吞吐估计整体剩余2至3天；24槽位启用后以实际epoch/完整G耗时再校准，不按进程数比例承诺加速。

## 实际接管核验

**VERIFIED / RUNNING**：资源代码 `bed988c895da628d5ca9a4ecab34c018aa3cc6af` 已提交、推送且remote OID一致。9项聚焦测试及一次独立P0/P1审查通过。新owner PID 810247；原精确匹配controller已退出，接管 16 条健康worker，全部保留原PID或正常完成。

实际每卡 PAMF 并发：`{"0": 3, "1": 3, "2": 3, "3": 3, "4": 3, "5": 3, "6": 3, "7": 3}`；所有活动worker都通过原 `3e880690` 科学release/CWD/完整argv核验，20 条活动日志已增长。科学矩阵、预算、seed、数据和默认头均保持，原完成行及唯一已知旧timing失败保留。

此为资源容量实测，暂不宣称1.5倍加速。启用前已完成F/G中位耗时为1.53/5.10小时，整体ETA暂保留2至3天，后续按24并发真实吞吐更新。正式机制时延仍等待独占卡，不以共享训练状态计时。
