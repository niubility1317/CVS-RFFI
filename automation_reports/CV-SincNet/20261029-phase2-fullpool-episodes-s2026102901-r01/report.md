# PAMF Phase2 Full eq：同池 K-support 全补集 query 与逐套重新叠加

- run_id：`20261029-phase2-fullpool-episodes-s2026102901-r01`
- group_id：`phase2-fullpool-episodes-r09`；类别：`diagnostic`；阶段：`Phase2-data-builder`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原6旧类20新类、7RX、4日共554823条原始记录；8综合+每种纯场景5套，逐套fresh practical residual，actual-K support及全补集query；数据制作，无模型。

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

## r09 最小预登记

数据规则与实际容量见 [r09 分配说明](../../../docs/PAMF_PHASE2_FULLPOOL_EPISODES_R09_20261029.md)。矩阵为原六旧类＋原二十新类、7 RX、23 套、K1/5/10/20、N0/2/5/10/20。原池 554,823 条；161 bank、3,220 split、12,760,929 次重新叠加观测。每任务 support K/类，query 全补集；同套 K 嵌套且 H 相同，跨套 H/support 重采样。原池、未来类别和真值只由离线 builder 组织，不送入方法。

本轮只制作数据，identity train/predict/score＝0，DURA 目标缓存＝0；F/G checkpoint smoke 不适用于不加载模型的数据作业。源 L/U/V 不变，未来身份任务每 episode/K 从匹配源模型重置，模型 seed 不重抽数据。

环境为 N607 普通账户的既有 CVS-RFFI Python；CWD、命令、独占 run/release root、owner、输出和日志均在 experiment.json。发布器仅归档已提交且已推送 Git HEAD，实际 commit 写入 runtime/submit，启动后回填本记录。最多 8 个 CPU worker，CUDA 不可见、OMP/MKL/OpenBLAS 线程均为 1。一次 direct-only preflight 通过，CPU96、当时可用内存约395GB；20ZIP 复用既有 CRC 与 Full 输入验证，不重复上传。

停止规则见 execution.stop_rule：首次输入数量/格式变化、容量不足、当前任务角色重叠、输出碰撞、非有限值或 worker 错误停止本任务子进程，保留 partial，不动其他 owner，不自动更换参数重跑。

5 项聚焦检查 PASS。唯一一次独立 P0/P1 审查 PASS，无未解决 P0/P1；审查确认固定原26类每日期非空，最小四日池590。预期最终产物为 builder_only 原池、161个 received/IDs/manifest/stage_rows/actual-K split/标签、独立 score sidecar、catalog 和 builder_report。启动用独立 PID/CWD/argv 与产物/日志增长证明；全部数据完成必须有最终报告，不将 submitted 当 complete。

## 已验证启动

VERIFIED：实际执行 commit 为 `42e6dca573508f511d2f33861ac7dbd58eaeee45`；发布归档 SHA 对比一致、远端编译通过。独立回读 controller PID2486872 与 pool worker2486876 的 CWD/完整 argv/线程设置一致，CUDA_VISIBLE_DEVICES 为空；首读已有4个 RX 原始池 manifest 和对应增长日志。当前是 RUNNING，尚无最终 builder_report，不宣称数据全部制作完成。证据为同 run 的 preparation_status.json 以及本地 latest_readback.json；无身份训练、身份预测、评分或 DURA 缓存。

随后独立回读确认进入 episodes 阶段：7个 RX 原始池均完成，共554,823条；8个 CPU worker 正在生成 satellite received。进度文件合计至少80000条观测已写出。当前尚无完整 bank 的结束报告，最终数据完整性状态仍为 RUNNING。

## r10 三池数据输入复用

用户最终选择三个互斥物理池，r09全池布局不再作为后续身份任务输入。此作业的逐episode原始信道观测仍合法且可精确分片复用，因此健康r09工厂继续，不停止、不热改、不重生成。同一received及匿名ID在r10按原始物理身份归属拆为3池，并新抽pool内support/query；r10 run为`20261029-phase2-threepool-episodes-s2026102901-r01`。复制/分片不计新信道或新独立物理样本，r09历史输出保持。
