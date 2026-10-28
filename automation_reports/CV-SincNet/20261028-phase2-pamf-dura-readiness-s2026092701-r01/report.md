# PAMF Phase2 真实接口验证预登记

用户已授权检查后启动、复用合规 Phase1、每 GPU 最多 3 个进程。本 run 为 source-only 无更新真实权重验证；目标数据范围正在澄清，不构造目标数据、不预测 query。

当前三视图 PAMF seed01 固定 E50/11100，自身 scratch20→10→20，严格原生 loader 验证全部父阶段。五种特征为完整 composite z_id160、t_emb160、f_emb160、pa_local160、当前输入 FFT96；DURA 为既定 R69 stage2/Graph-RR/FP32/TF32off，逐窗口估计，零 F50 reader。

只读取当前原 Main split 的前 2 条合法 L_s IQ。核对联合特征与原 reader.features 精确相同、原默认头维度、DURA 批量与逐条一致及冻结参数/缓冲未变。优化器更新 0、target/query 读取 0。其结果只证明接口，不代表 Phase2 性能。

## 输入与复用证据

2026-10-28 N607 只读核对：4 个当前三视图 F 完整权重存在，固定最终 E50/11100、target_access=false。DURA 原 source/source_contract.json 与当前 F0 的 role_ids、counts、source_rxs、source_days、ratios、split_seed 全部一致；旧契约不单列 classes，原 ManySig 六个 TX 元数据已核对。外部 20260930 源基线同项也一致，9 种方法 seed01 的 completion 均为 SOURCE_TRAINED/E200/target_access=false；其他 seed 在真正接入前按相同原生来源检查。

DURA 沿原冻结模块验证 R69 元数据及其固定 R65 来源；源训练 L/U TX 隐藏、F-free、target-free 的原训练记录保留。历史用户指定版本和既往目标基准暴露如实披露。本 run 不重新选择权重。

## 数据容量缺项

ManyTx 每个当前目标 RX/TX 每日期 50 条，共 200 条。原 8 综合＋3 单场景各 36 support 池＋30 query 共需 726 条，不能执行。本 run 的 preparation_only=true 令队列仅包含 source smoke，目标作业数为 0。已向用户请求对会改变科学口径的数据调整作选择；不能将时间经过视为回复。

## 执行与停止

唯一 owner 为 codex-phase2-practical-20261028；不可覆盖 run/release，命令、环境和全部路径以 experiment.json/runtime 为准。先一次真实 checkpoint 无 query smoke；本 run 仅此作业。每卡总进程不超过 3，先铺每卡再增加槽；其他账号占用整卡跳过，其他 owner 进程保留。第一次技术/协议/输入/非有限/碰撞错误停止新派发，失败不自动重试。

本地 8 项聚焦测试通过；一次独立 P0/P1 静态审查通过，覆盖本次 source-only 首发和 publisher。发布后补唯一代码 commit、单 archive 传输比较、远端编译与 PID/CWD/argv/GPU/log 独立读回。

## 发布与独立读回

VERIFIED：代码版本 `71f1428c8ac85695ebffb2efd791fe34061b325b` 已提交、推送并独立核对 upstream OID；归档只比较一次传输 SHA，远端编译通过。控制器 PID 2043771 的 CWD、argv 与 run-root 已独立核实，source-only smoke 尚为 QUEUED。

首个客户端 SCP 在本地配置定位层失败，尚未连接远端。已独立核实 release/run/archive 不存在，再用项目根已验证的连接配置继续同一归档。恢复后传输、编译与控制器落地均 VERIFIED，没有重复启动。后续 publisher 支持显式 `--ssh-config`，配置缺失会在打包前失败；该回归项通过、唯一一次原问题定点复核通过，未热改本次远端冻结代码。

初次 preflight 有 7 张空闲卡；发布时其他账号已占用全部 8 张卡。队列遵守其他账号整卡避让，当前 worker 尚未进入 GPU；控制器会等待合适资源。不能将控制器存活或归档编译通过写成真实 checkpoint smoke 通过。目标性能实验未启动，数据容量选择尚未收到。

本机私有读回为 `local_artifacts/pamf_phase2_readiness_20261028/latest_readback.json`。完整数据问题及可行调整见 `docs/PAMF_PHASE2_CAPACITY_AUDIT_20261028.md`。当前无性能结果，无 query/truth 读取、无优化更新。

## 外部源权重复用核查补充

36/36 VERIFIED：8 个主比较家族×4 个种子共 32 份，以及旧版 Feature Separation 辅助版本×4 共 4 份。检查实际 completion/initialization/source_contract 与 last.pt 内容，role_ids/RX/day/ratio/split、类别、equalized1/256/unitRMS、scratch-only 来源、固定最终 E200 选择和 target_access 均通过；需 source base_state 的方法也已核实文件存在。没有进行目标推理或任何优化更新。具体逐行检查见 native_source_reuse_summary.json；实际 B/C 接口与当前版本身份仍需在相应 run 绑定，不能将此来源核查写成全部 29 条链已经运行。
