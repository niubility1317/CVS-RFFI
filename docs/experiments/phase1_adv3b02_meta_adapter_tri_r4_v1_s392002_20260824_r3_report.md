# CVS_META_ADAPTER_TRI_R4_V1 r3 N607预登记报告

- run ID：`phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3`
- 状态：`RUNNING`
- 时间：2026-08-25（Asia/Hong_Kong）
- 修复提交：`2c092018888153e91434b1bf2f418d18b63f2597`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`

## 候选与矩阵

P0为冻结base控制；P1为随机adapter；P2为source监督adapter；P3为FOMAML固定LR；P4为FOMAML+Meta-SGD。P1～P4顺序运行，科学失败不阻断后续候选。Phase1只使用source角色训练和选择，最终checkpoint分别评价clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`。

## 本地修复验证

- `r1`暴露launcher未支持部署输入覆盖；`r2`进一步暴露Phase1入口没有消费已经正确传入`train.py`的绝对输入路径。
- 修复后launcher和Phase1入口统一使用显式只读checkpoint／ManySig路径，不改变候选、训练算法、数据角色或Phase2边界。
- 第二轮RED测试在旧入口稳定复现release relocation失败；GREEN后Phase1入口31项、邻近trainer／Phase2 runner／scorer68项，共99项通过；`r3`配置加载和生产入口`py_compile`通过。

## N607最小预登记

- N607账户：普通`N607`用户`szu2070436088`
- GPU：0；每GPU并发训练数计划为1。
- release归档本地路径：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3_2c092018.tar.gz`
- release归档远端路径：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3_2c092018.tar.gz`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3/checkout`
- 冻结checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 启动命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py --config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r3 --base-checkpoint /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --gpu 0`
- expected artifacts：每个P1～P4子目录的`logs.jsonl`、`metrics.csv`、`selected_meta_bundle.pt`、`source_adaptation_curve.json`、`run_summary.json`、`p0_control_evaluation.json`、`final_checkpoint_evaluation.json`、`frozen_prototypes.npz`，以及矩阵级`candidate_matrix_summary.json`。
- 技术停止规则：仅在协议越权、错误checkout/output root、输出覆盖、无法产生规定artifact、launcher-wide故障，或至少两个候选出现相同确定性pre-artifact异常时停止；不得因低准确率停止。

`r1/r2`均已封为技术失败且不再使用。`r3`在归档SHA核对、远端编译和启动健康检查完成前不声明`RUNNING`，当前没有性能结果。

## N607发布与启动状态

- 发布时间：2026-08-25 01:06（Asia/Hong_Kong）。
- release归档本地与远端SHA256均为`c73699c902981ced9d3131b74dab52f366c8b926378dbcad4648ed432a7ac433`，比较一致；远端checkout已加载本次修复代码和`r3`配置。
- checkout内六个预登记生产入口已生成Python3.10编译产物；实际launcher和训练入口随后成功加载，未再出现`r1/r2`的路径传播异常。
- launcher PID为`2498514`，训练子PID为`2498587`；两者CWD均为预登记`r3/checkout`，cmdline分别绑定`r3`配置、output root、绝对checkpoint、ManySig和GPU0。
- 启动前stdout日志不存在，启动后为6399字节；GPU0仅见训练子进程。15秒复核中CPU ticks从6740增至8240，子进程保持100%CPU，GPU0显存增至496MiB，符合数据初始化阶段的持续计算状态。
- 当前最高状态：`RUNNING`。尚未形成P1完成artifact、候选矩阵或Phase2性能结果，不能声明正向收益。

## 运行中非阻断效率修复

- 2026-08-25 01:44只读复核时，P1训练子进程仍保持100%CPU和`R`状态，CPU ticks持续增长、错误扫描为空，但stdout仍为6399字节，P1仅有`config_snapshot.json`。证据表明它仍在全量source/clean索引清单阶段推进，而非僵死。
- 根因是物理样本互斥检查通过数据集`__getitem__`逐条读取、裁剪并归一化IQ，但该检查实际只需要已经构建的WiSig索引元数据。
- 本地后续代码改为优先从`WiSigCompactDataset.index`和`WiSigSubsetDataset.index`生成相同`physical_sample_id`；不具备索引的兼容数据集仍沿用原路径。新增RED→GREEN负测明确禁止清单扫描解码IQ。
- Phase1入口32项、模型／训练器／Stage2适配47项、checkpoint／内循环／目标函数88项，共167项回归通过。
- 该修复没有同步到正在运行的`r3`，也没有改变`r3`的checkout、进程或output root；`r3`仍严格归属于提交`2c092018888153e91434b1bf2f418d18b63f2597`。只有后续新的不可覆盖run才可消费此优化。

## Stage2正式输入接线修复

- 等待r3期间只读核对Stage2现有入口，发现`stage2_target_row_export`虽然把选中的support token保存在审计JSON中，但输出NPZ只有IQ和标签；Meta-Adapter runner严格要求`received_iq/support_labels/support_physical_ids`三个字段，原接线会在正式Target5启动前合法失败。
- RED测试稳定证明缺少`support_physical_ids`；GREEN后导出器把与IQ／标签相同rank-prefix选中的不可变token写为非object字符串向量，不改变K、support内容或query处理。
- exporter、Meta-Adapter Stage2适配、模型和trainer联合56项通过；仅有既存AMP弃用提示。
- 该修复只为Phase1完成后的新Stage2不可覆盖run准备，不修改或重启正在运行的r3。

## Stage2真实truth sidecar评分兼容

- 只读核对既有独立scorer代码和冻结类绑定后确认：正式`truth_sidecar.json`使用`true_class_handle`，同一场景含全部520个query token；REG0旧类指标只定义于其中`evaluation_role=target_old`的120条。原Meta-Adapter scorer只接受整数`true_class_index`并会把全部token都当旧类，无法合法评分真实sidecar。
- 修复后的scorer先验证receipt及DA0_REG0／DA1_REG0两份完整prediction，再验证冻结`d19`类绑定的class index集合与bundle注册类一致，最后才打开truth。它按receipt场景精确连接全部opaque token，只把`target_old`通过冻结handle→class index映射送入旧类均值和floor；target new token只参与完整性连接，REG0新类指标仍为`N/A`。
- 简化整数truth保持向后兼容；真实sidecar RED→GREEN及scorer／runner／exporter联合52项通过。
- 独立scorer根已定位为既有`.../before/scorer/truth_sidecar.json`，当前只核对路径名和文件大小，尚未为本方法打开truth内容。

## 2026-08-25 02:28只读运行复核

- r3 launcher PID`2498514`与P1训练子PID`2498587`仍存在，cmdline继续精确绑定r3 checkout、output root、绝对checkpoint、ManySig和GPU0。
- P1子进程累计运行约4886秒，保持`R`状态和100%CPU，RSS约3.67GiB；GPU0为0%利用率、496MiB显存。日志仍为6399字节，run root仍只有`_configs/P1.json`与`P1/config_snapshot.json`。
- `/proc/2498587/io`显示累计`rchar=2394463497`且进程持续获得CPU；日志未发现`Traceback/Error/Exception/FAILED`。最高可证状态仍为`RUNNING`，没有P1训练完成、四场景评价或性能结果。
- 该状态符合旧release在全量manifest阶段逐条解码IQ的已知慢路径，不构成预登记技术失败；不得终止、重启或重复启动r3。

## Phase2 Target5数据切片核对

- 正式Meta-Adapter设计和项目既有125定义均固定五个切片为`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`和`K1/new20`；不得因现有文件方便把`K10/new10`替换成`K2/new20`。
- 只读核对晚块实验复制的25份`VALIDATED_ONCE`manifest后确认，该资产属于另一套`K1/K2/K5/K10/new5/new20`矩阵，每个receiver只有`K10/new5`、`K10/new20`、`K5/new20`、`K2/new20`和`K1/new20`，没有`K10/new10`。
- N607的既有`stage2_inputs`与run目录定向文件名搜索均未找到权威`K10/new10`manifest；现有package根也只包含`before/new5/new20`。因此当前不得把`K10/new20`句柄改名或伪装为`K10/new10`。
- Phase1 source选择通过后，Target5启动前必须从固定received-IQ资产按一次性builder边界补齐真实`K10/new10`切片并取得其`p2_min_v1/VALIDATED_ONCE/capsule_id/split_id`；这属于缺失数据row补建，不改变Meta-Adapter方法，也不触发其他已验证切片重验。

## Target5 truth-blind矩阵执行器

- 现有正式入口只能执行单个row，无法直接闭合15行Target5；因此新增`stage2_meta_adapter_matrix.py`和CLI，仅顺序调用既有单row runner，不接受、打开或传递truth。
- RED阶段5项测试以模块缺失稳定失败；GREEN后联合runner、scorer和target-row exporter共58项通过，两个生产入口`py_compile`与`git diff --check`均通过。
- 执行器在创建输出root前验证完整Target5/Target25笛卡尔积、固定operating point与K的对应关系、单一candidate/bundle/seed、row ID安全性和单row配置allowlist；`K2/new20`不能替代`K10/new10`。
- 任一row技术失败时只停止后续派发，写入矩阵失败状态并保留所有已完成或失败诊断产物；不删除、不覆盖、不自动重跑。prediction完整后仍由独立scorer在执行器之外truth-last评分。
- 这项结果只证明执行链闭合，不是性能证据；r3仍为`RUNNING`，Target5尚未启动。

## K10/new10缺失输入补建预登记

- 任务范围：只为receiver=`20-1`、seed=`713101`、method/support/query/draw seed=`7282101/7282201/7282301/7282401`补建一个`K10/new10`输入；不重建received-IQ，不改动既有new5/new20、K1/K5切片，也不触发其他`VALIDATED_ONCE`数据重验。
- 固定母缓存：`/home/szu2070436088/2510044040/CV-SincNet/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix/rx_20_1/seed_713101/cache_set.json`。
- 固定builder release：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_full_ablation_phase2c_t1_20260730_v2_0903163e`；该release已成功闭合相同receiver/seed的before、new5和new20封包。
- 新不可覆盖root：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/adv3b02_meta_adapter_target5_new10_s713101_20260825_v1_ce07a101`。
- new10严格取同一draw seed的new20排列前10类：`1-16,18-8,2-16,14-11,11-19,4-10,2-5,19-8,19-9,20-12`；禁止手工改序或从new20改名复制。
- package完成后，以既有before包和新new10包生成一个K10 Stage2-B manifest；仅使用空闲GPU进行固定feature/manifest构建。任何已有output、builder异常、协议字段不匹配或产物不完整都立即停止本次补建，不影响r3和其他任务。
- 该补建只产生Target5输入，不读取性能，不启动Meta-Adapter Target5，也不代表Phase1已通过source选择。

## K10/new10补建结果

- 2026-08-25 03:02～03:04完成固定母缓存到new10 package及单个K10 feature/manifest构建；物理GPU1只用于该固定feature构建，结束后显存回到1MiB。r3的GPU0进程、release、日志和output root均未改变。
- new10 predictor package root SHA256=`d6faa799aefc6b739afb23d66e225ac328eb960e4ec3fe84361c38c70b09dade`，seal SHA256=`07e67f8f50dfaba724d305c29c23b5e95751f1d9365e885bef615f08b13f6338`；注册类16、support pool160、query320，三个场景文件均完整。
- `offline_build_audit.json`状态为`PASS`，确认正式Phase1 class binding被使用、support/query物理ID同场景互斥、跨场景物理ID互斥、单物理样本单LEO观测合规、clean sample access=false。
- 权威Stage2-B manifest：`.../artifacts/feature_k10_new10/stage2b/features.manifest.json`，SHA256=`3a1b81815b9898d44e63e6f923d59192a1f16c72741cf0aa7e90c01de878c919`；payload SHA256=`593b4d52c29569bb4d1872358e513817d3eda3a50fe27cea62acb9746dcecd68`。
- manifest固定`phase2_data_status=VALIDATED_ONCE`、`capsule_id=d18-reuse-validated-once-rx20-1-seed713101-m7282101-k10-new10`、`split_id=p2_min_v1-rx20-1-m7282101-s7282201-q7282301-d7282401-k10-new10`、`stage_scope=stage2b`、`k_shot=10`；query truth/role、source sample/cache/label/replay访问均为false。
- Stage2-A和Stage2-C cache是同一次固定builder的完整副产物，本目标不消费其特征或旧D92判决；Meta-Adapter正式run仍只从固定received-IQ package导出raw support/query IQ，在新meta checkpoint上做真实梯度更新。
- 当前提交的target-row exporter已对真实`before/support_leo_clear_weak.npz`完成一次support-only no-query smoke：输入60行，K10输出6类×10行，物理ID60/60唯一且与IQ/标签对齐。
- smoke输出NPZ精确只有`received_iq/support_labels/support_physical_ids`；audit确认`query_input_opened=false`、query行数0、`query_truth_opened=false`、`query_role_opened=false`。该检查未复制或打开query文件，也没有性能结果。
- 最新分支随后执行覆盖Phase1 meta episode/inner loop/trainer/checkpoint/真实入口和Phase2 adaptation/handoff/target factory/runner/matrix/scorer/exporter的宽回归，共259项全部通过；仅有既存AMP API弃用警告。
- 结论：原Target5五切片中的`K10/new10`数据缺口已闭合；仍须等待r3 Phase1完成并通过source选择后，才能生成正式Target5配置和启动性能实验。

## Target5输入工厂

- 新增`stage2_meta_adapter_target_factory.py`及CLI，把5个Target5或25个Target25的权威Stage2-B manifest与固定raw-IQ package转换为15/75行runner配置。
- 工厂在创建输出root前核对完整receiver×operating-point集合、K映射、三个LEO_WEAK场景、`VALIDATED_ONCE`、非空capsule、`p2_min_v1` split，以及manifest中的query truth/role和source sample/cache/label/replay访问均为false。
- 每个row只调用已验证exporter，输出support的`received_iq/support_labels/support_physical_ids`和query的`received_iq/query_ids`；plan和CLI均没有truth/scorer输入字段。
- RED阶段4项测试以模块缺失稳定失败；GREEN后factory/exporter/matrix/runner聚焦46项通过，加入完整Meta-Adapter宽回归后259项通过，生产入口`py_compile`与`git diff --check`均通过；新增负测拒绝把K10/new20 manifest冒充K10/new10。
- 工厂需要最终选中的`selected_meta_bundle.pt`和`frozen_prototypes.npz`路径，因此本轮只闭合实现和测试；正式15行配置仍必须等待r3 source选择结果，不能用占位checkpoint提前发布。

## Phase1 episode引用快路径补全

- 只读分析r3旧release后确认：此前`12410f2a`只让source/clean物理ID重叠检查避开IQ解码，但`_build_refs`仍会遍历L_s、V_cal和V_select的全部样本并调用dataset `__getitem__`，因此后备新run仍可能重复长时间pre-artifact准备。
- 新RED负测用只允许读取`dataset.index`、禁止`__getitem__`的WiSig载体稳定复现失败；GREEN后`_build_refs`直接从索引读取`tx_i/rx_i/day_i/eq_i/sig_i`，并用项目既有函数生成完全相同的physical ID和capture block。
- ref仍保留原dataset index、source角色及`clean/leo_clear_weak/leo_low_elev_weak/leo_rain_weak`四个view；实际IQ只在确定性episode被选中后由`_episode_batch`物化，不改变样本划分、K、训练参数、目标函数或评价。
- 后续复杂度负测进一步证明：旧采样器重复同seed两次会重建计划2次，同一class/spec会扫描1152次而非一次扫描的576次，RX holdout首次计划构建会迭代descriptor集合73次。修复后candidate plan和class/spec pool按冻结refs缓存，五类计划按domain等价键分组；完整计划数固定为36/144/360/54/108，重复seed仍生成完全相同episode。
- clean与三类LEO弱场景评价原先会先把整个held-out split的单场景IQ堆叠为一个大张量，再由模型内部分128条推理；这会产生不必要的全split内存峰值。RED负测用129条样本禁止超过128条的任意`torch.stack`，旧实现稳定失败；GREEN后评价逐场景、固定128条流式物化IQ并累计总数、正确数和逐类结果。
- 流式实现仍按相同physical ID和seed生成每个LEO view；小型真实四场景变换测试证明其完整输出与旧物化算法逐字段一致。Phase1真实入口35项、入口/episode采样器/trainer联合88项及当前Phase1/Phase2 Meta-Adapter 14文件宽回归265项通过，仅有既存AMP API弃用警告。该修复只供r3发生预登记技术失败后的新不可覆盖run使用；当前r3仍保持只读运行，不重启、不覆盖。

## 2026-08-25 04:27只读运行复核

- r3 launcher PID`2498514`和P1训练子PID`2498587`仍存在；子进程累计运行约3小时20分，保持`R`状态、100%CPU、RSS约3.67GiB，CPU时间与墙钟时间继续同步增长。
- `/proc/2498587/io`仍显示已经完整读取ManySig pickle；run root仍只有`_configs/P1.json`和`P1/config_snapshot.json`，stdout仍为6399字节，异常指纹计数为0。
- 旧release在artifact前同时包含逐样本IQ解码ref构建和重复candidate-plan扫描，能解释当前CPU长时间占用；但预登记没有时间停止线，进程仍持续计算，因此证据只支持`RUNNING`，不支持技术失败、终止、重启或重复启动。
- r4本地release归档已按固定提交`70961b7a9e9f952cec6160036b6b09ea0db5e415`准备完成，SHA256=`0f54bf1c5ca587986de6c1789455c3aec867c6c2fd13fc64107288368d571a20`；尚未同步N607或启动。准备状态已提交并推送至`c8a3d7b60df590b1242e4726e7a36765a0bfa1ce`。

## Target5五切片输入路径闭合

- 从已完成的Phase2-C v5 binding registry中仅按row identity筛出receiver=`20-1`、method/support/query/draw seed=`7282101/7282201/7282301/7282401`的四个权威Stage2-B manifest：`K10/new5`、`K10/new20`、`K5/new20`和`K1/new20`；未打开scorer truth。
- 四个既有manifest与新补建`K10/new10`manifest均为`cvs.full_ablation.phase2.feature_cache_manifest.v2`、`stage_scope=stage2b`、`phase2_data_status=VALIDATED_ONCE`，receiver、K、capsule和`p2_min_v1` split逐项匹配；query truth/role及source sample/cache/label/replay访问标志均为false。
- 正式raw-IQ接线固定为：五个操作点的support均取v2 package中`before/predictor/support_<scenario>.npz`；query分别取`new5`、新建`new10`或`new20` package中的`query_<scenario>.npz`。`K5`和`K1`只由相同固定support按既有rank-prefix导出，不生成额外物理样本。
- 当前只完成manifest和路径只读核对，没有导出正式15行、没有打开query truth/role、没有启动Phase2。Target5工厂仍须等待Phase1选中的真实`selected_meta_bundle.pt`与`frozen_prototypes.npz`，不得用占位checkpoint提前生成正式配置。
