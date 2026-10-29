# Phase2 全部外部对比执行登记（2026-10-29）

本轮按原完整总表接入三套已验证数据。20 个可运行版本形成 28 条外部 S/B/C 链；CSIL 原始 MATLAB/Fisher、Tweak M10 延续原表 N/A，不能以修复版或单包版本冒名。

复用 44 个已核对实际源数据角色、物理 ID、表示、训练与继承来源的 Phase1 checkpoint。其余 6 组×4 seed＝24 个源基座从头训练；U_s TX 始终隐藏。源训练程序不接收目标数据句柄，正式访问目标前逐个通过真实 checkpoint 无 query smoke。

三池物理样本互斥；每池 8 个综合场景、3 个单场景各 5 个 realization，7 RX、4 model seed、K=1/5/10/20、N=0/2/5/10/20。每类剩余全部为 query，最小 176。三种单场景明确为 low 郊区、mid 城市、low 城市。

P0 使用固定 received IQ；P1 直接读取核心 run 已完成生产任务的同一 DURA v1 缓存，禁止另行恢复或训练 G。每个方法先完成第一 seed、首 pool/episode/RX 的全部 K/N 两路预测，再继续完整矩阵；此扩展只依据预测合法完成，绝不依据目标分数。

核心与外部统一调度，每 GPU 维持本轮 Phase2 两个 GPU worker，其他 owner 另计；进入时至少剩余 5000 MiB。只接管旧核心 CPU 调度器，原 GPU worker 的 PID、代码、命令、CWD 和输出路径保持。之后核心任务仍用原 release/runtime。CPU scorer 合计最多一个。

各 K 独立从对应源状态开始，C 的各 N 前缀/扩展独立复制实际 B；同一 B 供各 C 控制共享。原生注册仅取得当前新 support 和其声明允许继承的状态；组合控制取得当前全部新旧 support。query 只用于冻结 eval 推断，不改模型、BN、归一化、阈值或类配额。P0/P1 整组预测固定后独立 truth-last 评分。

## 方法、阶段性质与完整预算

| 版本 | 源处理 | B / C 性质 | 预算 |
|---|---|---|---|
| poster_ft | reuse；poster_ft | 原生DA机制＋明确项目C扩展;明确项目组合控制，不能叫原论文原生C；C=N_A_CLOSED_SET_METHOD | {'source_epochs': 200, 'source_batch': 128, 'source_optimizer': 'Adam', 'source_lr': 0.001, 'B_epochs': 10, 'B_lr': 0.001, 'B_batch': 256, 'native_C': False, 'source_U': False, 'implementation': 'existing_repo_POSTER_Homegrown_port', 'claim': 'author dense-layer-reset FT'} |
| radionet_ada_knn | reuse；radionet_ada | matched_WiSig_IQ_support_only_extension；C=AVAILABLE | ADA10000_iterations;cosine_distance_weighted_kNN_k=K |
| radionet_ada_ft_knn | reuse；radionet_ada | 原生B机制＋项目C追加扩展；C=AVAILABLE | ADA10000_iterations+support_FT30_epochs;cosine_distance_weighted_kNN_k=K |
| radionet_df_ft | reuse；radionet_df_ft | author final-layer-reset FT；C=N_A_CLOSED_SET_METHOD | {'source_epochs': 200, 'source_batch': 128, 'source_optimizer': 'Adam', 'source_lr': 0.001, 'B_epochs': 30, 'B_lr': 0.001, 'B_batch': 128, 'native_C': False, 'source_U': False, 'implementation': 'existing_repo_RadioNet_DF_port', 'claim': 'author final-layer-reset FT'} |
| radionet_triplet | train；radionet_triplet | all positive pairs, epochwise source-only semihard mining, cosine distance-weighted kNN; N-MEV support mean；C=AVAILABLE | {'source_epochs': 30, 'source_batch': 128, 'source_optimizer': 'Adam', 'source_lr': 0.001, 'B_epochs': 0, 'native_C': True, 'source_U': False, 'implementation': 'DF64_cosine_triplet_semihard_Torch_port', 'claim': 'all positive pairs, epochwise source-only semihard mining, cosine distance-weighted kNN; N-MEV support mean'} |
| protonet_cda | reuse；protonet | 原生episode/B机制＋项目C扩展；C=AVAILABLE | B0;C0;support_squared_Euclidean_prototypes |
| mrior_sda | reuse；mrior | 原生DA机制＋明确项目C扩展;明确项目组合控制，不能叫原论文原生C；C=N_A_CLOSED_SET_METHOD | B100_outer_updates;7_estimator_updates_per_outer;lr0.0006 |
| dadda_sda | reuse；dadda | 原生DA机制＋明确项目C扩展;明确项目组合控制，不能叫原论文原生C；C=N_A_CLOSED_SET_METHOD | B100_updates;support_true_label_conditional_alignment |
| twostage_sda | reuse；twostage | 原生DA机制＋明确项目C扩展;明确项目组合控制，不能叫原论文原生C；C=N_A_CLOSED_SET_METHOD | B100_DANN+100_LMMD+100_support_FT_updates |
| feature_separation_hu2024 | train；feature_separation_hu2024 | 原生DA机制＋明确项目C扩展;明确项目组合控制，不能叫原论文原生C；C=N_A_CLOSED_SET_METHOD | {'source_epochs': 200, 'source_batch': 256, 'source_optimizer': 'Adam', 'source_lr': 0.005, 'source_early_patience': 20, 'source_lr_patience': 10, 'source_lr_factor': 0.1, 'B_epochs': 100, 'B_lr': 0.005, 'B_batch': 256, 'native_C': False, 'source_U': False, 'implementation': 'existing_repo_Hu2024_independent_reproduction', 'claim': 'shared attention encoder; TX/RX CE, Frobenius separation, both entropy terms; target RX frozen'} |
| feature_separation_legacy | reuse；feature_separation | auxiliary_legacy_feature_separation;NOT_Hu2024；C=N_A_CLOSED_SET_METHOD | B100epochs_one_update_per_epoch_Adam0.001;legacy_TX_RX_port |
| ra_collab_single_ft | train；ra_collab_single_ft | 原生DA机制＋明确项目C扩展;明确项目组合控制，不能叫原论文原生C；C=N_A_CLOSED_SET_METHOD | {'source_epochs': 100, 'source_batch': 64, 'source_optimizer': 'SGD', 'source_lr': 0.001, 'source_early_patience': 20, 'source_lr_patience': 10, 'source_lr_factor': 0.2, 'B_epochs': 20, 'B_lr': 1e-05, 'B_batch': 32, 'native_C': False, 'source_U': False, 'implementation': 'existing_repo_RA_Collab_GRL_single_node', 'claim': 'single-node FT extension; no cross-query fusion'} |
| tweak_m1 | train；tweak_m1 | 原生DA机制＋明确项目C扩展;明确项目组合控制，不能叫原论文原生C；C=AVAILABLE | {'source_epochs': 100, 'source_batch': 64, 'source_optimizer': 'SGD', 'source_lr_grid': (0.01, 0.001, 0.0001, 1e-05, 1e-06), 'B_epochs': 0, 'native_C': True, 'source_U': False, 'implementation': 'existing_repo_Tweak_12D_batch_hard', 'claim': 'first 128 samples of each 256-IQ received window, M=1 explicit extension; centroid and mean L2 radius'} |
| tweak_m10 | N/A | Native M=10 needs ten legally linked same-transmitter observations per query track. The current capsule has one fixed 256-IQ window per opaque physical ID and no predeclared same-TX track ID. Query truth, reshape and random grouping cannot supply those tracks. | 0 active |
| csil_corefix | reuse；csil | 原生C核心＋数值修复版＋B任务扩展；C=AVAILABLE | B10epochs_Adam0.001;C3epochs_masked_SGDM0.01;source_empirical_Fisher |
| csil_author | N/A | Original MATLAB workspace/author source Fisher is not a runnable Python version of the current capsule. The historical empirical-Fisher corefix source cannot be relabeled as original exp(gradient_squared) Fisher; that original numerical path overflowed even float64. | 0 active |
| mopc_paper | reuse；mopc_hr | 论文语义复现＋匹配B/数据扩展；C=AVAILABLE | B10epochs_Adam0.001;C20epochs_per_5class_session_SGD0.01;paper_equations |
| mopc_public_code | reuse；mopc_hr | public_code_loss_and_correction_semantics_matched_IQ_task_extension;not_original_dataset_reproduction；C=AVAILABLE | B10epochs_Adam0.001;C20epochs_per_5class_session_SGD0.01;public_code_semantics |
| orthogonal | reuse；orthogonal | 原生C机制＋B/任务扩展；C=AVAILABLE | B10epochs_SGD0.08;C50epochs_SGD0.08 |
| issl_fixed_v1 | train；issl_fixed_v1 | 原生C流程修复版＋B10扩展；C=AVAILABLE | {'source_epochs': 300, 'source_batch': 128, 'source_optimizer': 'Adam', 'source_lr': 0.03, 'B_epochs': 10, 'B_lr': 0.0001, 'B_batch': 32, 'C_SSL_epochs': 2000, 'C_transfer_epochs': 300, 'C_SSL_lr': 1e-05, 'C_transfer_lr': 0.03, 'native_C': True, 'source_U': False, 'implementation': 'explicit_repaired_v1_independent_topology_port', 'claim': 'repaired ISSL extension; not author-behavior parity'} |
| issl_author_behavior | train；issl_fixed_v1 | detached KD, accumulated SSL gradients, reshape queue and parameter-only momentum retained; B10 is project extension；C=AVAILABLE | {'source_epochs': 300, 'source_batch': 128, 'source_optimizer': 'Adam', 'source_lr': 0.03, 'B_epochs': 10, 'B_lr': 0.0001, 'B_batch': 32, 'C_SSL_epochs': 2000, 'C_transfer_epochs': 300, 'C_SSL_lr': 1e-05, 'C_transfer_lr': 0.03, 'native_C': True, 'source_U': False, 'implementation': 'explicit_author_semantics_independent_topology_port', 'claim': 'detached KD, accumulated SSL gradients, reshape queue and parameter-only momentum retained; B10 is project extension'} |
| lora_triplet_knn | train；lora_triplet_knn | 原生追加enrollment行为；Torch/256点/阶段映射为扩展；C=AVAILABLE | {'source_epochs': 1000, 'source_batch': 32, 'source_optimizer': 'RMSprop', 'source_lr': 0.001, 'source_early_patience': 20, 'source_lr_patience': 10, 'source_lr_factor': 0.2, 'B_epochs': 0, 'native_C': True, 'source_U': False, 'knn_k': 15, 'implementation': 'independent_Torch_LoRa_256IQ_STFT64_overlap32', 'claim': 'Torch port with short-IQ STFT extension; TF2.1 RMSprop numerical parity unverified'} |

ISSL C 保持 SSL2000＋transfer300；MoPC 每 5 新类 session20，N20 为 4 session；RadioNet ADA10000＋FT30。Tweak 的源 LR 搜索明确为 5×E100＝总 E500，不冒称单次 E100 总成本。POSTER/DF 复用原 E200 的真实 clean＋sat、batch128 配方，历史增强和 source-only 选模身份保留。

## 逐链 S/B/C 与继承

| ID | 方法 | B | C | 状态继承 | 性质 |
|---|---|---|---|---|
| J01 | RadioNet ADA＋FT＋kNN 注册扩展 | source L_s＋旧 support 做 ADA→FT→kNN | 冻结 B encoder，追加新 target kNN memory | B encoder＋旧 memory＋距离/投票规则 | 原生B机制＋项目C追加扩展 |
| J02 | ProtoNet CDA 注册扩展 | 冻结自身 episode encoder，旧 support 类均值 | 追加新类 prototype，全部类平方欧氏竞争 | B encoder＋旧 target prototype | 原生episode/B机制＋项目C扩展 |
| J03 | CSIL corefix＋B扩展 | 旧 target support SFT；明确 B 扩展 | 通道隔离/mask/KD/Fisher；C仅新support及合法历史 | 实际B模型/参考＋固定Fisher；不恢复另一个源base | 原生C核心＋数值修复版＋B任务扩展 |
| J04 | MoPC-HR论文语义＋B扩展 | 旧 support SFT＋声明的原型校正；B 扩展 | 最多5新类/session；原型增强/HR/漂移校正 | B权重/已校正旧prototype/参考；session继续前传 | 论文语义复现＋匹配B/数据扩展 |
| J05 | Orthogonal SEI＋B扩展 | 冻结 encoder，校准旧分类权重；B 扩展 | 冻结encoder/旧权重，新权重初始化与校准 | B旧权重＋源正交结构；不替换encoder | 原生C机制＋B/任务扩展 |
| J06 | ISSL fixed_v1＋B10 | 旧 support 全网 SFT；B10 扩展 | 扩头→SSL→CE transfer，保留合法旧训练样本replay | 实际B→teacher/key等；按原生C初始化/更新queue，不沿用另一run | 原生C流程修复版＋B10扩展 |
| J07 | LoRa triplet＋KNN enrollment | 自己的冻结 triplet 表示，旧 target enrollment | 冻结表示，追加新 enrollment；保持KNN规则 | B encoder＋旧 target enrollment | 原生追加enrollment行为；Torch/256点/阶段映射为扩展 |
| J08 | POSTER FT→原生输出扩展FT | 冻结卷积，训练声明的 Dense | 复制B旧输出权重/bias，新增类输出后联合target CE | B全部前置层/旧输出权重/bias；新行合法support初始化 | 原生DA机制＋明确项目C扩展 |
| J09 | MRIOR-SDA→原生输出扩展FT | GAD/DV-KL＋support CE；CPL关闭，SDA扩展 | 复制B旧输出权重/bias，新增类输出后联合target CE | B全部前置层/旧输出权重/bias；新行合法support初始化 | 原生DA机制＋明确项目C扩展 |
| J10 | DADDA-SDA→原生输出扩展FT | source/support CE＋动态MMD/LMMD；SDA扩展 | 复制B旧输出权重/bias，新增类输出后联合target CE | B全部前置层/旧输出权重/bias；新行合法support初始化 | 原生DA机制＋明确项目C扩展 |
| J11 | Two-stage SDA＋FT→原生输出扩展FT | source/support DANN→LMMD→support FT | 复制B旧输出权重/bias，新增类输出后联合target CE | B全部前置层/旧输出权重/bias；新行合法support初始化 | 原生DA机制＋明确项目C扩展 |
| J12 | Hu2024 新完整实现→原生输出扩展FT | 更新共享 encoder/TX 路径，RX 路径冻结 | 复制B旧输出权重/bias，新增类输出后联合target CE | B全部前置层/旧输出权重/bias；新行合法support初始化 | 原生DA机制＋明确项目C扩展 |
| J13 | RA-Collab 单节点→原生输出扩展FT | 单节点全网旧 support FT；GRL关闭 | 复制B旧输出权重/bias，新增类输出后联合target CE | B全部前置层/旧输出权重/bias；新行合法support初始化 | 原生DA机制＋明确项目C扩展 |
| J14 | Tweak M1→追加类校准 | 旧 support centroid/mean-L2-radius 校准 | 保留旧centroid/radius，追加新类校准并用原closed-set判决 | B encoder＋旧centroid/radius | 原生DA机制＋明确项目C扩展 |
| J15 | POSTER FT→NCM | 冻结卷积，训练声明的 Dense | B表示下重新编码全部新旧support，归一化类均值cosine判决 | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J16 | MRIOR-SDA→NCM | GAD/DV-KL＋support CE；CPL关闭，SDA扩展 | B表示下重新编码全部新旧support，归一化类均值cosine判决 | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J17 | DADDA-SDA→NCM | source/support CE＋动态MMD/LMMD；SDA扩展 | B表示下重新编码全部新旧support，归一化类均值cosine判决 | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J18 | Two-stage SDA＋FT→NCM | source/support DANN→LMMD→support FT | B表示下重新编码全部新旧support，归一化类均值cosine判决 | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J19 | Hu2024 新完整实现→NCM | 更新共享 encoder/TX 路径，RX 路径冻结 | B表示下重新编码全部新旧support，归一化类均值cosine判决 | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J20 | RA-Collab 单节点→NCM | 单节点全网旧 support FT；GRL关闭 | B表示下重新编码全部新旧support，归一化类均值cosine判决 | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J21 | Tweak M1→NCM | 旧 support centroid/mean-L2-radius 校准 | B表示下重新编码全部新旧support，归一化类均值cosine判决 | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J22 | POSTER FT→JointCE | 冻结卷积，训练声明的 Dense | B表示下新建全部类共享线性头；全部新旧support CE | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J23 | MRIOR-SDA→JointCE | GAD/DV-KL＋support CE；CPL关闭，SDA扩展 | B表示下新建全部类共享线性头；全部新旧support CE | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J24 | DADDA-SDA→JointCE | source/support CE＋动态MMD/LMMD；SDA扩展 | B表示下新建全部类共享线性头；全部新旧support CE | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J25 | Two-stage SDA＋FT→JointCE | source/support DANN→LMMD→support FT | B表示下新建全部类共享线性头；全部新旧support CE | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J26 | Hu2024 新完整实现→JointCE | 更新共享 encoder/TX 路径，RX 路径冻结 | B表示下新建全部类共享线性头；全部新旧support CE | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J27 | RA-Collab 单节点→JointCE | 单节点全网旧 support FT；GRL关闭 | B表示下新建全部类共享线性头；全部新旧support CE | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |
| J28 | Tweak M1→JointCE | 旧 support centroid/mean-L2-radius 校准 | B表示下新建全部类共享线性头；全部新旧support CE | 只保留B表示/buffer；替换原B判决，不继承其分类权重 | 明确项目组合控制，不能叫原论文原生C |

原闭集 DA 方法的原生 C 记 N/A；输出扩展 FT100、NCM、JointCE100 分列为项目组合，不能作为原论文原生注册结果。前置 B 实际只拟合一次，各 C 分支复制该 B。FT100 继承真实 B 前置状态与旧输出行；NCM/JointCE100 只继承 B 表示，替换判决规则。

## 启动、停止与产物

运行 ID：`20261029-phase2-external-threepool-m4-r01`。配置：`configs/pamf_phase2_external_threepool_execution_20261029.json`。唯一 launch owner：`codex/root/phase2-external-threepool-20261029`。实际代码以发布 Git commit 记载，原核心执行 commit 保持 `20f920985a97376aad2ac9663d33dcb9b81e3c49`。

命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.pamf_phase2_practical.external_dispatch --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase2-external-threepool-m4-r01-release/runtime.json`；环境：N607 原 CVS-RFFI Python；CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase2-external-threepool-m4-r01-release`。

Any protocol/numerical/worker/output failure blocks that source/profile and descendants, preserves partials and unrelated healthy workers; no score-based stop/retry/selection. Core systemic failure stops new core work and unavailable shared-G descendants.

预测任务 77280，独立评分任务 38640，源权重资格/训练任务 68；核心现有任务继续并复用已完成结果。预算大，登记和排队不等于完成。源/目标训练日志保留实际 loss、更新数、耗时和状态，失败项不得补成零分或悄悄缩短训练。

完整 B 权重只保存每方法首 cap/seed 示例，所有任务保存 B 来源、资源与实际继承说明，并在执行中验证 source/B 未被 C 修改；避免全量复制大模型耗尽磁盘。预测和原始日志留在服务器，不提交 Git。

当前状态：PREREGISTERED，启动后的独立进程/GPU/日志证据将回填本报告。
