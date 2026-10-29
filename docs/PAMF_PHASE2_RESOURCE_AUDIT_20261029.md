# PAMF Phase2 域适应与新类注册资源记录核验

**结论：当前有实测资源记录，但覆盖不完整、计量范围不统一，尚不足以支持全部方法的公平效率比较。** 本次将资源消耗作为与识别性能并列的独立报告项，保留缺失值；不把已有准确率结果当作资源证据完整。

核验时刻：2026-10-29T14:57:32Z UTC（香港时间 UTC+8）。检查全部20个已接入外部版本的首capsule／首modelseed元数据，优先P0、K10、N10；未完成K10时仅观察已写入B状态。36/40条P0/P1路径已有完整manifest。该抽查用于发现计量覆盖缺口，不是完整矩阵资源统计。未读取query IQ、预测数组、truth或性能分数。

外部B阶段有独立实测耗时的版本为11/20，记录CUDA allocated峰值的版本为11/20。其他方法仍有epoch/参数或日志信息，但不能据此补造耗时和显存。

| 方法／版本 | B实测耗时 | B实测显存峰值 | C现有记录 |
|---|---|---|---|
| poster_ft | 缺失 | 缺失 | J08：耗时/步数；缺显存；J15：耗时/步数；缺显存；J22：耗时/步数；缺显存 |
| radionet_ada_knn | 288.405235 s | 167.206 MiB | radionet_ada_knn：耗时/步数；缺显存 |
| radionet_ada_ft_knn | 308.364490 s | 166.988 MiB | J01：耗时/步数；缺显存 |
| radionet_df_ft | 缺失 | 缺失 | N_A_NATIVE_CLOSED_SET |
| radionet_triplet | 缺失 | 缺失 | radionet_triplet：缺独立计时；缺显存 |
| protonet_cda | 0.015646 s | 377.992 MiB | J02：耗时/步数＋显存 |
| mrior_sda | 6.877619 s | 293.176 MiB | J09：耗时/步数；缺显存；J16：耗时/步数；缺显存；J23：耗时/步数；缺显存 |
| dadda_sda | 12.635895 s | 2148.409 MiB | J10：耗时/步数；缺显存；J17：耗时/步数；缺显存；J24：耗时/步数；缺显存 |
| twostage_sda | 6.928101 s | 150.270 MiB | J11：耗时/步数；缺显存；J18：耗时/步数；缺显存；J25：耗时/步数；缺显存 |
| feature_separation_hu2024 | 缺失 | 缺失 | J12：耗时/步数；缺显存；J19：耗时/步数；缺显存；J26：耗时/步数；缺显存 |
| feature_separation_legacy | 4.703568 s | 798.915 MiB | N_A_NATIVE_CLOSED_SET |
| ra_collab_single_ft | 缺失 | 缺失 | J13：耗时/步数；缺显存；J20：耗时/步数；缺显存；J27：耗时/步数；缺显存 |
| tweak_m1 | 缺失 | 缺失 | J14：缺独立计时；缺显存；J21：耗时/步数；缺显存；J28：耗时/步数；缺显存 |
| csil_corefix | 0.192237 s | 60.137 MiB | J03：耗时/步数＋显存 |
| mopc_paper | 0.149321 s | 51.100 MiB | J04：耗时/步数＋显存 |
| mopc_public_code | 0.132429 s | 51.100 MiB | mopc_public_code：耗时/步数＋显存 |
| orthogonal | 0.522590 s | 28.042 MiB | J05：耗时/步数＋显存 |
| issl_fixed_v1 | 缺失 | 缺失 | FIRST_UNIT_NOT_YET_COMPLETE |
| issl_author_behavior | 缺失 | 缺失 | FIRST_UNIT_NOT_YET_COMPLETE |
| lora_triplet_knn | 缺失 | 缺失 | J07：缺独立计时；缺显存 |

## 已确认的主要缺口

POSTER、RadioNet DF/triplet、Hu2024、RA、Tweak、LoRa和ISSL接口没有统一的B/C阶段耗时、峰值显存及实际更新数记录。部分可由epoch、batch和实际support/replay数量精确还原更新数；历史wall time、CPU/GPU峰值与能耗不能靠配置恢复。ISSL首完整C单元未闭合与接口未计量是两个不同事实。

原生适应器和CSIL/MoPC/Orthogonal记录了耗时、更新数、参数量和CUDA allocated峰值，但CPU峰值为明确null。组合C控制记录耗时、100次更新/曝光数/参数及常驻state bytes，缺阶段峰值显存。该state bytes不等于训练峰值；Fisher、旧模型、teacher/key/queue、优化器和输入张量也需明确计入范围。

MILKR已有fit、Gram构建、线性求解时间及support/Gram/系数/持久head字节数。audit.seconds在拟合后经过全部query判决才结束，不能当纯训练时间。F特征提取、DURA恢复和cache生成另在外层，没有统一阶段成本分摊记录。

CUDA allocated峰值来自单进程PyTorch分配器；常驻source/B/参考模型可包含其中，缺start基线时不能解释成该阶段新增显存。它也不是整卡nvidia-smi总占用。当前每卡2个本轮worker且可能有其他owner，耗时受共享资源影响，只能称当前并行环境下的运行耗时。

部分计时包含model copy、预处理、enrollment；另一些在copy后才开始。公开字段名fit_seconds/training_seconds相似不代表范围一致。所有cuda:0为进程逻辑设备，正式汇总须结合调度产物还原物理GPU和同期并发。

## 实际记录示例（单一单元，不是均值或速度结论）

同P01/MIX01/RX1-1、model seed2026092701、P0、K10、N10：MoPC-HR修复版B记录约0.149321 s/10次更新，C记录约2.420639 s/160次更新。此为现有定义下的实测阶段耗时。

MILKR同单元：B纯fit 0.027148 s；C纯fit 0.144326 s，C的fit＋query判决字段则为32.280680 s。C持久head 965152 bytes，Gram 204800 bytes。不能把其CPU核拟合时间与外部方法含网络前向/反向的时间直接相除宣称加速。

## 正式资源表的报告口径

| 项目 | B域适应 | C新类注册 |
|---|---|---|
| 时间 | support就绪至B可用的端到端耗时；另列纯优化 | 同一实际B至C可用的新增耗时；另列总B＋当前C |
| 计算预算 | 实际epoch/step、source及旧support曝光、各网络优化次数 | 新/旧support与source replay曝光、SSL/transfer/session分别统计 |
| 参数/状态 | 总参数、有效更新参数、mask/辅助网络 | 新增输出/原型/核系数、继承B及新增状态bytes |
| 内存 | 进程CUDA基线/峰值/增量峰值/reserved；CPU RSS基线/峰值 | 同口径，含必要teacher/queue/optimizer及复制开销 |
| 特征与DURA | 分列原始输入／恢复输入、support恢复及特征成本 | 区分新类特征成本、旧特征缓存与全support重拟合 |
| 计算/能耗 | MACs/FLOPs或理论复杂度注明实测/估计与算子覆盖 | 同口径；共享GPU整卡功耗不能冒充进程能耗 |
| 推理 | query单独计时、batch/吞吐/延迟定义 | 全注册类共同竞争；不计入训练时间 |

每条部署链只计一次B；NCM/JointCE/原生扩头各自从同一B分叉，不能重复计B后宣称某组合更重。N2/5/10/20是独立前缀任务，不能把四条C耗时相加当一次注册过程；MoPC单个N20内部四个session则确实累计。

MILKR的0优化器更新代表解析拟合，仍有特征提取、核构建、求解与状态存储成本。P1的DURA只生成一次供方法共享，资源表同时给出基准cache摊销与实际部署恢复成本；共享缓存不能记成部署0开销。Phase1权重复用不等于历史训练成本0。

通信量的0值只在记录明确的“support已在本地、无新增节点间传输”范围成立；设备上的原始IQ获取、标签下发、模型部署和跨节点传输分别说明，未定义的不补0。原闭集方法原生C继续N/A，组合扩展单列。

## 当前处置与补测边界

已保存统一资源字段规范与逐方法缺失清单，机器可读见同名audit JSON及configs/pamf_phase2_resource_reporting_20261029.json。现有日志可精确恢复的预算/状态量可按原artifact汇总；无法恢复的实测量保持缺失。若开展统一资源补测，必须作为独立资源测量记录，固定相同support/输入、GPU/并发、precision、batch、warmup和同步边界，统计多次中位数/离散度，说明算子计量范围；不冒充原运行历史实测值。

本次仅审计和记录要求：未热改健康worker、重启训练、读取query真值、启动额外训练/预测/评分/性能profiling；未增加实验发布阻断门槛。性能产物完成状态与资源证据完整性分别报告。
