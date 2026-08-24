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
