# 冻结信号表征的匿名注册结果（2026-10-31）

**结论：NO_PROMOTION。** 唯一主方法 `predicted_REG1` 没有注册出匿名类别，未来两条件的 A_N、H_KN、H_OW 均为 0。旧类决策保持不变不代表匿名注册成功；原 predicted 重点候选和 pooled 冻结对照保留，完整 Phase3 目标为 ACTIVE。

Run `20261031-phase3-signal-register-s2026092701-r01`；实际代码 `e48f3b357eccb12e778c1eb348776e5c6b457922`。方法见[计划](PAMF_PHASE3_SIGNAL_REGISTER_PLAN_20261031.md)，全部分组及区间见[完整评分](../local_artifacts/pamf_phase3_signal_register_20261031_r01/score/results.json)与[私有分析](../local_artifacts/pamf_phase3_signal_register_20261031_r01/analysis.json)。

## 方法、数据和权限

原 F0 Tri-SIG-TIME SSL E50/11100、r03 开放信号分支、DURA v1、known90/OE32 GMM 和两组六阈值全部冻结。新 ε 只用 L6300 六类×5 RX×3 日、每格 70 条的第七个其他近邻距离之 95% 分位；完整无标签 U 经冻结 DURA→F0/r03，执行 min_samples=8 的 DBSCAN。簇至少 16 条且原 predicted 拒识比例严格过半才注册，全部成员估计匿名 Gaussian，与旧身份竞争；真实 TX、K、配额不进入算法。

源 RX1/3/4/6/8 对应 1-19/18-2/19-2/2-19/3-19，日期为 2021-03-08/15/23。旧六类为 14-10、14-7、20-15、20-19、6-15、8-20。继承 split392005、L6300/OE6134；本轮 OE、U_s 不读。L 占原 L+U_s 的 10%，额外 V27000 标签只作风险校准；NV3150 为历史开发未知 19-15/19-16/7-1 各 1050，源 RX 已见。

U384 为 RX24-1/index26、03-08；future Q 为同 RX、03-15 的 864 条物理记录，旧六类各 96=576、三非旧类各 96=288。clean/star 为同一物理 Q 的两条件且均经过 DURA；本轮 Q 只复用冻结预测。G10 MIX01 为 600 km、25 MHz、2.45 GHz、residual/post_sync/noEQ、reference SNR 30 dB、reflect，郊区/城区各0.5、原覆盖面积六分量；固定 receiver2027、scene2026103001/channel2026104001 和一次 received。不增加独立物理样本，clean 是事前测试例外。

匿名状态先冻结、V 校准后冻结，四方法×三条件预测全部完成后才连接四份真值。匿名映射只用 U 真值做一次 Hungarian，之后固定；实际到达类为 6/7（19-15/19-16），后来类为 8（7-1）。全部真实到达类进入 A_N，未发现计错；未匹配匿名接收计错。

## 完整 12 运行点

均为真实 TX 宏平均百分比。A_K 是旧类正确接收率，C_K 是接收率，H=harmonic(A_K,R_U)。**REG0 的 R_U 分母是三非旧类，REG1 是后来类 8；跨 REG 的 R_U/H 不能直接当作收益。** REG0 的 A_N 为 N/A。

| 条件 | 方法 | A_K | C_K | R_U | H | A_N |
|---|---|---:|---:|---:|---:|---:|
| source | pooled_REG0 | 89.7704 | 97.1074 | 60.8889 | 72.5613 | N/A |
| source | predicted_REG0 | 89.7444 | 96.7593 | 61.9365 | 73.2914 | N/A |
| source | pooled_REG1 | 89.7704 | 97.1074 | 70.7619 | 79.1408 | 0 |
| source | predicted_REG1 | 89.7444 | 96.7593 | 71.5238 | 79.6048 | 0 |
| clean_DURA | pooled_REG0 | 68.9236 | 85.4167 | 99.6528 | 81.4874 | N/A |
| clean_DURA | predicted_REG0 | 70.8333 | 87.3264 | 100 | 82.9268 | N/A |
| clean_DURA | pooled_REG1 | 68.9236 | 85.4167 | 100 | 81.6033 | 0 |
| clean_DURA | predicted_REG1 | 70.8333 | 87.3264 | 100 | 82.9268 | 0 |
| star_DURA | pooled_REG0 | 63.0208 | 84.8958 | 76.3889 | 69.0639 | N/A |
| star_DURA | predicted_REG0 | 64.0625 | 85.5903 | 77.0833 | 69.9723 | N/A |
| star_DURA | pooled_REG1 | 63.0208 | 84.8958 | 79.1667 | 70.1770 | 0 |
| star_DURA | predicted_REG1 | 64.0625 | 85.5903 | 79.1667 | 70.8182 | 0 |

K=0 时实际未注册集合仍是三非旧类；主 source/clean/star 的该集合拒识率仍为 61.9365%/100%/77.0833%，与 REG0 相同。REG1 表中后来类拒识率对应 751/1050、96/96、76/96，分母变化不构成预测改进。

## 发现和配对结果

ε=17.987383312044177。原始两簇为 305/24 条，predicted 拒识 75/4 条（24.5902%/16.6667%），均达到大小要求，但未过严格半数条件；噪声 55 条中 54 条被拒识。全 U 的 133 条拒识候选中，40.6015% 落入噪声，噪声自身拒识率 98.1818%。这些是拒识事件，不能称为真实未知。

因此匿名 K=0、U 映射为空、新 Gaussian 为 0。没有原始两簇的真 TX 组成或纯度证据；不能据此断言真实设备不可分或表征没有身份信息。

source known/NV 和两 Q 的 REG1 原始标签、决策分数均逐条等于同 base REG0。两组 REG1−REG0 在三条件及全部分组的旧正确、旧接收、到达类正确、固定 later 拒识及全部非旧拒识 gain/loss/net 全为 0；固定 later 的 R_U 变化也为 0。

predicted_REG1 对 pooled_REG1 的旧正确净数为 source −7、clean +11、star +6，全部非旧拒对净数为 +33/+1/+2；star later 为 1 增、1 失、净 0。这是既有读出差异，不能归因注册或据较高控制替换唯一主方法。

## 科学门槛和覆盖

主 source known12 校准保护通过，42 组整体保护未全过。K=0 的两套校准过程各只有一个零候选，注册阈值保存为 0/0；没有有效匿名阈值优化，不能把两套 procedure 夸大为学得两个有效注册参数。

主 clean/star 的同 DURA 原 F0 闭集基线为 80.0347%/72.9167%，旧 A_K 损失 9.2014/8.8542 个百分点，均超 2；旧 C_K 为 87.3264%/85.5903%，均低于 90%。clean later 拒识单项达标，但 A_N=0；star later 拒识 79.1667% 也未达 90%，不通过。

`registered_macro_*` 是**预期注册集合**（旧六类＋全部真实到达类，包括未发现类）的指标，不表示实际部署类别数。主 source/clean/star 的该集合正确率为 67.3083%/53.1250%/48.0469%，接收率为 83.2837%/65.4948%/70.1823%，FRR 为 16.7163%/34.5052%/29.8177%；实际部署仍为六旧类＋零匿名。主 star 旧类接收 493/576、正确仅 369/576，含 124 条错误旧身份接收；接收不等于识别正确。

主 star 六景（high/mid/low suburban、high/mid/low urban）A_K 为 72.8571/69.7222/66.2404/75.3307/75.5093/54.2402%，later R_U 为 100/83.3333/83.7838/100/66.6667/67.7419%。high 缺到达类 19-15，A_N、预期注册 C_K、H_KN、H_OW 为 N/A；其旧 A_K/later R_U/H 可描述，但不认证完整覆盖。其余主景 A_N 为 0，全景门槛未达。

完整 TX/RX/day/scene、source42 风险、三组配对和逐类 Wilson 区间保留于链接产物。区间仅为单一历史开发 RX/seed 的记录级描述；REG1 AUROC/OSCR 未固定定义，保留 N/A，不挪用 REG0 曲线作注册质量。

## 成本和执行证据

NN 更新 0、原 GMM/旧阈值拟合 0。新增 signal 前向 6684=L6300+U384，F0/G 各 384，三实际 checkpoint 各两条合成烟测另计。新统计为 source ε 拟合一次，无有效匿名 Gaussian/注册阈值学习；F0 E50/11100、r03 11100+2220、MLP2220、DURA 和源/Q 缓存成本继承。

神经参数 834665。原 density 数组 364340 B；source_radius 156 B（ε 数值 8 B、schema 148 B），空 anonymous schema 160 B。旧两组六阈值 96 B 精确保持，零占位两阈值 16 B，合计 112 B；不是新增神经参数，不部署 source/U 逐行 bank。

pipeline 为 ANALYZED_DEVELOPMENT，两 job 均 returncode=0/COMPLETE。predict/CPUscore/全流程为 25.931523/11.398478/37.415431 s，是过程用时，非逐包延迟。三份完整生产 stdout 共 11 行，warning/error/traceback 为 0；测试旧 np.trapz 提示与生产日志分开。

三次观察中独立捕获 driver1206800、producer1206891 的 CWD/完整 argv/CUDA4；producer 仅有一次 live snapshot，UUID95d925a1 的显存样本为 510 MiB。日志 642→终态796 B，显存不是峰值。CPU1207091 仅有 pipeline gpu=None、完成产物及代码 CUDAempty，未独立捕获 live。终态 live/ownedGPU 为空、foreign_operations=0，不补遥测重跑。

24 项唯一测试＋29 子项及唯一 P0/P1 审查通过；单包 645 files/data0，SHA `6bd7bae78e4fa81d2071ed9dc075d6ed7cc2e7a5c92367c4ab071014a584e98d` 比较、compile/runtime 均 VERIFIED。本次未重评分、重读私有真值或调阈值。

源 NV/Q 历史暴露、单 seed/RX，日期非源未见；formal RX24-18 未读。本轮无 target DA、无持续多 session 注册成功，发现/适应/未来匿名识别完整目标保持 ACTIVE。
