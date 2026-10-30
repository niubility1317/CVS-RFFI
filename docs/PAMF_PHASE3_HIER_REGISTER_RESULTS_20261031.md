# 层次密度发现与匿名注册结果（2026-10-31）

**结论：NO_PROMOTION。** HDBSCAN 注册出一个匿名类，但漏发现另一个真实到达类，并明显误接收后来类。唯一主 `predicted_REG1` 的 clean/star 完整 A_N 仅为 47.9167%/33.8542%，旧类正确接收不增、固定后来类拒识下降，未达联合门槛。原 `predicted_ratio` 重点候选和 `pooled_ratio` 冻结对照保留，新 HIER 不替换它们，完整目标 ACTIVE。

Run `20261031-phase3-hier-register-s2026092701-r01`；实际代码 `d6f3a19fe06252c2e4b9ed199939896b4504ab62`。方法及固定参数见[计划](PAMF_PHASE3_HIER_REGISTER_PLAN_20261031.md)；[完整评分](../local_artifacts/pamf_phase3_hier_register_20261031_r01/score/results.json)与[私有分析](../local_artifacts/pamf_phase3_hier_register_20261031_r01/analysis.json)保留全部分组、配对和区间。

## 方法和数据权限

唯一发现机制变化为 HDBSCAN：实际 sklearn 1.7.0，固定 eom、min_cluster_size=16、min_samples=8、euclidean/brute、n_jobs=1、alpha=1、epsilon=0、allow_single_cluster=False、copy=True、max_cluster_size=None。它不使用原 source ε 切割。仍只允许大小至少 16 且原 predicted 拒识比例严格过半的完整簇注册，全部成员估计 Gaussian，保留原收缩公式与真正旧/匿名竞争。

原 F0 Tri-SIG-TIME SSL E50/11100/default6、source-only r03、DURA v1 E20/4940、known90/OE32 GMM 和两组六阈值冻结。只复用上轮 signal-register/e48f3b 的 source 聚合和 U 的 ids/logits/closed/z160；不使用旧分配、注册阈值、score 或 NV/Q 拟合。匿名聚合带无标签 U 祖先，不能把整个部署状态称为 source-only。

源 split392005/VALIDATED_ONCE：L6300、OE6134、V27000（旧六类各 4500）、NV3150（19-15/19-16/7-1 各 1050）。L 占 L+U_s 的 10%，V 是额外已知标签校准；OE 祖先只有 outside-six 成员监督。本轮 L/OE/U_s IQ 均不读。源 RX1/3/4/6/8 对应 1-19/18-2/19-2/2-19/3-19，日期 2021-03-08/15/23。

U384 为开发 RX24-1/index26、03-08 的原 G10 星地+DURA 缓存；future Q 为同 RX、03-15 的同一 864 物理记录，旧六类各 96=576、三非旧类各 96=288，clean/star 均经 DURA。源/U/Q 角色物理互斥；G10 MIX01、receiver2027、scene2026103001/channel2026104001、一次 received 保持，不增加独立物理样本，clean 仅作冻结测试例外。

真实三 checkpoint 合成烟测→source/U 缓存→匿名冻结→V 两个注册阈值→全部阈值冻结→完整四方法×三条件预测→独立 CPU 四份真值。仅 U 真值做 Hungarian；真实到达类为 6/7（19-15/19-16），后来类为 8（7-1）。全部到达类进入 A_N，漏发现计错，不用 Q 重排映射或调参。

## 全部 12 运行点

表中为真实 TX 宏平均百分比。A_K=旧类正确接收率，C_K=旧类接收率，H=harmonic(A_K,R_U)。**REG0 的 R_U 分母是全部三非旧类，REG1 是后来类 8；跨 REG 的 R_U/H 不直接比较。** REG0 的 A_N、H_KN、H_OW 为 N/A。

| 条件 | 方法 | A_K | C_K | R_U | H | A_N |
|---|---|---:|---:|---:|---:|---:|
| source | pooled_REG0 | 89.7704 | 97.1074 | 60.8889 | 72.5613 | N/A |
| source | predicted_REG0 | 89.7444 | 96.7593 | 61.9365 | 73.2914 | N/A |
| source | pooled_REG1 | 89.7185 | 97.2778 | 50.0000 | 64.2138 | 6.0000 |
| source | predicted_REG1 | 89.7037 | 96.9481 | 50.5714 | 64.6792 | 6.1429 |
| clean_DURA | pooled_REG0 | 68.9236 | 85.4167 | 99.6528 | 81.4874 | N/A |
| clean_DURA | predicted_REG0 | 70.8333 | 87.3264 | 100 | 82.9268 | N/A |
| clean_DURA | pooled_REG1 | 68.9236 | 85.5903 | 66.6667 | 67.7764 | 47.9167 |
| clean_DURA | predicted_REG1 | 70.8333 | 87.5000 | 66.6667 | 68.6869 | 47.9167 |
| star_DURA | pooled_REG0 | 63.0208 | 84.8958 | 76.3889 | 69.0639 | N/A |
| star_DURA | predicted_REG0 | 64.0625 | 85.5903 | 77.0833 | 69.9723 | N/A |
| star_DURA | pooled_REG1 | 63.0208 | 86.4583 | 60.4167 | 61.6913 | 33.8542 |
| star_DURA | predicted_REG1 | 64.0625 | 87.1528 | 60.4167 | 62.1862 | 33.8542 |

## 发现正确不等于未来注册完成

五个原始簇大小为 190/21/28/27/31，predicted 拒识数为 26/21/1/6/10，噪声 87；只有 21 条簇通过严格半数，匿名 K=1。已有 scorer 证明该注册簇 21 条均为真 TX19-15、旧类污染 0，冻结映射为 anon0→6；漏发现 19-16。其他原始簇没有真 TX 组成证据，不能推断其纯度或设备不可分。

已发现类 6 在未来 clean 正确 92/96=95.8333%，star 为 65/96=67.7083%；未发现类 7 的正确数均为 0。因此完整两到达类 A_N 为 47.9167%/33.8542%，不能丢掉漏类改报单类高分。U 注册簇 100% 纯度不是未来 A_N，也不是跨 RX 完全注册成功；源 NV 主 A_N 也仅为 6.1429%。

REG1 的实际未注册集合为 {7,8}，主 clean/star 对该集合拒识率为 82.8125%/67.1875%；它不同于 REG0 的三类集合，也不同于主后来类 8。主 star H_KN=44.2985%、H_OW=48.6224%，不能遮盖后来类误接收。

当前两方法的选择依据保持为[原 class-density 三 DURA 条件结果](PAMF_PHASE3_CLASS_DENSITY_RESULTS_20261031.md)，本表 REG0 已精确重现；不按新 HIER 的局部 A_N 或 U 纯度重新选择基线。

`registered_macro_*` 是**预期注册集合**（旧六类＋全部真实 U 到达类，包含未发现类）的宏指标；实际部署是六旧类＋一个匿名类，不是八个已成功注册类别。主 clean/star 预期集合正确率为 65.1042%/56.5104%，接收率 77.7344%/79.5573%，FRR 22.2656%/20.4427%，均须与 A_N、实际 K 分开解释。

## 同一批记录的注册效应

主 predicted_REG1−REG0：source 旧正确 0 增/11 失，接收净 +51，到达类正确 +129，固定后来类拒对 0 增/220 失；clean 旧正确净 0、接收 +1、到达类正确 +92，后来类拒对 96→64（−32）；star 旧正确净 0、接收 +9、到达类正确 +65，后来类拒对 76→58（−18）。未来新增旧接收均是错误身份，不能把 C_K 上升称为身份改善。

pooled_REG1−REG0 的 source 旧正确 −14、接收 +46、到达类正确 +126、later 拒对 −218；未来两条件的对应净数与主相同。predicted_REG1−pooled_REG1 在 source/clean/star 的旧正确净数为 −4/+11/+6、later 拒对 +6/0/0，到达类正确 +3/0/0；star 到达类为 1 增/1 失。完整三组逐景/源42 组配对保留于私有分析，控制不能替换唯一主。

在“旧 margin 固定、只增加匿名竞争”的结构中，旧正确接收和同一 later 的拒识集合只能保持或减少。这一注册部件单独无法补上原旧 A_K 的绝对保护缺口或将原 later R_U 提升至 90%；结果支持另行检验联合拒识/无标签适应，而不是用 A_N 单项宣告成功。此类后续机制本轮未实施。

## 风险、场景和判定

两注册阈值为 −280.6977305402709/−281.16204591272106，均由同一源 V27000 合法校准。source known12 通过，42 组整体未全过；两方法各有 27000 个候选，紧邻更低阈值均因 RX3 的 5400 条 V、六类各 900 条而失败：原 F0 闭集−旧 A_K=2.0185185 pp，超过 2。该证据来自已保存校准报告，无重校准或新阈值扫描。

未来主 clean/star 旧 A_K 保持 70.8333%/64.0625%，同 DURA 原 F0 闭集为 80.0347%/72.9167%，损失仍 9.2014/8.8542 pp。旧 C_K=87.5%/87.1528%、预期注册 C_K<90%、A_N<80%、later R_U=66.6667%/60.4167% 均不满足 clean95/star90 的联合要求。source 校准保护不能代替这些未来结果。

主 star 六景（high/mid/low suburban、high/mid/low urban）的 later R_U 为 50/66.6667/64.8649/87.5/33.3333/51.6129%；A_N 为 N/A/50/43.75/50/37.5/22.9167%。high 缺 19-15，A_N、预期注册 C_K、H_KN、H_OW 为 N/A，完整覆盖失败；不能按 present-only 补零或跳过后认证。

全部 TX/RX/day/scene、source42 风险及逐类 Wilson 区间已完整读回。区间仅为单一历史开发 RX/seed 的记录级描述，源 V 又是校准复用数据；REG1 曲线定义未固定，AUROC/OSCR 保留 N/A，不继承 REG0 曲线作注册证明。

## 成本和执行证据

新 NN 更新、真实 F0/G/signal 前向、source L/OE 读取、ε 拟合均为 0；消费既有 U384 特征，actual 三 checkpoint 各两条合成烟测另计。新增 HDBSCAN、一个匿名 Gaussian 和两个 V 标量阈值，不能称为零统计学习。原 source L6300 signal、U384 G/F0/signal、ε、GMM、Q 缓存和 F0 11100/r03+2220/MLP2220/G4940 训练成本继承。

NN 仍为 834665 参数。density 数组 364340 B、继承但不用作聚类的 source_radius 156 B、匿名聚合 1484 B；旧 12 阈值 96 B 精确复用，新两标量 16 B，合计 112 B。不部署 source/U 逐行 bank。

pipeline ANALYZED_DEVELOPMENT，两 job COMPLETE/returncode0；predict/CPUscore/总过程 47.769382/12.666512/60.521157 s，非逐包延迟。三份完整生产 stdout 共 8 行，warning/error/traceback 为 0；合成测试旧 np.trapz 提示另记。sklearn 本地1.7.2/远端1.7.0签名已 VERIFIED，没有安装或环境修改。

三次独立观察捕获 driver1234212、producer1234312 的 CWD/完整 argv/CUDA7，UUID9c4eb614 显存 406 MiB 单次样本非峰值，日志 525→675 B。CPU1235017 未独立捕获 live，仅 pipeline gpu=None、完成产物及代码 CUDAempty；终态 live/ownedGPU 均空，foreign_operations=0，不为遥测重跑。

10 项唯一测试＋45 子项、唯一 P0/P1 审查通过。单包 653 files/data0，SHA `0364569d2d36083777e2809979d6f6fb2bb02eb239cfa07a58336590063f25a6` 比较、compile/runtime VERIFIED。本次只整理既有结果，未重新评分、拟合、扫阈值或补读私有真值。

历史 NV/Q 已曝光，单 seed/RX、日期与源重合，formal RX24-18 未读。本轮没有 target DA 或持续多 session 注册；保留 NO_PROMOTION 和完整目标 ACTIVE。
