# 固定 GMM 类别竞争与 OE 混合背景结果（2026-10-31）

**结论：NO_PROMOTION。** 唯一主 `hybrid_competition` 未超过原 `predicted_ratio` 的联合性能，跨 RX clean/star 均未满足已知保护；星地拒识也未达 90%。原 `predicted_ratio` 重点候选、`pooled_ratio` 冻结对照保持，新读出不替换它们，完整 Phase3 目标 ACTIVE。

Run `20261031-phase3-class-competition-s2026092701-r01`，实际代码 `b30db356419342dd15ecf811e444916aa1e5c4df`。方法、数据及验收见[计划](PAMF_PHASE3_CLASS_COMPETITION_PLAN_20261031.md)；全部细分与三组配对见[完整评分](../local_artifacts/pamf_phase3_class_competition_20261031_r01/score/results.json)和[私有分析](../local_artifacts/pamf_phase3_class_competition_20261031_r01/analysis.json)。

## 本轮检验的变化

固定原 F0 Tri-SIG-TIME SSL E50/11100/default6、source-only signal r03、DURA v1 E20/4940、known90/OE32 GMM 及 normalizer。每旧类由 15 个 RX/day Gaussian 等质量组成，log-density 为 ℓ_j；原 F0 默认头预测 c 保持，令 b_c=logmeanexp(其余五类 ℓ_j)。

诊断 `class_competition` 为 ℓ_c−b_c，不用 OE 背景。唯一主 `hybrid_competition` 为 ℓ_c−logaddexp(ℓ_OE,b_c)+log(2)，即分母固定为 0.5 OE＋0.5 其他旧类密度，温度 1、权重不搜索。此前 OVA/logits/cosine 已检验类竞争；新检验仅限这套冻结 GMM 公式，不宣称首次引入竞争。

旧 pooled/predicted 分数、阈值、默认 logits/closed、baseline、特征和 metadata 精确复用。模型、Gaussian、normalizer 新拟合为 0；**校准不为 0**：源 V27000 为两新列各拟合六阈值，旧 12 阈值不变，共 24 个。各新读出按相同规则独立校准，因此比较不能全归因于未经校准的单个算子。诊断公式虽不用 OE 背景，表征祖先已用 OE，不能称为完全无 OE 模型。

三实际 checkpoint 各两条合成烟测后，固定模型和分数公式→V 已知风险校准→全部阈值冻结→NV、clean Q、star Q 四列全部预测→独立 CPU 三次真值连接。源失败也不取消 Q；本次只整理既有产物，不重新评分、拟合或扫描阈值。

## 全部 12 运行点

下表均为逐真实 TX 宏平均百分比；A_K 为旧类正确接收率，C_K 为旧类接收率，R_U 为三非旧类拒识率，H=harmonic(A_K,R_U)。H 无独立晋级门槛。

| 条件 | 方法 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|
| source | pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 |
| source | predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 |
| source | class_competition | 89.8148 | 96.3593 | 36.8889 | 52.2979 |
| source | hybrid_competition | 89.6889 | 96.5148 | 61.3016 | 72.8267 |
| clean_DURA | pooled_ratio | 68.9236 | 85.4167 | 99.6528 | 81.4874 |
| clean_DURA | predicted_ratio | 70.8333 | 87.3264 | 100 | 82.9268 |
| clean_DURA | class_competition | 74.8264 | 93.0556 | 57.2917 | 64.8954 |
| clean_DURA | hybrid_competition | 68.5764 | 85.0694 | 100 | 81.3594 |
| star_DURA | pooled_ratio | 63.0208 | 84.8958 | 76.3889 | 69.0639 |
| star_DURA | predicted_ratio | 64.0625 | 85.5903 | 77.0833 | 69.9723 |
| star_DURA | class_competition | 64.5833 | 86.9792 | 22.5694 | 33.4495 |
| star_DURA | hybrid_competition | 62.6736 | 83.5069 | 76.7361 | 68.9956 |

主方法 source/clean/star AUROC 为 93.4227%/95.4735%/81.6750%，OSCR 为 87.1307%/77.0833%/60.5607%，仅描述固定排序，不能从曲线另挑阈值。诊断 class_competition 的星地 A_K 虽为 64.5833%，R_U 仅 22.5694%、AUROC 61.1376%，不能据局部旧类准确率换主。

## 配对变化与失败位置

主方法相对 predicted：source 旧正确 18 增/33 失、净 −15，未知拒对 33 增/53 失、净 −20；clean 旧正确 0 增/13 失、未知拒对净 0；star 旧正确 0 增/8 失、未知拒对 1 增/2 失、净 −1。star A_K/R_U/H 分别下降 1.3889/0.3472/0.9767 个百分点，未满足不劣于选定控制。

相对 pooled，主方法 source/clean/star 的旧正确净数为 −22/−2/−2、未知拒对 +13/+1/+1；微小拒识收益没有抵消已知损失。相对 class_competition，主 clean/star 旧正确净 −36/−11、未知拒对 +123/+156；混合 OE 背景明显改善这个诊断的拒识，但仍未胜既有 predicted 联合表现。三组全部 gain/loss 和逐源组/逐景配对均保留。

主 clean 接收旧类 490/576，其中正确 395、错误身份 95；star 接收 481/576，其中正确 361、错误身份 120。原 F0 闭集正确分别为 461/420；新增拒识另损失 66/59 条本可正确的样本。身份误分类与拒识是不同事件，不能用较高 C_K 替代 A_K。

同 DURA 原 F0 闭集 source/clean/star 为 90.9815%/80.0347%/72.9167%；主方法闭集到 A_K 损失为 1.2926/11.4583/10.2431 个百分点。source known12 通过，42 组未全过；clean R_U=100% 也不能救 C_K=85.0694% 和旧类损伤，star C_K=83.5069%、R_U=76.7361% 均不达标。

只读已存校准选项：主方法 common-tail 为 148/4241，六 predicted-class 前缀数为 148/148/156/173/157/159；更下一档 16/457 因 RX3 的 5400 条 V（六类各 900）旧类损失达 2.0185185 pp 而失败。诊断新列下一档亦绑定该 RX；30 个交叉组仅诊断，不把 known12 通过改称 known42 通过。本次没有重做候选扫描。

主星地六景（high/mid/low suburban、high/mid/low urban）A_K 为 72.8571/66.3889/65.3475/75.3307/73.6574/52.2156%，R_U 为 N/A/94.4444/87.3029/86.6667/81.1111/57.8584%。high 缺 19-15，R_U/H 为 N/A，完整主景覆盖失败，不能删景或补零认证。主 source 三未知 R_U 为 66.6667/44.6667/72.5714%，star 为 78.125/71.875/80.2083%；薄弱设备仍存在。

全部 TX/RX/day/scene、source42 风险及逐类 Wilson 区间已完整读回；它们是一个历史开发 RX/seed 的描述，源 V 又被用于校准，不构成独立盲测或跨 RX 置信保证。

## 数据、成本和执行边界

源 contract392005/VALIDATED_ONCE 的 L6300/U_s56700/OE6134 为祖先，本轮新读 0；V27000 为额外已知校准标签，NV3150 为 19-15/19-16/7-1 各 1050。源 RX1/3/4/6/8 与三日数据保持，旧六类顺序及信道细节沿[计划](PAMF_PHASE3_CLASS_COMPETITION_PLAN_20261031.md)。

clean/star Q 为 RX24-1/index26、2021-03-15 的同 864 物理记录，旧六类各 96=576、三非旧类各 96=288，三条件全部使用 DURA。G10 MIX01、固定 receiver2027/scene2026103001/channel2026104001、一次 received 复用；不重抽信道、不增加物理数据，clean 仅冻结测试。U384/正式 RX24-18 未读，日期与源重合、NV/Q 历史曝光、单 seed/RX 限制保持。

新 NN、Gaussian、normalizer 拟合为 0；真实 G/F0/signal 前向、source L/OE/U 读取均 0；actual 三 checkpoint synthetic 各 2 另计。新 12 个 V 阈值是真实校准成本，另有旧 12 个精确控制，24×FP64=192 B。原 density364340 B、NN834665 参数不变，无 prior 或 source/U 行 bank。F0 11100、r03+2220/MLP2220、G4940、源密度和缓存成本继承。

pipeline ANALYZED_DEVELOPMENT、predict/score 均 COMPLETE/returncode0；用时分别 17.873421/15.735417 s，全过程 33.693221 s，不是逐包延迟。三份完整生产 stdout 共 7 行，warning/error/traceback 为 0；focused_tests.log 的 464 个旧 np.trapz 提示仅属合成测试。

三次独立观察捕获 driver1253032、CPU scorer1253692 的 CWD/完整 argv/CUDAempty；predictor1253122 仅有 pipeline 指派 GPU4、日志和实际 checkpoint smoke/完成产物，未独立捕获其 live/GPU 显存。日志 0→predict674 B/score47 B/driver21 B，GPU 占用或峰值 UNKNOWN，不补遥测重跑。终态 live/ownedGPU 空、foreign_operations=0。

9 项唯一测试＋30 子项通过：worker7＋23sub/6.92s 有持久日志，root launcher2＋7sub/0.032s 为已记录 console 证据；AST7/镜像8、唯一 P0/P1 PASS。单包 661 files/data0，SHA `1296b317cc08f67a1512649dd5785e73b2313d62394ce4374819562fc426b157` 比较、compile/runtime VERIFIED。

无目标无标签适应、发现或匿名注册，A_N/H_OW 为 N/A；原 class-density 三条件仍是保留 predicted 重点候选和 pooled 对照的选择依据。本次新公式没有通过，完整目标保持 ACTIVE。
