# ADV3B02 CORE90 CVS-HSID最小实验最终报告v2

状态：ARTIFACTS_COMPLETE / ANALYZED / SCIENTIFIC_SIGNAL_NO_PROMOTION

分析完成时间：2026-08-23 19:09 CST

## 1.结论摘要

实验已经结束。五行矩阵S0_CORE90、R3_SPEC_PROTO、X0_HIER_PROTO、F0_HIER_FUSION、X2_RX_ROBUST均闭合；driver明确输出[HSID-COMPLETE]，当前无本run活动训练进程。四个训练行均为ARTIFACTS_COMPLETE、exit code0，最终checkpoint严格加载缺失键0、意外键0，clean及三个LEO_WEAK逐场景结果与same-row prediction均已保存。

本轮得到三个层次不同的结论：

1.分层频谱选择比旧R3频谱选择更好。X0相对R3的Spec准确率在clean及三个LEO场景提高2.8500～4.3186pp，cluster bootstrap的95%CI均高于0。但P0稳定频带只有16/64个，按common→nonlinear→domain顺序独占分配后形成common/nonlinear/domain=64/0/0个FFT bin；因此本轮只证明了common-only筛选优于旧mask，不能证明完整三角色分层成立。
2.独立证据融合成功保护了Raw。F0与X2没有再用弱Spec直接替代CORE90，而是将其作为低权重logit残差；候选在每个场景仅改变110～404/204,000个canonical预测，rescue均大于harm。由此避免了R3/X0的20pp以上灾难性退化。
3.增益太小，不足以晋级。最佳X2相对S0的clean为+0.0196pp，LEO overall均值为+0.0430pp，LEO Strict UDU均值为+0.0550pp，strict receiver floor为+0.0167pp。统计上能够识别稳定正方向，但效应不足0.1pp、仅单seed，且gate近似常数、Spec仍显著弱于Raw，因此不能替换ADV3B02_CORE90_SOFT_E200。

最终判决：R3、X0为SCIENTIFIC_FAILURE_NO_PROMOTION；F0、X2为SCIENTIFIC_SIGNAL_NO_PROMOTION。X2是下一轮机制迭代起点，不是新基线。

## 2.预登记、版本与协议

- run ID：phase1_advb02_hsid_minimal_s392002_20260823_v2。
- 基线：ADV3B02_CORE90_SOFT_E200；checkpoint=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth。
- seed=392002；训练行均为200epoch。
- Phase1 source-only角色：L_s/U_s/V_cal/V_select=5,880/52,920/12,600/12,600，source池84,000，L_s/(L_s+U_s)=0.10。
- source receiver为0–6，held-out receiver为7–11，交集为0；source day为0/1，target day为2/3。
- P0、训练和选模不访问target/query；P0覆盖5,880个物理样本、23,520个clean/三LEO视图，target_or_query_access=false。
- LEO场景：leo_clear_weak、leo_low_elev_weak、leo_rain_weak。它们是WiSig/ManySig上的地面代理压力测试，不是实际在轨验证。
- 实现提交：445a966b2f53fadcc9a807c625a776d295e93590；定点修复提交：45fd122d0d6d25d78fe2fb7b368eb64f486e5013；发布提交：5ae1930be85e750a2649288dbd80227461234041。
- release源码根：/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_hsid_minimal_s392002_20260823_v2-5ae1930b。
- release归档本地/远端SHA-256一致：faa50021573b203f6c9911bb34a715c0f6469cf6122dc8167d778e1f978146a6。
- 本地相关回归68项、10个Python入口编译、launcher语法、五行dry-run通过；独立P0/P1审查及唯一一次定点复审闭合。
- v1因源码根/项目根混用在训练前技术失败，无训练、prediction或性能结果；本报告仅分析v2。

## 3.完成性与数据质量

### 3.1运行闭合

- driver日志最终标记：[HSID-COMPLETE] run_id=phase1_advb02_hsid_minimal_s392002_20260823_v2 selected=S0,R3,X0,F0,X2。
- 2026-08-23 19:08 CST只读回查：本run活动driver/trainer为0。
- R3、X0、F0、X2均保存200条epoch记录、最终checkpoint、terminal status、resource summary、clean与三个LEO评估，以及各1,632,000条Raw/Spec/Fused诊断记录。
- 四个训练日志均为7,801行；未发现Traceback、RuntimeError、CUDA OOM、Killed、非有限loss或编码替换字符。
- 四个最终checkpoint严格加载均为missing0/unexpected0。

### 3.2真实checkpoint smoke

smoke使用真实CORE90 checkpoint和hierarchical mask，验证target/query输入为0、Raw可训练参数为0、HSID可训练参数为14,570、Raw主输出差异为0、全部输出有限。因此Raw冻结和no-query边界成立。

### 3.3P0频谱审计

- 旧R3保留32/64个band，即128/256个FFT bin。
- hierarchical P0执行64次bootstrap，只有16/64个band的选择稳定概率达到0.8。
- 角色选择按common→nonlinear→domain顺序消耗合格band，common目标20首先占用全部16个稳定band。
- 最终common/nonlinear/domain mask为64/0/0个FFT bin；三类mask均排除DC。
- bootstrap概率范围0–1，共8个离散值，非占位常数。

这是一项有效但退化的结果：稳定性门实际阻止了不可靠band进入模型，但独占式角色分配造成后续角色饥饿。X0/F0/X2不能被描述为完整的“身份频段+非线性频段+域频段”三路模型。

## 4.正式最终指标

以下均来自各行独立最终评估的同一checkpoint；括号内为相对S0变化。

| 行 | clean overall | clean Strict UDU | strict RX floor | strict RX×day floor | LEO overall均值 | LEO Strict均值 | 判决 |
|---|---:|---:|---:|---:|---:|---:|---|
| S0 | 90.1402% | 86.0900% | 77.0583% | 76.3333% | 76.4688% | 70.5628% | 冻结基线 |
| R3 | 64.1328%（−26.0074） | 56.8767%（−29.2133） | 36.0917%（−40.9667） | 28.3667%（−47.9667） | 52.5036%（−23.9652） | 45.3389%（−25.2239） | 失败 |
| X0 | 67.5167%（−22.6235） | 60.8517%（−25.2383） | 52.8417%（−24.2167） | 42.6167%（−33.7167） | 56.0770%（−20.3918） | 49.8783%（−20.6844） | 失败 |
| F0 | 90.1544%（+0.0142） | 86.1183%（+0.0283） | 77.0667%（+0.0083） | 76.3167%（−0.0167） | 76.5007%（+0.0319） | 70.6006%（+0.0378） | 微弱正信号 |
| X2 | 90.1598%（+0.0196） | 86.1367%（+0.0467） | 77.0750%（+0.0167） | 76.3667%（+0.0333） | 76.5118%（+0.0430） | 70.6178%（+0.0550） | 本矩阵最佳但不晋级 |

逐LEO场景：

| 行 | clear overall/Strict | low-elev overall/Strict | rain overall/Strict |
|---|---:|---:|---:|
| S0 | 78.4691%/72.5533% | 75.6461%/69.8633% | 75.2912%/69.2717% |
| R3 | 56.5368%/48.8083% | 51.2377%/44.1900% | 49.7363%/43.0183% |
| X0 | 60.8755%/53.9233% | 54.6216%/48.6550% | 52.7338%/47.0567% |
| F0 | 78.5049%/72.5833% | 75.6779%/69.9133% | 75.3191%/69.3050% |
| X2 | 78.5137%/72.5967% | 75.6897%/69.9383% | 75.3319%/69.3183% |

X2对S0在三个LEO overall上分别为+0.0446、+0.0436、+0.0407pp，在Strict UDU上分别为+0.0433、+0.0750、+0.0467pp；方向一致，但量级极小。

## 5.prediction级配对分析

### 5.1评分口径

每个场景从三个互斥主split取84,000+60,000+60,000=204,000条canonical记录；以TX×RX×day形成204个cluster，执行10,000次配对bootstrap。Raw、Spec、Fused来自同一row、同一checkpoint、同一前向诊断文件。LEO正式final_eval和prediction导出是两个随机信道重放过程，绝对值存在不超过约0.063pp的重放差异；因果rescue/harm和CI只使用same-row prediction，不混用两个过程。

### 5.2R3和X0：Spec直接替代的失败

| 行/场景 | 相对same-row Raw的Δ | 95%cluster CI | rescue | harm | net |
|---|---:|---:|---:|---:|---:|
| R3 clean | −26.0074pp | [−30.2687,−21.8941] | 5,664 | 58,719 | −53,055 |
| R3 clear | −21.8975pp | [−25.2178,−18.6078] | 14,767 | 59,438 | −44,671 |
| R3 low | −24.2627pp | [−27.3559,−21.3034] | 15,739 | 65,235 | −49,496 |
| R3 rain | −25.3304pp | [−28.3201,−22.4269] | 15,673 | 67,347 | −51,674 |
| X0 clean | −22.6235pp | [−26.5418,−18.8121] | 6,344 | 52,496 | −46,152 |
| X0 clear | −17.5789pp | [−20.8187,−14.3431] | 17,020 | 52,881 | −35,861 |
| X0 low | −20.9603pp | [−24.0452,−18.0304] | 17,241 | 60,000 | −42,759 |
| X0 rain | −22.4804pp | [−25.4309,−19.6082] | 16,858 | 62,718 | −45,860 |

X0相对R3本身有明确改善：clean/clear/low/rain分别+3.3838/+4.3186/+3.3025/+2.8500pp，对应95%CI均高于0；但它仍远低于Raw。层次筛选改善了Spec，却没有把Spec变成可独立决策的身份表征。

### 5.3F0和X2：保守融合的微弱正收益

| 行/场景 | 相对same-row Raw的Δ | 95%cluster CI | rescue | harm | net | 改变预测数 |
|---|---:|---:|---:|---:|---:|---:|
| F0 clean | +0.0142pp | [+0.0020,+0.0284] | 53 | 24 | +29 | 110 |
| F0 clear | +0.0338pp | [+0.0181,+0.0500] | 125 | 56 | +69 | 286 |
| F0 low | +0.0353pp | [+0.0191,+0.0525] | 129 | 57 | +72 | 311 |
| F0 rain | +0.0255pp | [+0.0113,+0.0407] | 104 | 52 | +52 | 282 |
| X2 clean | +0.0196pp | [+0.0054,+0.0363] | 67 | 27 | +40 | 141 |
| X2 clear | +0.0471pp | [+0.0265,+0.0681] | 167 | 71 | +96 | 378 |
| X2 low | +0.0441pp | [+0.0250,+0.0647] | 160 | 70 | +90 | 404 |
| X2 rain | +0.0333pp | [+0.0147,+0.0525] | 140 | 72 | +68 | 376 |

样本级McNemar精确检验中，F0四场景p值为0.00126、3.14×10⁻⁷、1.36×10⁻⁷、3.81×10⁻⁵；X2为4.50×10⁻⁵、4.31×10⁻¹⁰、2.79×10⁻⁹、3.49×10⁻⁶。即“rescue多于harm”不是随机方向波动，但统计显著不等于有足够科研效应量。

### 5.4X2相对F0的独立贡献

- clean：+0.0054pp，95%CI[+0.0010,+0.0103]，净增11条；
- clear：+0.0132pp，95%CI[+0.0039,+0.0225]，净增27条；
- low：+0.0088pp，95%CI[+0.0005,+0.0172]，净增18条；
- rain：+0.0078pp，95%CI[−0.0010,+0.0162]，净增16条。

cross-RX、receiver-CVaR和TX×RX interaction整体只提供了极小增量，rain场景尚不能排除cluster层面的零效应。不能把X2相对S0的全部收益归因于receiver-aware训练；主要收益已经由F0的低权重安全融合获得。

## 6.接收机、类别与几何诊断

R3在clean Strict UDU的RX7从Raw 86.3167%降至36.0917%，下降50.2250pp；RX10下降31.1667pp。rain下RX9下降38.5083pp。X0缓解但未消除该问题：clean RX7仍下降33.4750pp，rain RX9下降35.7750pp。

R3不是单类坍缩：Spec预测类别占比为14.69%～17.71%，prototype条件数3.12，类间余弦范围−0.406～0.151。真正问题是未见receiver上的类条件频谱几何错位。clean中class1 recall从86.3235%降至44.1118%，class3从82.9441%降至51.6500%；主要错误转换包括1→3、3→1和0→1。

X2相对Raw在clean Strict UDU各receiver变化为：RX7 +0.0500、RX8 +0.1583、RX9 0、RX10 +0.0083、RX11 +0.0167pp。low场景RX11仍有−0.0250pp，rain场景RX7为−0.0083pp。整体floor略增，但不是所有receiver都单调改善。

类别层面同样如此：X2 clean的class0/class1分别−0.0059/−0.0088pp，其他四类为正；low的class1为−0.0029pp。安全融合将伤害压到极低，但没有实现严格逐类非退化。

## 7.训练动力学与资源

| 行 | 选中epoch | 选模分数 | loss起止 | train TX起止 | Spec CE起止 | Raw/Spec一致率起止 | U_s接受率 | wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R3 | 198 | 80.6977 | 5.3889→1.4787 | 16.63%→83.72% | 2.6944→0.4658 | 16.72%→80.30% | 884/403,200=0.219% | 64.70min |
| X0 | 198 | 79.1889 | 5.5641→1.5808 | 16.65%→82.27% | 2.7817→0.5039 | 16.72%→79.10% | 719/403,200=0.178% | 160.71min |
| F0 | 150 | 94.4218 | 3.2959→3.9055 | 95.68%→90.66% | 2.7629→0.5088 | 16.72%→79.29% | 3,663/403,200=0.909% | 67.03min |
| X2 | 200 | 94.4218 | 3.8333→4.1984 | 95.68%→90.66% | 2.7631→0.5107 | 16.72%→79.18% | 3,663/403,200=0.909% | 95.54min |

F0/X2保留Raw主路径，故其训练TX与R3/X0的Spec主路径不能直接比较。F0/X2的伪标签precision为3,662/3,663=99.973%，但覆盖仍不足1%，63%的U_s没有形成足量训练信号。

每行均出现6个epoch的保护性非有限梯度skip，非有限loss skip为0；无系统故障。总参数1,064,235，可训练参数14,570；峰值CUDA allocated约545.56MiB，reserved为674～734MiB。wall time是在共享GPU调度下的吞吐观察，不能写作隔离单样本延迟。

X2使E200 cross-RX loss从F0的5.4072降至4.4478，interaction loss从0.004419降至0.002603，但receiver-CVaR近似不变，source选模分数与F0相同。这解释了为何receiver-aware约束在训练空间可见，在最终决策上却只产生约0.01pp增量。

## 8.融合gate与独立证据质量

F0 gate均值约0.1121，X2约0.1501；四场景gate变异系数分别仅约0.41%～0.49%和0.21%～0.22%。中位数、90%和99%分位几乎重合，rescue与harm样本的gate均值也非常接近。gate对绝大多数样本近似常数缩放，而不是强质量自适应路由。

七个质量量分别为valid-bin ratio、fade ratio、phase coherence、trend error、clip ratio、DC ratio和SNR proxy。rescue/harm之间可见少量趋势，例如X2 rain中rescue的trend error均值0.7912、harm为0.8397，SNR proxy为6.6540对7.2010；但没有一个变量形成清晰单调分界。当前增益主要来自“小幅、全局、校准后的Spec logit残差”，不能声称已经实现成熟的独立证据置信门控。

Spec本身仍很弱：F0/X2的canonical Spec在clean约67.96%/68.16%，LEO rain约49.32%/49.34%，而Raw为90.14%和75.26%。融合能够利用极少数边界样本的条件增量，但Spec尚不是可独立部署的身份证据源。

## 9.设计要求对照

| 设计目标 | 实验证据 | 判定 |
|---|---|---|
| 完整频谱坐标计算局部差分、精确mirror、趋势残差和真实质量量 | 实现与smoke闭合，训练稳定 | 工程成立 |
| RX/day/LEO分层可辨识性 | 稳定性筛选有效，但角色分配退化为common-only | 部分成立 |
| SID作为独立prototype证据而非嵌入替代 | prediction中Raw/Spec/Fused独立保存；F0/X2保护Raw | 架构方向成立 |
| harmed少于rescued | F0/X2四场景均满足，CI和McNemar支持 | 机制成立 |
| 跨接收机稳健目标改善floor | X2相对F0仅约0.003～0.022pp，rain CI跨0 | 信号过弱 |
| 质量gate按样本可靠性调节 | gate变异极小，rescue/harm不可明显分离 | 未成立 |
| 利用63%无标签source池 | 接受率0.178%～0.909% | 未成立 |
| 替换CORE90成为新基线 | 最佳效应<0.1pp且单seed | 不通过 |

## 10.根因排序

1.频谱角色分配饥饿：16个稳定band全部进入common，nonlinear/domain为空，设计中的三角色结构没有真正接受实验。
2.Spec互补性不足：它含有少量边界纠错信号，但整体身份几何比Raw低20pp以上。
3.gate近似常数：模型没有充分利用质量变量进行样本级风险选择，安全主要来自小alpha，而非可靠性辨识。
4.receiver-aware损失与最终决策脱节：X2显著降低训练cross-RX/interaction loss，却只带来约0.01pp输出变化。
5.U_s覆盖过低：伪标签精度高但接受率不足1%，弱标注资源基本闲置。
6.source-only选模分辨率不足：F0 E150与X2 E200得到相同最终选模分数，无法有效排序0.01pp级机制差异。
7.少量非有限梯度是次要稳定性问题：均被skip保护，没有形成loss异常或系统失败，不是性能瓶颈主因。

## 11.最终判决与下一步

- R3_SPEC_PROTO：SCIENTIFIC_FAILURE_NO_PROMOTION。
- X0_HIER_PROTO：SCIENTIFIC_FAILURE_NO_PROMOTION；保留“common-only稳定mask优于旧mask”的归因证据。
- F0_HIER_FUSION：SCIENTIFIC_SIGNAL_NO_PROMOTION；证明保守独立证据融合可以做到rescue>harm。
- X2_RX_ROBUST：SCIENTIFIC_SIGNAL_NO_PROMOTION；本矩阵最佳，但绝对效应、单seed证据和机制完整性不足以替换CORE90。

下一轮不应直接扩大到多seed或完整矩阵，也不应继续单独调大fusion alpha。建议保留X2的Raw保护结构，做一个单seed最小可证伪候选：

1.将角色mask从顺序独占改为预留配额或允许多角色重叠，确保common/nonlinear/domain均有非空、稳定频带；若某角色没有稳定band，则显式降级为两角色并单独命名。
2.训练gate直接区分Raw-correct harm风险与Raw-wrong rescue机会，把质量量、Raw/Spec margin、JS divergence和agreement纳入可校准风险目标；报告gate AUC、coverage-risk和分桶净收益。
3.保持fusion alpha上限不变，以“LEO mean至少+0.1pp、Strict mean至少+0.1pp、clean不低于−0.05pp、所有场景rescue>harm”为下一次最小screen目标；这是一项新建议，不追溯改写本轮预登记。
4.只有该候选达到screen目标，再进行多seed确认；当前X2不进入多seed扩展。

## 12.可声明边界

可以声明：在Phase1 source-only、WiSig/ManySig地面代理和三个LEO弱信道压力场景下，分层稳定选频改善了独立Spec头，保守Raw/Spec logit融合在单seed中实现了统计可辨但极小的同方向净救援。

不能声明：完整三角色频谱解混成立、Spec已经跨接收机稳健、质量gate已经可靠辨识、X2显著提高工程性能、已经完成Phase2新类注册/unknown拒识、或已经获得真实在轨验证。
