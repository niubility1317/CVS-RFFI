# ADV3B02 CORE90 CVS-HSID最小实验报告v2

状态：`RUNNING / PARTIAL_ANALYSIS_THROUGH_X0_E131`（S0与R3已完整分析；X0仍在训练，F0/X2尚未启动）

## 1.预登记

- run ID：`phase1_advb02_hsid_minimal_s392002_20260823_v2`。
- 基线：`ADV3B02_CORE90_SOFT_E200`；checkpoint=`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 数据角色：Phase1 source-only `L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`；mask、训练和选模均不得读取target/query。
- 矩阵：`S0_CORE90`、`R3_SPEC_PROTO`、`X0_HIER_PROTO`、`F0_HIER_FUSION`、`X2_RX_ROBUST`；seed=`392002`；训练行200epoch。
- LEO_WEAK：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；每行训练完成后必须保留clean及三个逐场景结果。
- GPU：GPU0/GPU1；发布前再次回读时GPU0空闲、GPU1有1个既有训练进程；本run每卡最多增加1个，保持每卡不超过2个。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_hsid_minimal_s392002_20260823_v2/`。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_hsid_minimal_s392002_20260823_v2/`。
- 技术停止：仅协议/query泄漏、错误checkpoint/checkout/row、输出冲突、无prediction闭合、确定性重复异常、OOM/NaN或进程归属不清；低性能不停止。
- 预期artifact：P0统计与分层mask、首步真实checkpoint无query smoke、每行checkpoint/训练日志、clean与三种LEO结果、same-row `y/raw/spec/fused` prediction、margin/gate/RX/day/scenario/质量字段。
- 结果边界：prediction完整并由独立scorer连接truth前，不声明性能提升。

## 2.实现与验证

- 主实现提交：`445a966b2f53fadcc9a807c625a776d295e93590`。
- 双根定点修复提交：`45fd122d0d6d25d78fe2fb7b368eb64f486e5013`。
- 相关回归：68项通过；10个Python入口编译、launcher`bash -n`、五行dry-run和`git diff --check`通过。
- 真实checkpoint无query smoke：`VERIFIED`；`query_input_count=0`、`target_input_count=0`、Raw可训练参数0、HSID可训练参数14,570、Raw主输出零漂移、输出有限。
- 独立审查：初审0个P0、5个P1；五项修复后的唯一一次定点复审`5/5 PASS`，剩余P0/P1为无。
- v1失败边界：P0在训练前因源码根/项目根混用失败，无训练、prediction或性能结果；失败产物保留，不复用v1。

## 3.发布命令

- release提交：`5ae1930be85e750a2649288dbd80227461234041`。
- release源码根：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_hsid_minimal_s392002_20260823_v2-5ae1930b`。
- 项目数据根：`/home/szu2070436088/2510044040/CV-SincNet`。
- P0准备：`cd <release> && env ROOT=<release> PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet RUN_ID=phase1_advb02_hsid_minimal_s392002_20260823_v2 GPU_0=0 GPU_1=1 MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_advb02_hsid_20260823.sh --prepare-p0 --only=R3,X0,F0,X2`。
- 正式启动：`cd <release> && nohup env ROOT=<release> PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet RUN_ID=phase1_advb02_hsid_minimal_s392002_20260823_v2 GPU_0=0 GPU_1=1 MAX_ACTIVE_PER_GPU=2 bash code/scripts/launch_phase1_advb02_hsid_20260823.sh --only=S0,R3,X0,F0,X2 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_hsid_minimal_s392002_20260823_v2/driver.out 2>&1 < /dev/null &`。
- release传输只对一个Git归档做一次本地/远端SHA-256比较；不增加成员hash、seal或receipt。

## 4.发布与启动证据

- release归档：`phase1_advb02_hsid_minimal_s392002_20260823_v2-5ae1930b.tar.gz`；本地与远端SHA-256均为`faa50021573b203f6c9911bb34a715c0f6469cf6122dc8167d778e1f978146a6`；远端Python编译与launcher语法检查通过。
- P0频谱审计：`VERIFIED`；`source_only=true`、`target_or_query_access=false`，覆盖23,520个source视图；bootstrap选择概率范围0–1、8个离散值且非全1；`common/nonlinear/domain`三类mask的DC位均为0。严格稳定性阈值下`nonlinear/domain`为空，作为本次真实P0退化选择保留，不作为协议或启动停止理由。
- 正式启动时间：2026-08-23 11:35 CST；driver PID=`1482192`。
- driver CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_hsid_minimal_s392002_20260823_v2-5ae1930b`；cmdline=`bash code/scripts/launch_phase1_advb02_hsid_20260823.sh --only=S0,R3,X0,F0,X2`。
- 首步真实checkpoint smoke：`VERIFIED`；`query_input_count=0`、`target_input_count=0`、Raw可训练参数0、HSID可训练参数14,570、`primary_raw_logit_max_abs=0`、所有输出有限。
- 启动后检查：driver存活并进入`S0_CORE90`；子进程PID=`1482425`绑定GPU0与本release评估脚本；`driver.out`和`SMOKE.out`已产生并增长；GPU1既有PID=`1269217`未被触碰。
- 当前仅完成发布与运行健康闭合；训练行、逐场景评估、same-row prediction及独立scorer尚未完成，因此不声明任何性能提升。

## 5.额外gate处理

除项目八项白名单外不增加审核、seal、receipt或逐文件哈希；旧要求若形成额外gate，记录`REJECTED_EXTRA_GATE`并继续最小流程。

## 6.阶段性深度分析（截至2026-08-23 13:15 CST）

### 6.1完成状态与结论边界

| 行 | 状态 | 已有证据 | 当前结论 |
|---|---|---|---|
| `S0_CORE90` | `ARTIFACTS_COMPLETE` | clean及三种LEO完整评估 | 冻结同row基线 |
| `R3_SPEC_PROTO` | `ARTIFACTS_COMPLETE / ANALYZED` | 200epoch、checkpoint、完整评估、1,632,000条prediction | `SCIENTIFIC_FAILURE_NO_PROMOTION` |
| `X0_HIER_PROTO` | `RUNNING` | 截至E131的全部结构化记录 | 仅允许阶段性诊断 |
| `F0_HIER_FUSION` | `PENDING` | 无 | 不作结论 |
| `X2_RX_ROBUST` | `PENDING` | 无 | 不作结论 |

driver PID=`1482192`仍存活。13:15 CST时X0位于E131/200，F0/X2目录尚未出现。因此本节是对当前全部可用数据的完整分析，不是五行矩阵终局；整体run仍为`RUNNING`。

### 6.2数据与协议核验

- source池共84,000条物理样本，`L_s/U_s/V_cal/V_select=5,880/52,920/12,600/12,600`，对应全池比例`0.07/0.63/0.15/0.15`；`L_s/(L_s+U_s)=0.10`。
- source receiver为0–6，held-out receiver为7–11，交集为0；source day为0/1，held-out day为2/3。
- P0仅读取`L_s`，覆盖5,880个物理样本和23,520个clean/三LEO视图；`source_only=true`、`target_or_query_access=false`。
- smoke使用真实CORE90 checkpoint，query/target输入均为0，Raw可训练参数为0，HSID可训练参数为14,570，Raw主输出零漂移。
- WiSig/ManySig与LEO_WEAK均为地面代理和物理启发的模拟压力测试；本结果不是实际在轨卫星验证，也不构成Phase2适配、注册或Phase3未知拒识结论。

### 6.3P0分层频谱审计揭示的结构退化

传统R3 mask保留32/64个band，即128/256个FFT bin。64次按物理样本cluster的bootstrap只有16/64个band达到稳定概率0.8；`select_hsid_role_masks()`按`common→nonlinear→domain`顺序互斥分配，而common目标上限为20个band，因此16个稳定band全部被common消费。最终分层mask为：

| 角色 | band | FFT bin |
|---|---:|---:|
| common | 16 | 64 |
| nonlinear | 0 | 0 |
| domain | 0 | 0 |

bootstrap概率范围0–1，共8个离散值，并非占位全1；三类mask均正确排除DC。问题不在bootstrap真实性，而在“稳定候选不足+顺序互斥分配”：X0/F0/X2实际接收的是common-only频谱，不是设计中的common/nonlinear/domain三证据分解。这一事实限制了后续行对完整分层架构的检验能力。

### 6.4S0与R3最终同row结果

R3选择E198 checkpoint，严格加载0 missing/0 unexpected。独立最终评估结果如下：

| 指标 | S0 CORE90 | R3 SPEC | R3−S0 |
|---|---:|---:|---:|
| clean overall | 90.1402% | 64.1328% | −26.0074pp |
| clean strict UDU | 86.0900% | 56.8767% | −29.2133pp |
| clean strict receiver floor | 77.0583% | 36.0917% | −40.9667pp |
| clean strict RX×day floor | 76.3333% | 28.3667% | −47.9667pp |
| `leo_clear_weak`overall | 78.4691% | 56.5368% | −21.9324pp |
| `leo_clear_weak`strict UDU | 72.5533% | 48.8083% | −23.7450pp |
| `leo_low_elev_weak`overall | 75.6461% | 51.2377% | −24.4083pp |
| `leo_low_elev_weak`strict UDU | 69.8633% | 44.1900% | −25.6733pp |
| `leo_rain_weak`overall | 75.2912% | 49.7363% | −25.5549pp |
| `leo_rain_weak`strict UDU | 69.2717% | 43.0183% | −26.2533pp |
| 三LEO overall均值 | 76.4688% | 52.5036% | −23.9652pp |
| 三LEO strict均值 | 70.5628% | 45.3389% | −25.2239pp |

R3不是小幅负收益，而是大范围跨receiver几何失效。clean strict RX×day floor下降47.97pp，说明平均准确率不能表达最严重的局部失效。

### 6.5逐样本配对、bootstrap与错误转换

独立scorer只选用三个互斥canonical split，共204,000条不重复记录/场景，并按TX×RX×day形成204个cluster进行10,000次配对bootstrap：

| 场景 | 同输入Raw | Spec | Δ | 95%cluster CI | rescue | harm |
|---|---:|---:|---:|---:|---:|---:|
| clean | 90.1402% | 64.1328% | −26.0074pp | [−30.2687,−21.8941] | 5,664 | 58,719 |
| clear | 78.4088% | 56.5113% | −21.8975pp | [−25.2579,−18.6833] | 14,767 | 59,438 |
| low-elev | 75.8034% | 51.5407% | −24.2627pp | [−27.3397,−21.2631] | 15,739 | 65,235 |
| rain | 75.2618% | 49.9314% | −25.3304pp | [−28.3080,−22.3980] | 15,673 | 67,347 |

四个场景的McNemar连续性修正检验均远离随机波动范围；clean的204个cluster中178个退化、20个改善。clean上Spec挽救5,664个Raw错误，却破坏58,719个Raw正确样本，净损失53,055个。R3的Raw与Spec预测不一致率为45.56%，不是低频偶发修正。

R3的`fusion_gate`全为0，`fused_pred`与Raw完全一致，这是spec-only行的预期配置。因此R3正式主结果必须读取Spec列；`fused_pred`不能被误写成R3融合收益。paired Raw是同一checkpoint、同一输入下的冻结Raw参照，三种LEO的paired Raw与独立S0行存在不超过0.06pp的信道流差异，因而统计因果判断以同输入配对结果为主，正式基线表仍保留S0行。

### 6.6receiver、类别与置信几何

- clean上RX7从Raw 84.2208%降至Spec 29.9250%，下降54.2958pp；RX10下降31.2125pp。rain场景RX9下降40.4167pp。
- clean类别1的recall从86.3235%降至44.1118%，类别3从82.9441%降至51.6500%。最大错误转换为`1→3`9,628条、`3→1`7,258条、`0→1`7,187条。
- Spec预测类别占比保持14.69%–17.71%，6个48维prototype条件数为3.12，prototype类间余弦范围−0.406至0.151。模型没有塌缩到单一类别，失败来自receiver-dependent频谱嵌入错位。
- Raw margin均值15.793，Spec margin均值1.700。7维质量变量单独区分Spec rescue与harm的最佳AUC为clean 0.629、LEO 0.565–0.593，单变量不足以承担可靠回退；F0仍可联合Raw/Spec margin、JS散度和一致性学习门控，需等待真实结果。

### 6.7R3完整训练曲线与系统健康

- 完整解析200/200个epoch和7,801行stdout；无Traceback、RuntimeError、CUDA OOM、Killed、非有限loss或UTF-8替换字符。
- 总loss从5.3889降至1.4787，训练TX准确率从16.6319%升至83.7153%，频谱CE从2.6944降至0.4658，Raw/Spec训练一致率从16.72%升至80.30%。训练收敛不等于held-out receiver有效。
- source-only选模在E198达到记录分数80.6977：`V_select`TX=89.6825%，source卫星均值=77.3095%、floor=73.3492%，source receiver floor=77.0000%、RX×day floor=75.6667%。这些指标覆盖已见source receiver，不能替代leave-one-source-receiver外推。
- E1、E39、E85、E137、E142、E198出现受保护的少量非有限梯度跳过；每次为2/90 batch，E1为8/90 batch。非有限loss始终为0，checkpoint与评估均闭合，因此这是次要数值稳定性问题，不是本轮大幅性能下降的主因。
- E131–200期间共接受884/403,200个`U_s`伪标签，接受率0.219%；每epoch中位数12.5/5,760、最大22/5,760。日志中的伪标签正确率为100%，但覆盖极低，R3实际主要由7%有标签样本驱动，63%无标签数据没有形成有效训练信号。
- 训练耗时3,881.73s（64.70min）；总参数1,064,235，可训练参数14,570；峰值CUDA allocated=545.56MiB、reserved=734.00MiB。资源记录是并发环境下的调度观测，不是隔离延迟基准。

### 6.8X0阶段性曲线

截至E131，X0无Traceback、RuntimeError、OOM、Killed或非有限loss；受保护的非有限梯度发生在E1/E45/E77/E125。当前记录为：`val_tx=88.8016%`、source卫星均值71.8810%、floor66.6429%、source receiver floor75.3889%、RX×day floor74.5556%，本epoch伪标签11/5,760。已冻结最佳记录仍是E110的`source_hsid=76.5867`。

X0到E120的source卫星均值72.4497%、floor67.1905%，低于R3同阶段的75.4101%和72.0397%；分层common-only mask尚未显示出source卫星收益。X0仍有69个epoch和最终评估，以上只能作为实时诊断，不能形成晋级或失败判决。

### 6.9根因判断与当前科研结论

R3的根因按证据强度排序如下：

1. 频谱身份空间保留了强TX×RX交互。prototype本身分离正常，但未见receiver使类别相对几何发生大幅旋转，RX7和RX9最突出。
2. R3只优化频谱CE，cross-RX、receiver-CVaR、TX×RX interaction和margin safety权重均为0；source验证在已见receiver内上升，无法约束未见receiver风险。
3. R3以Spec直接替代Raw，没有可靠度回退。结果表现为rescue存在但harm约为rescue的4–10倍。
4. 分层P0退化为common-only，使X0/F0/X2无法检验完整三角色证据分解；这会限制后续负结果的归因范围。
5. `U_s`门控过严，接受率仅0.219%，弱标注主张没有转化为足量无标签监督。
6. 少量非有限梯度需要后续定位到具体参数与质量batch，但它们稀疏、被跳过且不伴随loss异常，不能解释26–48pp的系统性下降。

当前可证结论是：修正频率坐标、镜像和相位数学后，独立频谱prototype能够稳定训练，也包含可测的少量Raw互补信息；但R3频谱证据本身不具备跨receiver可替代性，且远低于ADV3B02 CORE90。R3判定为`SCIENTIFIC_FAILURE_NO_PROMOTION`。CVS-HSID整体路线尚未被完整证伪，因为受限Raw融合与X2 receiver稳健目标尚未运行完成；同样也没有任何证据支持当前晋级。

### 6.10剩余闭合项

- 等待X0完成200epoch、最终checkpoint、clean及三种LEO评估和same-row prediction。
- 同一driver随后执行F0与X2；不得因R3低性能停止既定矩阵。
- 每行prediction完整后，以相同canonical split和TX×RX×day cluster scorer计算Raw/Spec/Fused配对变化、receiver/RX×day floor、rescue/harm、gate分布和质量条件效应。
- 五行均闭合后将状态更新为`ARTIFACTS_COMPLETE / ANALYZED`，给出X0是否改善频谱证据、F0是否安全保留Raw、X2是否减少receiver尾部风险的最终判决。
