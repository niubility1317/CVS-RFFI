# Stage2功能研发目标与证据门

状态：`ACTIVE / NEXT_R3_TSL_ROUTE_CLOSED_NO_PERFORMANCE_RESULT / NEXT_R4_FA_RDCE3_CER_PLR160_DESIGN_FROZEN`

## 0.0 2026-08-04当前活动目标与统一指标命名

NEXT-R3的`R3-RDCE160×TSL-160`已在N607最终r3的prepare阶段关闭。数值wire floor下穿已按同一冻结floor修复并通过独立复核，但真实physical-LOO仍出现`physical LOO fold has no correctly classified reference margin`。这说明TSL信赖半径依赖“参考头先正确分类”的前提在真实资产上不成立。r1/r2/r3均未产生完整prediction或score，严格记为`NO_PERFORMANCE_RESULT`；不创建r4、不修补TSL、不从技术退出推断性能。

下一唯一候选冻结为`NEXT-R4 FA-RDCE3×CER-PLR160`：

- `FA-RDCE3`只用REG0旧类support的类等权残差，在既有RDCE rank-3基上闭式估计一个跨类共享3维域位移；Phase1只封存与checkpoint共同密封的INT8多样本聚合中心、公共Fisher精度、残差方差和公共半径。R0先减一次`B^Ta`，再执行一次RDCE；R1 signed-unit输出后禁止再次位移、ReLU或L2归一化。DA1_REG1逐字节复用DA1_REG0的`a/κ`，不得以新类support重拟合DA。
- `CER-PLR160`保留qKNN为基座。K1的head逐logit精确alias qKNN；K5只用support构造类等权共享对角shrinkage的中心化prototype-logit残差，以无正确率、无LOO、无top-k的连续公共公式缩放。残差为零或量化后无函数时记`NO_HEAD_FUNCTION`并精确alias Q，不视为技术失败。
- 两组件都不更新完整网络参数，不在Phase2执行Fishr式梯度方差匹配。Fishr只可用于Phase1构造/保护稳定表征；K1阶段只允许可辨识的共享低秩闭式状态。

所有后续artifact、表格、报告和对话统一使用以下四个主状态；历史字段只可在括号中作映射，不再作为主名称：

|状态码|唯一中文主名称|必须报告的主指标|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|old BA、old-floor、总正确数；new/H=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|同一旧query上的old BA、old-floor、总正确数；new/H=`N/A`|
|`DA0_REG1`|域适应前/新类注册后|old BA、seen-new、H、all-floor、总正确数|
|`DA1_REG1`|域适应后/新类注册后|同上；作为联合主结果|

禁止单独使用“before/after”“old-after”“注册后”而不注明DA和REG状态。注册效应必须在固定DA状态内比较REG1−REG0；域适应效应必须在固定REG状态内比较DA1−DA0。

最小矩阵固定为`2 receiver(1-1,18-2)×6 held-class×K1/K5=24`个逻辑行，不扩receiver、seed、K或超参数。K1每行4个Q唯一预测，head仅保存alias receipt；K5每行4状态×Q/H，共8个唯一预测；全矩阵共144个唯一prediction、192个含alias的arm artifact。主要因果量为DA前注册、DA后注册、两种DA状态下的注册效应、K5的`H−Q`和DA×head交互。

完整矩阵回收后才使用性能裁决，不作为运行中健康早停：保留候选要求`DA1_REG0−DA0_REG0`的old BA至少`+0.25pp`，`DA1_REG1−DA0_REG1`的H至少`+0.25pp`，K5的`H−Q`同时使H至少`+0.25pp`且总正确数增加，并满足seen-new、all-floor非降及4个receiver×K聚合层至少3个H非负。未达到即按组件或整条路线关闭，不调`λ/γ/K/receiver`。

工作流只保留直接影响下一真实性能实验的步骤：设计已由独立监督`MERGE`；随后是科学核心实现、query零fit/update/selection等必要负测、真实checkpoint无truth smoke、独立`P0=0/P1=0`、Git提交和唯一runner发布。主agent`gpt-5.6-sol/high`负责协议、整合、数据/结果分析与裁决；`gpt-5.6-terra/max`负责DA/head科学实现和独立审查；`Luna/max`负责文件、hash、Git机械工作及冻结规格下的唯一N607 runner。不得把重复数据验证、额外签名层、完整125矩阵或文档美化加入发布gate。

## 0.2026-08-03闭环与目标重置

`D130-JOINT6-AFFINE-SCALE-R1`已在N607完成168/168条source-held LOCO prediction和独立score。CSPAR-2的K5 DA效应为`ΔH=-0.556pp、Δ总正确数=-9`；SRDH-2的K5 DA效应为0；两者均失败。D92-Lite160相对Full160的拟合解析MAC减少99.754%、显式峰值工作集减少90.607%，但`A_held_proxy`下降0.529pp、`F_retained`下降1.270pp，因此只有效率收益，没有满足联合目标的性能收益。两候选均关闭，不进入G0、fresh63、Target25或125，也不得调layer/rank/step/view/seed/shrinkage/阈值复活。

`D131-D92-Lite160-QTIE-Target125`在393条partial prediction后因7个确定性技术失败停止：5个K1 qKNN精确并列，另有2行虽存在有限full288表示但primary160为精确零。该run没有完整manifest、truth或score，严格记为`NO_PERFORMANCE_RESULT`；D131的r4补丁链和125重跑永久关闭。D131只提供两条实现约束：唯一160维表示必须对ReLU零行作同一forward内的pre-ReLU signed totalization；K1精确并列必须无truth fail-closed，不能借助类别顺序、ID或hash打破。

下一研发轮只允许一条原理不同的候选，冻结名为`NEXT-R1 FABR-TSL（Fisher-Anchored Block Residual + Tail-Safe Lite）`。它同时解决：

1.从特征空间共享变换转向参数空间局部残差，不更新全部checkpoint参数；
2.层选择不能预设为浅层。先用允许的Phase1数据计算各block的receiver敏感度、TX判别保持率、Fisher曲率和跨receiver方向一致性，预注册选择一个block；不得用Target query、truth或局部性能选层；
3.Phase1只联合封存类置换对称的低秩梯度/Fisher基和量化统计。Phase2只用当前K-shot support拟合2个共享残差系数，并用Fisher二次项约束更新；禁止encoder全参反向、checkpoint replacement、逐类adapter和query更新；
4.D92-Lite必须同步改为尾部安全的单仿射头：保留160维INT8/FP16 wire和低成本对角统计，以类对称的公共trust region约束support拟合相对冻结参考头的偏移；不得恢复old/new role分裂、两套稠密协方差、D62 row splice或按类门控；
5.设计必须先证明K1/K5可辨识性、公共变换对所有注册类的合法性、Fisher锚定与头部trust region不会把identity当收益，再进入实现。Fishr只作为Fisher重要性/曲率来源之一，不把“浅层Fishr”预设为答案。

NEXT-R1的最小性能矩阵固定为一个候选、42个receiver-held×seen-class-LOCO fold、`K∈{1,5}`，共84个candidate-row；每row保留`R0Q/R0F/R0L/R1Q/R1F/R1L`六个逻辑臂，以分别识别DA、Lite D92和联合替换效应。公共R0只计算一次，F臂只作为同表示的历史D92机制比较器，不复活D130方法。完整负结果立即关闭；完整正结果保持同一method lock依次进入G0真实588功能面、一次fresh63六臂矩阵和一个预注册单seed Target25 screen，不运行125。正式Target仍以完全同键的历史formal D92作外部基线。

NEXT-R1设计流程严格时间盒为`DESIGN_DRAFT -> FEASIBILITY_REVIEW -> DESIGN_FROZEN`：只允许一次方法作者与独立supervisor交叉复审，可行性摘要不超过20行；只要协议合法性、K1/K5可辨识性、真实forward功能路径和资源上界没有P0/P1问题，就立即冻结并实现。补充文档、通用工具、重复数据验证、额外签名/权限层、P2测试和论文叙事均不阻塞发布；若一次复审仍证明机制不可辨识，则直接拒绝该设计，不通过增加流程或盲调实验延长。

### 0.1NEXT-R1冻结方法锁

FABR只从`t1.norm.{weight,bias}`、`t2.norm.{weight,bias}`、`t3.norm.{weight,bias}`、`cls_head.joint_proj.0.bias`四个位点中按Phase1-only规则选择一个block，固定rank2。每个receiver-held×class-LOCO fold的资产构建严格排除held receiver和held class；以冻结TX交叉熵参数梯度先构造`G=E[gg^T]`，固定`eps_F=max(float32_tiny,64*float32_eps*tr(G)/P)`，再令`F=G+eps_F I`、`S=Cov_rx(E[g|rx])`并按top广义特征值排序；`tr(G)`非有限或不大于0时fail-closed。候选必须同时满足：在实际INT8反量化`B`上重算的leave-one-receiver子空间最小principal cosine`C>=0.9`；使用同一反量化`B`的`±2^-6`扰动时，Phase1验证TX总正确数和逐class floor均不低于R0；反量化`B`满rank；完整`K=B^T F B`满足正定和`cond(K)<=10^6`；真实forward作用超过重复前向数值抖动。按首个广义特征值降序选择，数值并列按`t1→t2→t3→joint`打破。target support/query和F臂均不参与选层。

R1只对当前调用作功能式覆盖`phi'=phi0+Ba`，不写checkpoint或`state_dict`。Phase1只封存一个INT8`B`、每方向FP16 scale、反量化基上的完整2×2对称FP16`K`及冻结常数。Phase2用当前全部registered support等权计算类间分离；K5紧致项使用physical-LOO均值，K1紧致项严格为0。以`delta=2^-6`做rank2中心差分，固定`lambda_F=1、lambda_0=10^-3、rho=0.25、m=0.20、tau=0.10`，解`a_tilde=-(H+lambda_F K+lambda_0 I)^-1 g`并投影到`a^T K a<=rho^2`。R0 support forward可复用，额外执行4个扰动support forward和1个精确R1 support forward；query只各执行一次R0/R1 forward，不参与拟合。非有限、病态、`a`为数值零、特征与Gram变化不超过重复前向抖动或量化后仅噪声变化时，直接`REJECT_REVISION_NO_FUNCTION`，不得alias/fallback或在target侧换层。

唯一表示由同一次真实functional forward捕获的`joint_proj.0`线性输出、即160维pre-ReLU向量`p`定义，不得误读该层的320维输入：先计算`h=ReLU(p)`；若`||h||2>0`则`z=h/||h||2`，否则若有限且`||p||2>0`则`z=p/||p||2`，只有`p`精确全零或非有限时拒绝。范数累积使用float64，不增加target侧阈值。R0、R1、4个`±delta`support forward、Phase1验证、Q/F/L共享cache及真实checkpoint smoke必须从该同一160维线性输出调用同一表示函数；不得读取full288、FFT96、RF32或另一个辅助表示。

FABR令`N=C*K`、`mu_c=L2(K^-1 sum_k z_ck)`，support目标唯一冻结为：

```text
L = L_sep + L_comp
L_sep = [2/(C(C-1))] * sum_{c<d} tau*log(1+exp((mu_c^T mu_d-m)/tau))
L_comp(K=5) = [1/(C*K)] * sum_c sum_k [1-z_ck^T L2((K-1)^-1 sum_{j!=k}z_cj)]
L_comp(K=1) = 0
m=0.20, tau=0.10
```

若`C<2`、任一类support数不等于K或任一均值无法L2归一化则fail-closed。参数、类别和样本的canonical flatten顺序分别为method lock保存的实际参数key列表及各段shape、冻结registered-class顺序和类内physical ID升序。以4个中心差分得到`g_r=[L(+delta e_r)-L(-delta e_r)]/(2delta)`及按上述canonical顺序展平的表示Jacobian`J`，并明确取`H_FABR=J^T J/N`为Gauss-Newton矩阵，不称为真实Hessian，也不与TSL仿射头`H_TSL`混名。`B`反量化后计算`K_F=B^TFB`并用于求解。连续解先按round-to-nearest-even舍入为FP16`a16`，再反解为float32`a32`；若`a32^T K_F a32>rho^2`，仅按`sqrt(rho^2/(a32^T K_F a32))`作一次同一`K_F`径向缩放并重新舍入FP16、反解float32，随后复验；第二次仍超界即fail-closed，不循环、不clip单个系数。

TSL的K1 F/L逐logit严格alias同表示Q，K1收益只归因于`R1Q-R0Q`。K5从Phase1共同封存的类/receiver对称对数方差先验`q_logv0∈INT8[160]`、FP16`scale_logv0/offset_logv0/nu0/rho_h`出发，唯一解码为`v0_j=exp(offset_logv0+scale_logv0*q_logv0_j)`。令`u_ck=L2(z_id(x_ck))`，`mu_c=K^-1 sum_k u_ck`，`mu_c`不再做第二次L2归一化；对每个独立physical support定义`e_ck=((K-1)/K)*(u_ck-(K-1)^-1 sum_{j!=k}u_cj)`，并构造：

TSL的公共API只允许`fit(support_z160,support_labels,registered_classes)`，不得接收`old_count`、role字段、F臂结果、query、truth或局部性能。Phase1先验按以下唯一规则构建。对每个允许receiver×class cell按physical ID排序，至少含2个signed-pre-ReLU160单位表示；以无偏对角方差`s_rcj^2=sum_i(z_rcij-mean_i z_rcij)^2/(n_rc-1)`，令`v_floor_p1=max(float32_tiny,64*float32_eps*mean({s_rcj^2:s_rcj^2>0}))`，无正方差时拒绝，并计算`ell_j=mean_{r,c} log(s_rcj^2+v_floor_p1)`，每个cell等权。固定`offset=(max_j ell_j+min_j ell_j)/2`、`scale=(max_j ell_j-min_j ell_j)/254`；`scale<=0`时拒绝；`q_logv0_j=clip(round_to_even((ell_j-offset)/scale),-127,127)`。`offset/scale`按FP16 round-to-nearest-even封存并以实际反量化值生成最终`v0`；receipt保存最大绝对/相对解码误差及全部`v0>0`。

`nu0`取全部cell自由度`n_rc-1`的几何均值，再按FP16 round-to-nearest-even封存。`rho_h`只从Phase1 physical-LOO验证构建：每个cell按同一TSL公式得到`H_ref/H_hat`及`D_cell`；对参考头判对的每个validation样本和每个错误类别计算pairwise参考margin`m_ic>0`与改变`d_ic=(hat_y-hat_c)-(ref_y-ref_c)`，令`eta_cell=min(1,min_{d_ic<0} m_ic/(-d_ic))`，空负集合取1，`rho_cell=eta_cell*D_cell`。`rho_h`取全部有限正`rho_cell`的固定Type-7 5%分位数；无有限正值时拒绝；FP16封存若向上舍入则用`nextafter(float16_value,0)`下调一格，确保实际`rho_h`不超过未量化值。最终资产绑定checkpoint hash、cell物理ID根和表示规则hash，并保存`nu0/rho_h`量化误差、正性及逐cell量化后margin-slack receipt。FABR的support/query forward成本与head因果拟合成本分开计量。

```text
v_post = (nu0*v0 + sum_c sum_k e_ck^2) / (nu0 + C*(K-1))
v_sph  = mean(v_post)
Wref_c = mu_c/v_sph;  bref_c = -sum_j(mu_cj^2)/(2*v_sph)
What_c = mu_c/v_post; bhat_c = -sum_j(mu_cj^2/v_post_j)/2
```

先分别执行唯一的全类仿射中心化`W-=mean_c(W_c)、b-=mean_c(b_c)`，不再作范数归一化，也不分old/new。令`D=||[What-Wref,bhat-bref]||F`、`eta=min(1,rho_h/D)`、`H=Href+eta(Hhat-Href)`。输入必须为每类恰好K个独立physical support；`u/mu/e/v0/v_post/v_sph/nu0/rho_h/D/eta/H`均须有限，`nu0>0、rho_h>0`。固定`v_floor=max(float32_tiny,64*float32_eps*mean(v0))`，任一`v0_j<v_floor`或`v_post_j<v_floor`均fail-closed，不作clip。若`D<=64*float32_eps*max(1,||Href||F,||Hhat||F)`或量化后仅有数值噪声变化，直接`REJECT_REVISION_NO_FUNCTION`，不回退。该全局Frobenius约束只限制相对spherical reference的logit扰动，不宣称保证真实floor；floor只由外部同row完整矩阵判定。H最终编译为单一`INT8 W[C,160]+FP16 scale[C]+FP16 intercept[C]`，F臂不得回流`v_post/eta/DA`。

所有K、所有臂都必须在float32最终logit上执行精确top-tie检测；任一query出现最高logit精确并列即记为`TIE_UNRESOLVED / NO_PERFORMANCE_RESULT`。K1的Q/F/L还必须逐logit共享同一结果。不得使用registry顺序、类别ID、physical ID、hash、role或truth消除并列。发布前只执行一次无truth的K1 liveness scan；它只验证能否形成唯一prediction，不读取性能，也不形成新增调参gate。

独立`FEASIBILITY_REVIEW`及差分复核最终确认上述修订已闭合，裁定`P0=0、P1=0、DESIGN_FROZEN`。现在立即进入唯一候选实现；不新增第二轮方法设计、第二表示或额外矩阵。实现若偏离本节任一公式、常数、输入边界或fail-closed语义，必须退回设计审查，不得直接发布实验。

### 0.2用户明确请求下登记的D138 D92-Lite修复候选

用户在D131路线永久关闭后再次明确要求“修复优化，然后跑D92-Lite的125实验”。因此新增独立candidate`D92-Lite-PR160/r1`，不修改D131，也不把NEXT-R1 FABR混入本次实验。该候选只修复D92-Lite的表示与头部边界，直接沿用已验证的Target125输入定位和750个单臂prediction surface；若无truth liveness或真实checkpoint smoke不能闭合，则不得启动完整125。

- 表示从D92的`registered_feature[0:160]`改为同一sealed TorchScript前向中`model.id_backbone.cls_head.joint_proj.0`的线性pre-ReLU`p∈R^160`；`||ReLU(p)||>0`时归一化ReLU视图，否则归一化有符号`p`；精确零或非有限直接fail-closed。新extractor与源runtime、checkpoint和method lock绑定，且以`ReLU(p)`与原160维输出做parity核验；不读取full288、FFT96、RF32或第二套表示。
- K1为该160维表示的全注册类qKNN逐logit精确alias；K5/K10为全注册类共享对角OAS仿射头，wire固定为`INT8 W[C,160]+FP16 scale[C]+FP16 intercept[C]`，不使用old/new角色分裂。拟合只读support，query端零fit、零update、零selection。
- 每个query独立对全部已注册类竞争；最终float32最高值精确并列统一`TIE_UNRESOLVED`，不得使用registry顺序、类别ID/hash、physical ID、role或跨query重分配打破并列。任何完整125性能结论仍需同row prediction、truth和score闭合；本candidate的表示修复本身不是性能成功。
- 状态映射固定为`DA0_REG0=before`与`DA0_REG1=after`；`DA1_REG0`和`DA1_REG1`对这个单一`M_JOINT`运输臂候选均为范围外，不生成四状态DA因果表，也不把运输臂名称当作已执行DA机制。
- D138锁文件为`configs/d138_d92_lite_pr160_r1.json`，正式实验ID预登记为`d138_d92_lite_pr160_target125_20260804_r1`。该候选与NEXT-R1保持独立，不新增数据复验；固定received-IQ、physical IDs、receiver/TX集合、scenario、K、support/query split和`p2_min_v1`均未改变。

### 0.3D138 r3远端门终态

D138的本地实现、36项回归、31项source闭包、远端compile和Torch 2.1 extractor load均已通过。r1、r2分别在prepare前因隔离依赖缺失停止；r3已补齐最后两项依赖，但唯一prepare在未读取数据、未生成plan或prediction前因run-owned空目录`prepared`已存在而触发不可变输出保护：`FileExistsError: immutable prepare output already exists`。因此r3标记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，没有smoke、shard、truth或score证据；不得从控制门结果推导性能结论，也不删除或覆盖r3远端run root，不自动启动r4。若后续重新释放，必须先重新登记不可覆盖run ID并修复该运行器输出目录生命周期问题。

## 1.最终目标

本轮研发必须在`p2_min_v1`下形成一个同时包含轻型共享域适应和精简D92分类头的Stage2-C方法。D92不是冻结的下游基线，而是必须与DA共同删改、共同归因和共同优化的组成部分。最终候选在单seed的25个Target job上必须满足：

|slice|注册后旧类准确率|最低旧类准确率|新类准确率|
|---|---:|---:|---:|
|K10/new5|≥92%|≥85%|≥92%|
|K10/new10|≥92%|≥85%|≥90%|
|K10/new20|≥92%|≥85%|≥86%|

K5/new20相对matched K10/new20的注册后旧类、最低旧类、新类和`H_old_new`下降均不得超过5pp。K1/new20必须相对同row冻结D92基线产生真实提升，不能依靠identity、整臂fallback或未改变prediction通过。

联合方法还必须同时满足下列效率目标。基线固定为完成Target125的正式288维D92，不得用160维source-held代理状态替代：

- 精简D92固定采用单平面对称INT8系数`W_q[C,160]`、每类FP16 scale和FP16 intercept，不保留FP32 sidecar。正式D92核心数组为`B_formal=1152+590C`字节，D92-Lite为`B_lite=164C`字节；在C=26时由16,492B降至4,264B，减少74.1%。计入rank2 DA资产后，联合新增数值状态仍至少减少50%；
- K5头部拟合MAC-equivalent至少减少90%，同机同线程实测墙钟中位数至少减少50%；“不增加backbone forward次数”只适用于同一已缓存表示上的`head_causal_resource_receipt`，联合FABR额外4个扰动support forward、1个R1 support forward及R0/R1 query forward必须在系统receipt中如实单列，不得宣称端到端forward不增加；
- K5的query端head MAC至少减少40%；K1严格alias到qKNN，其资源按qKNN正式口径单列，不得写成D92-Lite零成本；
- 正式D92的old/new稠密协方差属于拟合期瞬时量，部署态本来就是单仿射头。精简目标是删除重复拟合、288维辅助块和无效计算，不得把source-held代理额外保存的两块160×160矩阵冒充正式D92部署开销。

中间门只筛掉没有功能作用或协议不合法的实现，不降低、替换或重新解释上述最终指标。

## 2.证据起点

|方法|已验证结论|对新研发的约束|
|---|---|---|
|D62|历史target-capsule完整125诊断；K10/new20同row`A_old=68.68%、N=68.78%、H=68.56%、forgetting=17.34pp`，K1/new20为`44.03%/27.15%/33.41%/24.11pp`；注册后仅24/375个场景状态实际激活，K1整体fallback|禁止用离散安全门把大多数row退回基线；这些历史125不是当前定义的Target25|
|D91|仅历史K10/new5 development；15/15 outer prediction与D62逐值相同：`A_old=82.22%、N=84.67%、H=82.62%、forgetting=10.56pp`|support目标下降或内部几何变化不等于分类功能；不得外推到125或Target25|
|D92|历史target-capsule完整125诊断；K10/new20为`A_old=71.333%、N=68.150%、H=69.555%、forgetting=14.778pp`，相对D62主要改善旧类与遗忘、轻微降低新类；K1逐值不变。正式实现作用于288维D62/D81管线，在大量full/block/crossfit组件内分别拟合old/new收缩协方差，最终只部署单个紧凑仿射头；另一个160维source-held代理会额外保存两块协方差，不能混为正式资源证据|D92必须与DA联合重构；删除role分裂、重复稠密拟合、D62行拼接和无独立贡献的FFT96/RF32块；K1不得伪造不可辨识的类内方差|
|SVRN/r4.2|完整125且相对D62全面劣化|不再使用会放大注册后旧类崩塌的分支状态|
|D104|量化机制和release代码闭合，无Target性能|ANGQ只能作为实现组件，不能预设为最终分类头|
|D130|完整168条source-held LOCO方向性代理；CSPAR-2的K5 DA`ΔH=-0.556pp/正确数-9`，SRDH-2为零效应；Lite160拟合MAC减少99.754%、工作集减少90.607%，但held-proxy与最低类下降|两候选关闭；只复用数值缩放和低成本实现，不复用失败表示变换，不把效率收益写成性能收益|

## 3.指标与同row口径

原子scenario-row固定为：

```text
(receiver, seed, K, new_count, scenario,
 capsule_id, split_id, query_id_root, method_lock)
```

同一矩阵的全部臂必须共享全部字段、同一old query、同一new query和同一独立scorer。每个正式slice由同seed的5个receiver×3个互斥LEO场景组成15个scenario-row，并等权宏平均。

- `A_old`：注册后旧类准确率；
- `N`：已注册新类准确率；
- `H`：同一row的`A_old`与`N`调和均值；
- `F_old`：对每个旧类先聚合该slice全部15个scenario-row，再取最低类准确率；
- `forgetting=B_old-A_old`；
- 辅助报告：mean row-floor、worst row-floor、逐类、逐receiver、逐scene和正确数。

不得用不同row的边际最大值拼接结论。`F_old`是全部旧类的通用下界，不是预选弱类清单。

## 4.D130已完成候选与原理边界

以下2条候选是D130的历史冻结设计，已完成且关闭，不再是当前研发入口。二者共享相同160维输入、相同support/query、相同六臂头部和评分器，不为各候选另造head。

|候选|机制|K5状态|K1边界|冻结理由|
|---|---|---|---|---|
|C1=`CSPAR-2`|Phase1联合封存rank2 nuisance轴`B`，Phase2用全类等权类内散度估计轴向收缩，形成非标量PSD度量|2个共享收缩系数|使用Phase1封存`alpha0`；只能称sealed metric benefit，不能称target support DA|闭式、低状态、无encoder梯度|
|C2=`SRDH-2`|Phase1封存rank2非线性响应字典`P/Q`及summary标准化统计，Phase2从全类support共享响应生成低秩残差|2个共享响应系数|用跨类共享summary，允许形成可辨识状态；不得含类专属参数|与PSD度量原理不同，可改变非线性邻域|

`RDCE-r3`与C1同属“Phase1轴＋target scatter PSD”族，历史D106 source-held小幅正收益不足以构成独立原理，继续关闭。CSPAR-2与SRDH-2也因D130完整负结果关闭。NEXT-R1必须按§0证明参数空间局部残差与尾部安全头的独立可辨识机制，不复用D127/D128 checkpoint replacement链，也不把浅层更新写死为唯一选择。

C1在Phase1只保存`B∈R^(160×2)`及冻结`alpha0/alpha_max/eps`。K5按全部注册类等权估计：

```text
S_w = [C(K-1)]^-1 Σ_c Σ_i (u_ci-μ_c)(u_ci-μ_c)^T
v_j = b_j^T S_w b_j
v_perp = [tr(S_w)-Σ_j v_j]/158
α_j = clip(1-(v_perp+eps)/(v_j+eps), 0, alpha_max)
φ_C1(u) = normalize((I-B diag(α) B^T)^(1/2)u)
```

C2固定：

```text
a_j = a_max tanh(([C^-1 Σ_c K^-1 Σ_k tanh(q_j^T u_ck)]-m_j)/d_j)
φ_C2(u) = normalize(u + P[a ⊙ tanh(Q^T u)])
```

`B/P/Q/m/d/a_max`只能来自与checkpoint联合封存的允许Phase1聚合知识；Phase2不读取source/clean样本或FP32 sidecar。C1/C2都不得读取query、truth、role、class quota或全局批次计数，也不得包含TX/class ID专用参数。

首轮候选矩阵固定为7个receiver×6个class的receiver-held×seen-class-LOCO共42折；每折构造新增Phase1资产时同时排除held receiver与held class，随后在held receiver的固定received-IQ上，把其余5个Phase1已见类记为`retained`组、held class记为`held-proxy`组，分别执行K1/K5六臂。checkpoint已经见过全部6个TX，因此held class绝不能重命名为注册新类；42折只产生方向性代理证据，不输出正式`N/H_old_new`，也不代替Stage2-C。K1严格为K5 support前缀，support/query物理ID互斥，且每折Phase1资产seal必须同时绑定held receiver、held class和420条Phase1-fit物理ID根。42折结果可以在全部prediction封存后按§5的代理主比较关闭明显负收益候选；不得据此修改资产、超参数或fold。最终资产按同一冻结公式重建一次。不得复用D127/D128的Phase1 autograd、checkpoint replacement或outer-audit发布链。

预测生成和truth评分分离。发布前只保留下列必要检查：

1.状态只读取不可变Phase1 bundle、当前row合法support和冻结配置；
2.query零fit、零selection、零update，每条query独立面对全部注册类；
3.类别标签置换等价，两个任务组使用同一规则；source-held代理不得产生正式old/new声明；
4.真实checkpoint无truth smoke至少改变feature、neighbor、margin或argmax之一；
5.C1若退化为共同平移、正交或全局正缩放并保持邻居排序，直接拒绝；C2若共享summary为零或残差无功能，直接拒绝；
6.序列化字节、拟合MAC、同机同线程时延、瞬时工作集和backbone forward次数形成receipt。

协议错误记为`INVALID / NO_PERFORMANCE_RESULT`；机制合法但无可观测决策作用记为`REJECT_REVISION_NO_FUNCTION`。

## 5.D130历史共享缓存六臂联合筛选（已完成）

本节记录D130已执行的冻结因果矩阵，供结果追溯；它不再定义NEXT-R1的方法内容。NEXT-R1按§0只保留一个候选，但继续使用同一套六臂因果接口，避免失去Lite D92相对历史D92机制比较器的独立归因。

每个候选只缓存两种表示：`R0=normalize(z_id160)`与`R1=normalize(phi_Ci(R0; support-only state))`。每种表示只做一次support/query特征缓存，再供三个头复用：

|头|定义|目的|
|---|---|---|
|Q|冻结Phase1-lock qKNN|DA因果基线|
|F=`D92-Full160`|当两组都含多个类时，在160维输入上复现历史D92的两组自动收缩full covariance、0.5/0.5平均、等先验与同一仿射中心化；5-retained/1-held的source-held矩阵只能使用明确标注的`single-class proxy extension`，不是历史D92严格复现|同表示机制对照；正式D92比较推迟到Target25|
|L=`D92-Lite160`|相同双组对称语义，但只拟合diagonal OAS并直接编译为仿射头；source-held结果只称proxy|验证删减稠密拟合的方向与成本|

因此每个候选的最小完整矩阵固定为：

|臂|表示|头|
|---|---|---|
|R0Q|基础160维|qKNN|
|R0F|基础160维|D92-Full160|
|R0L|基础160维|D92-Lite160|
|R1Q|适应后160维|qKNN|
|R1F|适应后160维|D92-Full160|
|R1L|适应后160维|D92-Lite160|

F/L必须部署为同一wire：`W_q[C,160]+FP16 scale[C]+FP16 intercept[C]`，序列化均为`164C`字节，query端均为`160C`MAC；Lite的效率主张仅来自拟合时延、峰值瞬时工作集、dense matrix/solve次数，不虚构部署态差异。正式288维`D92-Formal`继续作为同row外部全管线参考和资源参考，不进入六臂head因果结论。

仿射编译若遇到有限但超过FP16范围的截距，只允许对同一head的全部类别共同乘一个正2次幂，并同步缩放权重与截距；这在量化前严格保持argmax和类别置换等价，且不增加wire字节或query MAC。不得逐类clip/scale、fallback到Q或把它宣称为INT8/FP16量化后对任意query严格等价。若共同缩放会使任一非零权重行的逐类scale低于FP16最小正规数，必须确定性失败并关闭该数值实现。receipt需记录指数、缩放前后峰值、截距归零/子正规计数和明确的等价范围。

K1中F/L严格alias Q并保存等价receipt，不重复计算，也不提出head改进声明；K1只比较`R1Q-R0Q`。K5使用全部六臂，并用以下三个预注册主效应判断：

```text
DA_EFFECT       = R1Q-R0Q
LITE_BASE       = R0L-R0F
JOINT_REPLACE   = R1L-R1F
```

source-held矩阵的三个效应必须在同row池化后满足`ΔH_retained_held_proxy>0`且retained＋held-proxy总正确数严格增加；`ΔA_retained>=0`、`ΔA_held_proxy>=0`、`ΔF_retained>=0`作为方向性“不牺牲”条件。这些字段不得写成`A_old/N/H_old_new`。逐receiver和逐类完整报告，但不添加0.5pp级小样本门。真正的`A_old/N/H_old_new/F_old`以及相对历史formal D92的联合收益只在Target25同键评分。`R1L-R0L`、`R1F-R0F`和六臂交互项只作解释性结果，不增加发布gate。

### 5.1D130历史最小实验矩阵与停止规则

首轮只执行42个receiver-held×seen-class-LOCO fold×`{K1,K5}`，每候选84个原子row。C1与C2可分配到不同GPU，但必须共享真实checkpoint、固定received-IQ、物理ID清单、K前缀、代码commit和代理评分定义。每个fold中公共`R0Q/R0F/R0L`只计算一次并由两个候选引用同一cache/receipt，每个候选只补`R1Q/R1F/R1L`；不得增加18row Target development或其他中间矩阵。

所有prediction完成并封存后一次性打开source-held truth。候选若任一K5代理主效应失败，立即记为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并关闭，不调层、rank、step、view、seed、shrinkage或阈值。若两候选都通过，只按`min(DA_EFFECT_H_proxy,LITE_BASE_H_proxy,JOINT_REPLACE_H_proxy)`、最差receiver代理联合增益、总正确数和端到端资源作冻结排序，选择一个进入G0/G1的方向性胜者；这不是Stage2-C晋级或正式正收益结论。若两候选都失败，本revision立即以完整负结果结束，不进入G0/G1/Target；下一研发轮最多补1条原理不同的新候选，必须先给出可辨识性与协议合法性推导，不得修改或重跑本轮失败候选。

### 5.2D130历史发布硬门

发布前只要求：实际Git方法入口；query零fit/update/selection及禁止clean/source/query-truth/role/quota/global-reassignment的聚焦负测；真实checkpoint-derived received-IQ archive no-query smoke；独立复核`P0=0/P1=0`；不可覆盖run ID/output；本地Git提交；N607预检与资源记录。既有`VALIDATED_ONCE`数据不因方法变化重验，不要求新签名层、通用执行平台、完整论文叙事或重复D62/D92/SVRN矩阵。source-held proxy发布只需保存解析字节/MAC和原始时延/工作集receipt，不以正式90%/50%/40%阈值阻塞；这些阈值必须由Target25同机同线程重复测量的中位数和实际峰值判定。

## 6.D130已取消与NEXT-R1沿用的G0→G1→Target25路径

D130没有胜者，因此D130的全部后续步骤均未触发并已取消。NEXT-R1若通过84-row六臂小矩阵，则沿用下列最小晋级顺序，但必须使用新的不可覆盖run ID、NEXT-R1的冻结method lock和独立报告，不得据此发布D130 G0、fresh63或Target25。

小矩阵只负责选出一个联合胜者。胜者保持同一method lock，按以下顺序扩展；失败即停止，不回头从已打开性能中修改候选：

1.`G0`：在既有588条Phase1功能面闭合非恒等、量化parity和两份资源receipt；不读取Target truth，不把功能变化写成性能收益；
2.`G1`：只运行一次未参与本轮设计的fresh63 source-held六臂矩阵；它仍是Phase1已见类代理，只检查跨receiver/class方向与负迁移，不输出正式new-registration指标，也不替代Target Stage2-C；
3.`G2`：G0/G1均闭合后，只运行一次单seed Target25，执行本节K10/K5/K1最终门。

不得为落地G0/G1重建通用发布平台或重复历史D62/D92/SVRN矩阵。两份资源receipt分别为：

- `head_causal_resource_receipt`：同160维、同INT8 affine wire下比较Full160/Lite160的K5拟合墙钟、峰值工作集、dense elements及factorization/solve次数；
- `system_formal_replacement_resource_receipt`：比较formal288 D92的`1152+590C`与Lite160的`164C`字节，并显式标记`representation_pipeline_changed / not_head_causal`。

### 6.1单seed Target25门

候选、method lock和全部超参数在Target访问前冻结。screen seed必须排除development seed`713102`，并从已完成D92 Target125中按数值升序选择第一个未参与本轮小矩阵/fresh63候选评分且具备全部同键D92 artifact的seed；完整性在打开该seed的本轮候选prediction前只读核验，若缺失则顺延到下一个完整seed，不得读取本轮性能后改选。它是`METHOD_UNSEEN_SCREEN`，不是全项目从未读取truth的盲测。矩阵固定：

```text
5 receivers × 1 seed ×
{K10/new5, K10/new10, K10/new20, K5/new20, K1/new20}
= 25 jobs
```

每个job覆盖3个物理ID互斥的`leo_*_weak`场景。一次Target25只评估一个revision，不得从25行中选择receiver、scene、class或slice重跑。

§5小矩阵及§6的G0/fresh63完成方向筛选。Target25运行胜者的六个逻辑臂`R0Q/R0F/R0L/R1Q/R1F/R1L`；K1的F/L按等价receipt alias Q，不重复计算。历史formal D92只从已完成125中按完全相同的`capsule_id/split_id/query_id_root/receiver/seed/K/new_count/scenario`键连接，不重跑D92。键不全同则该seed不得用于本轮K1 paired目标，应在prediction前改选另一个具备完整同键D92 artifact且本方法未见的seed。

Target25完整性单位保持：

```text
25 jobs × 3 scenarios × 6 logical arms
= 450个scenario-arm pair
= 900个state prediction surface（before/after各450）
```

每个pair必须同时封存Stage2-B的before旧类预测和Stage2-C的after旧类/新类预测。任一state或arm缺失、预测可覆盖、truth先于全部900个state prediction surface开放、键不唯一、哈希不匹配或只完成联合臂，均不得进入性能分析。K1 alias必须有逐logit等价receipt；`forgetting`只能由同一pair的before/after旧类预测计算，不能从其他方法或其他run补入。

K10执行§1全部硬门。

K5以同receiver、同scene的K10/new20为matched基线。预登记时必须锁定`K5 support physical IDs⊂K10 support physical IDs`，且两者的`query_id_root`逐scene完全相同；只有满足该嵌套关系时，`A_old/F_old/N/H`下降≤5pp才属于paired结论。若不满足，只能报告非配对差值，不能用于通过本门。单scenario-row退化完整报告，但不再另设边际灾难阈值。

K1以同row冻结D92为基线：

```text
ΔH >= +2pp
ΔF_old >= +2pp
ΔA_old >= 0
ΔN >= 0
old+new总正确数严格增加
```

单seed通过记为`TARGET25_SCREEN_PASS`，证明本轮研发目标在该预注册seed上达到；不能据此宣称多seed稳定。

## 7.本轮完成边界

当前NEXT-R1轮的第一终点是完成理论设计、独立复核并冻结一个候选；第二终点是一次完整84-row六臂source-held矩阵。若该矩阵失败，本轮结束；若通过，保持同一method lock依次完成G0真实588功能面、一次fresh63六臂矩阵，再进入一个完整、预注册、单seed的`TARGET25_SCREEN_PASS`；不运行125或自动追加第二个Target seed。需要对外形成多seed`PROMOTABLE`声明时另行预注册confirm seed；它不属于当前发布硬门，也不能延迟首个Target25。任何结构、block、rank、Fisher统计、trust region、量化、阈值或fallback修改都产生新revision，不得借用旧prediction。

## 8.研发工作包与模型分工

|工作包|职责|执行模型|
|---|---|---|
|WP-DA|轻型共享DA、层位/状态可辨识性、K1/K5 support更新与资源|`gpt-5.6-terra/max`|
|WP-D92|历史D92计算图删改、类置换对称共享统计、K1边界及部署状态压缩|`gpt-5.6-terra/max`|
|WP-CODESIGN|DA与精简D92共享view/统计/缓存、六臂因果接口和formal参照边界|`gpt-5.6-terra/max`|
|WP-DATA|目标、矩阵、同row指标、结果分析、拒绝语义和交叉审查|`gpt-5.6-sol/high`|
|WP-INTEGRATE|协议解释、方案冻结、代码整合、独立复审和最终决策|主agent，`gpt-5.6-sol/high`|
|WP-IMPLEMENT|复杂科学核心实现、改变或实现新机制、需要科学判断的复杂缺陷修复|`gpt-5.6-terra/max`|
|WP-RUNNER|完全冻结后的唯一N607落地、同步、启动、健康检查、监控和artifact回收；不得修改方法或矩阵|默认`Luna/max`；仅科学或P0/P1调试改用`gpt-5.6-terra/max`|
|WP-MECH|固定清单、hash、manifest、报告骨架、字段完整性、冻结helper实现及执行已冻结的本地测试命令|`Luna/max`|

方法agent不得自我认证。WP-DATA必须审查K-shot可辨识性、common-transform cancellation、support proxy过拟合、旧/新任务平衡、类置换、资源和query/role/quota禁区。

每个功能包由不同agent拥有非重叠文件面。服务器实验必须另设唯一runner；当commit、矩阵、命令、路径、健康规则和停止规则已完全冻结时默认使用`Luna/max`执行机械落地，只有需要科学判断或P0/P1调试时才使用`gpt-5.6-terra/max`。Luna可按冻结handoff执行SSH/SCP、启动、短连接监控与artifact回收，但不得修改科学方法、选择或改变method/loss/rank/threshold/quantization/matrix/receiver/seed/K、解释性能或作晋级判断。runner不得改方法、调参、按性能重跑或与主agent重复启动。主agent和WP-DATA使用sol-high读取完整25-job/450-pair/900-state预测与评分证据后再作晋级决定。

## 9.拒绝语义

|状态|含义|
|---|---|
|`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|预测前系统性技术失败|
|`PARTIAL_DIAGNOSTIC_BIASED_NOT_PROMOTABLE`|只有partial prediction或score|
|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|完整矩阵完成但性能门失败|
|`TARGET25_SCREEN_PASS`|单seed25达到本轮硬目标|
|`SCREEN_POSITIVE_NOT_CONFIRMED`|单seed通过但确认seed失败|
|`PROMOTABLE`|另行授权的多seed确认完成；不属于本轮默认硬门|

禁止按receiver、scene、class、seed挑选结果，禁止跨run拼接极值，禁止用source-held替代Target。

## 10.当前执行优先级

1.`D106-RCMR`真实588条功能面及source-held结果只作为历史非晋级证据；`D121`、`D122`组合项、`D123`已关闭，旧run不重跑、不沿用其G0/G1流程、不修旧通用release链；
2.`D106` Target25 r7仅完成46/600 state后技术退出，严格为`NO_PERFORMANCE_RESULT`；当前没有新的Target性能；
3.历史formal D92保留为固定参照，但不再视为最终头：K1整臂fallback、288维D62/D81管线、old/new重复稠密拟合和row-splice计算是本轮明确删改对象；160维held代理的额外协方差状态只作独立工程诊断；
4.D127/D128全部是prediction前技术停止，没有性能结论；其Phase1 autograd/checkpoint-replacement实现路线已按预注册规则关闭，不再修复、不创建新run；
5.D130的CSPAR-2与SRDH-2已完整失败并关闭；共同正2次幂FP16修复和Lite160低计算实现可复用，但不得据此宣称性能正收益；
6.当前方法目标是§0的NEXT-R1设计推导：先完成block选择准则、低秩Fisher残差、K1/K5可辨识性和Tail-Safe Lite公共trust region；在`DESIGN_FROZEN`前不启动实验；
7.冻结后只实现一个候选和`R0Q/R0F/R0L/R1Q/R1F/R1L`六个逻辑臂。真实checkpoint no-truth smoke、聚焦协议负测、独立`P0=0/P1=0`、Git提交完成后立即发布84-row必要矩阵，不重验数据、不建通用平台；
8.NEXT-R1完整小矩阵任一预注册联合主比较失败即关闭，不调参复活；通过才保持同一method lock进入G0真实588、一次fresh63和一个单seed Target25，不运行125或重复D62/D91/D92/SVRN矩阵。
