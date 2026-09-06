# ADV3B02-ECRS-V1R/V2设计落地计划

日期：2026-09-06。交付状态：**计划完成，实施待执行**。

本计划的目标是把用户提供的优化规范与补充分析转成可以逐项实施、验证和归因的工程路线。主线为：**先修复训练接线，再统一响应估计，随后学习跨接收机身份结构，最后验证融合是否值得其计算成本。**保留ADV3B02双骨干、原PA路径、160维身份接口和既定source接收机。

本次仅交付计划，不修改模型、训练器、旧实验记录，不启动或续训N607实验。附件中的命令、建议及“下一步”是设计材料，不构成本次执行授权。下文所有新参数和科学门槛均为实施建议，不是实测结果或已经启用的配置。

## 1.依据、基线与证据边界

### 1.1本次实际核对的资料

|编号|资料|用途与边界|
|---|---|---|
|S1|用户提供的`C:/Users/lh594/Downloads/ECRS_V2_optimization_spec.md`，第0–10节|主设计规范；明确标为研究提案、未实施|
|S2|用户提供的`E:/codex/home/attachments/1ca0016d-a8b0-4ece-8655-61992ed02b1e/pasted-text.txt`|逐轮复算解释、公式、加速与实验建议；其复算数字按材料引用|
|S3|[V1全面实现与实验报告](../experiments/phase1_adv3b02_ecrs_v1_manysig_src5_s392005_e200_direct8_20260902_r2_comprehensive_report_20260905.md)|历史事实、配置与局限；冻结快照为2026-09-05 23:40|
|S4|[V1设计追溯表](../CVS_PHASE1_ADV3B02_ECRS_V1_TRACE_20260901.md)及[旧实施计划](2026-09-01-adv3b02-ecrs-v1-implementation.md)|历史实现定位，不追改为V2完成记录|
|S5|关联任务“制定ADV3B02-ECRS实现计划”，任务ID`01a05d28-a153-7c50-b589-5b65b67d24ab`|已读取，定位上述正式交付物；历史任务指令不扩大当前范围|
|S6|`E:/type10-7/项目.md`，2026-08-30版；[最小流程](../../tools/optimizer_workflow_contract.md)|本机当前科学协议优先；本次不更改协议|
|S7|当前Git基线`b5515093f06dc4630d0a3ff58d5d0f631b44e220`下的模型、训练器、loss和schedule|本次静态核对；历史运行提交为`1fb9fe05d9dcaba5cd21e8fed16270d0745e2e72`|

不把历史快照称为服务器实时状态。本次没有重新下载日志、重新评分或读取原始IQ；因此不声称复现了附件全部统计、参考波形已对齐、R7已完成或R8已修复。历史数据包的1500条epoch与160条日记是设计依据，不是新实验。

### 1.2已知问题与改造优先级

|优先级|证据|正确解释|落实动作|
|---|---|---|---|
|P0|`train.py:942`下same-TX、different-TX和gate均嵌入`resp_cls`条件|配置开启不等于实际执行；历史R6不能作为有效same-TX消融|拆分开关、合法集合、计数与梯度验收|
|P0|`model_dual_cvsincnet.py:709`在响应投影前detach；`schedule.py:255`按pair开关解冻encoder|response CE不能直接训练响应编码器；解冻不代表梯度可达|截断移至物理输出，独立响应头|
|P0|`losses.py:4849`的gate使用概率BCE；S3记录R8在E106发生AMP异常|技术失败，没有最终性能结论|logit与幅度分离，AMP安全损失，新版本新run|
|P1|S1/S2：E41/E40训练耗时R3–R8约3.52–3.84倍；E91/E90的R7/R8约2.63/2.72倍|提示split-fit及配对路径，应profile；不是隔离测速|批量求解、减少重复前向、向量化配对|
|P1|S1/S2：记录评估时间66.374小时，其中常规轮58.303小时|逐行时间和，不是日历时间或严格GPU-hours|所有匹配行统一降低全量source验证频率|
|P1|主岭回归与`losses.py:4599`辅助拟合正则实现不同|主估计与辅助目标未必对应同一统计模型|统一尺度、正则、nuisance与回退规则|
|P1|8个实轴锚点、隐式记忆；质量乘入后L2归一化|可能丢失共轭方向，整体减权可能抵消|公共复相位显式历史锚点；质量单列|
|P1|缺逐轮response CE/source-LEO/梯度明细|source-clean平台不能证明响应充分学习|三路径指标、有效更新时钟、逐模块遥测|

S3报告R1–R6已闭合，R7冻结为E194，R8为E106技术失败。R1–R6的clean约78.37%–79.43%，三LEO均值约50.88%–52.87%。这些数字只说明原探索的局限，不能给V2设定“超过R3即获胜”的target调参规则。多个基线loss字段为零，只能说明未记录到相应贡献；不推断所有U路径关闭或存在这些loss之间的梯度冲突。

## 2.版本切分与保持不变的边界

|内容|V1R|V2|
|---|---|---|
|职责|修开关、detach、共享头、AMP、计数和恢复；建立可解释控制|重构参照、低维估计、响应判别与几何融合|
|raw双骨干、PA路径|保持原实现与实际生效目标|继续保留，第一轮不更换主干|
|响应字典与锚点|保留V1以隔离修复效果|统一低维字典、公共复锚点|
|物理梯度|身份目标不更新物理估计器；旧物理目标按明确开关更新其参数|初始固定估计规则，三项新目标只训练判别编码器及相应头|
|课程、采样、全局LR|原课程和原采样作匹配控制|变化分别消融，不混入B3/B4效果|
|gate|修正数值接口并测试，首轮性能比较关闭|固定小幅融合证明有效后才研究动态gate|
|版本身份|`ecrs_v1r`，与V1区分|`ecrs_v2`，配置与bundle独立版本|

V1R不是V1的无条件数值等价版：修正梯度、独立分类头必然改变学习行为。V1R中的纯工程加速另做等价验证；共享I/Q滤波核虽由S1建议为安全替代，但它改变内容模型，放入单独`B2-content`对照，不冒充接线修复。

本轮始终属于Phase1 source-only研发。沿用ManySig、`equalized=1`、source RX=`[1,3,4,6,8]`、source day=`[1,2,3]`与`L_s/U_s/V=6300/56700/27000`；物理ID互斥。U的TX真值不进入训练、采样、权重、阈值或估计。V只读，不更新BN、EMA、prototype或归一化统计，不拆成新的校准/选择角色。

target不参与B0–B7选择与调参。Phase2适配、注册、unknown和多节点协同均不在本轮实现范围；仅保留将来可导出的单IQ、160维接口。协议参考必须是公共协议知识，不能把source样本模板伪装成公共参考打包进入Phase2。

## 3.目标结构与接口

```text
单条received IQ，形状[B,2,T]
 ├─现有ADV3B02身份骨干及PA路径 → z_raw[B,160] → H_raw
 └─旁路参考对齐（仅本样本可估计量）
     → 统一nuisance/response岭估计 → theta及质量q
     → 显式复数参考状态Probe(theta) → r[B,Q,2]
     → stopgrad(r) → E_resp → z_resp[B,64] → H_resp
                                ↓
                       P:64→160 → 切向有界残差
                                ↓
                       z_fused[B,160] → H_fused

训练专用：domain骨干、跨RX匹配、U双视图EMA、cross-fit诊断
部署：单IQ身份路径＋必要响应估计＋编码器＋固定/门控融合
```

拟新增`code/cvsrffi/ecrs_v2.py`承载参照、字典、求解器、锚点和融合；`ecrs_training.py`承载loss装配与统计。保留`model_dual_cvsincnet.py`为集成入口，避免先大规模搬迁旧模型造成额外风险。V1R优先在现有类和函数边界完成小改动，V2通过显式版本路由进入新模块。

输出契约建议包含：`z_id_raw`、`z_resp`、`z_id_fused`、三路径无标签margin推理logits、`resp_coef`、`resp_quality`、`reference_version`、`solver_status`、`rho`。训练模式可附带拟合中间量；推理默认不分配这些大张量。分类接口不能要求传入TX标签才能得到推理logits。

零/退化响应统一返回有限的零响应及`quality_valid=false`，不把零向量当有效单位身份。残差为零时回到raw；退化率和失败原因独立记录，不能默默过滤困难样本后报告更高准确率。

## 4.V1R：首先修复真正的学习路径

### 4.1独立开关与label_mask

`same_tx_cross`、`resp_cls`、`diff_tx`、`gate_calibration`分别控制自己的计算，不依赖另一loss开启。日志分离：`configured`、`executed`、`valid_count`、`raw_loss`、`weighted_loss`。合法集合为空返回有计算图语义的零值及原因；hinge已满足时loss为零合法，不要求每轮大于零。

当前`raw_ce`在`valid.any()`后仍直接使用整批labels，必须改为逐元素合法mask。L/U混合批次、全U、部分未知标签、无合法pair均加入定点检查。所有TX监督只接收L的有效标签；U pair只按同一`physical_sample_id`绑定clean/LEO。

same-TX开关在V1R中保留原有目标含义以检查旧设计是否真正执行；V2再替换为表示级跨RX判别，不能把两个目标写为同一机制。

### 4.2梯度与分类头契约

|目标|E_resp/H_resp|物理估计参数|raw骨干/H_raw|P/H_fused/gate|
|---|---|---|---|---|
|基线实际生效目标|不更新|不更新|依基线|不更新|
|response CE|更新E_resp与H_resp|隔离|隔离|不更新|
|V2跨RX判别|更新E_resp|隔离|隔离|不更新|
|V2 U pair|更新学生E_resp，EMA仅由训练更新|隔离|初始隔离|不更新|
|V1R物理预测目标|仅确实依赖的表示目标更新encoder|仅已开放物理参数|隔离|不更新|
|固定融合监督|初始隔离E_resp，后续开放另做消融|隔离|初始不接收融合梯度，raw仍由基线学习|更新P/H_fused|
|动态gate校准|隔离|隔离|隔离|仅更新gate|

截断点为`r.detach()→E_resp→H_resp`，不是`z_resp.detach()→P→H_raw`。H_resp使用独立普通CE头，不复制CosFace margin。H_fused在融合起点由H_raw拷贝初始化后独立更新，确保响应监督不通过共享头暗改raw。B1/B2阶段融合关闭，主CE只计一次；V1额外0.30 raw CE不得无说明叠加到基线，要作为已识别差异移除或以独立控制保留。

定点验收用非退化L批次验证CE对E_resp梯度非零、物理侧无梯度、H_raw无响应CE梯度，并核对optimizer step后实际参数变化。`requires_grad=true`和loss非零都不足以替代该检查。

### 4.3零作用控制与数值修复

B1分两个局部检查：关闭旁路与B0一致；旁路仅计算、输出不融合且loss权重为零。核对raw输出、梯度、参数更新、优化器动量/weight decay、BN、Dropout、RNG以及AMP跳步。新增模块初始化不能消耗raw初始化随机流；数据采样、增强和旁路分别使用独立随机流。旁路无效梯度不得通过全局AMP skip改变raw更新。

gate输出拆成`help_logit`、`p_help=sigmoid(help_logit)`及`rho=cap*p_help`。训练使用`BCEWithLogits(help_logit,target)`；不能把sigmoid后的rho当logit。PyTorch明确说明概率BCE不适合autocast，而logits版本适用；实施仍以实际安装版本验证。[PyTorch AMP文档](https://docs.pytorch.org/docs/2.14/amp.html)

修复后不得把旧R8 E106继续写成“原run补齐”：同版本同配置的中断恢复才是resume；代码/目标变化必须新run，若沿用旧权重须标为改版续训，不能进入从头训练主对照。

## 5.V2：统一低维响应估计

### 5.1参考对齐先于扩大模型

WiSig官方提供前导前256采样的未均衡与均衡版本，并说明接收机缺乏时间同步。官方数据形式不能证明本地索引、采样率、均衡算法或幅度尺度已经匹配。[WiSig官方说明](https://cores.ee.ucla.edu/downloads/datasets/wisig/)

实施时只读本地合法source样本与加载/预处理代码，记录采样率、截取起点、均衡开关、归一化和可用前导片段。公共参考经相同确定性处理，包级只允许估计有界时移、公共相位、CFO和增益等明确量；参考不能按TX标签选择。先检查对齐残差、稳定性与激励覆盖，再决定进入协议辅助B3。

若本地元信息不足或equalized波形与理论参考失配，执行两条有边界的路线：保留V1R作为可完成基线；V2旁路使用受约束、I/Q共享实滤波核形成估计参考，明确标为`estimated_reference`，不宣称协议辅助已落地。不得扩大自由内容网络掩盖失配，也不从target或source样本集合建立可回放模板。对齐检验的数值阈值需由真实source残差尺度确定，当前未读IQ不能编造。

### 5.2参数化与固定尺度

先写出基项清单及实/复参数类型：增益、相位、频偏采用实Jacobian列；自由复响应系数转换为非冗余实块。例如复字典Phi对应`[[Re(Phi),-Im(Phi)],[Im(Phi),Re(Phi)]]`，实nuisance参数只对应一列实堆叠Jacobian，不多复制一套自由系数。

候选从8个复基项开始，12项仅作后续维数对照。初始候选可用记忆阶0/1上的三次、五次直接非线性共4项，以及当前/延迟共轭线性与三次共4项；先删除与既定nuisance完全重合或数值不可辨识的方向，实际有效维数随审计记录，不能为凑8项保留零列。该具体清单是本计划的可执行起点，未被实验验证。禁止同时保留自由复`s`与`j*s`作为独立复基。

公共参考尺度优先固定；若使用包级幅度尺度，必须定义其物理含义并把系数映射回公共规范。用于cross-fit的幅度尺度只在拟合侧估计。分别报告字典条件数、有效秩、响应/nuisance重叠和零列数，不把可拟合误差降低直接称为可辨识TX特征。

### 5.3一个求解器服务所有路径

对实堆叠观测令`y=x_bar-s_bar`，模型为：

`y=N eta+Phi theta+epsilon`。

最小化`||W^(1/2)(y-N eta-Phi theta)||²+etaᵀLambda_eta eta+thetaᵀLambda_theta theta`。定义：

```text
G_N = Nᵀ W N + Lambda_eta
C = Nᵀ W Phi
A_eff = Phiᵀ W Phi + Lambda_theta - Cᵀ solve(G_N, C)
b_eff = Phiᵀ W y - Cᵀ solve(G_N, Nᵀ W y)
theta = solve(A_eff, b_eff)
eta = solve(G_N, Nᵀ W(y - Phi theta))
```

主估计、A/B cross-fit与跨视图预测调用同一接口，传入拟合观测mask、评估mask、参考版本与正则策略。初始`W=I`、固定正则策略；尺度在公共source规范下固定，取正数，禁止一处按trace放缩而另一处使用无关绝对岭值。建议归一化列空间中从`alpha_eta=alpha_theta=0.01`起步；这是提案，非最优值，不从target挑选。

批量构造Gram/RHS与Cholesky，收集失败子集后一次处理。严格正则下仍失败先标记条件数/非有限输入；限定一次增强正则回退，再失败返回显式状态，不无限重试。CPU高精度QR/SVD仅供小规模数值参照；CUDA的`lstsq`只提供假设满秩的`gels`，不能充当通用秩揭示回退。[PyTorch lstsq文档](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.lstsq.html)

参考、字典、W、正则及对齐状态完全固定时可预分解或预计算`theta=M y`。缓存键用这些配置的版本/形状/设备/精度信息；动态模板、权重或对齐变化必须重建或转批量求解，不错误复用M。预计算后仍是原岭估计，不称为删除求解器。

### 5.4严格cross-fit和可解释预测

先按固定时间位置或公共参考激励划分A/B，再仅用A估计观测相关的对齐、nuisance、权重、尺度与系数，在B上预测；反向亦然。不能先用全包生成这些量再切mask；detach不消除前向回流。涉及滤波/记忆的边界保护宽度至少覆盖实际依赖范围，采用块状划分并报告有效样本损失，不声称相关噪声被变为独立噪声。

统一冻结评估侧目标与权重，主诊断为`Delta_pred=NMSE_nuisance_only-NMSE_nuisance_plus_response`，并报告分母能量、两项NMSE、有效点数。只替换B侧观测不应改变A侧估计，是直接信息路径负测。固定物理估计器下cross-fit仅为诊断，不加入名义上“训练物理前端”的零梯度loss。

对已知合成线性增强G，可以诊断`G(s+N eta+Phi theta)`与增强观测；不能要求仅CFO/相位/增益校正器消除所有LEO变化。噪声项另计，不用噪声种子重构观测来虚增预测能力。真实增强参数只存在训练监督/诊断，部署学生不读取；clean/LEO共享原RX与原噪声，不替代跨RX约束。

## 6.复锚点、判别目标与质量

### 6.1公共锚点

候选为3个幅度×4个相位×2种显式历史，共24个复状态；幅度初值`[0.3,0.6,0.9]`，相位`[0,pi/2,pi,3pi/2]`，历史分别为稳态`(s,s,s)`和跃迁`(s,0,0)`。幅度是公共归一化尺度，覆盖不足的状态标为外推；历史是每个锚点自身定义，不由数组前一个元素决定。

比较未压缩theta、8实锚点、24复锚点与64维学习表示；复锚点输入为48个实数。用`alpha*s+beta*conj(s)`合成例验证实锚点盲区与复锚点可区分性。交换锚点排列并同步排列编码器输入后结果应一致。虚拟锚点不能创造观测中不存在的有效秩。

### 6.2身份方向与可信度分离

`z_resp=normalize(E_resp(stopgrad(r)))`只承载身份方向；`q_resp`独立保存coverage、条件数、有效秩、残差和不确定度。初期不把整体quality乘入encoder后又归一化；也不利用quality删除难样本。

初版统计解释选择高斯线性条件后验：在声明的噪声模型和先验精度下，以`solve(A_eff,I)`得到响应后验协方差，再传播至小维锚点。若W只是优化权重，必须改标为正则敏感度/启发式量，不能称为概率协方差。采样协方差`K Sigma_noise Kᵀ`与后验协方差不能混用。模型失配另记录，source V上的区间覆盖校准不得更新估计器；不构造T×T波形协方差。

`-10log10(NMSE)`改名`fit_residual_score_db`，不再称独立SNR；缺独立估计时SNR为unknown。

### 6.3V2仅三个新增基础目标

```text
L = L_baseline_actual + lambda_r L_resp + lambda_x L_crossRX + lambda_u L_U
L_resp = CE(H_resp(z_resp), y)，仅L有效标签
L_crossRX = mean([m + d(z_i,z_j) - d(z_i,z_k)]_+)
L_U = mean(1 - cos(E_student(r_leo), stopgrad(E_ema(r_clean))))
```

跨RX正例i/j为同TX、不同RX；负例k为不同TX，优先匹配anchor的RX/day/view。使用余弦距离，明确按anchor平均，避免pair多的TX获得隐式更大权重。初始遍历全部合法pair的向量化等价形式；每anchor少量抽对是另一方法消融。各RX不同包只对齐响应表示，不当作同次发射波形逐点重构。

建议起点`lambda_r=0.15`、`lambda_x=0.05`、`lambda_u=0.03`、`m=0.2`、EMA衰减`0.99`。只有前三个为借用旧量级的研究起点，数值和新loss不构成已验证配方。U损失归一化按物理样本数；L与U分别归一化再加权，不因U量大改变全部loss尺度。EMA从学生复制，只在学生成功更新后推进；不在V/target前向更新。

response CE与跨RX负例提供防坍塌约束；记录embedding方差、类别可分性与有效pair覆盖。U进入loader、U执行loss、U确实更新E_resp分别记录。初版不启用复杂null、伪标签门、自由学习基、FastTrust或额外域对抗堆叠。

## 7.采样、课程与有效更新时钟

主矩阵B0–B4沿用旧采样与课程，所有行使用相同验证节奏。B5先在同采样下检验新判别目标，再做独立采样对照。

候选平衡batch为6TX×5RX×4包=120个L物理样本；每RX选择同一天供不同TX匹配，day随batch轮换。另取120个按RX/day平衡的U，生成clean/LEO共480个波形视图。按实际L角色建立格点，不假定原始数据平衡就代表6300个L仍格点充足。缺格点先减少该格点采样并记录；有放回补齐必须标记重复率，不能伪造跨RX正例或读取U的TX真值。

U使用跨epoch连续无放回队列，保存队列位置、排列RNG与循环次数。120个L时每epoch约53个L批次，若每步120个U，则每epoch约6360次U暴露、完整56700池约9个epoch；这是算术示例，尾批和格点回退改变实际值，日志以唯一ID覆盖为准。

内存不足时，物理估计与无BN响应编码器可分块；raw训练切microbatch会改变BN统计，不能宣称自动等价。若更换有效batch，所有匹配控制同时采用并单列比较。每卡最多两个训练实验；此文不分配当前GPU或推断空闲资源。

|日程项目|V1R及主矩阵控制|V2单独对照|
|---|---|---|
|epoch|200|初轮仍200；延长须参数化原1–200硬边界，另立配置|
|LEO视图|E1–40 clear p=.30；E41–90 low/rain p=.60；E91–200三场景p=.80|保持场景与概率，不混入结构对照|
|LEO CE|E80开始，权重0.68|`S-LEO`候选E1–40由0线性升至0.68；显式覆盖默认、独立对照|
|新模块启动|V1R按旧阶段检查接线|V2 E_resp从E1参与身份监督；融合阶段在其已有source互补证据后开始|
|LR|raw保留原日程，响应V1R先保持同全局时钟|`S-LR`每组按成功有效更新次数计时，建议峰值2e-4、5%warmup、末值1e-6|
|optimizer|沿用基线，不反复重建|新增参数组完整注册，未启用时grad=None；保存所有动量与步数|

S-LR与S-LEO分别相对固定同一结构进行，不先叠加两项。U新一致性与提前LEO CE属于明确的方法级训练扩展，应记录对默认`concat_sat_ce_only`的覆盖，不假称严格Core90复现。平衡采样对照为`S-BATCH`。同epoch与相近时间预算分别报告；训练长度不看target曲线决定。

## 8.融合与可部署性

固定融合先采用：

```text
z0 = normalize(z_raw)
u = P(z_resp)
u_perp = u - dot(u,z0)*z0
v = u_perp / max(1, norm(u_perp))
z_fused = normalize(z0 + rho*v)
```

切向残差与z0正交且范数不超过1，所以夹角不超过`atan(rho_max)`。初值P=0、固定rho=0.05（非零），P在零点附近仍有可学习梯度；禁止P与全部有效系数同时锁零。上限沿用0.25只作几何能力界，不是默认最优幅度，固定0.05约对应2.86度界。几何有界不保证分类无害。

B6第一步是响应准备阶段＋融合阶段，整个row总计200轮，raw全程继续基线学习；建议E1–90学习响应、E91–200学习P/H_fused，初始化H_fused来自E90的H_raw。比较双方执行同样两阶段时序，不能把额外110轮训练隐藏在融合收益里。初始融合CE只回传P/H_fused，E_resp与raw不受其反传；原raw CE仍是保护目标，权重不重复。后续开放E_resp或主干层属于单独对照。融合监督权重建议1.0，按成功有效更新时钟运行。

保存逐物理样本raw/fused错误，统计rescue、harm、两者差及`(rescue-harm)/N`，按RX/day/场景分层。H_fused与H_raw独立时，上述为完整部署路径净收益；另用同一个冻结头比较raw/fused特征，隔离分类头贡献。

B7仅在B6于source规则下显示互补后启动。用训练L的无margin候选融合/原始风险变化生成help标签，教师候选与目标detach；训练只更新gate，不把当前rho改变后的自身表现循环当标签。gate只读本样本可得质量或无标签logits，不读true class、day/receiver专属阈值或增强真值。校准在同一V上只读完成，最终规则冻结后才能target确认。

部署bundle含必要身份权重、公共参考生成参数、尺度约定、字典、正则、锚点历史、encoder、P、H_fused及可选gate。训练恢复包另含domain/optimizer等；不混入部署。新增`feature_schema`版本保持160维但不能让旧V1 loader无声接受V2配置。保存/重载检查逐样本输出与batch组成无关；未来Phase2还须检查其输入白名单，160维相同不自动证明Phase2合规。

## 9.工程加速顺序与验收方法

|顺序|动作|是否改变方法|直接验证|
|---|---|---|---|
|E0|增加分段计时，锁定E40/41、E90/91的调用变化|遥测|求解次数、pair数、神经前向数与阶段开关对应|
|E1|Gram/RHS/Cholesky批量化，失败子集集中回退|拟保持等价|同输入输出、梯度、回退码与高精度参照一致|
|E2|固定部分预分解/缓存|条件成立时等价|与动态求解比较；权重/模板变更时强制失效|
|E3|复用同输入同状态clean前向，去除U无用domain路径|需检查状态语义|输出、梯度、BN、RNG、AMP；不合格则单列方法变更|
|E4|配对mask与距离向量化|保持同权重时等价|全部合法pair及reduce结果一致；抽样不归入此项|
|E5|训练与推理路径分离|推理应等价|不调用masked/cycle/split-fit/domain；输出一致|
|E6|神经部分compile可选|条件优化|固定形状收益与数值检查；动态复数回退不强制compile|

`cholesky_ex`支持批量输入，默认不检查错误时可避免其错误检查引起的同步；后续逐包`.item()`仍可能导致同步，因此必须一起改掉。[PyTorch cholesky_ex文档](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.cholesky_ex.html)

profile分训练前向、物理求解、cross-fit、配对、反向、数据加载、source验证及最终评估；短窗口warmup后使用CUDA计时，汇总p50/p95、峰值显存、物理样本/秒，不逐步同步污染正式训练。实际单卡硬件、并发和batch一起记录。

全量source验证设为每5轮、阶段边界两侧及末轮：200轮下为`{5,10,...,200}∪{1,39,40,41,79,80,81,89,90,91,199,200}`，合并为48次。固定平衡小V子集只作健康快检，仍属同一V、不是新数据角色，不用于另一路选模。所有比较行同一候选checkpoint时点，公开“验证节奏改变了选择机会”，不声称与历史每轮选择严格等价。最终评测不削减。

## 10.分阶段实验矩阵

### 10.1主线与可归因拆分

主线B0–B7对应S1第9节。相邻行仅在初始化、数据顺序、增强流、训练暴露和选择规则匹配时才解释差值；不要求共享收敛R0。共同子模块使用按名称保存的同seed初始状态，新增模块用独立RNG；不能仅写相同seed就认为初始化匹配。

|行|父控制|唯一主要问题与改动|融合|结果用途|
|---|---|---|---|---|
|B0|本轮匹配ADV3B02|本轮真实source基线；统一新验证节奏|无|所有收益参照|
|B1|B0|旁路纯计算、零新增loss；完成零作用契约|0|工程正确性，不是性能创新|
|B2|匹配V1实现控制|V1R接线与独立响应头；旧字典/锚点/课程|0|原设计是否得到身份梯度|
|B3|B2及下述桥接控制|公共/估计参考＋统一低维估计|0|估计规范性、代价及可分性|
|B4|B3|实锚点换公共复相位显式历史|0|读取方式是否丢身份信息|
|B5|B4|跨RX表示判别＋U编码器学习|0|可迁移结构和互补潜力|
|B6|B5|固定rho切向融合与独立融合头|0.05|相对raw净收益|
|B7|B6|source收益校准动态gate|≤0.25|是否优于固定融合，条件性研究|

B2是修复包，若要区分开关、梯度、独立头各自效果，局部拆为`B2-switch`、`B2-grad`、`B2-head`，同一旧物理配置，不能只以B2−B1声称某一修复贡献。历史R1–R8不是本轮匹配V1控制；无需为计划预先重跑全部旧矩阵。

需要正式量化修复包差值时补充`B2-V1`：采用V1的R7非融合配置、旧接线及旧loss权重，匹配本轮初始化/数据/验证节奏，明确关闭gate校准以避开已知R8错误；其差值只能解释为整套修复效果。B2的每个修复子行相对明确父行展开配置差异。B1通常只需定点与短程等价验证，不机械增加一条无科学必要的200轮纯计算实验。

B3同时牵涉参考、统一求解与冻结物理前端，必须保留桥接：`B3a`旧参考/旧维数仅统一估计器；`B3b`更换参照并固定估计规则；`B3c`删除冗余并降维。每个父控制使用相同有效loss集合，物理冻结后诊断项均不计入训练loss。若只执行整包B3，则只能称“估计器包效果”，不能声称单项因果收益。

B5拆为`B5-X`（B4＋跨RX）、`B5-U`（B4＋U）、`B5-XU`（两者），先保持旧采样；`S-BATCH`再相对B5-XU改采样。`S-LR`、`S-LEO`各自单独比较，证据返回后才选择是否组合。这些是按问题需要展开的局部对照，不是要求一次性运行所有组合。

### 10.2执行批次与预算

第一批只做必要功能/数值检查、B0/B1及B2关键对照，seed=392005、200轮为主参考；不八卡直接铺满所有新结构。第二批按source结果执行B3桥接、B4和B5的必要拆分。第三批B6；B7仅条件性进入。单seed为可证伪探索；对通过source门槛的候选与B0建议用`392005/392006/392007`三seed确认，新增seed清单为提案。

不在未profile时承诺训练小时数。预算表应分别填写每row训练小时、source验证小时、最终评估小时、峰值显存、部署延迟；总任务时间为各row之和，墙钟下界取最大row时长和总卡时/可用卡数的较大者，实际还受共享负载影响。S2的343.02小时是历史记录和，不能作为V2确定预算；每5轮验证的节省比例仅是历史算术估算。

完成首批profile后可计算计划预算，但不能以预算未精确预测阻断已授权的正确性检查。低性能不停止健康训练；科学“不继续扩大该方向”在约定row完成后判定，不等于中途杀进程。

## 11.指标、source选择与科学判定

### 11.1必须分开报告的三类证据

|证据|记录内容|不能替代|
|---|---|---|
|执行正确性|每项loss执行数、有效数、grad_norm、实际step、LR、requires_grad、非有限跳步|不能替代识别收益|
|响应机制|独立预测增益、TX/RX/view probe、秩/覆盖/协方差区间、表示方差|低NMSE不能替代TX结构|
|任务与资源|同row raw/response/fused的clean与三LEO、receiver/day分层、rescue/harm、时间显存延迟|均值不能替代floor，source不能替代target|

source V中同一物理样本的clean和LEO派生视图保持绑定，probe训练只用L或另行声明的合法source训练表示；V用于只读probe检验。不能把同一物理样本不同视图分别放probe训练与测试。跨RX泛化诊断采用source内预先定义的留RX方式训练probe，不改主模型L/U/V角色；它仍是source proxy，不等于target表现。

source正式选择规则建议：候选先满足clean与floor保留，再按三LEO等权均值最大选择；并列按更低部署延迟、较早epoch。选择头、场景权重和checkpoint时点预先固定，同一V统一使用。不得训练后改成哪个头更高就选哪个头。

### 11.2建议的初轮科学门槛

以下是本计划提出的source-only可证伪门槛，不来自已有已冻结用户数值目标。实施预登记沿用这些值，或在接触任何target结果前说明修改；不能事后追着结果改变。

|对象|建议判定|
|---|---|
|B1工程等价|确定性FP32小批次raw输出/梯度与一步更新误差`atol=1e-6, rtol=1e-5`；BN计数/RNG/跳步一致。AMP容差另据实际精度声明|
|批量求解与固定M|良态合成输入对高精度参照`rtol=1e-4, atol=1e-5`；病态输入看预测残差和失败状态，不能用系数相对误差掩盖不可辨识|
|B3/B4机制|独立预测增益为正，并同时报告TX probe与RX/view泄漏；不凭NMSE单项进入融合|
|B6单seed筛选|相对同row raw：source-clean下降≤0.5pp，clean最弱RX下降≤1pp；三LEO均值提升≥1pp，最弱RX×LEO单元下降≤1pp，rescue−harm>0|
|资源约束|部署batch=1 p95延迟建议≤B0的1.25倍；训练成本如增加必须明示，不以隐藏诊断耗时满足条件|
|多seed确认|报告三seed均值/标准差、各seed差值；至少2/3满足收益方向、无seed突破上述clean/floor退化界；不是统计显著性自动保证|
|B7必要性|同source条件下相对固定B6有净收益，floor不更差且成本可接受；否则保留固定融合|

上述B6的raw/fused同row条件用于检验响应互补性；还必须对独立匹配B0检查同样的clean/floor退化界及LEO收益，防止把“融合救回了一个已退化的raw分支”误当总体进步。B0→候选和候选raw→fused两组差值同时列出。

不满足时保留负结果并标`NO_PROMOTION_TO_DEFAULT`。若统一估计与足够判别学习后仍无TX结构，或响应有TX信息却持续无法补充raw，停止扩大该响应表示；不以增加字典、门控、伪标签系统解释所有失败。

target确认只发生在source选定并冻结模型、配置、阈值后。闭集Phase1仍需保留clean与三LEO逐场景最终检验；source筛选落选row记`SOURCE_SCREEN_COMPLETE`，不能称完整target实验闭合。若开展registered/unknown确认则按当前项目协议仅对预登记单一候选实行role/truth-blind预测、预测先固定、独立scorer后接truth，且单物理样本单LEO；不把历史闭集四场景诊断偷换成该确认协议。target结果只确认既定声明/准入，不用于B行重排、重训或选择性重跑。

## 12.工程工作包与直接验收

下表文件均相对Git仓库根；“新增”表示拟建文件，当前未实现。

|工作包|文件/入口|交付内容|直接验收与依赖|
|---|---|---|---|
|W1执行修复|`code/train.py:825`、`code/cvsrffi/losses.py:4849`|独立开关、mask、logit gate、loss日志|原R6阻断条件构造；空pair/全U；CUDA AMP合法正负例|
|W2梯度/头|`code/model_dual_cvsincnet.py:525/689`、`code/cvsrffi/schedule.py:233`|r侧detach、独立头、显式参数组|响应梯度可达；raw/物理隔离；B1零作用；依赖W1|
|W3统一估计|新增`code/cvsrffi/ecrs_v2.py`；替换旧`WeightedRidgeLayer`调用边界|参考/字典/Schur求解/缓存/cross-fit|联合块解与Schur一致；只改评估侧不改拟合参数；缓存失效；依赖source对齐检查|
|W4锚点质量|同W3模块，旧`SurfaceAnchorEncoder`作为对照|复状态、64维encoder、独立质量与协方差|共轭可分合成例、显式历史、零响应、区间覆盖；依赖W3|
|W5训练判别|新增`code/cvsrffi/ecrs_training.py`，`code/train.py`集成|三目标、pair mask、EMA与分组日志|U无truth；CE/跨RX/U各自梯度；无pair与坍塌诊断；依赖W2/W4|
|W6采样日程|`balanced_tx_rx_sampler.py`、`schedule.py`、训练入口|L格点与U队列、有效step LR、阶段参数化|缺格点、尾批、恢复后队列与step一致；作为独立对照|
|W7融合部署|V2模块、`identity_only_forward.py`、`checkpoint.py`|切向固定融合、版本化单IQ导出|角度界、P零初始化可更新、重载一致、无训练分支；依赖W5|
|W8效率遥测|训练入口、`logging.py`、分析工具|批量/前向复用、source验证48次、资源拆账|同语义输出/梯度/BN/RNG；只改变有证据等价部分|
|W9实验交付|拟新增`code/scripts/launch_phase1_adv3b02_ecrs_v1r_v2.sh`与对应report|版本参数→模型→schedule→loss→日志完整接线|dry-run展开每row配置；正式执行时遵守最小流程|

复用现有`code/tests/test_ecrs_gradient_routing.py`、`test_ecrs_schedule.py`、`test_ecrs_losses.py`、`test_ecrs_checkpoint.py`、`test_ecrs_pair_metadata.py`等测试入口；新增V2数值/参考/锚点测试。删除或更新“断开encoder身份梯度才正确”等过时断言，不让旧测试迫使错误行为继续存在。仅跑改动相关测试，文档阶段不跑GPU训练测试。

拟新增CLI命名：`ecrs_version`、`ecrs_reference_mode`、`ecrs_solver_mode`、`ecrs_anchor_mode`、`ecrs_resp_ce_enabled`、`ecrs_cross_rx_enabled`、`ecrs_u_pair_enabled`、`ecrs_fusion_mode`、`ecrs_fixed_rho`、`ecrs_update_resp_from_fusion`、`ecrs_lr_clock`。这些当前不是可运行参数；实现时由一处配置解析，禁止launcher与schedule两处默认值漂移。旧V1参数保留兼容，V2不套旧R1–R8编号掩盖语义变化。

训练恢复需保存model、optimizer、scaler、每参数组scheduler/有效step、EMA、Python/NumPy/Torch/CUDA RNG、sampler/U队列、epoch及batch位置；验证中断前后下一步输入与更新一致。不可只恢复epoch和权重就宣称等价resume。

## 13.设计追溯总表

状态描述的是**实现状态**，不是计划是否写完：`pending`待实施，`deferred`条件性后置；本次没有新功能`implemented/verified`。目标文件可通过W编号定位到第12节。

|ID|源章节|规范要求|目标文件/工作包|状态|验证|处理说明|
|---|---|---|---|---|---|---|
|T01|S1§0|保留双骨干/PA/source RX/单IQ|W2/W7|pending|B0/B1与接口检查|不替换主干|
|T02|S1§1|阶段耗时与有效日志|W8|pending|E40/41、E90/91调用计时|历史比值只作定位|
|T03|S1§2.1|四个loss独立开关|W1|pending|单开关组合与空集合|不要求loss恒正|
|T04|S1§2.2|r侧detach与独立响应头|W2|pending|梯度与参数更新|raw/物理隔离|
|T05|S1§2.2|真正零作用控制|W1/W2|pending|输出/状态/RNG/跳步|不只rho=0|
|T06|S1§2.3|logit gate与AMP修复|W1|pending|CUDA autocast有效样本|性能gate暂不开|
|T07|S1§3.1|协议参考与本地预处理匹配|W3|pending|source对齐与残差|未读IQ，不声称已匹配|
|T08|S1§3.1|共享I/Q核及cycle定位|W2/W3|pending|内容对照与算子往返|共享核单独B2-content|
|T09|S1§3.2|实/复非冗余参数化与尺度回映射|W3|pending|合成系数、秩与尺度|8–12为候选|
|T10|S1§3.3|统一Schur求解及正则|W3|pending|联合解/主辅路径一致|不用显式inverse|
|T11|S1§3.3|固定M及严格失效条件|W3/W8|pending|动态等价与失效检查|动态对齐不复用旧M|
|T12|S1§3.4|先划分后估计及边界保护|W3|pending|评估侧替换负测|冻结时仅诊断|
|T13|S1§4.1|复锚点、显式历史与probe|W4|pending|共轭盲区/历史/分层probe|外推单列|
|T14|S1§4.2|协方差统计解释及覆盖校准|W4|pending|高精度/小规模覆盖|不把1/diag当diag逆|
|T15|S1§4.2|质量分离及残差评分改名|W4/W8|pending|标量coverage与日志|SNR缺失为unknown|
|T16|S1§5|仅三项新增基础目标|W5|pending|独立梯度与归一化|旧物理诊断不冒充loss|
|T17|S1§5|label_mask与U真值隔离|W1/W5/W6|pending|混合/全U负测|采样U不看TX|
|T18|S1§5|G前向诊断与单视图边界|W3/W5/W7|pending|删除增强元数据仍可推理|合成参数仅训练诊断|
|T19|S1§6|120L/120U候选与缺格点|W6|pending|格点/pair/尾批计数|独立S-BATCH|
|T20|S1§6|U队列与唯一覆盖|W6|pending|跨epoch/恢复读回|不固定重复小子集|
|T21|S1§6|有效step LR与完整optimizer|W6|pending|零有效step/恢复|独立S-LR|
|T22|S1§6|提前LEO CE与长度参数化|W6|pending|阶段边界与默认覆盖|独立S-LEO，不立即组合|
|T23|S1§7|切向固定融合与可学零初始化|W7|pending|角度界/梯度/rescue-harm|rho=.05提案|
|T24|S1§2.3/7|收益校准动态gate|W7后续|deferred|B6有效后与固定融合比较|尚无互补证据|
|T25|S1§8|批量求解/前向复用/配对向量化|W8|pending|输出/梯度/BN/RNG|抽对不称等价|
|T26|S1§8|验证频率与三路径输出|W8|pending|48次时点与只读V|target范围不削减|
|T27|S1§8|推理裁剪与数值精度|W7/W8|pending|单IQ不执行训练分支|线代FP32/complex64|
|T28|S1§8|可选compile|W8后续|deferred|固定形状收益验证|动态回退先不编译|
|T29|S1§9|B0–B7与同配置拆分|W9|pending|配置diff与公共RNG|不强制共享收敛R0|
|T30|S1§9|三路径预测/净收益/分层/资源|W8/W9|pending|逐物理ID输出与汇总|不重算旧缺失预测|
|T31|S1§9|完整恢复状态与改版新run|W6/W7/W9|pending|下一步恢复一致|不覆盖旧R8|
|T32|S1§9；S2§10|无TX互补则保留负结果；不堆复杂模块|W9|pending|source预登记判定|自由基/null/FastTrust不纳入首版|

合计32项：pending=30，deferred=2，implemented=0，verified=0，rejected=0，blocked=0。参考对齐为最高风险待查项，尚未实施检查，不提前标为失败或阻塞。设计忠实度为“保持S1主线、补齐可执行细节”；表中显式候选参数、桥接对照和数值门槛是本计划细化，不冒充原文既定结论。

## 14.后续执行顺序与交付完成定义

依次完成W1/W2→B0/B1与B2接线控制→W3参考/统一求解→W4锚点→W5判别→W7固定融合；W8的等价优化按对应模块成熟度穿插；W6的采样/时钟/课程独立验证；最后才考虑T24/T28。每次只扩大解决当前问题所需的对照。

未来获得实施/实验任务后，在已核实Git基线创建独立V1R/V2工作分支，保留当前V1研究分支与历史产物。正式发布按八项最小流程：本地代码提交及相关检查、候选一次P0/P1直接正确性审查、最小预登记、唯一run与launch owner、资源preflight、一次归档传输校验、远端编译与启动后读回，最终预测完整后独立评分。文档本身不触发独立实验审查、checkpoint smoke或GPU运行。

本计划完成要求为：来源边界清楚、全部规范条目有去向、文件与阶段依赖明确、实验可归因、数值/梯度/协议验收具体、未决技术问题有处理路径，并完成UTF-8、引用路径及Git交付检查。代码完成、训练完成、artifact闭合和科学晋级分别由后续真实证据证明，不能由本计划的“完成”推导。
