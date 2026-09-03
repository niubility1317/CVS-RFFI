# ADV3B02-DAOT-STN-RX-V2 Implementation Plan

> **执行要求：**实施阶段按本计划逐任务推进，每个任务先写失败测试，再做最小实现；完成本地实现与验证后暂停，不启动性能实验，直到用户另行明确授权。

**Goal：**在保留ADV3B02成熟骨干、ManySig单V协议和现有Phase1边界的前提下，把报告提出的“锚定轨道教师、预算化切向约束、方向性Jacobian路由、显式接收机轨道、TX条件接收机对齐、连续U可信度、尾部风险优化和高效Temporal Orbit Memory”落到可测试、可配置、可消融的`ADV3B02-DAOT-STN-RX-V2`实现中，重点改善未知接收机下的receiver×scenario最差组，而不是只抬高LEO均值。

**Architecture：**沿用现有`DualCVSincNetDisentangle`、EMA教师和单V训练主链。物理视图生成、轨道目标、切向/路由、接收机风格、U可信度、尾部风险与调度分别保持模块化；`SSDG/train_ssdg.py`只负责编排。部署默认采用“两个新鲜视图+Temporal Orbit Memory”，三新鲜教师视图仅保留为A2/A3上界或明确消融，不作为默认实现。选择性切空间投影是第二阶段高风险开关，首版默认权重为0。

**Tech Stack：**Python、PyTorch、NumPy、pytest、现有CV-SincNet/SSDG训练入口、ManySig数据协议、Git工作树。

**Spec：**用户提供的《ADV3B02-DAOT-STN下一阶段全面优化方案》、`项目.md`、现有`docs/CVS_PHASE1_ADV3B02_DAOT_STN_V1_TRACE_20260901.md`以及A1～A7结果报告。

**Global Constraints：**

- 只做Phase1 source-only弱标签/半监督域泛化；不得读取target训练样本、target标签或query反馈。
- 数据协议固定为`tx_rx_day_1_7_2`、seed=`392005`、source receiver=`[1,3,4,6,8]`、source day=`[1,2,3]`、`L/U/V=0.07/0.63/0.30`且V只读。
- 最终目标测试只保留`clean`、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。
- 明确不实施、不发布、不作为门槛：使用上一轮A4/A7等checkpoint测试任何非LEO_WEAK场景；报告中的D0与C3相应路径标记为`REJECTED_BY_USER_SCOPE`。
- 不把ManySig的LEO弱信道代理结果表述为真实轨道部署结论。
- 三教师性能视图只用于A2/A3上界及其明确消融；V2部署默认必须是两次新鲜前向加Temporal Memory。
- 本轮先完成代码、配置、聚焦测试、真实checkpoint无query smoke和一次P0/P1审查；不启动训练或完整评测实验。

---

## 1.交付边界与完成定义

本轮“落地完成”必须同时满足：

1. 报告中的低风险主链全部有真实代码入口、CLI/config开关、日志字段和单元测试。
2. 默认路线为`ADV3B02-DAOT-STN-RX-V2`，但不改变未开启V2时的旧训练行为。
3. Temporal Memory、接收机风格、切向方向注册表、连续U可信度均可保存/恢复checkpoint状态。
4. 高风险选择性切空间投影可独立启用，但默认`lambda_subspace=0`。
5. 本地聚焦测试与真实checkpoint无query smoke通过；只做语法/功能验证，不产生性能结论。
6. 形成后续实验的最小预登记草案，但本轮不发布到N607、不启动实验。

不属于本轮交付：

- 非LEO_WEAK场景上的旧checkpoint诊断、cross-family泛化或相应CVaR门槛。
- Phase2域适应、新类注册、support/query处理或四状态DA/REG实验。
- 真实卫星链路、轨道高度或真实接收机硬件的外推结论。
- 为提高报告完整度而增加额外seal、hash、receipt或审核门。

---

## 2.报告要求到代码的追踪矩阵

|ID|报告机制|现有基础|V2落点|默认状态|验收证据|
|---|---|---|---|---|---|
|V2-OT-01|锚定球面轨道中心|`orbit_teacher.py`鲁棒球面均值|加入clean锚点、离散度自适应锚定系数和非对称student路由|开|单位球、clean退化、权重下界测试|
|V2-OT-02|coverage-aware教师分布|`deployment_orbit.py`固定重要度|按可恢复度、部署先验和coverage floor生成样本级权重|开|权重归一化、floor、极端样本测试|
|V2-TG-01|统一切向方向注册表|4训练方向+8诊断方向分散定义|新增`TangentDirectionRegistry`，统一方向类型、单位、步长、预算和可用分支|开|注册完整性和非法方向负测|
|V2-TG-02|弦长切向能量|`acos/delta`角灵敏度|改为`2(1-cos)/delta_j^2`并保留兼容模式|开|近零有限性、尺度关系、反传测试|
|V2-TG-03|逐方向步长/预算|统一`delta=0.05`、近零预算|离线有限差分校准接口；纯nuisance近零、混合方向非零、TX方向不进入nulling|开|线性误差、预算分型测试|
|V2-RT-01|随机单方向TX干预|固定PA/IQ/clock组合|每样本只采一个TX方向并记录方向ID与符号|开|可重放、单活跃方向测试|
|V2-RT-02|Directional Jacobian Routing|仅fingerprint下界|比较`z_id/z_dom`对nuisance与TX方向的相对响应，并约束`z_dom`尺度|开|四距离、margin、尺度作弊负测|
|V2-RX-01|Source-only Receiver Style Bank|无显式RX轨道|从source log-spectrum、IQ协方差、相位增量、幅度统计拟合低秩bank|开|仅source输入、PCA重建、状态恢复测试|
|V2-RX-02|虚拟接收机视图|普通信道增强|在source风格凸包及`lambda∈[0.8,1.2]`小幅外推中采样RX变换|开|范围、确定性、有限输出测试|
|V2-RX-03|TX条件接收机对齐|全局prototype能力|按TX×RX×excitation-bin维护原型，只对同TX同激励桶做RX对齐|开|禁止跨TX对齐、稀疏桶回退测试|
|V2-U-01|连续可信度与三态U|硬argmax全视图共识|融合可恢复度、JS不一致、时间不一致、prototype margin，输出Core/Ambiguous/Irrecoverable|开|边界单调性、无label依赖测试|
|V2-U-02|class/RX/severity配额|全局阈值|加入每类阈值和receiver/severity覆盖上限，避免easy-view垄断|开|配额和空组测试|
|V2-TAIL-01|receiver×day×severity CVaR|若干通用CVaR函数|新增EMA组损失、最小组样本数、权重上限和`alpha:0.4→0.2`调度|开|尾组选择、EMA、权重上限测试|
|V2-SC-01|独立辅助损失调度|orbit/tangent两尺度|实现S0～S6独立权重，不再用orbit scale控制nuisance/fingerprint|开|各epoch边界与独立性测试|
|V2-GR-01|主任务保护的冲突投影|已有梯度日志/其他路线手术|记录base/orbit/tangent/route/RX/CVaR余弦；仅持续冲突时投影identity backbone辅助梯度|开，延迟触发|无冲突不改梯度、持续冲突才投影测试|
|V2-MEM-01|向量化Temporal Orbit Memory|字典式memory|Tensor Bank保存特征、时间、可靠度、场景桶、接收机桶；动态momentum与TTL|开|批量更新、TTL、checkpoint测试|
|V2-EFF-01|两新鲜视图部署默认|三教师默认|默认`clean+rotating fresh+memory`；三新鲜视图仅上界模式|开|实际前向计数测试|
|V2-SUB-01|选择性nuisance子空间|无|广义特征分解、soft projection、每5～10 epoch更新、stop-gradient|默认关|子空间维度、正交性、关闭零影响测试|
|V2-XF-00|旧checkpoint非LEO_WEAK测试|报告D0/C3建议|不实现、不配置、不验收|排除|计划、launcher和验收规则中均无此场景|

---

## 3.实施任务

### Task 1：冻结V2配置契约与排除项

**Files：**

- Modify：`code/cvsrffi/deployment_orbit.py`
- Modify：`code/SSDG/train_ssdg.py`
- Create：`configs/phase1_adv3b02_daot_stn_rx_v2_s392005.json`
- Modify：`tests/test_adv3b02_daot_stn.py`

**步骤：**

1. 先写失败测试，断言V2是显式opt-in，默认教师模式为`temporal_memory_rx`，fresh view count为2，`lambda_subspace=0`。
2. 写负测，禁止V2最终场景列表出现`clear_leo`、`low_elev_leo`、`rain_leo`、`storm_mp`、`mixed_orbit`、`geo_clear`或其他非`*_weak`场景。
3. 新增V2配置对象和CLI字段，保留V1/A1～A8旧行为不变。
4. 配置文件固化ManySig单V数据、seed392005、source/target receiver与只允许的四个最终场景。
5. 运行聚焦测试，确认旧V1默认值和V2新默认值互不污染。

### Task 2：实现锚定、coverage-aware的轨道教师

**Files：**

- Modify：`code/cvsrffi/orbit_teacher.py`
- Modify：`code/cvsrffi/deployment_orbit.py`
- Modify：`code/cvsrffi/daot_training.py`
- Modify：`tests/test_adv3b02_daot_stn.py`

**步骤：**

1. 为样本级覆盖混合权重写测试：`alpha=(1-gamma)*normalize(recoverability*deployment_weight)+gamma*pi`。
2. 为clean锚定球面目标写测试：所有视图一致时退化到共同方向；困难视图失真时目标不应远离clean锚点；输出始终单位化。
3. 实现扩展可恢复度向量接口，接收SNR、elevation、K-factor、deep-fade、clip、occupancy、canonical/phase/spectral error；缺失字段使用显式中性值并记录mask。
4. 实现离散度相关锚定系数，不在训练主链内偷偷拟合target统计。
5. 将student loss拆为channel student→orbit target和clean student→clean teacher，clean权重默认在`0.15～0.30`配置范围内。
6. 日志新增覆盖熵、clean权重、有效视图数、轨道离散度与缺失元数据比例。

### Task 3：统一TangentDirectionRegistry并改为预算化弦长能量

**Files：**

- Modify：`code/cvsrffi/deployment_orbit.py`
- Modify：`code/cvsrffi/selective_tangent.py`
- Modify：`code/cvsrffi/daot_training.py`
- Create：`code/cvsrffi/tangent_calibration.py`
- Modify：`tests/test_adv3b02_daot_stn.py`
- Create：`tests/test_daot_tangent_calibration.py`

**步骤：**

1. 先写注册表测试，覆盖pure Doppler、Doppler rate、fractional STO、RX-only SFO、RX filter basis、multipath basis、SNR/noise、RX phase noise、AGC residual、total CFO、total IQ imbalance、clipping、quantization。
2. 为每个方向声明`kind={pure_nuisance,mixed,tx_fingerprint,secant_only}`、物理单位、默认`delta_j`、预算来源和允许的分支。
3. 禁止clipping/quantization进入局部tangent；它们只走secant/orbit视图。
4. 用弦长能量替换V2的`acos`灵敏度，同时保留V1兼容函数，避免改写历史实验语义。
5. 实现有限差分线性误差`e_lin`校准工具，选择满足`e_lin<0.1～0.2`的最大稳定步长。
6. 实现按方向预算：pure nuisance近零；mixed取source自然同TX变化分位数；TX fingerprint从nulling集合排除。
7. 实现单方向分层采样，概率与部署方差、EMA灵敏度相关，并保证方向覆盖。

### Task 4：实现随机TX干预与Directional Jacobian Routing

**Files：**

- Modify：`code/cvsrffi/deployment_orbit.py`
- Modify：`code/cvsrffi/selective_tangent.py`
- Modify：`code/cvsrffi/daot_training.py`
- Modify：`code/SSDG/train_ssdg.py`
- Modify：`tests/test_adv3b02_daot_stn.py`

**步骤：**

1. 为PA、IQ gain、IQ phase、TX CFO、clock skew、DAC nonlinearity写“每样本仅一个方向”的失败测试。
2. 将固定组合指纹干预替换为随机单方向、随机正负号、可重放强度采样。
3. 计算四个距离：`Δ_nui_id`、`Δ_nui_dom`、`Δ_fp_id`、`Δ_fp_dom`。
4. 实现两项margin routing loss，使nuisance变化优先进入`z_dom`，TX变化优先进入`z_id`。
5. 对`z_dom`使用归一化角距离或显式方差/上界约束，阻止通过无限放大尺度作弊。
6. 输出每方向路由margin、违规率和TX/nuisance响应比；保持旧`fingerprint_keep`可作为独立消融。

### Task 5：实现Source-only Receiver Style Bank和虚拟RX轨道

**Files：**

- Create：`code/cvsrffi/receiver_style_bank.py`
- Modify：`code/cvsrffi/deployment_orbit.py`
- Modify：`code/SSDG/train_ssdg.py`
- Create：`tests/test_receiver_style_bank.py`

**步骤：**

1. 写协议负测：构建bank只接受source split及其receiver/day元数据；任何target/test角色立即拒绝。
2. 实现source统计抽取：log-spectrum、IQ covariance、phase-increment、amplitude-transfer；所有统计支持流式累计，避免一次装入全数据。
3. 对source receiver统计做低秩PCA，保存均值、基、source系数、解释方差和版本字段。
4. 在source系数凸包中采样，并允许围绕均值做`lambda∈[0.8,1.2]`的小幅扩展；超界、非有限或不可逆变换回退到恒等视图并计数。
5. 把虚拟RX变换接入教师视图轮换：channel、receiver、channel+receiver按step/epoch轮换，不在每一步全部前向。
6. 将bank状态写入checkpoint并验证恢复后采样可重放。

### Task 6：实现TX条件接收机原型与组尾风险优化

**Files：**

- Create：`code/cvsrffi/receiver_conditioned_alignment.py`
- Modify：`code/cvsrffi/daot_training.py`
- Modify：`code/SSDG/train_ssdg.py`
- Create：`tests/test_receiver_conditioned_alignment.py`

**步骤：**

1. 实现激励描述量：PAPR、幅度分位数、频谱占用、slew、circularity；首版使用固定物理分桶，默认3～6桶。
2. 维护TX×RX×excitation-bin原型；只允许同TX、同激励桶跨RX对齐，禁止全局域对齐。
3. 稀疏桶低于最小样本数时回退到同TX邻近激励桶，仍不得跨TX。
4. 实现receiver×day×severity组损失EMA和CVaR；`alpha`从0.4独立退火到0.2，并设置最小组样本数和最大组权重。
5. 把`L_RX`与`L_tail`作为独立loss项接入总目标，默认权重分别在`0.05～0.10`和`0.05～0.15`范围。
6. 日志输出组覆盖率、最差组、worst-two、receiver标准差代理和CVaR权重分布。

### Task 7：把U路径改为连续可信度和三态选择

**Files：**

- Create：`code/cvsrffi/daot_unlabeled_trust.py`
- Modify：`code/cvsrffi/daot_training.py`
- Modify：`code/SSDG/train_ssdg.py`
- Create：`tests/test_daot_unlabeled_trust.py`
- Modify：`tests/test_adv3b02_daot_stn.py`

**步骤：**

1. 写无标签边界测试：函数签名不得接收ground-truth label；改变隐藏label不得改变选择结果。
2. 实现连续可信度，组合recoverability、view-JS、temporal inconsistency和prototype margin，并输出可解释分项。
3. 映射为Core/Ambiguous/Irrecoverable三态：Core可走soft logit/prototype/pseudo-CE；Ambiguous只走feature/orbit一致性；Irrecoverable默认不做身份伪标签学习。
4. 实现每类动态阈值和receiver/severity配额，防止easy receiver或clean-like样本垄断U梯度。
5. 保留U0～U5消融开关，但不要求一次性跑完作为实现门槛。
6. 日志输出三态比例、每类覆盖、每RX/severity覆盖、JS、时间不一致和prototype margin分布。

### Task 8：实现分支专属不变性预算

**Files：**

- Create：`code/cvsrffi/branch_invariance.py`
- Modify：`code/cvsrffi/daot_training.py`
- Modify：`code/model_dual_cvsincnet.py`
- Create：`tests/test_branch_invariance.py`

**步骤：**

1. 基于现有`id_feat_imp/id_feat_dac/id_feat_pa/id_feat_joint`输出建立branch×direction策略表。
2. time分支抑制STO、pure Doppler、RX delay但保留瞬态/时钟/非平稳信息；frequency分支抑制RX filter/平滑multipath但保留TX谱不对称。
3. DAC/IQ分支只抑制可分离RX IQ残差；PA分支只抑制全局AGC/path loss，不抹除激励与PA记忆效应。
4. joint分支约束共同信道方向，并加入“无单方向支配”的软约束。
5. 对缺失/退化分支输出做显式回退；不得静默把所有预算施加到`z_id`。

### Task 9：实现S0～S6独立调度与主任务保护

**Files：**

- Modify：`code/cvsrffi/orbit_teacher.py`
- Create：`code/cvsrffi/daot_gradient_control.py`
- Modify：`code/SSDG/train_ssdg.py`
- Create：`tests/test_daot_gradient_control.py`
- Modify：`tests/test_adv3b02_daot_stn.py`

**步骤：**

1. 写epoch边界测试：S0基础/EMA；S1轨道feature；S2soft logit/prototype/RX原型；S3预算tangent；S4route/TX干预；S5CVaR/SWAD；S6收敛期降低强增强。
2. 每个loss有独立schedule scale，修复nuisance/fingerprint受orbit scale连带控制的问题。
3. 记录base、orbit、tangent、route、RX、CVaR在identity backbone上的梯度余弦和范数比。
4. 只在同一辅助项连续多个窗口与base冲突且超过阈值时，投影该辅助项在identity backbone上的冲突分量；domain/nuisance head不做全量PCGrad。
5. checkpoint保存冲突EMA与触发计数；恢复训练不重置保护状态。

### Task 10：向量化Temporal Orbit Memory与前向效率

**Files：**

- Modify：`code/cvsrffi/orbit_teacher.py`
- Modify：`code/cvsrffi/identity_only_forward.py`
- Modify：`code/SSDG/train_ssdg.py`
- Modify：`tests/test_adv3b02_daot_stn.py`
- Create：`tests/test_daot_efficiency_contract.py`

**步骤：**

1. 将字典式memory升级为Tensor Bank，保存normalized feature、last_seen、reliability、scenario_bin、receiver_bin和valid mask。
2. 实现可靠度相关动态momentum、TTL衰减和低可信条目重置。
3. 默认教师集合固定为`{clean,fresh,memory}`；若memory miss，则只用两个fresh来源，不临时增加第三次教师前向。
4. 教师、tangent和fingerprint视图优先走identity-only forward；不支持时安全回退完整forward并记录比例。
5. 同一步需要的多视图拼接为一次批量前向，再按slice拆分，测试前向调用数和数值等价性。
6. tangent样本比例在10%～25%内按recoverability×sensitivity×group-loss自适应分配。

### Task 11：实现选择性nuisance子空间投影，默认关闭

**Files：**

- Create：`code/cvsrffi/selective_nuisance_subspace.py`
- Modify：`code/cvsrffi/daot_training.py`
- Modify：`code/SSDG/train_ssdg.py`
- Create：`tests/test_selective_nuisance_subspace.py`

**步骤：**

1. 先写关闭测试：`lambda_subspace=0`时输出与未接入V2前严格一致，不更新子空间状态。
2. 从nuisance delta与fingerprint delta分别估计协方差，使用正则化广义特征分解选择高nuisance/低fingerprint方向。
3. 实现soft projection，默认候选维度8～16；basis每5～10 epoch更新一次并stop-gradient。
4. 记录特征值比、选择维度、TX margin变化和数值条件数。
5. 非有限、秩不足或条件数超限时跳过更新并保留上一个有效basis，不得让训练中断或悄然全投影。

### Task 12：整合总损失、checkpoint与日志契约

**Files：**

- Modify：`code/cvsrffi/daot_training.py`
- Modify：`code/SSDG/train_ssdg.py`
- Modify：`code/model_dual_cvsincnet.py`
- Modify：`tests/test_adv3b02_daot_stn.py`

**步骤：**

1. 总目标接入`L_base+λzL_orbit,z+λlogitL_soft-logit+λprotoL_proto+λtanL_budget-tan+λrouteL_route+λRXL_RX+λtailL_CVaR+λcleanL_clean-anchor+λsubL_subspace`。
2. 初始范围按报告固化：`λz=.30～.45`、`λlogit=.05～.10`、`λproto=.10～.15`、`λtan=.02～.05`、`λroute=.03～.08`、`λRX=.05～.10`、`λtail=.05～.15`、`λclean=.02～.05`、`λsub=0`。
3. 所有loss先做各自EMA尺度归一化，再乘独立schedule和权重；输出raw、normalized、weighted三层日志。
4. checkpoint保存/恢复style bank、conditioned prototype、group EMA、U trust状态、direction sampler、Temporal Memory、loss normalizer、gradient controller和可选subspace。
5. 配置负测覆盖负权重、非法alpha、非法TTL、跨TX原型、未知方向与禁用场景。

### Task 13：结构化batch与source-only模型选择接口

**Files：**

- Modify：`code/cvsrffi/balanced_tx_rx_sampler.py`
- Create：`code/cvsrffi/daot_source_selection.py`
- Modify：`code/SSDG/train_ssdg.py`
- Create：`tests/test_daot_structured_batch.py`
- Create：`tests/test_daot_source_selection.py`

**步骤：**

1. 扩展batch采样为P×R×M×K结构，默认`R>=3`；目标比例为30% cross-RX labeled anchor、45% receiver/day-balanced U、15%当前hard group，tangent/fingerprint作用于10%～25%。
2. 在样本不足时使用记录明确的层级回退，保证TX标签与source receiver条件仍正确，不复制target样本。
3. 实现5折rotating pseudo-unseen source receiver切分；超参数选择后支持全部5个source receiver重训。
4. source-only代理分数包含CVaR20、receiver floor、`H(clean,LEO_WEAK mean)`、receiver probe惩罚和训练成本。
5. SWAD区间只由pseudo-unseen source receiver指标选择；单V仍只读且不拆成V_cal/V_select。

### Task 14：聚焦验证、真实checkpoint smoke与一次P0/P1审查

**Files：**

- Modify：`code/scripts/smoke_adv3b02_daot_real_checkpoint.py`
- Create：`code/scripts/smoke_adv3b02_daot_stn_rx_v2.py`
- Create：`tests/test_adv3b02_daot_stn_rx_v2_smoke.py`
- Update：`docs/CVS_PHASE1_ADV3B02_DAOT_STN_RX_V2_TRACE_20260903.md`

**步骤：**

1. 运行V2纯函数与集成测试；确认V1回归测试不变。
2. 运行`py_compile`覆盖新增模块、训练入口、模型和smoke。
3. 用一个真实ADV3B02 checkpoint做source-shaped、无query、单batch smoke，验证checkpoint兼容、前向、反传、state round-trip与日志字段。
4. 做一次独立P0/P1正确性审查，只接受会导致真实实验跑错、越权、覆盖输出、误杀进程、不能启动或不能产生合法prediction的问题。
5. 若有P0/P1，只做一次针对原问题的定点修复与复审；P2和文档完善不阻塞。
6. 更新追踪表的`pending→verified/deferred/rejected`状态，并记录实际测试命令和结果。
7. 到此暂停；不打包release、不scp、不登录N607、不启动性能实验。

---

## 4.后续实验预登记草案（本轮不启动）

实施验证完成后，性能实验仍遵循先小矩阵证伪、后扩展确认，且只评估clean与三类LEO_WEAK。

### 4.1最小同row候选

|Row|目的|启用机制|教师成本|
|---|---|---|---|
|B0|同配置基准|A4机制配置重跑，不做非LEO_WEAK扩展|三视图，仅作为历史上界对照|
|V2-P1|锚定教师|asymmetric anchored orbit|两fresh+memory|
|V2-P2|稳定tangent|P1+registry+chordal+budget|两fresh+memory|
|V2-P3|方向路由|P2+random single-TX+route|两fresh+memory|
|V2-P4|接收机主线|P3+Style Bank+TX条件RX prototype|两fresh+memory|
|V2-P5|尾部主线|P4+continuous U+receiver CVaR+independent schedule|两fresh+memory|
|V2-E1|效率确认|P5+vectorized memory+identity-only forward|两fresh+memory|
|V2-R1|高风险增量|E1+selective nuisance subspace|两fresh+memory，默认不晋级|

这不是要求一次发布8个实验。首次只发布能回答单一机制问题的最小相邻行；未达到预登记门槛时停止扩展，但负结果必须保留和报告。

### 4.2固定结果字段

- 总体：clean accuracy、三类LEO_WEAK各自accuracy、LEO_WEAK mean、`H(clean,LEO_WEAK mean)`。
- 尾部：receiver×scenario strict floor、receiver std、worst-two receiver average、CVaR20。
- 机制：orbit dispersion、coverage entropy、selectivity、TX route margin、receiver probe、U三态覆盖、各方向tangent能量与budget违规率。
- 资源：峰值显存、wall time、样本吞吐、教师前向次数、identity-only回退比例。
- 稳定性：non-finite loss次数、skipped-gradient比例、辅助/主任务梯度余弦与投影触发次数。

### 4.3晋级门槛

- clean accuracy≥81.8%。
- LEO_WEAK mean≥73.0%。
- receiver×LEO_WEAK strict floor≥63.5%，并且相对A4同row基线至少+1.0pp。
- receiver std≤6.5pp，worst-two receiver average≥64.0%。
- selectivity相对基线提升15%～20%，receiver probe下降且TX margin不退化。
- non-finite loss=0，skipped-gradient<0.5%，不得出现A1式domain loss支配。
- 高效实现相对对应性能路线：LEO_WEAK mean下降不超过0.15pp、strict floor下降不超过0.20pp，wall time下降10%～20%。

不设任何非LEO_WEAK、cross-family或旧checkpoint外推门槛。

---

## 5.建议执行顺序与里程碑

1. **M1契约闭合：**Task1，冻结V2配置、默认两fresh+memory和非LEO_WEAK排除项。
2. **M2低风险数值核心：**Task2～4，完成锚定教师、预算化弦长tangent和方向路由。
3. **M3接收机主线：**Task5～7，完成Style Bank、TX条件RX对齐、CVaR和连续U可信度。
4. **M4结构与优化控制：**Task8～10，完成分支预算、独立调度、冲突保护和向量化memory。
5. **M5高风险可选件：**Task11，完成但默认关闭选择性子空间。
6. **M6整合闭合：**Task12～14，完成总损失、结构batch、source-only选择、聚焦测试、真实checkpoint smoke和审查。
7. **STOP：**提交并push实现与追踪报告后等待用户明确指令；不自动启动实验。

## 6.计划级风险与控制

- **风险：接收机仿真不真实。**首版只在source统计凸包及小幅外推中采样，任何非有限/超界变换回退恒等并计数。
- **风险：tangent抹除TX指纹。**严格区分pure nuisance、mixed和TX fingerprint；混合方向非零预算，TX方向不进入nulling。
- **风险：`z_dom`尺度作弊。**路由距离归一化并增加上界/方差约束。
- **风险：U路径再次被easy样本支配。**使用连续可信度、三态和class/RX/severity配额，而不是全局硬共识。
- **风险：辅助损失压过主任务。**独立schedule、EMA尺度、梯度余弦监控和持续冲突才触发的主任务保护。
- **风险：训练进一步变慢。**两fresh+memory为默认、视图拼批、identity-only forward、自适应10%～25%切向采样和低频完整诊断。
- **风险：方案一次性过大无法归因。**每项有独立开关和相邻消融；高风险subspace默认关闭，实验按最小相邻行发布。
- **风险：范围漂移到非LEO_WEAK。**配置白名单与测试双重禁止，D0/C3标记`REJECTED_BY_USER_SCOPE`。

