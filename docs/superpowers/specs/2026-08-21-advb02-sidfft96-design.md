# ADV3B02 SID-FFT96前端可辨识分解设计

状态：待书面规范确认  
日期：2026-08-21  
基座：`ADV3B02_CORE90_SOFT_E200`  
协议：`项目.md`与`docs/PROJECT_PROTOCOL.md`  
指导来源：用户提供的《CVS项目优化的核心方向：从“后验修正特征”转向“前端可辨识分解”》

## 1.目标与核心判断

本轮把Phase1优化重心从嵌入形成后的nuisance修正，前移到身份嵌入形成前的复频谱可辨识分解。第一轮不实现完整CVS-SIDG，而是完成两个相互依赖、可独立核验的最小阶段：

1. P0频谱可辨识性审计：仅在source侧估计逐频带发射机信息与跨域变化之比，生成固定的多频带身份掩码；
2. P1 SID-FFT96：在冻结的ADV3B02成熟路径旁增加96维复相位感知频谱证据，通过零初始化残差进入`z_id`，同时保留完全不变的raw路径用于同checkpoint逐样本比较。

本设计要回答一个可证伪问题：在不改变成熟时域、PA、DAC和分类几何的条件下，显式保留幅度趋势残差、相位残差、相位曲率、正负频率耦合与带边残差，能否提高LEO弱信道与Strict UDU表现，同时不损伤clean和最差receiver/scenario。

## 2.协议与实现边界

### 2.1必须保持的Phase1协议

- 数据角色固定为`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- 四个角色均只来自`R_s`，并保持物理样本ID两两不交；`R_s∩R_t=∅`。
- P0的TX可辨识性统计只读取`L_s`的TX标签。`U_s`不提供TX真值；`V_cal`与`V_select`不参与掩码拟合或候选选择。
- Phase1不读取Phase2 support/query、target receiver、target-known、target-new或target-unknown信息。
- source clean样本可生成对应LEO弱视图，但不改变任何Phase2单物理样本单观测语义。

### 2.2必须保持的Core90训练与评测路径

- 星地训练继续采用`concat_sat_ce_only=true`。
- 卫星TX CE有效权重固定为`0.68`，从E80开始计入训练总损失。
- 视图日程固定为：E1–40使用`leo_clear_weak,p=0.30`；E41–90使用`leo_low_elev_weak,leo_rain_weak,p=0.60`；E91–200使用三场景并集`p=0.80`。
- 完成训练的候选必须分别保留clean、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`结果。
- 不增加clean—satellite特征MSE、强KL、强GRL、Fishr、NTRS嵌入修正或CRRA分支插值。

### 2.3本轮明确不实现的内容

- 结构化干扰算子`O(q)`及其正则化逆；
- sample-conditioned完整Koopman矩阵；
- MUSE Raw—Canonical—Prototype三方路由；
- source episodic pseudo-new几何与Phase2 support-only算子适应；
- RF32职责重分配、Phase1 bundle格式扩展与Phase2正式性能矩阵；
- 全量S0–S12、多seed或完整125确认。

这些内容只在SID-FFT96满足本设计第9节的晋级标准后进入后续候选，不作为本轮发布条件。

## 3.现有实现与缺口

当前Phase2辅助FFT96由`code/export_spaceborne_features.py::_spectral_logmag_sketch_batch`生成。它对去均值、RMS归一化后的复IQ执行FFT，只保留插值后的对数幅度，因此不包含显式相位残差、相位曲率或正负频率耦合。

当前Phase1模型的`DSQFreqStabilityStem`从正负频率对数能量、比值和不对称中减去局部平滑趋势，仍以幅度型统计为主。它没有实现指导建议中的96维复相位感知描述，也没有由source侧$J_b$固定的多频带身份掩码。

因此，本轮不能把已有DSQ重新命名为SID-FFT96。严格设计对齐要求新增可独立测试的复频谱描述模块、P0审计脚本和零初始化残差接入路径。

## 4.P0频谱可辨识性审计

### 4.1输入

P0读取固定Phase1划分中的`L_s`：

- 原始clean IQ；
- TX标签、source receiver、day和physical sample ID；
- 对每个clean样本按固定seed生成的三类LEO弱视图。

P0不得读取`U_s`的TX真值、`V_cal/V_select`标签或任何target数据。审计脚本必须将实际角色计数、receiver/day集合、样本ID互斥检查和LEO seed写入输出摘要。

### 4.2频带描述

对每个频带$b$构造：

\[
\psi_b(x)=[\log A_b,\sin\Delta\phi_b,\cos\Delta\phi_b,\Delta^2\phi_b,M_b^{\pm}].
\]

实现先对复IQ去均值并做带保护的RMS归一化，再使用Hann窗、中心化复FFT和有效频点掩码。相位特征使用相邻复频点乘积的`atan2`或等价的正余弦形式，避免直接展开相位造成分支切割错误。

### 4.3可辨识性统计

对每个频带计算：

\[
J_b=\frac{\operatorname{tr}S_{\mathrm{TX},b}}{\operatorname{tr}S_{\mathrm{DOM},b}+\sigma^2_{\mathrm{est},b}+\epsilon}.
\]

其中(S_{\mathrm{DOM},b})由同TX跨receiver、day和LEO视图的散度组成。分子和分母必须分别输出，禁止只保存最终比值。频带排序规则在脚本参数和JSON中固定；同分时按频带索引稳定排序。

### 4.4输出

P0输出目录至少包含：

- `spectral_identifiability.json`：频带边界、全部统计量、排序、输入角色计数和配置；
- `sid_mask.npz`：S3固定多频带身份掩码、中心低通S1掩码和有效频点掩码；
- `spectral_identifiability.csv`：逐频带可审计表；
- `spectral_identifiability.png`：TX散度、跨域散度与(J_b)的诊断图。

`sid_mask.npz`是S3的科学输入，不是额外seal、receipt或发布许可。

## 5.SID-FFT96描述

### 5.1公共预处理

输入为`[B,2,T]`实张量表示的复IQ。模块必须：

1. 拒绝错误rank、错误IQ通道数和不足以形成频带的长度；
2. 将NaN/Inf替换为有限值，并在输出中记录无效样本计数；
3. 去除复均值，执行带保护RMS归一化和Hann窗；
4. 生成中心化复FFT，不强制正负频率数值对称；
5. 应用与模式对应的固定掩码。

### 5.2固定96维分配

|分组|维数|定义|
|---|---:|---|
|幅度趋势残差|24|对数幅度减去局部平滑趋势后，在身份掩码内稳定池化|
|相位残差|24|去除加权线性相位趋势后的相邻相位增量正余弦压缩|
|相位曲率与局部相干|16|二阶相位差及相邻频带复相干|
|正负频率耦合|16|镜像幅度比、复共轭相关和不对称残差|
|带边残差|16|带内边缘的局部幅度/相位高阶残差，不声明完整带外谱再生|

五组拼接后执行有限值检查、逐样本中心化和带保护L2归一化。输出维数必须严格为96；掩码为空、有效频点不足或输出非有限时立即失败，不使用全零静默回退。

### 5.3三种实验模式

- `center`：S1固定中心低通掩码，其他描述公式不变；
- `phase`：S2使用全有效带宽但只启用相位残差、曲率和局部相干，未启用分组以确定性零填充保持96维接口；
- `sid`：S3使用P0生成的固定多频带掩码和完整五组描述。

`off`表示完全不构造SID描述，模型行为与当前ADV3B02一致。

## 6.模型接入与基线保护

### 6.1接入位置

SID分支位于原始IQ与最终身份嵌入之间，但不改变现有Sinc、频率卷积、DAC或PA路径。冻结ADV3B02产生原始身份嵌入(z_{\mathrm{raw}})，SID模块产生96维描述(s)，再通过零初始化投影形成残差：

\[
\Delta z_{\mathrm{sid}}=P_{\mathrm{sid}}(s),
\qquad
z_{\mathrm{sid}}=\operatorname{normalize}(z_{\mathrm{raw}}+\alpha\Delta z_{\mathrm{sid}}).
\]

`P_sid`的最后线性层权重和偏置均初始化为零。初始化时必须满足逐元素`z_sid==z_raw`的数值容差测试。

### 6.2双输出

同一次forward保留：

- `z_id_raw`与`logits_raw`：冻结成熟路径的结果；
- `z_id_sid`与`logits_sid`：增加SID残差后的结果；
- `sid_fft96`、各分组范数、有效频点比例和残差范数。

raw与SID使用同一个冻结分类头。训练损失只作用于SID输出；raw输出只用于基线核对和逐样本转换分析。禁止用SID结果覆盖raw字段。

### 6.3checkpoint加载

从历史`ADV3B02_CORE90_SOFT_E200`checkpoint加载时，只允许SID模块新增参数缺失。任何旧参数缺失、旧参数unexpected或shape不匹配均立即失败。加载报告必须分别列出允许新增的SID键和所有其他不匹配；不能用通用`strict=False`掩盖基座漂移。

### 6.4训练范围

S1–S3冻结全部原ADV3B02参数和分类头，只训练SID描述中的可学习缩放、SID投影和残差门。数据顺序、checkpoint、seed、优化器、epoch数和Core90卫星日程保持一致。第一轮不学习频带拓扑；S1和S3掩码均为固定输入，避免7%标签下的早期门控锁定。

## 7.训练损失与阶段

本轮不引入完整SIDG总损失。SID分支沿用Core90现有身份监督：

\[
\mathcal L=\mathcal L_{\mathrm{CE}}^{\mathrm{clean}}+0.68\mathcal L_{\mathrm{CE}}^{\mathrm{sat}}+\mathcal L_{\mathrm{existing,Core90}}.
\]

现有Core90损失只在其原本合法、可达的训练路径中生效；冻结参数不会因损失存在而更新。`U_s`仍按现有Phase1弱标注规则工作，不读取其真实TX标签。E80前不计算卫星CE；E80后只对SID输出增加与原策略一致的卫星TX CE，不增加clean—satellite嵌入一致性。

## 8.首发实验矩阵

首发run只包含一个共享基线和三个单seed机制臂：

|ID|模式|可训练参数|回答的问题|
|---|---|---|---|
|S0|`off`|0；直接评估冻结checkpoint|固定raw基线与新代码关闭时的等价性|
|S1|`center`|SID缩放、投影和残差门|收益是否只来自中心低通|
|S2|`phase`|SID缩放、投影和残差门|复相位规范化能否独立产生收益|
|S3|`sid`|SID缩放、投影和残差门|P0多频带完整SID-FFT96是否产生净收益|

共同设置：

- checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；
- seed：`392002`；
- epoch：S1–S3均为200；
- split：`0.07/0.63/0.15/0.15`；
- 星地训练：Core90三阶段LEO_WEAK日程；
- 评测：clean及三种LEO弱场景逐项报告；
- 输出目录：不可覆盖的统一run ID下按S0–S3分目录。

S0是同checkpoint冻结评测，不伪装成一次新训练。S1–S3的raw输出应与S0保持数值一致；若不一致，属于实现错误而非性能结果。

## 9.指标与预登记判断

### 9.1主要指标

- clean accuracy；
- 三种LEO弱场景accuracy及其均值；
- Strict UDU均值；
- receiver×scenario floor；
- raw→SID的`rescued`、`harmed`和净转换；
- TX probe与RX probe；
- SID残差范数、有效频点比例和各分组范数。

### 9.2进入下一阶段的门槛

S3只有同时满足以下条件，才进入多seed或P2结构化算子：

- `Δclean>=-0.3pp`；
- `ΔLEO_mean>=+1.0pp`；
- `ΔStrict_UDU>=+1.0pp`；
- `ΔLEO_floor>=+0.5pp`；
- `rescued>harmed`；
- RX probe相对下降至少20%，且TX probe不下降。

Phase2指标不属于首发矩阵。只有Phase1门槛通过后，才以独立后续实验比较现有FFT96与SID-FFT96；本轮不能声称Phase2已经改善。

## 10.实现文件与职责

|文件|职责|
|---|---|
|`code/cvsrffi/spectral_identifiability.py`|复IQ预处理、SID-FFT96描述、掩码验证和零初始化残差模块|
|`code/scripts/audit_phase1_spectral_identifiability.py`|P0 source-only频带统计与artifact输出|
|`code/model_dual_cvsincnet.py`|双raw/SID身份输出、冻结基座接入和模型配置|
|`code/SSDG/train_ssdg.py`|SID参数、严格checkpoint兼容加载、训练范围、损失与日志|
|`code/eval_feature_diagnosis.py`|逐频带统计、raw/SID转换、TX/RX probe和诊断汇总|
|`code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh`|P0与S0–S3不可覆盖发布矩阵|
|`code/tests/test_spectral_identifiability.py`|描述维数、有限值、掩码、相位与镜像行为|
|`code/tests/test_advb02_sidfft96_model.py`|零初始化等价、raw不变、SID梯度范围和checkpoint拒错|
|`code/tests/test_phase1_advb02_sidfft96_launcher.py`|协议、矩阵、Core90日程、路径和dry-run行为|

## 11.错误处理与安全边界

- P0发现角色越界、物理样本重叠、空频带或非有限统计时直接失败，不发布掩码。
- launcher拒绝已存在的run/log目录，拒绝缺失P0掩码的S3，拒绝非`0.07/0.63/0.15/0.15`划分。
- dry-run必须输出每个候选的精确命令、checkpoint、GPU、输入输出和预期artifact。
- 一次独立P0/P1审查只检查会导致真实实验跑错、越权、覆盖输出、误杀进程、无法启动或无法产生合法prediction的问题。
- 不创建设计SHA、逐文件SHA、seal、signature、receipt或额外发布许可。Git提交固定代码与配置；N607仅对一个release归档做一次本地/远端SHA比较。

## 12.验证策略

实现采用TDD：先写行为测试并观察预期失败，再写最小实现。完成前至少执行：

1. `py_compile`覆盖所有新增/修改Python文件；
2. SID描述、模型和launcher聚焦pytest；
3. launcher`bash -n`与`--dry-run`；
4. 真实ADV3B02 checkpoint、无query、单batch CPU或GPU smoke；
5. `git diff --check`；
6. N607远端编译、dry-run及启动后一次PID/CWD/cmdline/GPU/log增长核对。

实现完成只证明发布路径可运行，不构成性能改善。性能结论必须等待S0–S3完整prediction和独立评分结果。
