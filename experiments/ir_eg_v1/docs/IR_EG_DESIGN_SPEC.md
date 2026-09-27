# IR-EG / BR-IR-EG 设计规范 v1.0

状态：设计与实施依据；未实现新训练算法，未运行新的 CVS-RFFI 训练。
日期：2026-09-27。
证据基准：`niubility1317/CVS-RFFI@acf0a6407c4a2cd114851e9f60d0cd346e4b2c77`。

## 1. 目标与范围

在完整 DAOT＋FastTrust 联合训练背景下，首先验证有限步隐式域头响应 IR-EG 是否优于完整 EG；随后独立验证分块历史预测 BR-IR-EG 的效率。新方案是待检验的研究设计，不预先宣称性能提升、全局收敛或首创。

本规范的主算法为 IR-EG。NR-EG 子空间筛选、TR 坐标旋转、CF 风险候选、DRIC 局部约束、XT 跨 TX 元梯度、额外物理分支均不加入主实验。IR-ref 与 IR-cached 是同一算法的两个实现；BR-IR-EG 是另一个近似算法，不是工程等价优化。

### 1.1 不可改变的数据与监督边界

- 全部允许的源 RX 参与主训练，不永久留出某个源 RX。证据包内为 RX1/3/4/6/8、day1/2/3，L/U/V 数量为 6300/56700/27000；执行前按冻结数据 manifest 核验，差异报错，不自动改成另一个数据协议。[S1]
- 保留 clean＋LEO 同物理样本配对、U 曝光、原有教师视图、阶段、损失权重和归约。不因为修改求解器而给 LEO 增加域对抗损失。
- U 是源域无 TX 标签样本，不是目标域适配数据。U 域标签合法，U 隐藏 TX 标签不得进入训练、响应求解、诊断路由或控制。
- U 域损失对身份骨干的直接梯度为 0；U 的 DAOT/FastTrust 身份梯度仍合法。不能将二者混淆。
- V 仅按既有协议作校准、监测和选模；临时域头拟合用合法源训练样本，不用 V 反向拟合。
- 目标数据不得决定响应强度、历史刷新、风险、收敛、重跑或 checkpoint。候选及 checkpoint 冻结后，先固定预测，再由独立评分器连接目标真值。
- E200/44400 接受主步是历史比较读数，不是自动收敛证明。达到安全上限而未收敛，状态为 `safety_cap_not_converged`。
- 本计划不授权启动、恢复或停止任何远端任务。

### 1.2 环境与版本

首版支持单 GPU、FP32、现有 AdamW、固定预测比例 1。禁用 AMP、DDP、新优化器、梯度累积、`torch.compile` 和尚未验证的融合算子组合；后续逐项验收，不临时升级 PyTorch。

证据代码根目录为：

```text
docs/research/dynamic_game_20260927/implementation/response/code/
```

它是冻结证据副本，不等同于仓库当前默认导入路径。实施时创建独立研究执行包 `experiments/ir_eg_v1/`，从冻结执行包复制代码、必要依赖和测试，不编辑历史证据目录。本文实施路径中的 `CODE_ROOT` 指新包的 `code/`。

启动时记录关键模块 `__file__`、SHA-256、Git 基准、工作区 diff、Python/PyTorch/CUDA/cuDNN、设备、优化器 groups、随机种子与完整配置。由于原模型代码会修改 `sys.path`，必须逐个核验实际导入来源；存在多个同名模块且来源不匹配时拒绝运行。[S2]

## 2. 术语、参数归属与唯一算法定义

令 `q=(psi, phi)`。`phi` 是现有 `model.adv_head` 的全部可训练参数；`psi` 是除此以外全部参与原优化器的参数。`theta` 仅表示 `psi` 中身份编码器相关参数。不能把 `dom_head`、`dom_backbone`、TX 反向头或 FastTrust local identity head 当成 `adv_head`。

同一参数存在共享别名时，按对象身份去重；参数打包顺序使用冻结的参数清单，不在每步按字典临时排序。

原始带符号更新场：

```math
F_theta = grad_theta T - lambda_E grad_theta D_L
h = F_phi = alpha_L grad_phi D_L + alpha_U grad_phi D_U
lambda_E = alpha_L gamma_L
```

`T` 包括全部非该编码器域对抗路径的原任务，不仅是 TX CE。`gamma_L` 是 L 的 GRL 系数；`alpha_L` 为 L 域头 CE 的有效系数。二者不能重复相乘。U 的 GRL 系数为 0。[S3]

域头目标必须按原生归约写成：

```math
H = alpha_L / n_L * sum_i CE(f_phi(z_Li), d_Li)
  + alpha_U / n_U * sum_j CE(f_phi(z_Uj), d_Uj).
```

禁止拼接 L/U 后用一个总 batch mean 替代上述加权均值。`alpha_U` 含原有 tail 调度等实际系数，不硬编码一个看似相同的常数。L/U 域损失使用的确切物理 ID、视图和权重必须在 loss ledger 中登记。主配置中 L 域对抗来自合法 clean 路径，U 来自原生 strong 路径；其他辅助前向即使计算了 `adv_dom_logits`，也不自动成为域头训练样本。[S3,S4]

启动/验收断言：`grad_phi(T)` 为 None 或数学零，且 `grad_phi(native_loss)` 与相同随机条件的 `grad_phi(H)` 一致。断言失败意味着二玩家简化不适配当前配置；不得丢弃额外头损失后继续运行。

### 2.1 四个位置必须区分

|符号|含义|是否持久提交|
|---|---|---|
|`q0, phi0`|主步原点参数|起点|
|`qp, phip`|从原点执行一次虚拟 AdamW 后的位置|否|
|`phir=phip+kappa*delta_phi`|用于最终梯度场评估的域头响应位置|否|
|`q_next`|恢复原点后，用最终场执行一次正式 AdamW 的结果|是|

`phir` 通常不等于 `phi_next`。禁止在 `phir` 上继续做正式优化器更新。

## 3. IR-EG 数学规格

### 3.1 共同上下文

一个接受主步使用同一份 L/U、配对视图、教师输出、路由、损失权重、有效归约、BN 起始 buffer 和随机前向语境。

原点和预测点比较不重新校准、不重新抽取 U、不重新生成卫星信道、不重新计算 hard/partial 路由。DAOT loss normalizer 先按原点真实调用顺序计算并记录实际除数，预测点复用实际除数，而不是重估 EMA。[S5]

### 3.2 完整预测与原始域头梯度

在原点计算完整原始梯度 `g0`，保存：

```math
h0 = g0[phi]
qp = AdamW_virtual(q0, optimizer_state0, clip_global(g0)).
```

`h0` 是加权、未裁剪、未预条件、无 AdamW weight decay 的域头梯度。不得把参数位移、裁剪梯度或历史域头梯度混进 `h0`。

在预测点同一上下文计算：

```math
hp = grad_phi H(psi_p, phi_p)
e = hp - h0.
```

在 IR-cached 中，`hp` 通过真实预测特征的 detached 副本求得，只在域头求导。

### 3.3 AdamW 响应度量 R

本算法不是精确的隐式 AdamW；它采用以真实虚拟预测 `phip` 为锚、冻结二阶矩和全局裁剪系数的局部仿射模型。

对域头参数坐标 j，设参数自己的原点更新计数为 `k_j`，原点二阶矩为 `v0_j`，原点实际全局裁剪系数为 `c0`，当前预测学习率为 `eta_pred_j`。构造：

```math
v1_j = beta2_j*v0_j + (1-beta2_j)*(c0*h0_j)^2
vhat1_j = v1_j/(1-beta2_j^(k_j+1))
R_j = eta_pred_j*c0*(1-beta1_j)
      / ((1-beta1_j^(k_j+1))*(sqrt(vhat1_j)+eps_j)).
```

所有值从实际 optimizer group/state 读取，不能猜默认 betas 或 eps。优先核验 `v1` 与真实虚拟步骤产生的状态相等。空 AdamW 状态按该优化器第一次更新语义初始化；不拿 solver 全局步数替代每参数计数。

`c0` 只出现一次。不能先用裁剪后的 `e`，再乘含 `c0` 的 R。`c0` 应来自真实裁剪 helper 的同一次计算，不重新估计范数而改变浮点归约。

局部仿射映射：

```math
Phi_t(h) = phip - R*(h-h0).
```

因此 `Phi_t(h0)=phip`。历史一阶矩和 weight decay 的影响包含在锚点内。GGN 中不额外添加 weight decay 的 L2 曲率。AdamW 的二阶矩导数与裁剪系数导数被有意忽略，不作精确 Newton/隐式 AdamW 声明。[S6]

首版拒绝 `amsgrad=True`、`maximize=True`、稀疏梯度、非 FP32 状态和未经验证的优化器变体。冻结或零学习率坐标从响应活动子空间移除，增量置零；不能对 R=0 的坐标取逆。`grad=None` 与真实零梯度保持不同的优化器语义。

### 3.4 域头 GGN

当前冻结 `adv_head` 为 `Linear -> ReLU(inplace=True) -> Dropout -> Linear`，没有域头 BatchNorm。[S2]

对每次真实域头调用 c，使用其预测点特征、参数、固定随机掩码，令 `J_c` 为 logits 对域头参数的 Jacobian，`p_c=softmax(logits_c)`：

```math
G v = sum_c J_c^T [w_c * (diag(p_c)-p_c p_c^T) * (J_c v)].
```

`w_c` 包含逐样本有效权重/归约。域标签必须在冻结的domain映射中有效；某块样本为空且权重为0时整体跳过，不对空集合mean，也不以虚构样本补齐。禁止对已经加权的 logits 再平方权重；禁止将 signed/GRL loss 的 Hessian 当成 GGN；禁止把 `g g^T` 当成同一个矩阵。

只实现 JVP/VJP，不建立 `|phi| x |phi|` 稠密矩阵。计算VJP时，把概率和输出空间的乘积向量视为常量；不得对含未detach乘积的标量再次求梯度而额外引入概率或JVP导数。G 在单次 CG 内固定：phip、概率、掩码、特征、权重均不更新。生产求解不建立对骨干参数的高阶图。

### 3.5 线性系统与有限迭代

```math
(R^{-1}+G) delta_phi = -e
A(u) = u + sqrt(R)*G(sqrt(R)*u)
b = -sqrt(R)*e
A u = b
Delta_phi = sqrt(R)*u
phir = phip + kappa*Delta_phi.
```

默认 `kappa=0.25`，CG 从零向量开始，最多 2 次迭代。`rtol=1e-3` 只用于早停；达到 2 次仍未满足该阈值，应标记 `truncated_usable`，不能伪称精确收敛，也不能因此自动回退到 EG。这样才真正验证固定预算近似响应。

A 的单位阵已经提供正定项，首版不引入新的可调 damping 或信任半径。`b=0` 直接返回零响应；`h0=0` 但 `hp!=0` 必须正常求解。

CG 的内积与残差范数建议用 FP64 累加，向量和域头运算保持 FP32。任何非有限值、明显非正的搜索分母或活动参数结构变化视为数值/结构错误。2 次 CG 的残差不一定比初始 Euclidean 残差更小，不以这个不成立的断言作回退依据。

诊断另外记录二次模型能量 `0.5*u^T*A(u)-b^T*u`；不把这一局部数值量当成分类风险保护。

### 3.6 最终场与正式更新

固定 `phir` 的数值，但不冻结域头输入的梯度：

```math
F_theta_IR = grad_theta T(psi_p) - lambda_E*grad_theta D_L(psi_p, phir)
F_phi_IR = grad_phi H(psi_p, phir)
q_next = AdamW_formal(q0, optimizer_state0, clip_global(F_IR)).
```

响应求解链路停止梯度。最终域头参数是新的独立叶子或在完整重建图之前安装的数值。不能对 `phir` 的来源求 hypergradient。

最终对抗方向通过完整身份骨干传播，不只更新末端投影。不对 DAOT、FastTrust、TX CE 的共享特征总梯度做统一 hook。U 域头损失继续使用原生 GRL=0 语义，避免把应存在的零梯度变成 None 导致优化器行为变化。

## 4. 随机性与域头重放

### 4.1 不可用 eval 或重新抽样替代同一随机函数

每个真实 head 调用有稳定 call key，例如 `L/clean/main/adv`、`U/strong/rc4/adv`，并记录调用序号、样本 ID、shape、dtype、GRL 系数、有效权重和 dropout 前 RNG。

L 与 U 分开调用时，不合并成一个 batch 重放。CUDA dropout 的随机消耗可能受 shape 和调用次数影响。

### 4.2 掩码重放实现

原生前向保持不变。在 Dropout 执行前记录 RNG。隔离恢复该 RNG，对同 shape/dtype/layout 的全 1 张量调用原生 dropout，得到缩放后的掩码 M；恢复调用者 RNG。使用原输入时核验 `native_dropout(x)==x*M`。不能通过 `output/input` 推断掩码，因为零激活不可辨识。

用于 JVP/VJP 的纯 head 函数：

```text
Linear -> ReLU(inplace=False) -> multiply(frozen_M) -> Linear
```

这里只消除导数计算中的原地写入，不改变网络结构、激活值或随机掩码。非原地 ReLU 与原生路径的数值、参数梯度、输入梯度都必须测试，包括恰好为 0 的输入。遇到其他 head 结构先拒绝，而不是猜测兼容。

调用时冻结 dropout 掩码，不冻结 ReLU 激活模式；在 phir 处按新参数重新计算 ReLU。GGN 的 Jacobian 仍固定在 phip。

## 5. 两个实现，不混作两个算法

### 5.1 IR-ref：可审计的参考实现

1. 固定主步上下文，保存完整原点事务 S0。
2. 运行原点完整场，得到 g0/h0；保存本次原点闭包结束后的 buffers B0+ 与 RNG R0+，以及原点实际 normalizer 除数与待提交尺度。
3. 执行原生虚拟 AdamW，读取 phip 和虚拟二阶矩，构造 R；保存 qp 数值。
4. 保留 qp 参数，恢复闭包前 buffer/RNG，运行预测点捕获前向。正常启用必要的 autograd，但不对大骨干求完整梯度；捕获 detached 的 L/U 域头输入与随机调用记录。
5. 用小域头重放获得 hp/G；求解 delta_phi。
6. 删除捕获前向的全部图；保留 detached 记录。再次恢复闭包前 buffer/RNG，将域头数值设为 phir，其他参数保持 psi_p。
7. 重新运行完整正确器闭包，得到所有原始最终梯度 gIR。该次图建立之前已经安装 phir，不会修改存活图里的参数。
8. 删除正确器图，恢复 S0；从同一原点 AdamW 状态安装 gIR、全局裁剪、正式 step 一次。
9. 验证有限值；恢复 B0+/R0+，保留原点闭包产生的持久语义。随后按原生顺序提交教师、原型、归一化、路由和接受计数。
10. 写入遥测和 checkpoint（按原保存规则）。

这是三个完整场前向遍历、两个完整大参数梯度场求值的参考模式，不宣传为加速。一次“场前向遍历”内部包含多少模型调用由原任务决定。

### 5.2 IR-cached：正式性能研究采用的缓存实现

预测点只执行一次完整骨干前向，产生 `ObjectivePacket`：

- 所有原生非对抗任务张量及计算图；
- 两个有标签槽位的域头 loss 输入、原生 post-GRL 图和 detached 原始特征；
- 原生 loss assembly 的纯重组函数；
- 实际 head call tape、归一化记录和原生状态记录。

先在 detached 特征上完成 hp/G/响应求解，然后新建：

```python
phi_leaf = [p.detach().clone().requires_grad_(True) for p in phi_response]
```

用 `phi_leaf` 和缓存的原生 post-GRL 输入生成新的域头 CE，代入原生 loss assembly。一次 `autograd.grad` 计算 `psi` 与 `phi_leaf` 的最终梯度；按名字映射 `phi_leaf` 梯度到正式 `adv_head` 参数槽位。

禁止在存活图上 `copy_` 或 `.data` 改 `model.adv_head`。禁止用 `native_total + new_H - old_H` 作为主实现：这种抵消可能留下真实头梯度、改变 None 语义并掩盖重复对抗。应提取并复用原有 loss assembly，保持加法与权重顺序，仅替换两项已登记的对抗 CE 叶子。

原生任务不能通过未登记的 `adv_dom_logits` 消费影响路由或控制；检测到这种依赖，缓存版拒绝该配置。IR-ref 作为语义参照，不因缓存失败而静默混入同一运行。

`kappa=0` 在进入新分支前直接调用未修改的完整 EG 路径；不创建 hook、mask、CG、phi_leaf 或新归一化状态。另设 debug 测试强制走 packet 的零增量分支，证明 packet 本身与 EG 等价。

## 6. 主步事务与恢复

|状态|虚拟计算规则|正式接受规则|
|---|---|---|
|参数/AdamW矩/每参数step|允许临时预测，提交前恢复|正式推进一次|
|BN等注册buffers|每次场重放从闭包前状态开始|恢复原点完整闭包结束的状态；不是机械地令BN计数+1|
|Dropout/MixStyle/随机信道|复用本步上下文，局部重放隔离RNG|主随机流与原点闭包结束相同|
|EMA教师|本步冻结，不在预测点更新|按原生接受后顺序更新一次|
|DAOT归一化|记录原点实际除数；其余重放|只提交原点pending状态一次|
|RC4 route/calibration|本步冻结；校准边界在准备阶段完成|路由history只提交一次|
|原型/记忆/支持状态|不得在虚拟路径累积|沿用原生正式来源与顺序|
|solver/调度/history|虚拟调用不推进|接受后推进一次|
|loader/样本曝光|回退沿用已物化同一batch|不因回退调用next(loader)|
|日志计数|尝试次数允许记录|尝试、接受、回退、曝光分开|

现有 TrainingState 覆盖参数、梯度、optimizer、buffers、RNG 与显式 stateful；但不是所有 `ctx` 缓存和外部教师必然被自动覆盖。新增外层事务需要枚举全部持久对象，不得只恢复模型。模块training标志、临时requires_grad修改、上下文开关和hook生命周期也必须在finally中恢复；它们不一定属于state_dict。[S7]

有限响应求解数值失败：回到该主步的普通完整 EG；相同 batch、qp、随机状态和归一化条件下得到普通 EG 第二场，接受一次，记录明确 failure stage。2次迭代未精确收敛不算失败。

原点场非有限、缓存结构不匹配、梯度所有权断言失败、普通 EG 回退仍非有限：恢复完整事务并终止技术故障；不跳过样本后继续。若正式 optimizer step 后的外部 commit 失败，也要恢复外层事务；保存上一个完整接受 checkpoint。

## 7. 调度与配置

首轮继承冻结 native joint 配置：L/U batch=128/256、lr=2e-4、weight_decay=1e-4、max_grad_norm=5、predictor_lr_ratio=1、FP32、DAOT与FastTrust开启。实际beta/eps/所有原生阶段值以manifest为准。[S8]

响应主路线显式使用 `reuse_origin_used_scales`；不能沿用 `joint_config.py` 默认的 `legacy_reestimate` 后仍称与新响应矩阵同协议。EG对照也使用相同政策。

E1–20 继续完整 EG；E21 起每步进行 IR 响应，kappa 固定0.25。不继承 TR 的 E21–60 gamma ramp，不继承 DRIC 的每4步间隔。它们是不同方法的调度。[S9]

建议配置是新schema，实施前不能当作现有CLI运行：

```yaml
method: IR_EG
base_solver: full_EG
implementation: cached
ir:
  start_epoch: 21
  kappa: 0.25
  head_scope: adv_head_all
  metric: adamw_frozen_second_moment
  curvature: weighted_cross_entropy_ggn
  cg_max_iterations: 2
  cg_rtol: 0.001
  exact_hypergradient: false
  numerical_fallback: same_step_full_eg
  origin_scales: reuse_origin_used_scales
  source_risk_gate: false
  extra_domain_samples: false
br:
  enabled: false
  refresh_interval: 8
  cache: previous_accepted_final_raw_gradient
precision: fp32
launch: false
```

未知字段、布尔值冒充整数、kappa不在[0,1]、非法样本归约、优化器不支持等情况启动前报错。基线原生参数由完整冻结配置注入，不建立部分字段的“近似基线”。

## 8. BR-IR-EG：单独验收的后续加速算法

### 8.1 定义

`psi` 包括全部非域对抗头参与参数，不仅身份骨干。第t步从上一个接受主步取最终未裁剪梯度 `g_psi[t-1]`；用当前原点只读闭包取得真实 `h0[t]`、normalizer除数、BN/RNG终态。

```math
g_predict[t] = (g_psi[t-1], h0[t])
qp = AdamW_virtual(q0, state0, clip_global(g_predict[t])).
```

此时 c0 是混合预测场实际采用的裁剪系数，不是当前完整原点场的未知裁剪系数。R仍由真实虚拟更新及当前h0构造。预测点使用当前完整任务和当前 IR 响应，最终从原点提交一次。

缓存每个接受主步都更新为当前最终原始梯度，不是每8步才换一次。`refresh_interval=8` 指每个周期执行一次当前完整原点梯度，其余7步使用上一接受步的历史。历史年龄通常为1，不是8。

### 8.2 强制当前完整预测

首次步、E21进入BR阶段、每8步刷新、离散任务启停、校准状态版本改变、参数活动集合改变、恢复时缓存不完整或layout/version不一致时强制刷新。缓存存在结构问题时不能直接补零。

不能仅因为每步batch/route不同就刷新；否则近似算法永远不执行。也不能把每次连续normalizer变化当作新阶段。具体边界signature只含冻结配置版本、离散任务开关、校准版本和参数活动清单。

若无法在不进行完整反向的条件下可靠确定当前参数活动集合，则该主步执行完整原点求导。严格区分 `None` 与零梯度，尤其是分阶段启用的head。

### 8.3 成本边界

为保留原点normalizer、BN/RNG和部分内部autograd语义，仍保留必要的原点前向。通常省去一次完整大参数反向，不声称只剩一次前向。Fishr等原点内部导数单独计数，不能粗暴 `no_grad` 删除。

退化关系：`refresh_interval=1` 的 BR-IR 等于 IR；BR-IR 的 kappa=0 等于 BR-EG，不等于完整EG；只有同时每步刷新且kappa=0才回到完整EG。

BR不是执行等价优化。保存额外FP32历史梯度的显存、CPU读回、mask捕获、CG和日志成本全部纳入实测。

## 9. 诊断与日志规范

每主步轻量记录：accepted index、physical exposure、method、effective kappa、h0/hp/e norm、response norm、CG迭代数与status、linear residual、fallback原因、原点与正式裁剪系数、实际参数位移、optimizer commit次数、field/model/backward/head-only/JVP/VJP计数。

每1000个接受步及阶段边界执行隔离重诊断。关闭重诊断应使loss、梯度、参数、RNG、buffer完全不变。重诊断时间单独列出，但总训练成本仍包含它。

### 9.1 非线性响应残差

对原始完整delta和实际kappa*delta分别记录：

```math
R_nl(delta)=R^(-1/2)*delta
 + R^(1/2)*(h(psi_p,phip+delta)-h0).
```

保留signed变化，不截断负收益，不把它作为默认训练gate。kappa<1时不是要求把完整固定点残差解到零。

### 9.2 可恢复性与身份关系

临时域头从当前头复制，在合法源训练记录上固定拟合10次，骨干冻结、训练head Dropout按预登记规则处理，在不同物理记录的固定V监测集上评价。副本参数、优化器、RNG和BN不回写。记录恢复前后CE、准确率及差值，保留负差值。

记录源侧TX margin、最弱TX/RX、RX×TX切片、跨RX类间关系Delta_ab,r及TX×RX统计交互e_y,r。不能把域头变差单独解释为成功；不能把统计交互称为物理链路的精确参数恢复。

### 9.3 裁剪归因

稀疏地从同一原点状态比较普通EG与IR的实际AdamW位移；同时在诊断副本上使用共同参考裁剪系数重算一次，以区分域方向变化与全局缩放影响。该探针不改变正式更新。

### 9.4 结果字段

`configured/eligible/executed/applied/nonzero/fallback/accepted`分别记录。缺失值用null并注明原因，不能填0。2次CG截断状态不能记为converged。所有比例写分母。

## 10. 实验与判定

### 10.1 顺序

A. 数学与状态单元测试。
B. 已有早中晚checkpoint上的隔离单步/短窗口探针，不形成新性能声明。
C. IR-ref/IR-cached等价验收与成本测量。
D. 原生联合SIM、完整EG、IR-EG性能比较；先固定一个种子做源侧机制筛查，再在冻结候选上做392005/392006/392007配对确认。
E. BR-EG与BR-IR-EG效率验证，完整版本保留为性能参照。

明确的消融：IR无曲率（G=0，此时delta=-R e，不是CG0步返回零）；OR-EG原点特征响应控制：保持phip和R不变，以原点特征Z0计算ho=grad_phi H(Z0,phip)、Go和eo=ho-h0，执行同样2次CG，最终仍在psi_p和相应phir求正式梯度场。它隔离了预测特征对响应的贡献，不是新增持久域头更新。另做稀疏反事实编码器对抗关闭而域头监督保留。不得用同时关闭整个域损失的ADV0替代后一项。

不从种子或目标分数中选择性重跑。源侧筛查也需记录全部尝试和成本。

### 10.2 收敛与历史预算

保留历史E200/44400主步的成对读数，标签为budget audit。科学完成使用独立冻结的source-only收敛协议，全部比较臂一致，禁止目标成绩控制停机。

执行包现有200epoch校验不能直接覆盖掉。新增独立`convergence_confirmation`配置族，保留历史profile不变；完整写入学习率后续政策、安全上限、最后机制启动点和收敛窗。

优先沿用项目已批准的收敛manifest。若不存在可执行manifest，先做协议准备，不自动启动长训练。可用本计划预设的确认规则生成待冻结manifest：最后一次离散机制切换后至少40个完整epoch；连续4个10epoch窗口source V选择指标变化不超过0.2pp且CE相对改善不足0.5%；机制激活后已发生至少一次预定学习率下降；固定源监测panel相邻10epoch窗口的logit均方根差除以前窗logit均方根（分母至少1）不超过0.02；连续4窗中任务/域梯度与实际位移范数的窗口中位数，任意后窗不得比紧邻前窗增加超过20%（全零单独登记）；无未解释非有限步。数值阈值是预注册起点，不是数学收敛证明。最后学习率延续策略须来自同一确认manifest，不能执行者临场依据target决定。

以上规则不授权擅自延长或终止已有任务。到安全上限仍不满足条件，报告未科学完成而非已收敛。

### 10.3 性能与效率

性能指标：clean、LEO均值、Macro-F1、最弱TX、最弱RX、day0跨天、RX×TX条件差值。成对种子报告每个差值、均值和离散性；3个seed一致不等于统计显著。多个视图不能充当独立seed。

候选不默认接受clean或弱RX下降。“机制确实运行”“源侧有改善”“目标结果提升”“值得替代默认版本”是四个独立结论。单次负差值需要完整报告，不用总体均值掩盖。

效率同时报告：同物理曝光的墙钟/显存、同冻结源侧收敛判据的总成本、分组件profiling。隔离GPU/固定并发条件测量，包含teacher、数据、CG、snapshot、日志和评估成本。IR-ref不是效率候选。BR节省多少只能实测。

## 11. 验收底线

- kappa=0直接旧EG路径全状态一致。
- 同一位置、同一掩码下h0=hp，IR增量为0。
- 原点h0=0、任务导致hp非零，IR增量不能被短路为0。
- 用极小softmax线性头显式构造GGN验证矩阵自由乘积、对称性与PSD。
- 两次CG允许截断，但不能伪报高精度收敛。
- 有历史AdamW矩、非零weight decay、真实全局裁剪下，R与冻结二阶矩仿射映射相符。
- 最终phi梯度来自phir的独立叶子，求解过程不向theta传二阶梯度。
- U域损失对身份骨干为零；DAOT/FastTrust身份梯度仍可非零。
- LEO调用不因hook捕获而新增域头监督。
- 所有对抗项只出现一次，实际head字段梯度与显式H对齐。
- IR-ref与IR-cached在同状态下loss、每参数梯度、正式位移、优化器矩、RNG、buffers、teacher及normalizer对齐。
- 所有失败注入点回滚；没有用新batch掩盖同一步失败。
- 中断恢复包含缓存与随机状态，连续运行与分段恢复一致。
- stage表中实际执行次数符合登记值，不能仅开关为true。

## 12. 证据来源与边界

以下均为上述提交中的代码阅读依据，不代表本轮重跑测试：

- [S1] `implementation/response/README.md`：数据与执行包身份；其中历史49/222预算并存，不能直接混用。
- [S2] `implementation/response/code/model_dual_cvsincnet.py`：MLPHead、GRL、adv_head归属与导入路径。
- [S3] `.../cvsrffi/xuc_fusion/dr_objective.py`、`response_runtime.py`：U梯度权限及有效域头系数。
- [S4] `.../cvsrffi/xuc_fusion/objective.py`：L clean、LEO任务与完整目标入口。
- [S5] `.../cvsrffi/xuc_fusion/joint_normalization.py`：原点真实除数重放。
- [S6] PyTorch官方AdamW文档：优化器公式。R是本设计在冻结条件下的推导，不是官方现成API。
- [S7] `.../cvsrffi/game_tracking/state.py`、`solvers.py`：事务、完整EG与全局裁剪。
- [S8] `.../cvsrffi/xuc_fusion/joint_config.py`：冻结联合配置与默认normalizer。
- [S9] `.../cvsrffi/xuc_fusion/response_config.py`：现有各响应方法的调度差异。

新算法的定义、数值容差与实验次序为本计划提出的设计。没有声称已获得新的RFFI结果，亦没有将局部线性方程的正定性解释为全网络全局收敛。
