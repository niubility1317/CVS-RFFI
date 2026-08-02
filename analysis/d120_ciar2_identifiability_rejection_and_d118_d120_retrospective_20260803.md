# D120 CIAR-2可识别性裁决与D118-D120三轮回顾

状态：`REJECT_ALL / NO_STAGE2_CANDIDATE / NO_EXPERIMENT_RELEASE / NO_NEW_PERFORMANCE_RESULT`

## 1.结论

D120不发布G0，也不启动125矩阵。唯一值得进入原理推演的候选`CIAR-2`试图逆转二维IQ幅相失衡；但正式`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`场景均明确设置`enable_iq_imbalance=False`，该干预分支根本不执行。候选在当前正式场景上的目标变换因此是identity，无法构成非恒等域适应方法，故在实现前关闭。

该结论不是“CIAR-2性能差”，而是“候选针对的干预不属于当前正式场景”。没有实现、没有Phase1 falsifier、没有N607发布、没有Target truth和性能结果。

## 2.唯一推演候选

广义线性IQ-image模型写为

\[
y=\alpha x+\beta x^*,
\]

其中二维实状态`ρ`编码幅度与相位失衡。若同一row内`ρ`共享，则可用六类old support相对随checkpoint联合封存的Phase1类锚和两列函数Jacobian，解一个固定`2×2`正规方程；随后冻结`ρ`，以同一逆变换重编码old/new support和每条query。该反事实机制具有以下性质：

- K1仍有六个独立物理support，每个类贡献160维残差，不依赖同IQ多view增加K；
- fit只读old support，new support不重估状态，query零fit、零update、零selection；
- Phase1数值payload设计上约2.9KiB；每条记录额外计算不超过约2.6k实MAC，外加原模型forward；
- 作用于encoder之前，函数上不同于D62/D91的Fisher或row score、D92的协方差、D102的pre-ReLU MetaBias，以及D106/SVRN/qKNN后端规则。

这些性质只说明“若正式场景启用并共享该状态则可构造”，不能证明该干预在当前场景真实存在。

## 3.一票否决事实

`code/sat_channel.py`第280-286行只在`enable_iq_imbalance=True`时执行IQ幅相失衡；类默认值不能覆盖正式scenario配置。`code/training_controls.py`中的实际三场景分别在第156、179和202行固定：

- `leo_clear_weak.enable_iq_imbalance=False`；
- `leo_low_elev_weak.enable_iq_imbalance=False`；
- `leo_rain_weak.enable_iq_imbalance=False`。

因此`CIAR-2`在正式received-IQ上没有一个由场景生成器施加的IQ-image状态可逆，所有针对该分支的Jacobian、锚和闭式fit都只是在为未发生的干预建立模型。若把候选改称为纠正原始物理receiver硬件失衡，则需要重新证明该状态确实存在、跨support/query共享并且不与TX指纹混杂；当前没有这种独立观测，不能把方法名称替代证据。

D119又确认现有Phase1 loader、source archive、manifest和NPZ没有独立CFO或接收机校准真值，不能把外部可测receiver telemetry作为另一条识别路径。由同一IQ临时估计IQ imbalance再充当独立truth属于循环证据，禁止采用。

最终裁决：`REJECT_CIAR2_ABSENT_FORMAL_INTERVENTION / STAGE2_CLOSED / NO_TARGET_RESULT`。

最终独立复审：理论作者修正后确认`D120_THEORY_REJECT_ALL`；反方复审`MERGE，P0=0，P1=0`。复审明确区分“正式模拟场景未启用该干预”和“真实物理硬件绝不存在该失衡”，本报告只主张前者。

## 4.与既有方法空间的去重

|函数空间|既有路线|D120裁决|
|---|---|---|
|共同平移或球面运动|D112、D113|已覆盖，不能包装成新DA|
|末端非等距feature变换|D93、D94、D110|落入PSD/transport族|
|协方差、带宽或不确定性|D92、D110、D114、SVRN|已覆盖，不能仅换闭式估计器名称|
|support score或head重构|D62、D91、D106、qKNN|属于分类头/评分族，不是新的前端域状态|
|早层参数干预|D102、D119 GN-ISF|无独立状态证据时不可识别；D119已关闭|
|前端IQ-image逆|D120 CIAR-2|反事实下函数非等价，但正式三场景未启用该干预|

域映射在观测下不唯一时，增加闭式公式、rank或矩阵规模不能恢复可识别性。该边界与域适应可识别性文献对spurious maps的结论一致：[Identifiability Conditions for Domain Adaptation](https://proceedings.mlr.press/v162/gulrajani22a.html)。

## 5.D118-D120三轮回顾

|轮次|候选与目标|终态|是否发布真实实验|停止原因|
|---|---|---|---|---|
|D118|轻型快速DA可识别性前沿|无可立即发布的非恒等Stage2候选|否|现有合法观测不足以识别新的共享域状态|
|D119|`GN-ISF-48/r1`早层GN干预|`REJECT_GN_ISF_UNVERIFIABLE_CONFOUND`|否|预注册CFO混杂检验所需独立数值元数据不存在|
|D120|`CIAR-2`前端IQ-image逆|`REJECT_CIAR2_ABSENT_FORMAL_INTERVENTION`|否|正式三种`leo_*_weak`场景均禁用IQ imbalance|

三轮都没有产生完整预测，因此均没有post-registration `seen_new_acc`、`H_old_new`、逐类old accuracy或forgetting证据，不能进入性能排序或宣传。协议边界保持不变：仅`leo_*_weak`、无clean访问、无query truth/role/class quota/global reassignment。

共同教训是：目前瓶颈不在求解器速度，也不在实验矩阵规模，而在“候选所针对的干预是否真实属于正式场景，以及Stage2可观测量是否包含一个support与query共享且不与TX身份混杂的域状态”。在这两个问题同时得到肯定答案前，继续轮换早层、rank、正则或闭式估计器只会重复D118-D120。

## 6.停止与重开条件

本轮停止，不进入D121实现或实验。只有出现以下新增且部署合法的观测之一，才重新打开前端轻量校准族：

1.正式协议明确启用或观测到可验证的IQ幅相失衡；
2.同一TX/pilot跨receiver的物理配对；
3.每个record都具备可信、独立、部署时可用的receiver telemetry；
4.协议确认该接收机校准状态在support/query之间共享。

重开后最小Phase1 falsifier固定为receiver-held、class-LOCO、`rank(I)=2`、TX泄漏置换检验，以及相对D102和固定PSD函数span的held残差。任一失败即关闭；不得先跑G0或Target25再倒推门槛。
