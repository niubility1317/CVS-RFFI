# APSTA-P1时间融合稳健适配设计

## 目标

基于冻结`ADV3B02_CORE90_SOFT_E200`checkpoint，在`p2_min_v1`Stage2-B中只使用固定LEO received IQ、合法旧类target support标签、冻结类原型/映射和预登记配置，真实反向更新`id_backbone.t3+t_proj+fuse`，验证冻结特征空间是否为CAPTA-P0的主要瓶颈。

## 协议收敛

复盘建议的rank-4目标CosFace头、temperature和class bias与`项目.md`5.3.1“冻结类原型不得扩展为可训练判决状态或持久分类头”冲突，也与用户此前“不要D92式分类头”冲突。本轮因此使用协议安全候选`APSTA_P1_TIME_FUSION_ROBUST`：source CosFace头和ground prototypes始终冻结，只有原编码器三个非分类块可训练。该实现是报告的协议安全近似，不是`CAPTA_P1_TIME_HEAD_R4_ROBUST`严格复现。

## 输入与状态边界

- 适配API只接收model、target support received IQ、support标签、冻结类原型/映射、四句柄context和预登记config。
- 不存在source/clean/replay/cache/query/truth/role/quota输入面。
- 先完成support训练与support-only状态选择，再冻结全部参数，最后才能打开query。
- 每个query独立推理；query不得更新参数、buffer、checkpoint选择、门控或早停状态。
- 冻结教师路径保留为step-0候选和query审计分数；正式prediction使用support-only规则选出的单一安全状态。

## 可训练结构与资源

- 候选：`APSTA_P1_TIME_FUSION_ROBUST`。
- 可训练前缀：`id_backbone.t3.`、`id_backbone.t_proj.`、`id_backbone.fuse.`。
- source CosFace头、domain branch、频率分支、ground prototypes及其余checkpoint状态冻结。
- 记录真实训练参数、比例、结构参数、改变参数、非选择参数和buffer变化；参数比例只报告，不作为代码硬失败门。
- 优化器：AdamW，`lr=2e-4`，`weight_decay=0`。

## 训练目标

设归一化学生特征为`z_i`、冻结原型为`P_s`、标签为`y_i`。

1. 冻结头support CE：使用原CosFace头对target support监督，保持最终冻结判决路径可用。
2. 可微LOO：每个样本对应类别原型从`kappa*P_s`与同类其余support特征构成，严格排除自身；计算LOO CE。
3. worst-class tail：先求每类LOO CE，再用温度化`logmeanexp`强调最弱类。
4. topology：由全部support构造的锚定类中心Gram矩阵接近冻结原型Gram矩阵。
5. L2-SP：选择参数相对checkpoint初值的平方漂移。
6. 删除逐样本冻结特征MSE，不把学生特征强制拉回冻结特征。

预登记默认：`anchor_strength=3.0`、`head_ce_weight=0.25`、`loo_mean_weight=1.0`、`tail_weight=0.5`、`tail_temperature=0.5`、`topology_weight=0.25`、`l2sp_weight=1e-3`。

物理一致性在本轮`deferred`：复盘要求增强范围必须经地面校准，但未提供与当前checkpoint绑定的合法校准参数；本轮不自行发明CFO/IQ失衡范围。

## 检查点与安全选择

- 固定检查点：`0,10,30,100,300`，最大训练300步。
- 每个检查点只在support上计算robust LOO risk、最差类LOO margin、topology drift和参数漂移。
- step 0为冻结教师基线。
- 适配checkpoint只有同时满足`robust_risk<=step0`且`worst_class_margin>=step0-epsilon`才进入安全集合。
- 安全集合按`robust_risk`升序、`worst_class_margin`降序、`topology_drift`升序、步数升序选择；若无适配checkpoint满足条件，回退step 0。
- `epsilon=0`，不允许以support最差类退化换平均风险改善。

## Prediction与truth-last诊断

prediction保存selected student scores、冻结teacher scores、checkpoint选择审计、loss trace与资源审计，不保存truth/role。独立scorer之后连接truth，生成同row`DA0_REG0/DA1_REG0`指标。

最终诊断汇总至少包括：每row/每类A0与DA1 accuracy、selected step、support robust risk/margin、teacher→student翻转、teacher/student disagreement、每场景mean/floor和全矩阵mean/floor。oracle gate属于truth-last分析，不能反馈predictor或触发重跑。

## 最小实验

- run ID：`adv3b02_stage2b_apsta_p1_t5_s713101_20260824_v1`。
- 单seed=`713101`；5个receiver×3个LEO场景；`K5/new20`；15个同row。
- A0复用既有冻结prediction；DA1只运行本候选。
- 晋级：DA1相对A0旧类等权均值至少`+1.0pp`且全矩阵旧类floor至少`+0.5pp`，同时协议与资源审计通过。
- 未达门槛记为`SCIENTIFIC_FAILURE_NO_PROMOTION`，本轮不启动Target25、多seed或第二个频率候选。

## 非本轮范围

- `rejected`：可训练/持久低秩CosFace头、temperature/class bias、D92式协方差/LDA、多域类条件source原型包。
- `deferred`：物理一致性、低秩metric、SSF、频率专家、20%/30%梯度选择、identity backbone上界、地面episodic元训练、class/sample gate。
- 以上项目不得成为首个P1候选的发布阻断项。
