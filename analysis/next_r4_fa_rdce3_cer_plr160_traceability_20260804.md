# NEXT-R4 FA-RDCE3×CER-PLR160设计与实现追踪

状态：`DESIGN_FROZEN / IMPLEMENTING / NO_PERFORMANCE_RESULT`

## 1.问题与证据起点

近期真实证据只有D122中RDCE的source-held小幅正信号：`ΔH=+0.4447pp`、每行总正确数`+0.9206`、`Δall-floor=+0.4438pp`。D130 Lite160虽然显著降算力，但held-proxy和floor下降。NEXT-R3的TSL在真实physical-LOO中因球形参考头没有正确margin而无法校准信赖半径，未产生prediction或性能结果，路线已关闭。

NEXT-R4必须同时回答两个问题：用K-shot support得到可辨识的轻型域适应状态；在不重现D130尾部损伤的前提下简化D92头。不得使用query truth、role、正确率、quota、跨query重排或超参数搜索。

## 2.联合候选

### 2.1 FA-RDCE3

Phase1与checkpoint共同封存RDCE基`B∈R^(3×160)`、类聚合3维中心`m_c=Bμ_c`、跨类公共Fisher精度`D_F`、公共残差方差`D_v`、公共半径`ρ`和量化尺度。只允许INT8多样本聚合资产，不允许Phase1逐样本、LOO或source cache在Phase2重现。

REG0旧类support给出：

```text
r_ci = B z_ci - m_c
a = (D_F + C K D_v^-1)^-1 D_v^-1 Σ_cΣ_i r_ci
a <- a * min(1, rho / sqrt(a^T D_F a))
```

所有类对残差等权，最终只有一个跨类共享3维`a`。表示顺序固定为`R0 canonical z→z-B^Ta→一次RDCE S_kappa→R1 signed unit`。R1输出后禁止再位移、ReLU或L2归一化。DA1_REG1逐字节复用DA1_REG0的`a/κ`，不得用新增类support重拟合DA；新增类support/query只消费同一个已冻结变换。

K1不估计协方差、旋转或网络层参数；若后验为零或Fisher方向未定义，确定性取`a=0`并保留固定RDCE，不从query或support正确率选择回退。新增动态状态6B；拟合上界`C K×3×160 MAC`，C=6时K1约2,880、K5约14,400MAC；RDCE query约960MAC。

### 2.2 CER-PLR160

qKNN始终是分类基座。K1的H逐logit精确alias Q，不宣称head收益。K5只从全部注册类等权support估计共享对角形状：

```text
mu_c = K^-1 Σ_k u_ck
s_j^2 = C^-1 Σ_c (K-1)^-1 Σ_k (u_ckj-mu_cj)^2
v_bar = mean_j s_j^2
lambda = C(K-1)/(C(K-1)+160)
D = diag([lambda s_j^2 + (1-lambda)v_bar + eps_v]^-1)
D0 = (v_bar+eps_v)^-1 I
```

由`D-D0`生成prototype affine residual并在类别维中心化，去除自由类偏置。公共连续缩放固定为：

```text
gamma = Sq / (Sq + 4 Sr + 64 eps32 max(1,Sq,Sr))
logit = qKNN_logit + gamma * centered_shape_residual
```

`Sq`只由support qKNN bank的`nu,d_eff,h_c`解析得到。`Sr²=[C(C−1)]^-1Σ_cΣ_{a≠c}[r_c(mu_c)−r_a(mu_c)]²`，即每个类原型上自身类与其它类的残差pairwise gap RMS；不得替换为包含对角项的全残差矩阵RMS。没有LOO、support/query正确率、top-k、role或候选选择，并满足`gamma Sr≤Sq/4`。`Sr=0`或量化残差全零时记`NO_HEAD_FUNCTION`并精确alias Q；非有限fail-closed；最高logit精确并列统一`TIE_UNRESOLVED`。

部署wire固定为`INT8 W[C,160]+FP16 scale[C]+FP16 intercept[C]`，共`164C B`，C=6时984B；无`d^2`协方差或分解。相对既有Q的增量query为`Cd=960MAC`。

## 3.统一四态与最小矩阵

|状态码|唯一中文名称|适应状态|注册状态|
|---|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|R0|旧类|
|`DA1_REG0`|域适应后/新类注册前|FA-RDCE3|旧类|
|`DA0_REG1`|域适应前/新类注册后|R0|旧类＋新类|
|`DA1_REG1`|域适应后/新类注册后|复用DA1_REG0|旧类＋新类|

矩阵为`2 receiver(1-1,18-2)×6 held-class×K1/K5=24`逻辑行。K1每行4个Q唯一prediction，H只保存alias receipt；K5每行4状态×Q/H=8个唯一prediction。全矩阵为144个唯一prediction、192个含alias的arm artifact。K1 support是K5逐类前缀；K、四态和Q/H必须复用同一query physical/observation ID及相同顺序。

主比较为：`DA1_REG0−DA0_REG0`、`DA1_REG1−DA0_REG1`、每个DA状态内的`REG1−REG0`、K5的`H−Q`和`[(DA1_H−DA1_Q)−(DA0_H−DA0_Q)]`。

## 4.性能裁决边界

只有完整144 prediction/192 artifact封存并由独立score打开truth后才裁决：

- 保留：域适应后/新类注册前old BA至少`+0.25pp`；域适应后/新类注册后H至少`+0.25pp`；K5的H-Q同时使H至少`+0.25pp`且总正确数增加；seen-new和all-floor非降；4个receiver×K层至少3个H非负。
- 关闭FA：DA1_REG1−DA0_REG1的H不正，或任何主效应伴随all-floor≤`−0.5pp`。
- 关闭CER：K5 H-Q未同时提高H与总正确数，或在R0/R1两种表示下均降低old-floor。
- 关闭联合：DA1_REG1(H)−DA0_REG1(Q)≤0、总正确数不增，或任一receiver在K1/K5平均后H≤`−0.25pp`。0至`+0.25pp`的微弱正值也不授权调参。

这些阈值不是运行中健康早停；技术停止仍只允许协议/安全违规或预注册的重复确定性异常。

## 5.实现所有权与追踪

|职责|实现文件/状态|验证|
|---|---|---|
|FA-RDCE3科学核心|`code/cvsrffi/stage2_next_r4_fa_rdce3.py`，已实现，待独立P0/P1审查|wire/类置换/共同状态/R1边界/资源负测；聚焦测试6项通过|
|CER-PLR160科学核心|`code/cvsrffi/stage2_next_r4_cer_plr160.py`，已实现并关闭首轮2项P1|K1 alias/K5 shrinkage/pairwise Sr/gamma/量化/tie/资源负测；R0/R1只读表示验证与alias前query闭合经独立复审，聚焦测试6项通过|
|矩阵与method lock|`code/cvsrffi/stage2_next_r4_matrix.py`、`configs/next_r4_fa_rdce3_cer_plr160_proxy24_20260804.json`，已实现|24/144/192、共同query、K前缀、state复用；移除R3固定query/Phase1数量gate后聚焦测试6项通过|
|四态runtime|`code/cvsrffi/stage2_next_r4_runtime.py`，Terra/max按冻结接口实现中|四态显式`state_id`和中文主名称；DA1_REG1复用DA1_REG0状态；K1 H逐logit alias Q；K5 H=CER；query零fit/update/selection；REG0新类/H=`N/A`|
|artifact/scorer/CLI|`code/cvsrffi/stage2_next_r4_score.py`已实现；artifact/CLI待runtime后接合|truth分离、不可覆盖、144 prediction/192 artifact闭合、同row四态score、中文主名称和REG0的`N/A`；scorer聚焦测试5项通过|
|N607发布|唯一Luna/max runner|Git/hash、真实checkpoint smoke、24行完整artifact|

当前文件只冻结设计和可验证映射，不构成性能结果。

## 6.逐项验收台账

|ID|来源章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|R4-DA-01|2.1|REG0旧类support只拟合一个共享3维FA状态；DA1_REG1逐字节复用|`stage2_next_r4_fa_rdce3.py`|verified|聚焦测试6项；state wire/reuse receipt|不含query或新类重拟合|
|R4-HEAD-01|2.2|K1 H精确alias Q；K5 CER使用冻结pairwise Sr与164C B wire|`stage2_next_r4_cer_plr160.py`|verified|聚焦测试6项；独立复审P0=0、对应P1关闭|不恢复Full160稠密组件|
|R4-MAT-01|3|24逻辑行、144 unique prediction、192 arm artifact、K前缀和共同query|`stage2_next_r4_matrix.py`、配置JSON|verified|聚焦测试6项|已移除R3固定query/Phase1数量gate|
|R4-NAME-01|3|所有输出显式使用四态码与中文主名称；REG0新类/H=`N/A`|目标文档、matrix、scorer、报告|verified|目标提交`45e148c1`；scorer测试|禁止独立before/after|
|R4-RUNTIME-01|2.1、2.2、3|truth-free四态runtime；R1后不再ReLU/L2/位移；query零fit/update/selection|`stage2_next_r4_runtime.py`|pending|待聚焦runtime测试|当前最高风险：public qKNN会二次normalize|
|R4-ART-01|3、4|只投影预测与receipt，不携带truth/features/logits；严格闭合scorer单一schema|`stage2_next_r4_artifact.py`|pending|待artifact+scorer测试|机械实现由Luna/max负责|
|R4-SCORE-01|3、4|完整预测闭合后才开truth；输出same-row四态、per-class old、forgetting及比较|`stage2_next_r4_score.py`|verified|聚焦测试5项；与matrix合计11项|不做promotion决策|
|R4-CLI-01|5|prepare→predict→score三阶段、不可覆盖、动态metadata、真实checkpoint无query smoke|`code/scripts/run_next_r4_proxy24.py`|pending|待CLI聚焦测试与真实checkpoint smoke|不继承R3的588/14固定gate|
|R4-REV-01|5|独立科学复审P0=0/P1=0|上述全部实现|pending|待Terra/max复审|P2缓存优化不阻塞发布|
|R4-RUN-01|5|唯一Luna/max runner完成N607 preflight、同步、启动、监控和artifact回收|实验报告与远端不可变run root|pending|待Git冻结与preflight|不得按性能中止|
|R4-AN-01|4|主agent按四态和同row证据完成性能裁决|实验报告|pending|待完整score artifact|弱/负即关闭，不调参|
