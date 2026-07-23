# ADV3B02-TS-DRQKNN-BCRR/r1选择性吸收与DESIGN_FROZEN（已被r2-affine替代）

> 终态：`SUPERSEDED_TECHNICAL_REVISION / NO_PERFORMANCE_RESULT`。r1在实现终审和真实checkpoint support-only检查中暴露after teacher自指、125无系统故障停派及共享对称INT8门失败；后续唯一活动设计见`docs/ADV3B02_TS_DRQKNN_BCRR_R2_AFFINE_DESIGN_FROZEN.md`。本文件只保留历史设计，不再授权实现或发布。

## 1.身份与裁决

- candidate：`ADV3B02-TS-DRQKNN-BCRR/r1`
- 状态：`DESIGN_DRAFT -> FEASIBILITY_REVIEW -> DESIGN_FROZEN`
- 监督裁决：`MERGE / P0=0 / P1=0`
- 外部设计输入：`C:\Users\lh594\Downloads\ADV3B02_双分支双注册qKNN快速适应设计报告_20260723.md`
- 输入SHA256：`cfe29eb87519c7582a2822ddbd2c8d2e80c363bede471d06b8c310c50a5a42a1`
- 吸收原则：只保留与`p2_min_v1`、现有qKNN和已有正证据相容的机制，不做整份报告的全面实现。
- matched reference：当前DSSC完整125结果完成后作为普通reference；不得根据其target query结果修改本设计。

## 2.冻结候选

### 2.1三个互补部分

1.域适应：用Stage2-B target-old support在`z_dom`中拟合TX抑制的低秩类内域邻域，Stage2-C冻结旧状态并append新类。
2.qKNN：`z_id`INT8 Student-t qKNN始终承担全部注册旧类和新类的统一逐query跨类竞争。
3.OTHER：只复用连续BCRR，针对qKNN剩余的类级分数残差、old/new竞争和floor；不加入第二head。

`z_dom`不得直接产生跨类logit。它只重加权每个候选类内部的`z_id`邻域证据，因此DA与统一分类的职责可分离；BCRR不读取`z_dom`，用于修正双qKNN仍未解决的残差。

### 2.2合法输入与状态

- Phase1：SHA绑定的ADV3B02 checkpoint，使用strict `exact_state_dict_rebuild`。
- 双特征前向：必须调用`code/cvsrffi/dual_feature_forward.py:79-133`的head-bypass路径；不得调用会执行`dom_head/adv_head/tx_adv_head`的aux forward。
- Stage2-B：只读当前row target-old support IQ、标签、physical ID和注册表；拟合后冻结`Q/A/alpha`、旧类`mu_c`及旧bank。
- Stage2-C：只追加新类`z_id`codes、`mu_c`和冻结投影下的`z_dom`残差codes；不得重拟合旧状态。
- query：只逐样本前向并读取冻结state；不得进入fit、可靠度、温度、fallback、BCRR、量化或任何选择。
- 不读取clean/source样本、target receiver/TX标签、query truth、old/new角色、类别quota或跨query状态。
- GEOFF/r8 dual archive只可作为Phase1设计/lock证据，不得在row predictor打开或作为Phase2输入。

## 3.TX抑制双qKNN

令`u=L2(z_id)`、`v=L2(z_dom)`。Stage2-B对每个旧类`c`计算：

```text
mu_c = mean_i(v_i | y_i=c)
mu   = mean_c(mu_c)
S_W  = mean_c class_scatter(v_i-mu_c)
S_B  = mean_c (mu_c-mu)(mu_c-mu)^T
G    = sym(S_W-S_B)
```

`G`最多保留2个方向`Q=[q_1,q_2]`。固定2槽可靠度为：

```text
eps   = max(1e-8, 1e-6*trace(S_W)/160)
rho_j = [q_j^T(S_W-S_B)q_j / (q_j^T(S_W+S_B)q_j+eps)]_+
A     = diag(sqrt(rho)) * (Q^T S_W Q + eps I)^(-1/2)
alpha = 1[K>1] * 0.5*(K-1)/K*(rho_1+rho_2)/2
```

负方向、缺失槽、零方差或非有限状态令相应`rho=0`；若整体不可用则`alpha=0`并精确identity。固定2槽避免active-rank从2变1时的不连续，且`0<=alpha<0.5`。

对每个候选类`c`使用同一类中心计算support与query域坐标：

```text
r_i      = A Q^T(v_i-mu_c)
r_{q|c}  = A Q^T(v_q-mu_c)
pi_{qi|c}= softmax_i(r_{q|c}^T r_i)
w_{qi|c} = (1-alpha)/K + alpha*pi_{qi|c}
```

禁止原草案的“support减`mu_c`、query减全局`mu`”混合原点。query对每个候选类分别使用其`mu_c`，这是标签置换等价的全类计算，不读取query真值。

最终dual score复用基础qKNN的同一INT8 `z_id`bank、类带宽`h_c`、自由度`nu`和Student-t kernel：

```text
S_c^dual(q) = logsumexp_i(log_kernel_id(q,i)+log w_{qi|c})
```

当`alpha=0`时，逐值等于基础类归一化`logsumexp(log_kernel)-log K`，不仅argmax相同。

## 4.BCRR与四臂

|arm|冻结决策|
|---|---|
|`M0`|基础`z_id`INT8 Student-t qKNN|
|`M_DA`|TX抑制`z_dom`类内条件化＋同一`z_id`qKNN|
|`M_OTHER`|基础`z_id`qKNN＋BCRR|
|`M_JOINT`|双qKNN＋BCRR|

- M0/M_OTHER逐字节共享raw qKNN state；M_DA/M_JOINT逐字节共享dual qKNN state。
- BCR codes与数值类权重由同一`z_id`support生成并共享。
- `omega_raw`和`omega_dual`分别用各自同步physical-ID LOO logits按完全相同的预锁安全规则拟合。
- BCRR不直接读取`z_dom`，不得用一支gate决定另一支是否启用；receipt封存两套LOO SHA、`omega`及fallback原因。

主协同量：

```text
I_syn = H(M_JOINT)-H(M_DA)-H(M_OTHER)+H(M0)
```

## 5.K1/K5/K10与决策几何

- K1：类内散度不可辨识，`Q`为空或`alpha=0`；精确满足`M_DA=M0`、`M_JOINT=M_OTHER`。
- K5：6个旧类提供可用类内自由度，rank固定不超过2，是首个正式DA falsifier。
- K10：按完全相同公式确认，不增加rank、kernel、阈值或fallback。
- 方法不是所有类共同平移或正交变换；`z_dom`改变同一候选类内部不同`z_id`support样本的证据权重，必须审计domain neighbor、identity contribution、margin、argmax及净正确决策。

## 6.选择性吸收追踪

|设计报告元素|裁定|本revision处理|
|---|---|---|
|ADV3B02冻结双分支|采用|同SHA checkpoint head-bypass输出`z_id/z_dom`|
|Identity/Domain双注册|采用并收缩|最终跨类决策只由`z_id`qKNN完成|
|target-old域状态冻结、新类append|采用|Stage2-C不得重拟合旧状态|
|INT8 support bank、类归一化|采用|复用既有Student-t qKNN合同|
|K1关闭域条件化|采用|逐值identity|
|直接双余弦融合|拒绝|原始`z_dom`存在TX泄漏|
|硬membership/兼容性门|拒绝|避免D62式低coverage和大面积fallback|
|低margin选择性domain rescue|延期|会增加阈值并降低实际覆盖|
|共享身份残差或domain→ID transport|拒绝|是第二主要机制且有D93/D94负证据|
|Phase1双episodic重训|延期|需要新checkpoint，不能与本delta同时归因|
|地面原型直接投票/max-old重排|拒绝|容易重演old/new交换|
|Stage2轻量optimizer/adapter|拒绝|本revision固定0参数、0step闭式更新|
|连续BCRR|采用|已有独立净正确`+78`，只作OTHER|
|DSSC|保留为reference|不与本revision的DA臂混塞|

追踪状态：采用5项、采用并修订3项、延期2项、拒绝4项。未列出的设计报告细节不自动进入实现。

## 7.资源与生命周期

- 参数：0；optimizer step：0。
- 当前锁定MAC：ID-only backbone约`9,927,476/sample`；full dual forward约`38,890,840/sample`；双分支增量约`28,963,364/sample`。
- rank2域qKNN在`C=26,K=10`时额外约840 MAC/query。
- 增量域state预计小于4KiB；identity qKNN、dual state与BCRR总state目标小于128KiB，硬门256KiB。
- 全部query执行双分支，不使用选择性rescue；必须实测build/predict mean、P95、VRAM和最终wire bytes。
- INT8 teacher使用support FP64未量化状态；deployment必须从序列化bytes反解并满足top1一致率不低于99.5%、large-margin flip为0。

## 8.立即证伪与完整125

本地停止条件：checkpoint不能head-bypass输出双特征；K5/K10均rank0或`alpha=0`；类名/support顺序置换改变结果；Stage2-C修改旧状态；INT8或资源门失败；domain/adversarial head被调用。

完整125停止条件：

1.K5/K10的M_DA净正确不为正，或old/new任一净变化为负；
2.M_DA只改变`z_dom`或logit，几乎不改变identity contribution、margin或argmax；
3.M_OTHER不再有独立正收益；
4.`H(M_JOINT)<=max(H(M_DA),H(M_OTHER))`或mean`I_syn<=0`；
5.正协同不足188/375个scene slice或不足2/3个scene均值为正；
6.JOINT损害old-before、old-after、old gain、seen-new、BA、floor、min-old或min-new；
7.JOINT增加forgetting、old→new或new→old；
8.收益主要来自单个receiver、scene、K或seed；
9.协议、INT8、state、MAC、时延或显存门失败。

正式矩阵固定为`5 receivers×5 seeds×5 slices=125 jobs`，每job覆盖3个LEO弱场景；四臂闭合`375 scene slices/1500 score rows/1000 arm-state prediction artifacts`。不发布N607窄性能实验，不用125调参。

## 9.冻结改动范围

只新增：

1.`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`
2.`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`
3.`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`

不得修改现有模型、qKNN、BCRR、数据builder、GEOFF/r8、coverage、authority或scorer。正式实现必须包含`dom_head/adv_head/tx_adv_head`替换为raise仍能完成双特征前向的负例，并在receipt封存checkpoint SHA、feature keys/dims及`heads_called=0`。
