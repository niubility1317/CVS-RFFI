# Stage2功能研发目标与证据门

状态：`ACTIVE_GOAL_REFACTOR / FUNCTION_FIRST / TARGET_NOT_OPENED`

## 1.最终目标

本轮研发必须在`p2_min_v1`下形成一个同时包含共享域适应和qKNN分类头的Stage2-C方法。最终候选在单seed的25个Target job上必须满足：

|slice|注册后旧类准确率|最低旧类准确率|新类准确率|
|---|---:|---:|---:|
|K10/new5|≥92%|≥85%|≥92%|
|K10/new10|≥92%|≥85%|≥90%|
|K10/new20|≥92%|≥85%|≥86%|

K5/new20相对matched K10/new20的注册后旧类、最低旧类、新类和`H_old_new`下降均不得超过5pp。K1/new20必须相对同row冻结D92基线产生真实提升，不能依靠identity、整臂fallback或未改变prediction通过。

中间门只筛掉没有功能作用或协议不合法的实现，不降低、替换或重新解释上述最终指标。

## 2.证据起点

|方法|已验证结论|对新研发的约束|
|---|---|---|
|D62|完整125；注册后仅24/375个场景状态实际激活，K1整体fallback|禁止用离散安全门把大多数row退回基线|
|D91|仅K10/new5 development；与D62的15/15 outer prediction相同|support目标下降或内部几何变化不等于分类功能|
|D92|K10/new20旧类+2.622pp、floor+4.600pp，但新类−0.653pp；K1逐值不变|域适应与注册必须同等优化；K1不能依赖类内协方差|
|SVRN/r4.2|完整125且相对D62全面劣化|不再使用会放大注册后旧类崩塌的分支状态|
|D104|量化机制和release代码闭合，无Target性能|ANGQ只能作为实现组件，不能预设为最终分类头|

## 3.指标与同row口径

原子scenario-row固定为：

```text
(receiver, seed, K, new_count, scenario,
 capsule_id, split_id, query_id_root, method_lock)
```

四臂和基线必须共享全部字段、同一old query、同一new query和同一独立scorer。每个正式slice由同seed的5个receiver×3个互斥LEO场景组成15个scenario-row，并等权宏平均。

- `A_old`：注册后旧类准确率；
- `N`：已注册新类准确率；
- `H`：同一row的`A_old`与`N`调和均值；
- `F_old`：对每个旧类先聚合该slice全部15个scenario-row，再取最低类准确率；
- `forgetting=B_old-A_old`；
- 辅助报告：mean row-floor、worst row-floor、逐类、逐receiver、逐scene和正确数。

不得用不同row的边际最大值拼接结论。`F_old`是全部旧类的通用下界，不是预选弱类清单。

## 4.G0功能机制门

G0不读取性能truth，只回答实现是否具备可观测功能：

1.状态只读取不可变Phase1 bundle、当前row合法support和冻结配置；
2.query为零fit、零selection、零update，每条query独立面对全部注册类；
3.算法对类别标签置换等价，不含TX/class ID专用参数；
4.DA使用同一共享规则处理所有类；HEAD不读取ground bank，ground与target关系全部归入DA；
5.K1自由度必须来自跨类共享低维结构或Phase1预锁先验，不能估计单类方差或类专属adapter；
6.support physical-LOO/OOF只用于构造support-only状态，support准确率不能作为晋级依据；
7.no-query真实特征smoke必须改变feature、邻居、margin或argmax中的至少一项；
8.INT8、state、MAC、临时空间和序列化资源闭合；
9.若DA只是共同平移、正交变换或全局正尺度，且在变换后重建欧氏/余弦qKNN导致邻居排序不变，则直接证伪。

协议错误记为`INVALID / NO_PERFORMANCE_RESULT`；机制合法但没有可观测决策作用，记为`REJECT_REVISION_NO_FUNCTION`。

## 5.G1 source-held四臂因果门

冻结四臂：

|臂|DA|HEAD|
|---|---|---|
|M0|旧表示|旧头|
|M_DA|新共享DA|旧头|
|M_HEAD|旧表示|新support-only头|
|M_JOINT|同一新共享DA|同一新support-only头|

`M_DA/M_JOINT`必须复用同一DA state；`M_HEAD/M_JOINT`必须复用同一HEAD公式。HEAD不得直接使用ground prototype对query计分，否则会污染2×2归因。

对指标`y`定义：

```text
E_DA   = 0.5 * [(M_DA-M0) + (M_JOINT-M_HEAD)]
E_HEAD = 0.5 * [(M_HEAD-M0) + (M_JOINT-M_DA)]
I      = M_JOINT-M_DA-M_HEAD+M0
```

晋级条件：

- DA简单效应：Stage2-B的`M_DA−M0`必须满足旧类BA>0、`F_old`≥+1pp；Stage2-C还必须满足`N/H`均≥−0.5pp。联合后`M_JOINT−M_HEAD`必须同向复现；
- HEAD简单效应`M_HEAD−M0`和`M_JOINT−M_DA`方向一致，二者平均后满足`N`≥+1pp、`F_old`≥+1pp、`H`≥+0.5pp、`A_old`≥−0.5pp、`F_all`≥−0.5pp，并报告paired CI和negative tail；
- JOINT相对M0的`A_old/F_old/N/H`同向非负，old+new总正确数严格增加；
- 任一receiver或scene subgroup不得出现超过2pp的灾难性均值退化；
- `I`只报告协同或拮抗，不要求强制为正。

若某revision根据已经打开的source-held结果修改结构或超参数，该held立即降级为development。新revision必须使用尚未打开的source-held split，或明确跳过G1并承担直接进入Target25的更高风险。

## 6.G2单seed Target25门

候选、method lock和全部超参数在Target访问前冻结。矩阵固定：

```text
5 receivers × 1 seed ×
{K10/new5, K10/new10, K10/new20, K5/new20, K1/new20}
= 25 jobs
```

每个job覆盖3个物理ID互斥的`leo_*_weak`场景。一次Target25只评估一个revision，不得从25行中选择receiver、scene、class或slice重跑。

K10执行§1全部硬门。

K5以同receiver、同scene的K10/new20为matched基线。预登记时必须锁定`K5 support physical IDs⊂K10 support physical IDs`，且两者的`query_id_root`逐scene完全相同；只有满足该嵌套关系时，`A_old/F_old/N/H`下降≤5pp才属于paired结论。若不满足，只能报告非配对差值，不能用于通过本门。任一核心指标出现>15pp的单scenario-row灾难退化则不晋级。

K1以同row冻结D92为基线：

```text
ΔH >= +2pp
ΔF_old >= +2pp
ΔA_old >= 0
ΔN >= 0
old+new总正确数严格增加
```

单seed通过记为`TARGET25_SCREEN_PASS`，证明本轮研发目标在该预注册seed上达到；不能据此宣称多seed稳定。

## 7.G3顺序确认

需要形成可推广的`PROMOTABLE`声明时，保持同一method lock，再运行一个预注册fresh confirm seed的25-job实验，不合并成125大矩阵。screen seed和confirm seed必须各自单独重过G2，并在两seed池化后再次重算。只有当两次结论冲突或置信区间极不稳定时，才单独决定是否增加第三seed；第三seed不是默认硬流程。

确认门：

- K1的paired`ΔH/ΔF_old`95%CI下界>0；
- K5四项下降的95%CI上界≤5pp；
- confirm seed失败记为`SCREEN_POSITIVE_NOT_CONFIRMED`。

修改结构、rank、loss、step、量化、阈值或fallback会清零确认资格并回到G0。

## 8.研发工作包与模型分工

|工作包|职责|执行模型|
|---|---|---|
|WP-DA|ground压缩知识、target old/new support、共享非等距低维DA、K1可辨识性|`gpt-5.6-terra/max`|
|WP-HEAD|纯target-support qKNN、带宽/密度/不确定度校准、弱类floor和新类注册|`gpt-5.6-terra/max`|
|WP-DATA|目标、矩阵、同row指标、结果分析、拒绝语义和交叉审查|`gpt-5.6-sol/high`|
|WP-INTEGRATE|协议解释、方案冻结、代码整合、独立复审和最终决策|主agent，`gpt-5.6-sol/high`|

方法agent不得自我认证。WP-DATA必须审查K-shot可辨识性、common-transform cancellation、support proxy过拟合、旧/新任务平衡、类置换、资源和query/role/quota禁区。

## 9.拒绝语义

|状态|含义|
|---|---|
|`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|预测前系统性技术失败|
|`PARTIAL_DIAGNOSTIC_BIASED_NOT_PROMOTABLE`|只有partial prediction或score|
|`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`|完整矩阵完成但性能门失败|
|`TARGET25_SCREEN_PASS`|单seed25达到本轮硬目标|
|`SCREEN_POSITIVE_NOT_CONFIRMED`|单seed通过但确认seed失败|
|`PROMOTABLE`|G0–G3全部通过|

禁止按receiver、scene、class、seed挑选结果，禁止跨run拼接极值，禁止用source-held替代Target。

## 10.当前执行优先级

1.按已通过独立设计审查的`D105-CBRC-MB4`冻结共享DA，明确K1公式和ground bundle生命周期；
2.按已通过独立设计审查的`LPO-RC-qKNN`冻结不读取ground的纯support HEAD；
3.先实现两者最小可运行功能与2×2接口；
4.完成G0合成负测和真实no-query smoke；
5.再决定是否释放已预登记但尚未落地的D104 source-held run；
6.G1通过后只运行一个冻结revision的Target25。
