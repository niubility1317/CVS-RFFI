# Stage2域适应与分类头联合研发目标

版本：2026-07-22
状态：可直接作为新`/goal`目标Prompt
协议：`protocol_schema=p2_min_v1`
初始化文档：`docs/STAGE2_RESEARCH_AGENT_INIT.md`

## 1. 单一目标

在`E:\type10-7`中，严格遵循当前`AGENTS.md`、`项目.md`和`p2_min_v1`，基于ADV3B02 final checkpoint持续研发、实现并验证可逐样本部署的极轻型Phase2方法。方法必须同时包含：

1. 显式域适应：利用合法target support、`z_id=feat_joint`、`z_dom=feat_imp`及与checkpoint共同封存的Phase1聚合知识，校正目标接收机与LEO弱信道偏移；
2. 统一分类头：在全部实际注册旧类与新类中，联合利用局部qKNN证据和全局Shrinkage RDA/SRDA证据；
3. 联合协同：域适应与分类头必须分别产生独立正收益，联合方法还必须证明正交于简单相加的协同增益，即`1+1>2`。

工作重心放在方法设计、最小因果实现、Phase1 LODO、锁定窄实验和真实N607证据。不得把大部分时间消耗在重复数据验证、authority/hash重建、跨run原始SHA对齐、报告格式重构或无关文献扩展上。普通负结果不是停止理由；只有新增数据权限、改变科学场景、干预用户现有任务或其他高影响动作才请求用户授权。

## 2. 当前研发起点

不得从头重造方法。沿用以下已审查代码链：

- `A`：Patch A，纯`z_id160` identity Student-t single-qKNN；
- `B`：Patch B，在A不变的前提下增加raw-`z_id` Shrinkage RDA/SRDA全局头；
- `C-id`：Patch C-id v0，K≥2从合法support估计rank≤2的类内—类间保护nuisance metric；K1严格identity；
- `C-dom`：下一主研发项。新Phase1共同封存bundle提供`z_dom→z_id`低秩cross-map和receiver nuisance basis；Phase2由全部注册类support估计类无关receiver context；
- `C-joint`：仅在C-id和C-dom分别超过A后，联合去除共享receiver偏移与剩余sample/channel扰动；
- `D`：最佳C的局部qKNN expert＋与B完全相同的raw-`z_id` SRDA expert，使用Phase1锁定或合法support-only可靠度融合。

已有证据必须作为机制边界，而不是重复探索：

- D92证明注册均衡协方差头可改善old/floor，但会牺牲new；分类头有价值，却没有解决域偏移；
- D93/D94在ground coverage仅约`0.144–0.227`时搬动整体坐标，support达到100%仍使held query退化；不得再做全坐标transport或以support-fit自证；
- D99在Phase1 LODO出现正信号，但真实target K10相对D81的old-after、seen-new、H和floor均下降约24pp；不得把Phase1代理等同target泛化，也不得一次混改metric、kernel和fusion；
- D100的有效融合系数为0，预测与D99相同；增加第二个头不自动产生互补；
- Role-Oracle只证明跨角色竞争存在较大上限，不得作为协议合法方法、参数选择信号或晋级证据。

## 3. 主方法：Coverage-Coupled Cross-Branch DA＋SRDA

### 3.1 C-dom共享receiver-context适应

Phase1仅从合法single-observation source archive学习并封存：

- `z_dom`中的receiver/domain basis`U_R^dom`；
- rank≤4的`z_dom→z_id`cross-map`W_dom→id`；
- 中心、尺度、奇异值、`D_eff`、LODO/LOCO稳定性和coverage证书；
- Phase1锁定的收缩、rank和identity回退规则。

`U_R^dom`、`W_dom→id`及其coverage证书必须作为int8多物理样本聚合Phase1知识共同封存，或由该int8聚合知识确定性重建；不得保存sample-level feature、成员ID、归属清单、可替换FP16/FP32 sidecar或target访问后生成的Phase1组件。

Phase2只用当前row全部已注册旧类与新类support，按类别平衡估计共享receiver context`r_t`。K1只能估计该跨类共享context，不能估计类内scatter或sample-channel basis。校正采用低秩、非正交、强收缩残差：

```text
z_id' = z_id - q_dom * W_dom→id * r_t
```

`q_dom`只能由Phase1可靠度、support-only coverage和预锁定数值条件共同决定。coverage低、LOCO不稳或receipt不闭合时，`q_dom=0`并逐值回退identity。不得使用receiver ID、TX ID、query角色、query置信度或场景真值设置`q_dom`。

### 3.2 C-id剩余nuisance适应

K≥2时，C-id从`z_id`合法support的类平衡类内scatter与类间保护中提取rank≤2方向，只做PSD软抑制；K1严格回退identity。C-id只负责receiver context校正后的剩余sample/channel扰动，不重复学习C-dom共享方向。C-joint必须报告两支子空间重叠、更新norm及关闭cross-map消融。

### 3.3 B分类头

B保持Patch A局部Student-t qKNN不变，并增加全注册类统一SRDA头：

- 所有旧类和新类均值只来自当前row target support；
- ground只通过共同封存的int8多样本聚合知识提供class-agnostic共享协方差basis/spectrum，不提供旧类class mean、bias或logit；
- K1不估target covariance方向；K≥2只允许class-balanced scatter和极低rank residual；
- qKNN与SRDA必须在Phase1 LODO中证明非零disagreement、双向rescue、NLL改善和worst-class不退，才允许融合；
- 融合权重不得从query估计。K1使用Phase1锁定权重；K≥2可使用预登记support cross-fit可靠度。

### 3.4 D联合协同

D不得把B在C变换后重新拟合却仍声称是同一B。冻结结构为：

- local expert：最佳C变换后的Patch A qKNN；
- global expert：与B逐值相同的raw-`z_id` SRDA；
- fusion：与B逐值相同的融合公式、温度、权重来源、support可靠度函数和持久状态；D相对B唯一允许的变化是local logits来自最佳C。`q_dom=q_id=0`时D必须精确回退B；D的全局融合系数`alpha_global=0`时必须精确回退最佳C局部expert；B自身的`alpha_global=0`时必须精确回退A。

对主指标`m∈{H_old_new,soft-CVaR/floor}`定义协同项：

```text
Δ_syn(m) = [m(D)-m(B)] - [m(C*)-m(A)]
```

Phase1 LODO中要求`Δ_syn>0`且paired 95% CI下界>0；锁定target窄实验要求两个主指标的`Δ_syn>0`且D同时不弱于B和C*；最终125/1200确认要求paired mean `Δ_syn>0`且95% CI下界>0。若C*没有独立超过A，或D只复制最佳单支，则不得声称联合优化成功。

## 4. 强制六臂与负对照

每轮冻结以下同row因果臂，禁止省略或混改：

|臂|唯一变化|分类头|
|---|---|---|
|A|identity `z_id`|Patch A single-qKNN|
|B|只增加SRDA|A＋raw-`z_id` SRDA|
|C-id|只增加`z_id` nuisance DA|与A逐值相同|
|C-dom|只增加`z_dom→z_id` receiver-context DA|与A逐值相同|
|C-joint|C-id＋C-dom|与A逐值相同|
|D|最佳C局部expert＋固定B全局expert|联合头|

机器回执必须满足：

```text
C.classifier_hash == A.classifier_hash
B.DA_hash == identity_hash
D.DA_hash == best_C.DA_hash
D.global_head_hash == B.global_head_hash
D.fusion_hash == B.fusion_hash
```

至少包含：identity/no-adaptation、random/permuted`z_id`、random/permuted`z_dom`、关闭cross-map、关闭ground prior、coverage回退和共同正交变换负对照。共同可逆变换后若完整重估均值/协方差，LDA margin不变；不得把这种“对齐完成”写成收益来源。

## 5. 数据协议硬边界

Phase2只能读取：immutable Phase1 deployment bundle、匹配`VALIDATED_ONCE`的固定单LEO弱观测capsule、当前row合法target-old/target-new K-shot support与标签、query访问前锁定的数据无关配置。

必须保持：

- 一个物理IQ仅有一次随机允许的LEO弱信道观测；K-shot是K个独立物理support；
- support/query物理ID互斥，三个场景的物理ID集合互斥；
- 不访问clean/raw/source样本、sample-level source feature、source replay或可替换sidecar；
- query逐样本面对全部注册类，只前向和一次评分；不得更新任何状态；
- 禁止query伪标签、熵最小化、图、OT/Hungarian、quota、角色Oracle和batch reassignment；
- ground知识只能作为共享域/协方差/不确定度先验，不能直接给旧类加分或覆盖target prototype；
- target-old和target-new正式状态均采用int8，无FP32 sidecar。

匹配的`capsule_id/split_id/schema=p2_min_v1/VALIDATED_ONCE`只核对一次。只有received IQ字节、physical ID、receiver/TX集合、scenario、K、support-query划分或schema改变时重验；方法、adapter、head、超参数、checkpoint推理状态、bundle、资源或报告变化不得触发数据重建。

## 6. 高效研发顺序

1. 完成B独立Phase1 nested LODO runner与外部authority receipt；不得用builder自报fit/quant/resource晋级。
2. 并行完成C-id LODO与新的C-dom Phase1 bundle/LODO；两支都保持A分类头。
3. C-id和C-dom必须分别相对A在old/new/H/floor/min-class/NLL/forgetting联合门上取得独立正收益，才进入C-joint。
4. 只有B与最佳C都独立通过，才构建D并检验`Δ_syn`。
5. 冻结六臂后，仅运行预登记K1/new20与K10/new20、三个场景、代表receiver的matched窄实验；不得根据结果回调rank、`q`、alpha、loss或bundle格式。
6. 窄实验全部通过才运行历史125稳定性screen；125不选参。
7. 125通过后，以同一commit运行`5 receivers×5 seeds×3 scenes×K{1,5,10,20}×new{2,5,10,20}=1200`评价单元完整确认。

用户显式要求某个未过窄门方法跑125时，可以作为诊断执行，但必须预标记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不得据结果改参或晋级。

## 7. 性能与资源门

每个版本必须同时报告注册前old、注册后old、old adaptation gain、seen-new、H、BA、全部注册类floor、min-old、min-new、forgetting、old→new/new→old和完整逐类/receiver/scene/K/seed结果；不得只说明缺陷或拼接不同row极值。

K10完整确认硬门：

- `old_acc_after_increment≥92%`；
- `min_old_class_acc≥88%`；
- `seen_new_acc(new5)≥92%`；
- `seen_new_acc(new10)≥90%`；
- `seen_new_acc(new20)≥86%`。

同时满足：K5核心指标相对matched K10下降≤3pp；K1总体及每receiver old adaptation gain≥0；K1相对direct ADV3B02至少+2pp且paired 95% CI下界>0；K5/K10/K20遗忘不高于matched identity qKNN；D相对B/C*的old/new/H/min-old/min-new均不降、forgetting不增；`Δ_syn`通过。

资源硬门：trainable parameters≤80000、adaptation epochs≤30、optimizer steps≤50、persistent incremental state≤256KB、dense query graph=false、query-dependent batch optimization=false。正式int8要求top1一致率≥99.5%、大margin flip=0，并报告实际wire、MAC、平均/P95时延、峰值显存和前向次数。

## 8. 无用工作永久禁区

除非触发明确例外，不得进行：

- 重复建设或人工追溯已经`VALIDATED_ONCE`的数据、hash、allowlist、authority和准入系统；
- 把D18数据句柄、D81/D92基线、D93/D94 transport等不同对象混称为“版本”；
- 要求跨run row-specific opaque handle或封装artifact原始SHA bit-exact；跨run只比较稳定语义，raw SHA仅审计；
- 用125、Role-Oracle、development query或confirmation query选择候选、rank、量化格式、alpha、阈值或回退；
- support accuracy=100%、prototype重构余弦高、代码测试通过、进程启动或资源达标即宣称性能成功；
- 连续只改qKNN/RDA/温度/协方差/融合却声称完成域适应；
- 再做全坐标transport、共同正交变换或无coverage回退的ground强先验；
- 在没有disagreement和双向rescue证据时继续堆叠第三个头或固定凸融合；
- 多个agent重复读取全量历史、修改同一文件、独立启动同一run ID或主agent线性等待N607。

## 9. 完成条件

只有六臂因果证据、Phase1 LODO、锁定target窄实验、125、1200单元完整确认、协议证据、int8生命周期、资源审计、matched baselines、完整日志、报告、复现命令和Git提交全部存在且性能门全部通过，才能标记完成。

完成实验但性能未达标时，必须记录`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并返回下一单一机制假设。无prediction的运行只能记录`TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得混入算法结论。
