# D21 KNN原型生命周期、Phase1压缩几何与新类注册设计

## 1. 设计结论

D21不把“域适应”理解为在Phase2持续改写同一组KNN原型，而采用三层隔离状态：

1. `Phase1 immutable geometry`：ADV3B02和地面旧类压缩几何作为同一deployment bundle共同封存，Phase2只读。
2. `Stage2-B old snapshot`：每个旧类只由当前目标receiver上已经叠加单一`leo_*_weak`信道的K个support形成目标域原型与半径；完成support-only选择后提交不可变快照。
3. `Stage2-C append-only registry`：注册新类时只追加新类原型、半径及必要的稀疏碰撞边界；旧类原型、旧类半径、旧类打分列和已锁超参数均不得重估。

永久安全主干固定为`Frozen-Append+DALI(domain20,max-old lock)`。Phase1几何先只用于旧类内部身份重排，不改变逐样本`max(old)`，因此不会重复D19强地面anchor把seen-new压至6.67%–22%的失败。域适应的主信息仍来自目标域LEO_weak support；新类注册的主信息只来自新类LEO_weak support。

## 2. 不可突破的数据边界

- 每个support/query物理IQ在Phase2前只叠加一种LEO_weak信道状态；Phase2不读取clean样本、clean衍生信号、source样本或样本级source feature。
- Phase1组件只允许保存不可逆many-to-one聚合几何，不得保存原始IQ、单样本feature/logit、exemplar、sample index、成员距离列表、成员数量、协方差或full-precision sidecar。
- support标签只用于注册和support-only闭式选择；query输入schema不含truth、old/new/unknown角色、真实batch类别数、类别quota、顺序或global assignment。
- 每个query独立面对全部已注册类；禁止query拟合、query EMA、query伪标签更新和dense query图。
- 注册生命周期中的“旧类/新类”只表示registry提交前后，不是query侧角色Oracle。

## 3. Phase1最小几何包

### 3.1 当前正式v2

对旧类`c`和Phase1域`d`，先离线计算单位归一化聚合质心`p[d,c]`。固定global max-min中心域为domain20，将其6个旧类质心量化为core；其余域×类相对core的偏移按类做`R=3`低秩分解并int8量化：

```text
p_hat[d,c] = dequant(core[c])
             + sum_r dequant(coeff[d,c,r]) * dequant(basis[c,r])
```

每个域×类另保存一个稳健角半径：

```text
r_source[d,c] = P90_x(1-cos(z_x,p[d,c]))
```

其中`x`只来自Phase1授权训练split，成员值在Phase1内部聚合后立即丢弃；Phase2只看到`radius_q[D,C]`和逐类FP16 scale。

domain20和domain24对应不同优化目标。domain20最大化最差同类余弦，最差余弦为0.994498；domain24最小化平均重构残差，残差能量低9.88%。D21优先旧类floor且core直接服务DALI，因此锁定domain20。domain20在18个真实LEO_weak旧类support中心上的最小top-1 margin为0.011996，domain24为0.000927，前者约为后者13倍。

### 3.2 压缩审计

以下仅是相对历史int8组件的几何保持审计，不是任务准确率：

|表示|含96B半径的逻辑payload|随机原型argmax/margin保持|18个真实LEO_weak support中心保持|max angle|
|---|---:|---:|---:|---:|
|domain20+dense int4 residual|7,799B|99.9100%|18/18|0.9919°|
|domain20+lowrank R2 int8|3,527B|99.6533%|18/18|2.2967°|
|domain20+lowrank R3 int8|4,589B|99.8100%|18/18|2.0827°|

正式默认采用R3。它比R2多1,062B，但随机保持率提高0.1567pp、最大角误差下降0.214°；DALI只读core时不支付残差重建MAC。dense-int4保留为重构保真度对照，不作为最轻默认。

### 3.3 暂不上传的可选统计

若半径消融证明不足，下一步只考虑增加每域×类一个int8稳健分离余量：

```text
g_source[d,c] = P10_x(cos(z_x,p[d,c])-max_j!=c cos(z_x,p[d,j]))
```

该标量可用于限制floor类的更新幅度，但当前v2尚未授权，不能进入Phase2。最近竞争类可由重建中心直接推导，不重复存储；精确成员数、协方差、Fisher、BN统计和样本级残差均不需要且禁止上传。

## 4. 旧类原型如何适应而不遗忘

### 4.1 旧类不是一组被反复EMA改写的中心

旧类保持两个互不覆盖的对象：

- `source anchor`：Phase1只读core/低秩几何，永不更新。
- `target-old prototype`：由Stage2-B合法LEO_weak support一次性估计，提交后永不更新。

因此“域适应”发生在Stage2-B的目标域support估计和有界旧类内部重排，而不是query到来后的在线学习。Stage2-C加入新类时，旧类prototype、radius、DALI尺度、旧类GEMM路径和旧类score列必须逐位锁定。

### 4.2 旧类中心候选

K10开发工作点只在固定候选中进行support删除验证：

1. spherical mean；
2. spherical medoid；
3. Huber spherical mean；
4. fixed-trim spherical mean。

选择键依次为：三LEO场景共同的最差逐类LTO准确率、最差q25 margin、总体准确率、中心bootstrap角漂移、状态/MAC。任何类退化则整个场景回退spherical mean。K1只能用单点原型；K5只允许mean/medoid；K20复用K10已锁规则，不重新选超参数。

### 4.3 DALI安全旧类修正

DALI对每个query先计算target prototype基础分数，再用domain20 core和同一个received IQ的ADV3B02 direct logits产生有界旧类内部残差。修正后严格恢复原始`max(old)`，新类score逐位不变。它可以修复旧类内部身份混淆，但明确不能解决new prototype侵入old组；后者交给统一半径和稀疏碰撞边界处理。

## 5. 新类如何注册

### 5.1 单原型原子注册

对新注册类`n`，使用其K个LEO_weak support生成单位球鲁棒中心`p_n`。所有新类先独立生成，再按class handle确定性排序后一次append，避免注册顺序影响结果。旧状态复制为只读snapshot，新状态仅附加：

- 1个新类中心；
- 1个收缩半径；
- 至多1条稀疏局部碰撞边界；
- K10且双模态证书通过时可附加第2个中心。

注册失败不得静默改写旧状态。候选不通过support-only侵入门时依次回退：2-prototype→单prototype、碰撞修正→off、radius residual→off、鲁棒中心→spherical mean。

### 5.2 K依赖半径

目标类经验半径来自物理support删除后的角距离，而不是把同一IQ变换成多个view：

```text
lambda_r = K_eff/(K_eff+4)
r_target^2 = lambda_r*r_empirical^2 + (1-lambda_r)*r_prior^2
```

- K1：`K_eff=0`，完全使用Phase1全部域×旧类半径的固定global robust prior，禁止由单点产生零半径。
- K5：使用LOO残差，`K_eff=4`，`lambda_r=0.5`。
- K10：使用LTO残差，`K_eff=8`，`lambda_r=0.667`。
- K20：复用K10锁定公式，以18个LTO有效残差代入，不重新调参。

半径对所有注册类使用同一公式。候选标准化证据为：

```text
q_c(z) = clip((r_c-(1-cos(z,p_c)))/max(r_c,r_floor),-1,1)
s_c = s_base_c + beta_r*q_c(z)
```

`beta_r`和`r_floor`由开发K10一次锁定；radius residual必须小幅、有界，并通过全部注册support的逐类非退化门。禁止直接使用无界`cos/r`，否则窄半径类会获得人为高置信。

### 5.3 新旧与新新碰撞边界

仅靠原型中心会在新类数5/10/20增大时产生极值侵入。D21为每个新类最多保留1个support-only最近竞争类`j*`，形成稀疏局部判别方向：

```text
v_n = normalize(p_n-p_j*)
b_n = 0.5*(dot(v_n,p_n)+dot(v_n,p_j*))
delta_n(z) = beta_b*clip(dot(v_n,z)-b_n,-h,h)
```

该残差只附加到新类自身score：接近新类support时为正，接近碰撞旧类或另一新类时为负；旧类score列仍逐位不变。`j*`由全部注册support中心和半径的固定overlap规则确定，不读取query角色或query批次结构。每类最多1条边，状态和MAC为`O(CP)`，不存在dense query图。

碰撞边界必须同时通过old→new、new→old和new→new的self-excluded support混淆门。对已知开发floor类可重点报告，但正式算法不能硬编码TX；它以每类margin/radius自动识别风险。

### 5.4 两原型只作为K10条件分支

只有同时满足以下条件才允许单类2-prototype：

- K10或K20；K1绝对关闭，K5正式默认关闭；
- 两簇各至少3个物理support；
- LTO簇指派稳定率/Jaccard不低于0.8；
- 两中心角距大于2倍pooled radius；
- 目标类floor严格改善且其它每类support margin不下降；
- 三LEO场景使用同一个预登记规则。

多原型score必须按原型数归一化：

```text
s_c = tau*log(mean_m exp(cos(z,p_c,m)/tau))
```

禁止raw max，因为多一个prototype会产生隐式类别优势并加重新旧混杂。

## 6. 避免遗忘的四个独立不变量

|遗忘来源|必须保持的不变量|验证方式|
|---|---|---|
|参数漂移|Phase1 bundle和Stage2-B旧snapshot不可更新|成员hash、只读状态、update access=false|
|数值路径漂移|新增类前后旧类GEMM单独执行且score列逐位一致|bitwise old-score test|
|新类竞争侵入|old→new逐类self-excluded混淆和q25 margin不恶化|support-only intrusion guard|
|多原型极值偏置|每类固定状态上限且score按prototype count归一化|构造反例与资源审计|

注册后旧类准确率下降不再笼统称为“旧模型忘记”：若旧score列不变但winner变为新类，应单独记为`competition_forgetting`；若旧score列变化，则为实现违规并立即阻断。正式结果同时报告二者。

## 7. 候选路线与优先级

|路线|机制|优先级|主要价值|主要风险|
|---|---|---:|---|---|
|R1 Frozen-Append+DALI|单中心、旧snapshot冻结、max-old旧内重排|1|最轻、0参数、可证明不压new|不能主动修复new intrusion|
|R2 Robust Snapshot|mean/medoid/Huber/trim中support-only选择|2|改善受异常support影响的floor|小K删除验证方差高|
|R3 Hierarchical Radius|半径prior+LOO/LTO收缩+有界残差|3|统一新旧置信尺度|无界标准化会造成尺度偏置|
|R4 Sparse Collision Boundary|每个新类至多1个竞争方向|4|直接治理新旧/新新混杂|support过拟合时会伤new|
|R5 Certified 2-Prototype|稳定双模态类才分裂|5|覆盖真实多模态|原型数偏置和状态增加|
|R6 Low-rank Feature Residual|identity主分数+有界低秩残差|备选|可修特征几何|历史全替换低秩使old从77.78%降至55.83%|

不能重走的路线：D19式强Phase1 anchor直接融入old/new组间基础分数；仅继续扫描弱anchor权重；用低秩投影完全替换identity空间；默认给所有类多个原型；用support指标冒充formal query准确率。

## 8. ADV3B02离线复导出或重训

第一选择不是盲目重训，而是固定现有ADV3B02 checkpoint，在Phase1授权训练split上执行一次纯离线几何导出：域×类单位质心、P90角半径、domain20 global max-min中心和R3残差因子。若现有checkpoint、feature schema和Phase1 split能够完整复现，这是成本最低且不改变表示的路线。

只有出现以下任一情况才重训ADV3B02：

1. 现有checkpoint无法与准确Phase1 TX/domain registry逐列绑定；
2. 原训练流程未保存可复现的feature schema或授权split；
3. 固定checkpoint复导出的radius/中心几何在Phase1 held-out domain上不稳定；
4. 需要把几何导出代码、config、checkpoint和component作为一次共同生成的正式bundle重新签封。

即使重训，也不以Phase2 query性能选择epoch。checkpoint选择仍只用Phase1既定validation；几何组件在target不可达时生成。重训日志必须保存逐epoch loss/metric、split/registry、最佳epoch、checkpoint hash、导出器hash、component逻辑字节和量化误差。

## 9. 分级实验矩阵

### D21-A：Phase1几何导出

- 固定checkpoint复导出与必要时重训各自记录成本；不接触target。
- 比较v1 dense int8、v2 R3+radius、dense-int4 fallback的几何保持、逻辑状态、重建MAC和峰值内存。
- 半径必须来自真实Phase1成员P90；历史v1缺半径时不得用域偏移伪造。

### D21-B：support-only锁定

- receiver=`20-1`开发seed起步，3个LEO_weak场景，K=1/5/10/20，before/after均运行。
- 候选依次为R1、R1+R2、R1+R2+R3、再加R4；R5仅K10/20。
- 每个候选报告逐类LTO/LOO、floor、q25 margin、bootstrap角漂移、old→new/new→old/new→new混淆、状态/MAC。
- support-only结果只用于锁参/阻断，不作目标准确率声明。

### D21-C：开发query与独立确认

- 完成method lock和共同bundle后才生成不可变prediction artifact；独立scorer随后接truth。
- 开发seed统一锁定K10工作点、半径和碰撞阈值；K1/K5/K20只复用公式。
- 先运行5个seen-new开发屏，再扩展10/20类；成功后进入5 receiver×至少5确认seed×3场景正式矩阵。
- 同一行同时报告注册前old、注册后old、seen-new、`H_old_new`、每类old floor、competition forgetting和资源Pareto。

## 10. 当前决策

1. 正式Phase1状态锁定为`domain20 core+R3 int8 residual+P90 radius`；当前估算逻辑payload约4,589B，最终以实现审计为准。
2. R1为永久回退主干；R2和R3是下一批主实验；R4仅在注册后混杂证据存在时启用；R5只处理K10/20稳定双模态。
3. 旧类目标域原型只在Stage2-B变化一次；Stage2-C不因新类到来而重估。
4. 新类原型append-only注册；K1使用global radius prior，K5/10使用收缩半径，所有类使用同式逐样本打分。
5. floor优化依赖统一的逐类稳定性和碰撞门，不硬编码query角色、类别配额或TX配额。

