# D103-R1-RXID-DUALSPLIT-MB4重入卡

状态：`REENTRY_CARD / FEASIBILITY_PROBE_PREREGISTERED / TARGET25_NO_GO`

日期：2026-07-24

## 1.前序实例与重入理由

前序candidate为`D103-RXID-Episodic-MetaBias4-qKNN Revision2`，终审为`P0=0、P1=6 / NO_GO_TO_DESIGN_FROZEN`。首次微探针错误地把tap成员`z_id`绑定成`z_dom`，其K1机械数值无效；训练常量和数值门又在首次probe后写入，不能由同一candidate重跑自证。

新candidate固定为`D103-R1-RXID-DUALSPLIT-MB4`。本卡在任何R1微探针运行前冻结输入、常量、probe、资源公式和量化ABI；R1不得读取Revision2的K1数值或据新probe改参。

R1保留Revision2未被终审否定的科学机制：冻结基础checkpoint；以`r=Norm(Uz_dom)`做类无关domain表示；以`z(a)=Norm(ReLU(pre_relu+Ba))`做rank-4 MetaBias；Phase2继续使用D102同一类对称4维闭式support求解、box→ellipsoid约束、全部support统一重编码和typed INT8 Student-t qKNN。R1唯一方法变化发生在Phase1：线性TX零空间、多尺度MMD、跨day/cross-TX receiver自监督和K1/K5/K10 receiver-held元任务共同学习`U/B/bank`。Phase2仍为0 optimizer step，query逐样本只读并面对全部注册类。

## 2.R1输入和固定inner fold

|字段|冻结值|
|---|---|
|tap SHA256|`c6807d9156ab3ac8f7005707a3bd7eec342d2e4f0a43d4b96d5ea8a9574ec4c1`|
|dual SHA256|`dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0`|
|tap用途|同row`pre_relu`及metadata|
|dual用途|真实`z_dom`及同row`z_id` parity|
|row绑定|physical ID、label、receiver、day、class IDs逐数组相等；`z_id max_abs≤1e-5`|
|固定inner held receiver|`14-7`|
|固定episode receiver|`18-2`|
|seed|`103713`|
|outer结果读取|`false`|
|性能指标|不计算|

R1微探针仍只使用冻结source-val tap做资源/shape近似，不保存任何资产。它不授权source-val进入正式训练。

## 3.Phase1 split权限

正式D103-R1严格使用项目固定`0.07/0.63/0.30`：

- `L_s=0.07`：可读取TX、receiver、day；`P⊥`、TX-MMD、带TX标签MetaBias/qKNN元任务和类平衡bank只使用`L_s`；
- `U_s=0.63`：TX隐藏且不得恢复、伪造或从其他表连接；只允许receiver/day自监督、跨day正对和VICReg；
- source-val=`0.30`：完全不进入梯度、`P⊥`、`U/B/bank`、量化尺度、最终asset或训练early stop；只在状态冻结后执行outer receiver、双留出LOCO、双TX probe和matched证伪；
- final deployment asset只从`L_s+U_s`训练，source-val始终保持外部证伪面。

冻结基础checkpoint在M0/D102/D103之间逐字节相同，是LOCO唯一允许保留的历史知识。

## 4.singleton训练常量

R1不做超参网格或性能选参：

- Adam，learning rate=`1e-3`；
- 每fit20epoch×20meta step=400step；
- 每step固定包含K1/K5/K10三个episode；
- balanced batch每receiver×day×TX cell取2个互异物理样本；
- `μ=0.1`、`τ=0.1`；
- `λ_TX=λ_RX=λ_V=λ_O=1.0`；
- MMD`gamma={0.5,1.0,2.0}`取均值；
- qKNN训练温度=`0.2`；
- 不按loss、source-val或性能early stop。

7个receiver outer fold已经是唯一receiver-held审计，不再增加inner leave-one-receiver。每个outer只增加4个leave-one-day稳定性fit，day结果只允许reject，不允许选参。

## 5.K1数值门

下列门在R1正确probe前冻结：

- `rank(A_data)=4`；
- `min_singular_value(A_data)≥0.05`；
- `condition(Λ0+A_data)≤10`；
- `prior_fraction≤0.80`；
- `1e-4≤||a||_2`且满足冻结box/ellipsoid；
- 合法view消融top1 agreement≥99.5%、large-margin flip=0；
- 独立episode系数方向余弦中位数≥0.80。

R1微探针只机械检查前三项的shape/condition，不形成K1泛化或性能证据。正式任一K1 fold inactive时输出完整M0预测并标记`INACTIVE_NON_PROMOTABLE`，同时拒绝整个D103-R1实例。

## 6.双TX probe

outer-held receiver完全不进入D103资产训练，但在资产冻结后专用于攻击者probe。每个receiver×day×TX cell按

`SHA256(D103-R1-RXID-DUALSPLIT-MB4|receiver|day|TX|physical_id|probe_v1)`

升序切分，前60%为probe-train、后40%为probe-test；两侧physical ID互斥，每cell不足5个物理样本即fold失败。

容量固定为：

- 多项logistic regression：`C={0.1,1,10}`；
- RBF SVM：`C={1,10}`、`gamma={0.5/32,1/32,2/32}`；
- class weight=`balanced`、max_iter=`2000`、seed=`103713`。

每个容量同时计算pooled test BA和4个per-day test BA；该receiver的`fold_score`取所有容量×5个评分面的最大值。最终`mean=7个receiver fold_score均值`，`max=7个receiver fold_score最大值`；两者都必须≤25%。probe拟合状态只用于审计，不得回流D103资产。任何修改需新candidate和新`REENTRY_CARD`。

## 7.Phase2量化ABI

继承D102的MetaBias4求解和typed qKNN公式，但学习数组全部INT8：

- `U/B/g/t`：逐行对称INT8，`scale=max_abs/127`并以binary16保存；round-to-nearest-even；clip到`[-127,127]`；禁止`-128`；全零行code全0、scale=`1.0`；
- precision：先clip到`[0.05,20.0]`，再对log值做per-tensor affine INT8，offset/scale以binary16保存；
- sigma：先clip到`[0.05,2.0]`，再对log值做per-tensor affine INT8，offset/scale以binary16保存；
- 固定标量以IEEE754 binary16 manifest值编码：`T=0.25`、`Λ0=(1,1,1,1)`、`a_max=(0.25,0.25,0.25,0.25)`、`R=0.35009765625`；
- bank、邻居和class score平手按预先封存的稳定bank index、support enrollment index和opaque registry index升序；
- INT8 teacher/student top1 agreement≥99.5%、large-margin flip=0、无FP16/FP32学习数组或sidecar。

typed qKNN继续使用Student-ν=`3.0`、effective dim=`160`、shared h0=`0.2`、scale prior=`2.0`、scale ratio=`[0.5,2.0]`、temperature=`1.0`。

## 8.资源公式与失败封口

训练fit数固定为：

`(7 receiver outer+42 receiver×class双留出outer)×(4 leave-one-day+1 outer fit)+1 final fit=246fit`。

每fit400step，总计98,400step。新微探针必须重新测真实`z_dom`的step时间、显存和代表零值临时checkpoint磁盘；临时文件必须在返回前删除且不能含学习值。完整预算按：

`GPUh=98,400×seconds_per_step/3600×3.0×1.35`。

3.0是N607设备/实现安全因子，1.35覆盖双probe、量化、M0/D102 matched评估、I/O和失败artifact。最终上限向上取整到不低于估算值的下一个6GPUh；显存上限取不低于实测reserved的8倍且至少4GiB；磁盘上限必须大于`246×temporary_checkpoint_bytes`并至少20GiB。

任何资源上限、协议、资产闭包或数值门失败均保留artifact并以`NO_PERFORMANCE_RESULT`封口，不减少fold、K、day审计或LOCO覆盖。

## 9.R1微探针边界

允许3次warmup+3次计时，只输出：

- tap/dual hash和row parity；
- 跨day配对可构造性；
- 真实`z_dom`的K1机械rank/minSV/condition；
- tensor shape、参数量、step时间、峰值显存；
- 不含学习值且立即删除的代表临时checkpoint字节数。

禁止BA、TX准确率、LOCO性能、source-val训练、Target/capsule/query访问、deployment asset或任何基于结果的调参。R1微探针代码和本卡必须先进入本地Git commit，之后才允许运行。
