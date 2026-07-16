# D16-FCAR高效类局部非对称注册追踪

|ID|Requirement|Status|Verification|Notes|
|---|---|---|---|---|
|FCAR-01|只接收固定单LEO_weak runtime-authorized feature artifact；无clean/source/query truth/role/quota|verified|mapping拒绝逻辑沿用严格artifact入口；源码Oracle词扫描通过|base operator only|
|FCAR-02|按`support_rank`奇偶构造class-balanced 2-fold cross-fit|verified|K10/K5合成例均覆盖两个eval fold；每类每fold只拟合一次envelope/prototype|避免D15 nested `O(CN²)`重拟合|
|FCAR-03|评价fold A的OOF模型只由fold B拟合，反向同理|verified|逐record审计`model_train_row_indices`的rank parity与eval fold相反|每条物理support只有一个固定feature|
|FCAR-04|record `i`的阈值只使用同eval fold中除`i`外的OOF LLR，且这些peer模型也不含`i`|verified|逐record peer/model索引审计；将`i` feature乘`-1000`后其`q_pos/q_neg/mid/half_gap`不变|无self-threshold leakage|
|FCAR-05|每类独立选择`a_plus/a_minus`，固定grid为`{0,.005,.01,.02}`，`t=.5`，candidate固定margin band|verified|state grid验证与分段线性delta端点测试通过|没有shared gamma|
|FCAR-06|`delta=I[abs(base margin)<=b]*(a+*max(0,(h-t)/(1-t))-a-*max(0,(-h-t)/(1-t)))`|verified|正、负、阈值内和端点单测通过|门只读取未修正base margin|
|FCAR-07|类候选使所有truth类correct count不降、own Q20 true margin不降、零新增错误capture|verified|安全函数正反例通过；新增capture定义为`truth!=c, base_pred!=c, candidate_pred==c`|不能用总量互相抵消|
|FCAR-08|非floor非零候选也必须至少有一项严格收益|verified|完全neutral候选拒绝；任一truth Q20严格提升可通过一般收益门|零幅度永远fallback|
|FCAR-09|floor由OOF baseline accuracy底四分位自动识别；禁止硬编码class ID|verified|按`(accuracy, opaque handle)`稳定排序取`ceil(C/4)`；5类合成排序测试通过|tie不扩大为全类floor|
|FCAR-10|floor候选必须own correct或own Q20 margin严格改善|verified|floor严格门代码和一般收益门分离|floor重要性显式提高|
|FCAR-11|组合后使用最终full-support prototype、full-support envelope与全部2-fold OOF汇总`mid/gap`做`deployment_state_consistency_veto`|verified|专门构造parity prototype与full prototype预测相反的记录，veto输入严格采用full部署状态；enabled trace记录pass|该检查包含被评support自身，只能撤销，不能作为无泄漏性能证书|
|FCAR-12|joint退化按非floor后floor、收益小到大、opaque handle确定性撤销|verified|rollback key顺序测试通过|不依赖真实TX ID语义|
|FCAR-13|Before old state独立拟合冻结；After精确复用旧prototype/envelope/amplitudes，只追加new类|verified|所有旧数组逐位相同；positive state 257随机probe旧score逐位锁|old block使用独立GEMM和old-only rival|
|FCAR-14|K1 canonical true Z0；K2-K4 fail closed；K>=5可运行|verified|K1为rank0/空dims/零幅度；K2-4逐项拒绝；K5 OOF测试通过|不跨K借数据|
|FCAR-15|0参数、0epoch、无dense query图、1 backbone forward、0 FFT branch|verified|resource字段和<80KiB合成state测试通过|head ops有显式上界|
|FCAR-16|state内容哈希之外实施独立语义门：disabled全零、enabled非零幅度、dims边界、generation/old_count/K/resource fail closed|verified|恶意`state_content_sha=""`重新自封存仍因candidate/K/resource/disabled幅度违规被拒绝|哈希不替代合法性验证|
|FCAR-17|单query、all registered classes、runtime绑定|verified|单样本shape通过；双样本拒绝|没有真实query运行或标签接入|
|FCAR-18|真实runner只打开D8b strict-K10 before/after enrollment-only support，不打开query/truth/scorer|verified|`run_d16_support_only_fcar.py`复用D14 pre-open、payload与physical-batch-1 feature helper；真实artifact状态为`TRUE_Z0_NO_QUERY_OPEN`|authority固定为development diagnostic|
|FCAR-19|提供严格K10 joint leave-two-out入口，outer held2不得拟合state|verified|5 folds；每fold每类train K8/held K2；old score逐位锁；非K10拒绝|唯一无泄漏性能证书入口|
|FCAR-20|严格L2O输出joint overall/min/per-class、`H_old_new`、overall/per-class old forgetting、candidate-vs-Z0逐类非退化|verified|fold级与aggregate字段完整测试；after-new min和逐new类独立保留|不能由floor集合替代new类报告|
|FCAR-21|outer held2的物理样本、特征与标签不得进入训练决策|verified|fold trace保存train/held physical-ID SHA和`held_disjoint_from_selection=true`；held2 feature乘`-1000`后selection SHA、decision tensor SHA、floor handles、enabled不变|不比较绑定完整artifact的state content SHA|
|FCAR-22|K1、candidate/operator和resource派生字段必须与state精确一致|verified|K1必须等于canonical hp；K2-K4状态拒绝；正rank禁止K1；candidate/operator双绑定；enabled count/grid/head ops/state bytes/fits/forward/FFT/dense精确复算|防止自封存绕过|
|FCAR-23|module fit本身不以outer forgetting阻断候选，不能单独用于正式晋升|deferred to runner|本模块只计算并报告outer L2O overall/per-class forgetting|真实runner promotion gate必须对照项目阈值阻断旧类遗忘|
|FCAR-24|runner必须按全部scenario×全部outer fold逐类阻断遗忘与floor退化|verified|任一fold出现After-old低于Before-old、old/new低于Z0、joint/H/floor下降、无严格floor收益或old-score lock失败即回退true Z0|不允许aggregate平均抵消|
|FCAR-25|跨场景物理样本与parent received-IQ SHA两两不交|verified|runner在feature提取后、候选评估前核对三场景after enrollment union；真实audit三对均零交集|view不增加K，不生成第二LEO状态|
|FCAR-26|资源与Pareto审计必须覆盖参数、epoch、MAC、状态、延迟和峰值内存|verified-diagnostic|0参数、0epoch、3168 MAC上界、12859B数组状态、CUDA/Python峰值与support-row微基准均落盘|Python安全包装延迟不是正式部署query延迟|

## 机制

每类`c`按support rank奇偶切成两个class-balanced fold。eval fold的prototype和class/rest对角密度比统计只由相反fold拟合。对eval record`i`，其class-local conformal阈值为：

```text
q_pos_i = Q20({OOF LLR_j: fold(j)=fold(i), j!=i, truth(j)=c})
q_neg_i = Q80({OOF LLR_j: fold(j)=fold(i), j!=i, truth(j)!=c})
mid_i = (q_pos_i+q_neg_i)/2
half_gap_i = (q_pos_i-q_neg_i)/2
h_i = clip((LLR_i-mid_i)/(half_gap_i+eps), -1, 1)
```

因为同eval fold所有OOF模型均由相反fold训练，`i`既不在自身模型，也不在任何threshold peer的模型内。

固定幅度候选：

```text
a_plus, a_minus in {0, .005, .01, .02}
t = .5
delta_c =
  I[abs(base_margin_c)<=b] * (
    a_plus  * max(0, ( h_c-t)/(1-t))
    - a_minus * max(0, (-h_c-t)/(1-t))
  )
```

类级self-excluded OOF只用于support内幅度选择证据，不构成性能证书。选择后使用全部2-fold OOF LLR汇总部署`mid/half_gap`。随后执行含support自身的`deployment_state_consistency_veto`：同时改用full-support prototype计算base score，并用full-support envelope和部署`mid/half_gap`计算`h`，不混用parity-OOF base；该检查只能撤销候选。

唯一无泄漏性能证书来自严格K10外层L2O：按rank `(0,1)...(8,9)`形成5个held2 fold，每fold仅用剩余K8拟合。报告同时保留Before-old、After-old、After-new、joint、`H_old_new`、old forgetting和candidate-vs-Z0逐类门；After-new的min与逐new类结果独立输出。Before与After旧类始终使用相同old prototype block、相同old-only rival pool和相同幅度，因此旧score逐位锁定。

当前module fit只产生候选state和审计结果，不根据outer forgetting自动拒绝state。正式runner必须把aggregate与逐旧类forgetting接入promotion hard gate；在该门落地前，D16 module输出不得单独晋升为正式路线。

## 真实D8b strict-K10结果

runner在`receiver=1-20`、`seed=713201`的三个独立LEO_weak enrollment-only场景上统一比较true Z0、`margin_band=0.02`和`margin_band=0.04`。全部positive arm均未通过逐scenario、逐outer-fold、逐类门，最终选择`d16_z0_true_zero_base`。`margin_band=0.02`虽然在部分fold启用1–2个类，但三场景聚合预测与Z0完全相同；`margin_band=0.04`在low和rain分别改善一个旧类10pp和20pp，却使seen-new类下降10–20pp。

真实artifact：

```text
E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d16_fcar_strict_k10_v1
status=SUPPORT_ONLY_D16_DEVELOPMENT_TRUE_Z0_NO_QUERY_OPEN
training_log_sha256=9c665c3a06da4da077fee25e9a19a2f9c3b8dc0984e78ce0cade84a3badfa928
support_audit_sha256=fecfb8b42038bc86331190641e3df8a026ea3e842b697316c3bc089c32cb1efe
```
