# D15 CL-CDR保守类局部密度比包络追踪

|ID|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|
|CLCDR-01|只接收固定单LEO_weak runtime-authorized feature artifact；无clean/source/query truth/role/quota|module/tests/真实support诊断|verified_module_diagnostic_integration|普通mapping与runtime绑定反例通过；D8b三场景strict K10 enrollment-only实测未打开query|真实package authority仍为`LOCAL_PROTOCOL_REPAIR_REQUIRED`，不得作formal claim|
|CLCDR-02|strict K10 joint L2O，outer held2不可回流；每fold train K8闭式拟合|module/tests|verified|5-fold joint L2O；held old/new极端变异不改变selection/calibration SHA|K5/K20延期|
|CLCDR-03|每类相对pooled-rest拟合shrink diagonal Gaussian envelope/density ratio|module/tests|verified|pooled shrink variance、对角LLR和有限state验证通过|不构造class×class或query图|
|CLCDR-04|每类最多选择`r<=16`维；评价inner-held `i`时稳定率使用`rows\i`内部nested L2O维度的pairwise consensus|module/tests/trace|verified|state取所有`i`的nested stability下限；变异`i`不改变其nested dims/stability|full-train与含`i`模型均不参与`i`稳定门|
|CLCDR-05|维度选择稳定率低于锁定门时该类identity fallback|module/tests|verified|高噪声+stability1.0合成例出现disabled类，dims全`-1`|fallback只关闭该类correction|
|CLCDR-06|Before old envelope和old scores锁定；After只新增new类envelope，不修改old score|module/tests|verified|After将old/new prototype分成两次独立GEMM；Z0与positive state各513/12个随机probe旧列逐位相同|避免GEMM输出宽度改变浮点归约路径|
|CLCDR-07|候选必须三场景统一锁并先过逐类support门|真实support诊断|rejected_no_runner|共享gamma小网格无一候选同时满足Before-old、After-old和seen-new逐类非退化|保持Z0，不接runner/query/125|
|CLCDR-08|0参数0epoch，state只保存prototype和enabled类的稀疏dims/mu/var/pooled统计；<80KiB目标、256KiB硬门|module/tests|verified|合成6类rank0/8 state资源、只读和NPZ+JSON roundtrip验证通过|r固定上限16；真实new20字节待runner|
|CLCDR-09|真实Z0与alpha0 score/prediction逐位一致|module/tests|verified|随机feature score逐位相同，enabled全false|不保存latent envelope|
|CLCDR-10|formal API恰好单query/all registered/runtime绑定|module/tests|verified|双query与runtime漂移fail closed|不开放真实query|
|CLCDR-11|`ssr-gpu` py_compile/pytest/diff-check与选择性Git提交|trace/module/tests|verified|CL-CDR 17项、联合threshold control 27项pytest通过；py_compile与diff-check通过|只提交D15六文件，不混入共享脏树|
|CLCDR-12|v2 train-only class safety：稳定率门后，每类用inner held逐次重新拟合prototype/dims/stats，比较base与只修正c的cross-fitted预测|module/tests|verified|held物理sample删除后重新拟合；outer-held变异不改变selection/calibration SHA|outer held2不得回流|
|CLCDR-13|v2本类非退化：启用类c必须满足cross-fitted本类正确数不低于base|module/tests|verified|逐enabled类诊断断言`own_correct_candidate>=own_correct_base`|否则identity fallback|
|CLCDR-14|v2零抢占：对每个其他真实类，class c修正新增抢占数必须为0|module/tests|verified|逐其他truth class断言新增capture全0|不能只检查overall|
|CLCDR-15|v2 class-local conformal gap：`q_pos=Q20(own LLR)`、`q_neg=Q80(rest LLR)`且严格`q_pos-q_neg>min_llr_gap`；持久化`mid/half_gap`|module/tests|verified|部署state使用全部train的cross-fitted LLR汇总；安全评估对每条record删除自身后重算分位数；state SHA和资源字节纳入新数组|任一leave-self gap不足则identity|
|CLCDR-16|v2 margin band：可选只在class cosine距当前rival不低于`-margin_band`时触发conformal修正|module/tests|verified|`margin_band=0`时远离rival的new类score逐位不变|只读取未修正base score|
|CLCDR-17|推理修正使用`h=clip((llr-mid)/(half_gap+eps),-1,1)`与`gamma*h`，不再直接缩放raw LLR|module/tests|verified|逐元素公式对照通过；任一类修正绝对值不超过`gamma`；删除无效`llr_cap`超参数|v2锁中无死参数|
|CLCDR-18|K=1只能使用每类单一独立物理support；因无法类内估计方差而canonical true Z0，不借用其他K数据|module/tests|verified|state强制`rank=0,gamma=0,force_zero=true`、空dims；selection SHA仅由K1行生成；score与cosine逐位一致|K1不做伪增强或跨K池化|
|CLCDR-19|inner-held不得参与其自身conformal安全校准，包括其他校准record的LLR模型|module/tests|verified|评价`i`时，每个校准`j`均用`rows\{i,j}`重拟合；攻击性变异`i`后其Q20/Q80/mid/half_gap/nested dims/stability逐位不变|真正nested L2O，不是仅从LLR列表删除`i`|
|CLCDR-20|state必须可部署加载，不能只存在于内存trace|module/tests|verified|最小NPZ数组+JSON元数据；外部NPZ/JSON SHA双钉住；roundtrip逐位一致；拒绝覆盖、缺文件统一异常、顶层/hp exact key allowlist|原子双文件发布由runner层后续实现|

## 机制

对归一化feature`z`，每类`c`在support-only train中计算class-local均值/方差与rest pooled均值/方差。按shrink系数`lambda`得到：

```text
var_c = (1-lambda)*class_var + lambda*pooled_var + ridge
var_r = pooled_var + ridge
```

维度分数：

```text
F_cj = (mu_cj-mu_rj)^2 / (var_cj+var_rj)
```

取稳定top-r维。密度比：

```text
llr_c(z) = -0.5 * mean_j[
  (z_j-mu_cj)^2/var_cj
  - (z_j-mu_rj)^2/var_rj
  + log(var_cj/var_rj)
]
q_pos_c = Q20(inner-LOO own LLR)
q_neg_c = Q80(inner-LOO rest LLR)
mid_c = (q_pos_c+q_neg_c)/2
half_gap_c = (q_pos_c-q_neg_c)/2
h_c(z) = clip((llr_c(z)-mid_c)/(half_gap_c+eps), -1, 1)
score_c' = cosine_c + gamma*h_c(z)
```

对安全评价sample`i`，校准sample`j`的LLR必须由`rows\{i,j}`重新拟合后得到；因此`i`既不直接进入分位数，也不通过其他`j`的LLR模型间接回流。`i`的稳定率同样由这些`rows\{i,j}`模型所选维度两两计算，类级state使用所有`i`的nested stability下限。低于`min_stability`时该类`enabled=false`。只有所有`i`的nested gap都严格通过时才允许启用；部署state的`mid/half_gap`仍由全部train support的cross-fitted LLR汇总。

## v2 cross-fitted class safety

稳定率门通过后，不能直接启用。对每个class`c`和每个inner held sample`i`：

1. 删除`i`后重新拟合全部class prototype；
2. 删除`i`后重新选择class`c`的维度并拟合其class/rest统计；
3. 计算held sample的base all-class cosine预测；
4. 对record`i`的安全评估，对每个其余record`j`使用`rows\{i,j}`重拟合后计算`j`的LLR，再形成本条专属`q_pos_i/q_neg_i/mid_i/half_gap_i`；
5. 只对class`c`加入使用该leave-self-out阈值的CL-CDR修正，再计算candidate预测。

启用必须同时满足：

```text
own_correct_candidate(c) >= own_correct_base(c)
for every other truth class t:
    added_capture_to_c(t) = 0
for every inner held record i:
    Q20_own_minus_i(llr_c) - Q80_rest_minus_i(llr_c) > min_llr_gap
```

若任一条件失败，仅该类`enabled=false`。部署触发还可使用：

```text
base_score_c - max_{j!=c}(base_score_j) >= -margin_band
```

门只读取未修正base scores，多个类不能递归读取彼此的修正结果。

Before只构造old类CL-CDR state。After精确复用Before old prototype/envelope并只为new类增加同规则state。推理时old prototype block始终独立执行与Before相同形状、相同顺序的矩阵乘，new block另算后拼接，旧score列必须逐位相等。

## 真实support结论与声明边界

真实诊断使用D8b before/after strict K10 enrollment-only package，三个scenario中的物理support根ID互斥；每个物理样本只有一份固定`leo_*_weak`接收IQ，使用`base`单view、单backbone forward、0个FFT分支。外部seal SHA为：

```text
before=184a9167f3ea1fe341fd9d261e595bb0a42d97d49859486395dc33e0ba6fafe6
after=db83c9aaac7ab17ee102640bac2f512c44d88306a9535c64b365d1a76b012212
```

在修复独立old GEMM后的真实11类复跑中，所有候选的注册前/注册后旧score逐位锁均通过。共享`gamma`正候选仍未过逐类门：

|候选|clear After old/new/H|low After old/new/H|rain After old/new/H|失败模式|
|---|---:|---:|---:|---|
|Z0|0.6333/0.5000/0.5588|0.6833/0.4600/0.5499|0.6167/0.6200/0.6183|基线|
|rank8,gamma0.001|0.6500/0.4600/0.5387|0.6833/0.4600/0.5499|0.6167/0.6200/0.6183|clear旧类+10pp但`cls_f6a8...`-20pp；low同类-10pp|
|rank8,gamma0.0025|0.6667/0.4400/0.5301|0.6667/0.4600/0.5444|0.6333/0.5800/0.6055|改善部分旧类，但持续伤害`cls_1825.../cls_f6a8...`；low Before-old类退化|
|rank8,gamma0.005|0.6667/0.3600/0.4675|0.6833/0.4200/0.5202|0.6500/0.4800/0.5522|旧遗忘下降但new floor显著恶化|

其中Z0、`gamma=0.001`和`gamma=0.0025`已使用最终module SHA`EAFE04305599A9572F06E915EAD2C2CAF697EAE4F095409F40EE788E2A66428D`完成真实nested L2O最小复跑。所有row均`old_score_bitwise_locked=true`，但两个正arm仍未通过逐类门。

最终nested选择成本也不满足快速适配目标：三个场景feature提取共约6.34秒；`gamma=0.001`三个场景分别71.53/68.34/65.68秒，`gamma=0.0025`分别62.58/61.59/61.22秒，总support诊断397.77秒。该复杂度是development安全筛选成本，不是单query延迟，但`O(CN^2)`nested重拟合不适合作为星上主适配实现。

当前方法结论固定为：

```text
MODULE_SAFETY_IMPLEMENTED
PERFORMANCE_SUPPORT_GATE_NOT_PASSED
SUPPORT_ONLY_NO_GO_TRUE_Z0
QUERY_NOT_OPENED
125_MATRIX_NOT_OPENED
```

因此D15只作为共享幅度失败机制和D16逐类非对称幅度路线的实现基础，不得注册为promotable candidate或deployment evidence。
