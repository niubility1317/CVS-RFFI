# D15类局部pairwise margin threshold追踪

> 2026-07-17机制结论：该路线降级为control。D8b outer-held support上的post-hoc阈值Oracle诊断即使用held真值直接优化5个new常数阈值，三场景仍存在总逐类缺口clear 0.30、low 0.20、rain 0.60。该机制本质接近D13 class-local constant bias，不能同时保护old与new。禁止扩展真实runner、promotion或query；现有纯module/合成tests仅保留为阈值control实现证据。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D15-01|`项目.md`§7.1/7.1.1|只接收runtime-authorized固定单LEO_weak表征；不得读取clean/source/query truth/role/quota|D15 module/tests|pending|待artifact/binding反例|单物理sample不增加K|
|D15-02|strict K10|开发使用每类恰10个独立support；joint L2O每类held2/train8|D15 module/tests|pending|待exact-K和fold计数|K1/K5/K20必须独立package|
|D15-03|Before floor|train-K8内部LOO累计old-old对称碰撞，最多选择3条正权endpoint-disjoint pair|D15 module/tests|pending|待非贪心matching和held变异反例|无dense class/query图|
|D15-04|Before阈值|每条old pair只校准一个直接作用cosine margin`s_a-s_b`的有界threshold；仅在immutable base top2为该pair时应用|D15 module/tests|pending|待正/负threshold与score实现|old pair score用零和等价修正|
|D15-05|After注册|锁定Before old prototype、pair、threshold和全部old scores；每new类仅校准`new-max(Before old)`上的identity或有界正/负threshold|D15 module/tests|pending|待old score逐位锁和动态max-old反例|threshold>0抑制new，threshold<0抬高new|
|D15-06|train-only|pair、prototype、old threshold、new threshold全部只来自outer train K8内部LOO；held2只评分|D15 module/tests|pending|待held old/new极端变异|support selection SHA与state tensor不变|
|D15-07|统一cap|三场景统一候选网格`cap∈{0,0.01,0.025,0.05}`；Z0必须真实identity|D15 runner/后续|rejected|held-label Oracle仍无法关闭逐类缺口|禁止扩runner|
|D15-08|联合门|Before逐old类不低于base；After逐old类不低于Before/alpha0、逐new类不低于alpha0，floor/H/joint非退化|D15 module audit/tests|pending|待joint L2O字段|正/负threshold均须联合recheck|
|D15-09|资源|0参数、0epoch、state只保存prototype、最多3个old pair threshold和每new一个threshold；目标<80KiB，硬门256KiB|D15 module/tests|pending|待真实array bytes|无FFT、无dense图|
|D15-10|formal API|恰好一个query、全部注册类、runtime/code/checkpoint/operator绑定|D15 module/tests|pending|待单query反例|不开放真实query|
|D15-11|安全回退|全部正cap失败时真实保存cap0、空old pair、新threshold全0且alpha0逐位等价|D15 module/tests|pending|待随机score parity|不保存失败arm|
|D15-12|Git/验证|只新增D15独立trace/module/tests；`ssr-gpu` py_compile/pytest/diff-check；不提交Git|D15 files|pending|待验证|保护共享脏树|

## 预登记机制

### Before old-old阈值

outer fold的train K8内部逐物理sample LOO。对每个unordered old pair累计误分类到对端和固定近碰撞权重，再做确定性最大权endpoint-disjoint matching，最多3条正权pair。

对已选pair`(a,b)`收集内部LOO margin：

```text
m_ab(x) = s_a(x) - s_b(x)
```

只用真实类为`a/b`的train内部LOO margin枚举有界threshold候选：

```text
T_ab in clip({0} union all adjacent-margin midpoints, -cap, +cap)
```

以`min(acc_a,acc_b)`、pair overall、`-|T|`、稳定数值序选择。预测时只有immutable old-only base top2恰为`{a,b}`才使用：

```text
score_a' = score_a - T_ab / 2
score_b' = score_b + T_ab / 2
```

因此`score_a'-score_b'=m_ab-T_ab`，阈值直接作用cosine margin。

### After动态max-old阈值

After保持全部Before old scores逐位不变。对每个new类`n`，train K8内部LOO收集：

```text
m_n(x) = s_n(x) - max_c s_c^Before(x)
```

old样本负例使用其train-only D15 Before old scores；new样本正例使用内部LOO new prototype与同一Before old函数。枚举`[-cap,+cap]`内margin midpoint和0，以`min(old_negative_acc,new_positive_acc)`、balanced accuracy、`-|T_n|`选择。

正式After仅修改该new score：

```text
s_n'(q) = s_n(q) - T_n
```

不同new类都读取同一个pre-correction Before old max和自身未修正new score，不递归读取其他new修正。

## 本轮边界

优先完成纯module与合成tests。真实D8b runner若后续实现，必须继承D14修复后的外部expected seal SHA、authority fail-closed、development selection与confirmation apply-locked分离、真正state反序列化和NO_QUERY边界。
