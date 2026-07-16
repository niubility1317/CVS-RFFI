# D14稀疏pairwise Fisher双阶段门追踪

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D14-01|`项目.md`§7.1/7.1.1|每个物理sample只读取一个sealed LEO_weak received-IQ；任何base/operator representation均绑定同一父IQ SHA且不增加K|D14 module/runner/tests/audit|verified|runner pre-open、actual IQ SHA、physical batch1、跨场景token/SHA互斥；聚焦测试通过|禁止clean/raw、第二LEO状态和跨场景物理复用|
|D14-02|`项目.md`exact-K|K10开发只读取每类恰好10个独立物理support；fold按physical ID成组held；K1/K5/K20必须使用独立exact-K package|D14 runner/tests|implemented|strict K10/rank0..9与伪K反例已测；K1非法LOO边关闭已测|K1/K5/K20真实评估仍须独立package|
|D14-03|用户floor优化要求|Before从old fold-train support选择最多3条互不共享端点的旧—旧碰撞边，并以稀疏对角Fisher证据改善旧类floor|D14 module/tests/log|verified|train-K8内部LOO、正权deterministic maximum-weight matching、非贪心反例和端点互斥通过|无dense class/query图|
|D14-04|注册抗遗忘要求|After完整锁定Before旧prototype、旧pair和全部旧类score；每个new类最多追加1条new—old rival边，只修改对应new score|D14 module/tests|verified|随机probe与joint L2O均验证旧score逐位相等；每new单rival|old-like证据压低new，new-like证据抬高new|
|D14-05|固定received-IQ变换|全局operator仅可从`base/WL-IQ/FFT-EQ`中按K10 support-only统一锁定；所有派生representation共享physical ID、scenario、satellite seed和父IQ SHA|D14 runner/artifact audit|deferred|首轮显式锁`base_only_mvp`；D10辅助operator无完整统一provenance/global winner|WL-IQ/FFT-EQ不进入当前网格|
|D14-06|support-only选择|每fold pair、均值、方差、Fisher方向、rival和band触发统计仅来自old/new train K8；held2/query/scorer均不可影响state|D14 module/tests/log|verified|held2极端变异不改变selection SHA、edge/rival和压缩tensor SHA；rival使用train-only D14 Before|parent artifact SHA允许随held内容改变|
|D14-07|三场景统一超参|统一锁定`operator/ridge/gamma_old/gamma_new/B_select_old/B_old/B_new`；场景内只按合法support闭式构造状态|D14 runner/audit|verified|4-arm base-only lock SHA`d62763f4...9180a`覆盖三场景；全部正arm失败|不得按场景挑arm，候选未选择|
|D14-08|逐类硬门|Before每个旧类不低于base；After每个旧类不低于D14 Before和alpha0；每个new类不低于alpha0，且old/new floor、H、joint均非退化|D14 runner/audit|verified|C1虽保持new逐类并提高clear new overall至0.52，但三场景old forgetting仍为0.0833/0.0500/0.1000|未过门不开放query，候选未选择|
|D14-09|安全回退|全部正候选失败时真实保存无edge、gamma0、base逐位等价状态，并保持`NO_QUERY_OPEN`|D14 module/runner/tests/COMMIT|verified|真实保存`d14_z0_true_zero_base`；COMMIT SHA`a3df5083...709b`|不把失败arm保存为最终state|
|D14-10|逐样本部署|formal预测恰好1个物理query、面对全部注册类；无truth/role/quota/true batch count/global assignment/query fit|D14 formal API/tests|verified|单query/all-class与runtime/operator binding反例通过|runner无query/scorer输入|
|D14-11|资源硬门|0训练参数、0epoch、总序列化state≤256KiB；优先≤80KiB；报告forward/FFT触发、MAC、时延、显存和相对single-qKNN Pareto|D14 runner/audit|verified|after实际array state18,492B、NPZ+JSON 22,420B；MAC较single-qKNN低90%；0参数0epoch|Python safety wrapper微基准较裸矩阵慢，非正式query时延|
|D14-12|实验范围|先在D8b真实三场景strict K10 enrollment-only包完成support-only选择；只有全部门通过才允许candidate-bound query|D14 runner/artifact/report|verified|artifact`d14_sparse_pairwise_fisher_guard_v1`；状态`SUPPORT_ONLY_D14_NOT_SELECTED_NO_QUERY_OPEN`|未过门，不进入query/125，候选未选择|
|D14-13|AGENTS.md/Git|本地`ssr-gpu`验证、独立D14文件、检查diff并选择性提交|tests/trace/handoff|implemented|`py_compile`与17项聚焦pytest通过；`git diff --check`通过|主任务明确要求本子任务不stage/commit|
|D14-14|独立红队P0-1|before/after detached seal必须由调用方提供独立expected SHA，禁止从待验证seal自身现算|D14 runner/tests|verified|CLI强制`--before-seal-sha256/--after-seal-sha256`；自哈希反模式静态反例通过|expected与actual均写入audit/COMMIT|
|D14-15|独立红队P0-2|pre-open `control_state/formal_launch_authority`冲突必须fail closed；无独立authority PASS不得promotion|D14 runner/tests|verified|`LOCAL_PROTOCOL_REPAIR_REQUIRED`反例强制diagnostic；formal promotion同时依赖support gate与authority gate|当前D8b结构包永不promotion|
|D14-16|独立红队P0-3|NPZ+JSON必须真正重建state并完成预测逐位roundtrip；formal模式只接受FORMAL_SELECTED且promotion/support/authority全真，diagnostic模式必须显式开启；两种模式均绑定外部COMMIT、candidate/runtime/checkpoint/feature code|D14 module/runner/tests|verified|匹配formal COMMIT加载PASS；diagnostic默认拒绝；candidate/runtime/checkpoint/feature-code漂移均fail closed；17项聚焦测试通过|当前真实Z0仍无formal资格|
|D14-17|独立红队P0-4|全部正arm失败必须由完整runner选择Z0并实际保存/重建三场景before/after空edge gamma0 state|D14 runner/tests/artifact|partial|纯选择函数、手工COMMIT和单state Z0重建通过|尚缺最新完整`run()`三场景失败→六个Z0 state→audit/report/COMMIT端到端证据|
|D14-18|独立红队P0-5|开发选择与确认apply-locked必须分离；当前runner不得在确认receiver/seed重选|D14 runner/tests|implemented|当前CLI仅接受`development_select`，其他mode fail closed|confirmation apply-locked runner/API尚未实现，不能进入确认矩阵|

## 预登记机制

对固定received-IQ的锁定operator表征`z`，旧—旧或新—旧pair`(a,b)`使用对角Fisher方向：

```text
w_ab = (mu_a - mu_b) / (var_a + var_b + ridge)
denom_ab = sqrt(sum(w_ab^2 * (var_a + var_b + ridge))) + eps
direction_ab = w_ab / denom_ab
bias_ab = -direction_ab dot ((mu_a + mu_b) / 2)
e_ab(q) = z_q dot direction_ab + bias_ab
```

`var`、`w`和midpoint只允许在support-only闭式fit期间存在；持久化部署state只保存压缩后的`direction+bias`，不得保存全类variance或每edge完整midpoint。new edge只保存new类段或实际active edge，不为old类预留密集零矩阵。new20数组state目标低于80KiB。

Before只在base top-2命中预登记旧pair且base margin落入`B_old`时执行有界零和修正；After保持全部旧类score等于D14 Before，仅在new类与其唯一old rival处于`B_new`窄带时，对该new score加入`gamma_new*clip(e_new_old,-1,1)`。

所有计算只读取固定sealed LEO_weak IQ或由其生成的已登记接收后representation。任何representation均不是新物理sample，不增加K，也不得调用LEO channel simulator。

### fold-train-only边选择

每个joint K10 L2O fold的held ranks先固定为2个物理sample/类。下面全部pair/rival选择只能读取其余train K8：

1. 在train K8内部执行leave-one-physical-sample-out old-only cosine预测；每个内部held sample使用其余同类7个和其他类8个sample构造prototype。
2. 对每个unordered old pair累计对称碰撞权重：误分类到对端计1；正确但对端为runner-up且margin落入预登记`B_select_old`时，累计固定线性近碰撞权重。
3. 在最多6个old类上执行确定性最大权匹配，最多保留3条互不共享端点的正权edge；完全同权时按opaque class handle的稳定字典序破同。不得贪心生成共享端点或在held2结果上重选边。
4. 对每个new类`n`，使用train K8内部LOO prototype分数，分别统计每个old类`r`上的Before-correct old margin与new margin，并按固定quantile风险式选择唯一rival：

```text
R_rn = Q_qo(s_n - s_r | old-r train sample is Before-correct)
       - Q_qn(s_n - s_r | new-n train sample)
```

5. 缺少Before-correct old-r样本、new-n内部LOO样本、有限margin或有效非零Fisher方向时，该`new-old`edge关闭；不得用错误旧样本、held2或full K10补统计。

held2只能用于产生不可回流的support-only晋升指标。极端修改held2允许改变parent artifact/content SHA和held预测，但不得改变train selection SHA、edge集合、prototype/Fisher张量或任何校准诊断。

### 不可递归门控与真实零回退

- Before旧—旧edge的top-2和margin必须只由immutable old-only base scores计算；不得把new score加入旧edge触发条件。
- After new—old修正必须使用修正前的new score和锁定D14 Before old score触发；多个new edge不得递归读取其他new edge的已修正结果。
- 真正zero回退必须同时满足`operator=base`、`old_old_edges=empty`、`new_old_edges=empty`、`gamma_old=0`和`gamma_new=0`，并在随机feature上与alpha0 logit和prediction逐位一致。
- 方差统一使用`ddof=0`，ridge必须严格大于0；Fisher方向或归一化分母为0/非finite时edge关闭。K1不得通过NaN、Inf或伪随机方向继续预测。

## 晋升与回退

正候选必须在三个场景同时通过逐旧类、逐新类、floor、forgetting、`H_old_new`、joint和资源门。全部正候选失败时，最终状态必须无pair edge、实际gamma为0、与base逐位等价，状态为`SUPPORT_ONLY_D14_NOT_SELECTED_NO_QUERY_OPEN`。

## D8b strict K10真实support-only结果

输入仅为`receiver=1-20,seed=713201,new5`的before/after enrollment-only密封包；before为每scenario `6×10`，after为`11×10`，三个scenario的physical token与父IQ SHA集合互斥。未打开query、truth、prediction、score或scorer。

输出：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d14_sparse_pairwise_fisher_guard_v1`

该artifact生成于独立红队P0-1至P0-5修复前，只能保留为`PRE_FIX_DIAGNOSTIC_SUPPORT_ONLY`。它没有外部expected seal信任根，且输入pre-open audit为`control_state=LOCAL_PROTOCOL_REPAIR_REQUIRED`、`formal_launch_authority=false`；因此即使support门通过也不能promotion。其NO-GO性能结论仍可作为本地support诊断，但不能作为正式protocol PASS、可部署state或query开启依据。

- 状态：`SUPPORT_ONLY_D14_NOT_SELECTED_NO_QUERY_OPEN`
- 最终candidate：`d14_z0_true_zero_base`
- COMMIT SHA-256：`a3df5083474879ab1d2f4868fb838de63732891c1c4aee9339af6e695af7709b`
- support audit SHA-256：`7fbbe25e8e9297bfb5ef5cc868adeedfd6e9c24475bdcfaec17a02a39b99b4ec`
- training log SHA-256：`64d7223a472e9c2a895c0f331db5dc2878abd8c98544bc0c9bf6e0b559dd00fc`
- report SHA-256：`6e138f3a0d1886052f0f62bd8db95e01f4da3f45aa45537e93c4153807462589`

|场景|Before old/floor|Z0 After old/floor|Z0 new/floor|Z0 H/joint|Z0 forgetting|C1 new overall|C1 forgetting|
|---|---:|---:|---:|---:|---:|---:|---:|
|`leo_clear_weak`|0.7167/0.1000|0.6333/0.1000|0.5000/0.2000|0.5588/0.5727|0.0833|0.5200|0.0833|
|`leo_low_elev_weak`|0.7333/0.3000|0.6833/0.2000|0.4600/0.1000|0.5499/0.5818|0.0500|0.4600|0.0500|
|`leo_rain_weak`|0.7333/0.3000|0.6167/0.3000|0.6200/0.4000|0.6183/0.6182|0.1167|0.6200|0.1000|

C1在clear把new overall从0.50提高至0.52，并在三场景保持逐new类不低于alpha0，但没有恢复注册导致的旧类遗忘；C2/C3在clear把After old从0.6333提高至0.6500，却把new floor从0.20降至0.10，并在low/rain同时退化new、H或joint。因此D14证明：仅靠窄带稀疏pairwise new-logit修正无法同时跨越“旧类注册竞争遗忘”和“new floor非退化”两道门，不能开放candidate-bound query。

资源侧达到极轻目标：0参数、0epoch、1次backbone forward、0个FFT分支；每场景After array state为18,492B，实际NPZ+JSON为22,420B，均低于80KiB偏好门和256KiB硬门；估算head MAC相对identity-only single-qKNN降低90%。Python安全包装微基准比裸矩阵乘慢约84–109倍，仅作为support-row实现开销诊断，不能解释为正式query端到端时延。
