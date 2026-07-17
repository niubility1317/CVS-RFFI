# D21 M6 support-fold低秩投影诊断

## 结论

`NO_GO_SUPPORT_GATE`。四个预注册候选均未通过类均衡support-fold门，停止路线，不进行full-support refit，不生成最终patch、int8 head、prediction或score，也不打开任何query输入。

跨3个LEO_weak场景×2个fold，恒等基线的平均support-validation H为47.31%，四个M6候选均为47.00%，下降0.32pp；seen-new从46.67%降到45.33%。候选间没有产生可区分的验证预测，rank与损失权重无法由support证据支持，因此不能锁为部署方法。

## 严格SUPPORT_ONLY_NO_QUERY边界

- CLI只接受`--enrollment-root`与`--output-dir`，其中输入路径必须精确指向`predictor/after/enrollment_only`。
- 输入manifest必须满足`schema=cvs.phase2.somph_predictor_bundle.v1`、`profile=enrollment_only`、`registration_state=after`。
- manifest成员全集必须恰为6项：sealed runtime、method lock、overlay provenance和3个registered-support文件。额外成员、绝对路径、`..`以及含query/truth/scorer/apply_only/before的路径均fail closed。
- 实际只打开manifest、sealed runtime和3个support NPZ；NPZ只读取`support_leo_weak_iq`与`support_class_indices`。
- 未加载query IQ、query token、truth sidecar；没有query训练、适配、校准、选择、早停、回滚、候选排名、角色Oracle、真实batch类数、类别配额或全局分配。
- 没有prediction或score接口和产物。独立输入访问清单见`query_unreachable_proof.json`。
- 3个support文件均是密封的单一LEO_weak观测；未读取clean/source样本、特征或衍生信号。

## 方法与锁定协议

精确模型白名单仅含：

- `model.id_backbone.cls_head.id_proj.0.weight`，形状`160×160`；
- `model.id_backbone.cls_head.id_proj.0.bias`，形状`160`。

DOM分支及其余参数全部冻结，运行时保持eval模式，避免非白名单buffer更新。权重差分参数化为`ΔW=A·B`，rank仅取2或4；bias使用`Δb`。每次fold适配使用SGD、`lr=0.05`、momentum=0、5epoch=5个full-support step，没有optimizer持久状态。

损失为support-only CE、逐类CVaR、old pair保持、old/new双向分离和恒等近端。固定候选为rank2/4×`balanced/old_guard`两套预注册权重。K=10 support分成2个类均衡fold，每类5条训练、5条验证；所有rank和权重只由这些support folds评估。

门要求平均H至少提升0.5pp，同时old floor和new floor均不得退化。没有候选通过。

## 聚合support-fold结果

| rank | 权重preset | base old | adapted old | base new | adapted new | base H | adapted H | ΔH | worst old floor | worst new floor | 门 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | balanced | 48.33% | 49.44% | 46.67% | 45.33% | 47.31% | 47.00% | -0.32pp | 0→0% | 0→0% | FAIL |
| 2 | old_guard | 48.33% | 49.44% | 46.67% | 45.33% | 47.31% | 47.00% | -0.32pp | 0→0% | 0→0% | FAIL |
| 4 | balanced | 48.33% | 49.44% | 46.67% | 45.33% | 47.31% | 47.00% | -0.32pp | 0→0% | 0→0% | FAIL |
| 4 | old_guard | 48.33% | 49.44% | 46.67% | 45.33% | 47.31% | 47.00% | -0.32pp | 0→0% | 0→0% | FAIL |

floor为0说明仅靠当前K=10 support内部5/5切分无法形成可靠逐类门；这本身是NO-GO证据，不能通过放宽floor门掩盖。

## rank2 balanced逐场景/fold明细

| 场景 | fold | base old/new/H | adapted old/new/H | base old/new floor | adapted old/new floor |
|---|---:|---:|---:|---:|---:|
| clear | 0 | 46.67/56.00/50.91% | 50.00/56.00/52.83% | 0/0% | 0/0% |
| clear | 1 | 63.33/56.00/59.44% | 56.67/56.00/56.33% | 40/0% | 20/0% |
| low_elev | 0 | 46.67/44.00/45.29% | 56.67/44.00/49.54% | 0/0% | 0/0% |
| low_elev | 1 | 33.33/36.00/34.62% | 33.33/36.00/34.62% | 0/0% | 0/0% |
| rain | 0 | 50.00/44.00/46.81% | 50.00/36.00/41.86% | 20/0% | 20/0% |
| rain | 1 | 50.00/44.00/46.81% | 50.00/44.00/46.81% | 20/0% | 20/0% |

## 训练拟合判断

rank2 balanced的fold0训练loss从clear 5.822降到5.331、low_elev 7.467降到7.107、rain 7.888降到7.145；FP16 factor中约732–736/800个标量非零。但跨fold验证H仍下降，且rain fold0的seen-new从44%降到36%。这属于support训练子集改善未转化为support-heldout收益的非泛化/轻度support-fold过拟合信号，不能进入query测试。

## 资源审计

| 项目 | rank2 | rank4 |
|---|---:|---:|
| 低秩可训练状态参数 | 800 | 1440 |
| FP16 factor payload | 1600B | 2880B |
| merge触及的原模型参数 | 25760 | 25760 |
| merge后新增推理MAC | 0 | 0 |

24次fold拟合总计59.66秒；每次5step，远低于50step上限；峰值CUDA显存76566016B（约73.02MiB）。

NO-GO下没有物化最终patch或head。仅作部署预算上界审计：support排序首行是rank2，其FP16 factor上界1600B；11类`160-D int8 prototype+FP16 per-class scale`上界1782B；合计3382B，小于262144B。但`deployment_export_authorized=false`，该算术不能被表述为已有可部署产物。

## 证据

- `support_fold_log.jsonl`：120条epoch记录+24条fold验证记录；
- `selector_lock.json`：固定候选、support-only聚合和NO-GO门；
- `resource_audit.json`：参数、step、时间、显存和状态预算；
- `query_unreachable_proof.json`：输入schema、成员allowlist及实际访问审计；
- `fold_fp16_factors.npz`：仅用于复核的24组fold低秩状态，不是最终deployment patch；
- `../run_m6_support_fold_lowrank.py`：可复现runner。

本次未提交Git，未生成任何query结果。后续若研究M6，必须形成新的、预注册的support-only假设；禁止使用本次未打开的query来调rank、损失或门限。
