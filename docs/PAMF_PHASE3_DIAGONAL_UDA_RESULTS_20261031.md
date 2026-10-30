# 正对角仿射无标签适应结果（2026-10-31）

**结论：NO_PROMOTION。** 唯一主 `diagonal_predicted_ratio` 增加了旧类正确接收，但星地未知拒识损失更大：相对原 predicted，A_K +2.6042 pp、R_U −15.2778 pp、H −5.8282 pp。原 `predicted_ratio` 重点候选和 `pooled_ratio` 冻结对照保留，完整 Phase3 目标 ACTIVE。

Run `20261031-phase3-diagonal-uda-s2026092701-r01`，实际代码 `73f460c2e0c89b192e52af2e1b6ef01a576ca686`。固定公式与权限见[计划](PAMF_PHASE3_DIAGONAL_UDA_PLAN_20261031.md)；全部分组、配对及 42 条适应记录见[完整评分](../local_artifacts/pamf_phase3_diagonal_uda_20261031_r01/score/results.json)和[私有分析](../local_artifacts/pamf_phase3_diagonal_uda_20261031_r01/analysis.json)。

## 本轮方法与冻结顺序

原 F0 Tri-SIG-TIME SSL E50/11100/default6、r03、DURA v1 E20/4940、class-density/4af 的 known90/OE32 GMM、normalizer 和两组六阈值冻结。原 source whitener 得到 w 后，适配为 h=a⊙w+b，a 严格正，之后不再单位化；F0 默认身份 c 不变。SHIFT 固定 a=1、拟合 b160；DIAG 拟合 a/b320，两臂独立从单位变换开始，各固定 20 次 MAP-EM。

拟合源混合为 0.5known＋0.5OE；目标含 Σlog p_s(h)+NΣlog a+Nlog|W|，另加 λ=160 的单位变换锚定先验。λ 在 U 前固定，不是源数据调优值。Jacobian 必须进入 U 似然，在 known/OE 比值中抵消；本方法不是 CORAL 原算法，也不声称恢复物理信道。

只用旧 mixture-UDA/d72 的适应前 U 四字段 ids/default_logits/closed/signal_features；旧 prior/adapter/diag/cal/results 不进入拟合，F0 logits 不参与拟合，真实 TX/K/配额隐藏。实际三 checkpoint synthetic 各 2→source 状态冻结→U384 两适配器冻结→全部六列×三条件固定→独立 CPU 三份真值；U 真值读取 0。

source RX 不应用目标 RX 适配器，六列按原两列重复；新 RX 的 clean/star 共用同一 star-U 适配器，clean 不参与适应。原神经模型仍 source-only，整个新部署状态有 target-U 祖先，`source_only=False`，不能把 U 适应当作纯源训练。

## 全部 18 运行点

均为逐真 TX 宏平均百分比。A_K=旧类正确接收，C_K=接收，R_U=三非旧类拒识，H=harmonic(A_K,R_U)。source 重复行表示按接收机保持恒等，不能解释为适应在源域也产生收益。

| 条件 | 方法 | A_K | C_K | R_U | H |
|---|---|---:|---:|---:|---:|
| source | pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 |
| source | predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 |
| source | shift_pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 |
| source | shift_predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 |
| source | diagonal_pooled_ratio | 89.7704 | 97.1074 | 60.8889 | 72.5613 |
| source | diagonal_predicted_ratio | 89.7444 | 96.7593 | 61.9365 | 73.2914 |
| clean_DURA | pooled_ratio | 68.9236 | 85.4167 | 99.6528 | 81.4874 |
| clean_DURA | predicted_ratio | 70.8333 | 87.3264 | 100 | 82.9268 |
| clean_DURA | shift_pooled_ratio | 69.0972 | 85.5903 | 100 | 81.7248 |
| clean_DURA | shift_predicted_ratio | 71.0069 | 87.5000 | 100 | 83.0457 |
| clean_DURA | diagonal_pooled_ratio | 73.2639 | 90.7986 | 94.4444 | 82.5167 |
| clean_DURA | diagonal_predicted_ratio | 73.6111 | 90.6250 | 98.2639 | 84.1695 |
| star_DURA | pooled_ratio | 63.0208 | 84.8958 | 76.3889 | 69.0639 |
| star_DURA | predicted_ratio | 64.0625 | 85.5903 | 77.0833 | 69.9723 |
| star_DURA | shift_pooled_ratio | 63.0208 | 84.8958 | 76.0417 | 68.9217 |
| star_DURA | shift_predicted_ratio | 64.0625 | 85.4167 | 77.4306 | 70.1150 |
| star_DURA | diagonal_pooled_ratio | 66.8403 | 89.9306 | 60.4167 | 63.4663 |
| star_DURA | diagonal_predicted_ratio | 66.6667 | 89.2361 | 61.8056 | 64.1441 |

## 完整适应轨迹与配对效应

两臂 iteration0 至 20 共 42 条记录均已读取，数值有限、20 次更新目标全程非下降。共同初值 181695.05281440573；SHIFT 终值 182980.2905443705、offset norm=1.9081906559501747；DIAG 终值 186962.9954438197、尺度范围 0.6649160885532033 至 1.3917106154662513、offset norm=1.882684410540968。没有按 Q 早停或选迭代。

DIAG 的 known posterior mass 从 0.50756 到 0.58596，是固定混合模型的软责任质量，不是真实已知比例，也不能据此推断未知被吸收的 TX 组成。U 似然和固定 MAP 目标上升，未转化为未来联合指标达标。

主对原 predicted：clean 正确 16 增/0 失，接收 +19（另含 3 条错误身份），未知拒对 0 增/5 失；star 正确 15 增/0 失，接收 +21（另含 6 条错误身份），未知拒对 0 增/44 失。旧闭集身份未改变，增加的是原先被拒的正确接收，不是编码器或默认头识别率提高。

主对 SHIFT predicted：clean 正确净 +15、接收 +18、未知拒对 −5；star 正确净 +15、接收 +22、未知拒对 −45。主对 DIAG pooled：clean 正确净 +2、接收 −1、未知拒对 +11；star 正确净 −1、接收 −4、未知拒对 +4。全部三配对的 gain/loss、源42组和逐景结果保留；不能把 SHIFT 星地未知净 +1 的局部变化替换唯一主。

主 clean/star 正确接收为 424/576、384/576，接收为 522/576、514/576，仍含 98/130 条错误旧身份。未知拒对为 283/288、178/288；星地相对原 222/288 损失 44 条。A_K 增加不抵消 R_U 下降，也不能只看 clean 的 H 上升。

## 门槛与场景边界

同 DURA 原 F0 闭集 source/clean/star 为 90.9815%/80.0347%/72.9167%。主 clean/star 闭集到 A_K 损失仍为 6.4236/6.25 pp，均超 2；clean 的 C_K=90.625%、R_U=98.2639% 单项通过也不够，star C_K=89.2361%、R_U=61.8056% 未过。source known12 通过但 42 组未全过，且只是原校准复用；source 未知仍是开发诊断。

主 star 六景（high/mid/low suburban、high/mid/low urban）A_K 为 72.8571/75.8333/70.1114/77.7116/75.5093/55.9513%，R_U 为 N/A/74.0741/69.4726/82.5/70/43.6335%。high 缺 19-15，R_U/H 保留 N/A，完整覆盖未证实；不能删景、补零或以总体替代。

主 star 对 19-15/19-16/7-1 的拒识分别为 63/96、63/96、52/96。全 TX/RX/day/scene、source42 风险与逐类 Wilson 区间已完整保留；区间仅描述单一历史开发 RX/seed。星地 AUROC=85.2503%、OSCR=63.0528% 的排序诊断，不能证明当前固定阈值达到 90% 拒识，更不授权扫描新运行点。

## 数据、成本和执行

源契约392005/VALIDATED_ONCE：L6300/U_s56700/OE6134 为祖先，本轮新读0；V27000 是额外已知标签的原校准与本轮源诊断，NV3150 为 19-15/19-16/7-1 各1050。源5 RX/三日和 G10 MIX01 细节沿[计划](PAMF_PHASE3_DIAGONAL_UDA_PLAN_20261031.md)。U 为 RX24-1/03-08，future clean/star Q 为同 RX/03-15 的同864物理记录，旧576/非旧288、九类各96；三测试均 DURA，clean仅测试，不重抽 received 或增加物理样本。

新 NN/源 Gaussian/normalizer/阈值拟合及真实 G/F0/signal 前向、source L/OE 读取均0；三 CP synthetic 各2另计。**目标统计拟合不是0**：两臂各20次，共40次；每臂全U384更新对应7680条次、合计15360条次统计更新，另有iteration0诊断，不是更多独立物理样本或NN前向。

SHIFT 有效160参数，保存固定a及b数组共2720 B；DIAG 有效320参数、数组2716 B，两文件差4 B来自 kind 字符串长度。原密度364340 B、NN834665参数不变。原12个唯一τ96 B无新V校准，六列重复存36值288 B，单主6值48 B。不部署U/source逐行bank；原 F0 11100/r03+2220/MLP2220/G4940、源密度和缓存成本继承。

pipeline ANALYZED_DEVELOPMENT，两job COMPLETE/returncode0；predict/CPUscore/总过程为9.470567/4.272151/13.853247 s，非逐包时延。三份完整生产stdout共7行，warning/error/traceback为0；三次观察独立捕获driver1268991/CPU1269163的CWD/完整argv/CUDAempty。predictor1269086仅有pipeline指派GPU4、smoke/产物证据，未独立捕获live或显存；显存/峰值UNKNOWN，日志0→predict627 B/score42 B/driver21 B，终态live/ownedGPU空、foreign0。

最终验证为9 unique＋30 sub的并集PASS，不是初轮全绿：worker初轮6＋14sub通过，pipeline fixture错误阻止合法scorer读U，fixture-only定点1＋9sub通过；root CLI先1＋3sub通过，另一个旧marker fixture修正后仅该case1＋4sub通过。生产未因此改动，初失败记录保留，696个旧trapz提示属合成测试。AST7/镜像8、唯一P0/P1 PASS；单包669files/data0、SHA `a160a90887001dec92b806600c2b72e368aebda6124e68e232723698f385ed13` 比较、compile/runtime VERIFIED。

本轮U真值读取0，NV/Q历史曝光、单RX/seed、源日期重合，formal RX24-18未读。没有发现或匿名注册，A_N/H_OW为N/A；联合适应门槛失败，NO_PROMOTION、全目标ACTIVE。本次结果整理未额外拟合、评分或遥测补跑。
