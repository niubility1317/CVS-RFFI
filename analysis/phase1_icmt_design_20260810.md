# P1-ICMT冻结设计与实现追踪卡

状态：`LOCAL_VERIFIED_V2_P1_FIX_PENDING_INDEPENDENT_P0_P1_REVIEW`；设计审查：`P0=0/P1=0/ALLOW-MERGE`；postfreeze v2实现复审已关闭原两项P0，正式proxy输入冻结的`P1=1/REVISE`已完成本地最小修复，但尚未获得独立实现复审，实验结果亦未产生。

## 冻结对象

P1-ICMT（Independent-view Classwise Margin-Tail Tightening）以GeoSat-C continuation为共同基座。六个LOTO fold各有C/G两臂，共12臂；均从相同`training_final_only`warm-start模型权重开始，保持head/class order、物理样本与批序列、seed、sampler、40E、单次LEO forward、`(epoch+batch_idx-2)%3`场景循环、新AdamW/AMP初态及共同`L_base`不变。C不启用ICMT；G唯一增加ICMT，不增加forward、重采样或训练长度。

对同一L-batch的clean与单次LEO视图，令`z_id=feat_joint`，`ell_i^v=id_backbone.cls_head.head(z_id_i^v)`为raw pre-softmax local4 logits，`v∈{clean,leo}`。对类别`c`，

\[
m_i^v=\ell_{i,y_i}^v-\log\sum_{k\ne y_i}\exp(\ell_{i,k}^v),\qquad
\bar m_c^v=\frac{1}{n_c}\sum_{i:y_i=c}m_i^v,
\]

\[
L_{\mathrm{ICMT}}=\frac18\sum_{v,c}\frac1{n_c}\sum_{i:y_i=c}
\left[\max\{0,\operatorname{sg}(\bar m_c^v)-m_i^v\}\right]^2,
\quad L_G=L_{base}+0.05L_{\mathrm{ICMT}}.
\]

均值分母始终为该类全部`n_c`；仅严格`m_i^v<sg(bar m_c^v)`的行active，tie贡献0；`1/8`与`0.05`不按active重归一。ICMT是source-L中通用逐样本低margin尾部收紧假设，不预先声称可识别地修复某个RX/day；F2/F4/F5的18格floor和严格proxy双门是唯一后冻结检验。

G的每个辅助batch要求四个local TX均`n_c>=2`，且两视图`z_id`和logits有限；否则在backward前fail-closed。首个有效G batch以raw、unscaled VJP记录ICMT相对`L_base`对共享`z_id`encoder及精确分类head的norm/cos，二者都必须finite且nonzero；这只是诊断，不能反向调参。任何detach、head-only、None或零梯度均fail-closed。

每个训练批的冻结更新顺序是：读取L→共同clean forward→共同单次LEO forward→计算共同`L_base`→G计算ICMT并在首个有效批完成raw-unscaled VJP诊断→合成`L_G`→同一AMP/backward/clip/AdamW step；C跳过ICMT但保留其余序列。launcher固定`phase1_icmt12_20260810_v1`、6fold×C/G、seed`7281105`、40E与既有GPU映射，每张物理GPU至多两臂。

训练更新与ICMT严格`L-only`：仅`source_known_train`的L标签进入loss、backward和optimizer。共同trainer可以构建U loader，但不迭代U、不给U做forward；V仅执行C/G共有的`source-validation diagnostic forward`并读取V标签计算诊断指标，V不进入loss、backward、optimizer、校准或选模。proxy/held保持零loader、零forward。receipt中的`uses_*_rows=false`仅表示ICMT与训练更新不消费这些行，不表示共同trainer未构建U loader。C/G收据封存clean×4与LEO×4×3共16格的rows/active/finite：每场景全类覆盖、`active<=rows`，且终态每个clean类别rows等于其三种LEO场景rows之和。收据还绑定initial checkpoint SHA、head/class order、物理批/数据顺序SHA及新AdamW初态。C的ICMT字段为`N/A`或0。

后冻结42步已冻结为独立实现目标：12个ICMT专用clean export（单次输出L-fit、V-known、proxy，U只核hash且zero loader/forward/persist）、12个source-only三LEO export、12个proxy/binding step和6个同fold C/G pair。Gaussian-NLL仅在L上以同一路径`z_id=feat_joint`作float64 totalized-L2拟合，V/proxy零fit。门包括clean6/6、LEO18/18四floor、三场景及18格overall非负、每fold`ΔAUROC>0`和`Δ(mean u_proxy-mean u_V)>0`。任一完整门失败即`REJECT_P1_ICMT_PERMANENT`，均值不得补偿floor；实现与局部测试不构成门已执行或性能结论。

正式source proxy选择与已签GD postfreeze v2保持同一唯一口径：days=`2021_03_01,2021_03_08`、RXs=`1-1,1-19,14-7,18-2,19-2,2-1`、selection seed=`7281148`（训练seed`7281105+43`）、每个proxy TX最多`400`个物理样本、单fold总数严格为`400`。这些值不是CLI可调超参数；clean export、manifest、proxy JSON/CSV绑定、pair及F6 prior重算必须分别重新断言，不能接受prior自报的正整数count。

## 实现追踪

| ID | 冻结要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|
| ICMT-01 | 固定公式、raw logits、全`n_c`均值、strict tie0、固定`1/8×.05` | `code/cvsrffi/phase1_icmt.py` | local verified | 公式与tie单测 | 禁止温度、EMA、q、quantile和梯度投影 |
| ICMT-02 | local4`n_c>=2`、有限性和label permutation fail-closed | `code/cvsrffi/phase1_icmt.py`、`code/tests/test_phase1_icmt.py` | local verified | focused tests | 不读取RX/day/domain |
| ICMT-03 | 同一`z_id`—exact head路径和raw VJP双scope审计 | `code/cvsrffi/phase1_icmt.py`、`code/SSDG/train_ssdg.py` | local verified | focused tests | 不允许head-only或detach |
| ICMT-04 | 16格rows/active/finite闭合、终态clean=3×LEO、C字段N/A/0 | `code/cvsrffi/phase1_icmt.py`、`code/SSDG/train_ssdg.py` | local verified | focused tests | 每场景四类覆盖 |
| ICMT-05 | 严格warm-start、head/class与data-order绑定、新AdamW初态 | `code/cvsrffi/phase1_icmt.py`、`code/SSDG/train_ssdg.py` | local verified | focused tests | 只加载model权重 |
| ICMT-06 | 12臂40E同资源launcher与显式bash调用 | `code/scripts/launch_phase1_icmt12_20260810.sh` | local verified | `bash -n`、dry-run12 | 固定通过`bash scripts/launch_phase1_icmt12_20260810.sh`调用；run id为`phase1_icmt12_20260810_v1` |
| ICMT-07 | 实际lite_d no-query smoke和回归验证 | `code/tests/test_phase1_icmt.py` | local verified | `pytest` | 不实现postfreeze |
| ICMT-08 | 后冻结42步及非补偿门 | `code/export_phase1_icmt_features.py`、`code/export_phase1_icmt_leo_features.py`、`code/evaluate_phase1_icmt_postfreeze_pair.py`、`code/scripts/launch_phase1_icmt_postfreeze_20260810.sh`、`code/tests/test_phase1_icmt_postfreeze.py` | local verified v2 | postfreeze focused pytest、GD模板回归、dry-run42 | 独立ICMT schema/receipt，不复用GD pair JSON |

### Postfreeze实现追踪

| ID | 冻结要求 | 目标文件 | 状态 | 验证 | 备注 |
|---|---|---|---|---|---|
| ICMT-PF-01 | 逐checkpoint重建同一local4 L/U/V；只forward并保留L、V、proxy，U仅核hash且zero loader/forward/persist | `code/export_phase1_icmt_features.py` | local verified | split/hash/role/U-loader负测 | clean一次输出三角色；绑定训练receipt与final-only checkpoint |
| ICMT-PF-02 | 仅L的`z_id=feat_joint`拟合Gaussian；float64 totalized-L2；ddof=1、class-equal pooled、`.9/.1` shrink、`1e-6` floor；完整NLL与stable logsumexp连续`u` | `code/evaluate_phase1_icmt_postfreeze_pair.py` | local verified | 公式、positive等价、zero、nonfinite与L-only fit测试 | V/proxy零fit；所有L/V/proxy行保留 |
| ICMT-PF-03 | C/G×L/V/proxy封存total/zero/nonfinite/retained/dropped，L逐类计数闭合 | `code/evaluate_phase1_icmt_postfreeze_pair.py`、`code/tests/test_phase1_icmt_postfreeze.py` | local verified | L/V/proxy zero保留及计数篡改负测 | zero允许且按`T(0)=0`计分；nonfinite fatal |
| ICMT-PF-04 | clean6/6与LEO18/18四floor不低于`C-2pp`；每fold三场景overall及全18格overall不低于0；proxy两项逐fold严格正且6/6 | `code/evaluate_phase1_icmt_postfreeze_pair.py` | local verified v2 | 原始artifact重算、clean/LEO delta及同步摘要篡改负测 | 所有门非补偿；无阈值、选参或重试 |
| ICMT-PF-05 | pair严格绑定ICMT训练root、`F{fold}{C|G}_ICMT12`、checkpoint SHA、head/class、matrix/output root与fold1..6恰好一次 | `code/evaluate_phase1_icmt_postfreeze_pair.py` | local verified v2 | prior raw artifact当前SHA替换、跨run与计数篡改负测 | 不读取或复用GD pair receipt |
| ICMT-PF-06 | 固定v1 root、冻结GPU映射和`12+12+12+6=42`步 | `code/scripts/launch_phase1_icmt_postfreeze_20260810.sh` | local verified | `bash -n`、dry-run=`12+12+12+6=42` | 只后冻结导出/评分，不训练、不覆盖 |
| ICMT-PF-07 | 每个ICMT LEO步骤核冻结ManySig路径/SHA并封存NPZ SHA、source selection与可重建physical-key receipt；逐场景TX/RX/day完整 | `code/export_phase1_icmt_leo_features.py`、`code/evaluate_phase1_icmt_postfreeze_pair.py` | local verified v2 | 非冻结dataset字节、错误dataset绑定、source artifact替换及单场景缺day负测 | binding与LEO export同一步产生，不增加第43步 |
| ICMT-PF-08 | F6逐个重读F1–F5绑定的clean/LEO/proxy/sidecar，核当前SHA并重算分类摘要、连续几何与fold gates后逐字段比对 | `code/evaluate_phase1_icmt_postfreeze_pair.py`、`code/tests/test_phase1_icmt_postfreeze.py` | local verified v2 | clean/LEO delta、同步摘要+delta及prior raw artifact替换负测 | 聚合只消费重算结果，不信任prior派生delta |
| ICMT-PF-09 | 冻结proxy days/RXs/seed/max-per-TX/total=400并闭合clean manifest、physical-key、proxy JSON/CSV、pair与F6重算 | `code/export_phase1_icmt_features.py`、`code/evaluate_phase1_icmt_postfreeze_pair.py`、`code/scripts/launch_phase1_icmt_postfreeze_20260810.sh`、`code/tests/test_phase1_icmt_postfreeze.py` | verified locally | 1-row全链同步缩行、JSON/CSV/physical count及days/RXs/seed/max漂移隔离负测 | 正式值逐项来自已签GD postfreeze v2实现与launcher，不从prior receipt推断 |

## CB/CP/GD三轮复盘（不扩门）

已重读`项目.md`、当前P1目标与CB、CP、GD的完整设计及postfreeze报告。三轮共同结论是：LEO平均overall增益不能替代clean或逐LEO最差floor；proxy的AUROC与几何gap方向在fold间不稳定；任何均值补偿都不构成晋级证据。下一轮只检验source-L中的通用逐样本低margin尾部收紧假设，不以RX/day、proxy或query选择参数，也不把CB/CP/GD的已拒绝机制重新组合。主控又以`CB SFCE CP SFCE GD ProtoNLL LEO proxy floor`对现有项目会话索引做了只读检索，命中历史LEO部署主视图与floor优先条目；由于索引未命中本轮最新run名，本复盘的具体数值仍以已回收的CB/CP/GD同run报告为准，不用历史摘要替代当前证据。该复盘不新增训练臂、超参数、验证门或后冻结步骤。

## 实现边界

本轮在既有训练实现之外，仅新增独立ICMT postfreeze exporter、pair evaluator、42步launcher与focused tests，并更新本卡的实现追踪；不改变冻结公式、训练行为、超参数或训练launcher。不得修改GD、CB、CP、SCB、CARE、CIRF、`phase1_icmt.py`或`train_ssdg.py`，不得访问N607、启动性能实验、修改automation report/conversation index或提交Git。

postfreeze v1本地证据为：`py_compile`通过；ICMT专用测试16项与GD模板回归10项合计26 passed；`bash -n`与dry-run42通过。独立复审随后指出F6未从prior原始artifact重算、通用LEO导出缺少冻结dataset/source-selection/逐场景覆盖信任链，因此v1不得放行。v2已最小修复这两项P0，不改变Gaussian、门限、矩阵或42步数量；复审确认原P0关闭，但指出proxy days/RXs/seed/max-per-TX/total尚未作为正式常量贯穿全链，裁决为`P0=0/P1=1/REVISE`。ICMT-PF-09现已本地闭合：`py_compile`通过；ICMT专用测试31项与GD模板回归10项合计41 passed；`bash -n`通过；dry-run仍为12个clean、12个LEO、12个proxy和6个pair，共42步。1-row攻击同步修改C/G clean、manifest、proxy JSON/CSV、artifact SHA及prior expected count后，在当前pair与F6 prior路径的首个固定400门失败；四类selection和三类count receipt均有隔离负测。当前仅标记`LOCAL_VERIFIED_V2_P1_FIX_PENDING_INDEPENDENT_P0_P1_REVIEW`。未访问N607、未运行性能矩阵，不构成实验放行、性能结论或最终实现签署。
