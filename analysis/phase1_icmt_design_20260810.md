# P1-ICMT冻结设计与实现追踪卡

状态：`IMPLEMENTING`；设计审查：`P0=0/P1=0/ALLOW-MERGE`；实现审查与实验结果均待独立完成。

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

后冻结42步仅预注册、当前不实现：12clean、12source-only三LEO、12proxy、6pair；Gaussian-NLL仅在L上以同一路径`z_id`作float64 totalized-L2拟合，V/proxy零fit。门包括clean6/6、LEO18/18四floor、三场景及18格overall非负、每fold`ΔAUROC>0`和`Δ(mean u_proxy-mean u_V)>0`。任一完整门失败即`REJECT_P1_ICMT_PERMANENT`，均值不得补偿floor。

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
| ICMT-08 | 后冻结42步及非补偿门 | 本设计卡 | deferred | 后冻结独立实现 | 当前明确`PENDING_NOT_IMPLEMENTED` |

## CB/CP/GD三轮复盘（不扩门）

已重读`项目.md`、当前P1目标与CB、CP、GD的完整设计及postfreeze报告。三轮共同结论是：LEO平均overall增益不能替代clean或逐LEO最差floor；proxy的AUROC与几何gap方向在fold间不稳定；任何均值补偿都不构成晋级证据。下一轮只检验source-L中的通用逐样本低margin尾部收紧假设，不以RX/day、proxy或query选择参数，也不把CB/CP/GD的已拒绝机制重新组合。主控又以`CB SFCE CP SFCE GD ProtoNLL LEO proxy floor`对现有项目会话索引做了只读检索，命中历史LEO部署主视图与floor优先条目；由于索引未命中本轮最新run名，本复盘的具体数值仍以已回收的CB/CP/GD同run报告为准，不用历史摘要替代当前证据。该复盘不新增训练臂、超参数、验证门或后冻结步骤。

## 实现边界

本次仅落地训练路径、收据、launcher和局部测试。不实现postfreeze scorer、42步执行器或任何proxy拟合/选择逻辑；不修改GD、CB、CP、SCB、CARE或CIRF文件；不访问N607、不启动实验、不提交Git。

本地验证不替代独立实现审查：当前状态仅为`LOCAL_VERIFIED_PENDING_INDEPENDENT_P0_P1_REVIEW`，不构成实验放行、性能结论或最终实现签署。
