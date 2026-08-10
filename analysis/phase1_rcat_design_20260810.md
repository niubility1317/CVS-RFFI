# P1-RCAT冻结设计卡（2026-08-10）

## 状态与边界

本卡记录`P1-RCAT`（Receiver-Conditioned Angular Transport）的已终裁设计冻结，用于本地实现和技术验证；它不是性能结论、N607发布授权或对RX/day/unknown的修复声明。训练只可读`source_known_train L`的TX标签与物理`rx_i`元数据；`U`零forward，`V`仅共同诊断且零回流，proxy/held/target/day/fold均不得参与训练、选参或模型选择。

## 唯一辅助项

对同一物理`L`行的既有clean和单次LEO观测，令`z_C,z_L∈R^d`为raw encoder的`feat_joint`，并定义精确分段totalized-L2：

\[
T(z)=\begin{cases}z/\lVert z\rVert_2,&\lVert z\rVert_2>0\\0,&\lVert z\rVert_2=0\end{cases},\qquad
q_i=\lVert T(z_{L,i})-\operatorname{sg}(T(z_{C,i}))\rVert_2^2.
\]

固定`R_s={0,…,6}`、local4类别`C={0,…,3}`与28格分母：`I_rc={i:rx_i=r,y_i=c}`，`g_rc=0`若`n_rc=0`，否则`mean_{i∈I_rc}q_i`。唯一G项为`L_RCAT=(0.02/28)∑_{r,c}g_rc`；空格以可微零保留，不按活跃格、样本数或场景重归一。训练时以float32累计；后冻结使用float64的同一分段规则，二者不声称字节完全相同。

clean锚点完整stopgrad，LEO特征与shared encoder保留梯度；对于正范数行，`q_i=0`当且仅当`z_L=a z_C,a>0`。零向量按规则保留而非删除；任一`z`、范数、`T`、`q`或辅助损失非有限即在backward前fatal。

## 与共同基线及拒绝路线的区别

共同`L_base`从epoch1保留`0.10·KL(sg(logits_clean)||logits_LEO)`。RCAT不是该KL的低层重命名：若raw head为`Wz+b`，非共线`u∈ker(W)`且`z_L=z_C+u`时KL可为零而RCAT大于零；反之若`z_L=a z_C,a≠1`且`W(a-1)z_C`并非全常量，RCAT为零而KL大于零。因此其相对KL的新增可识别部分是head-nullspace方向漂移，且RCAT约束所有clean→LEO方向漂移。它不用GD的DRO/EMA/prototype-NLL、ICMT的尾均值、CAGM的centroid/radius/Gram或RCRMD的raw-logit margin/positive-drop；不拼接任何永久拒绝机制。

该项不直接驱动logit margin、head范数或proxy半径，因而没有RCRMD那种“为降低margin-drop而压缩/重排logit”的直接优化通路；这只是一条可证伪的机制边界，而非proxy保持或性能改善承诺。若fixed400 proxy双门、全局/fold门或clean/LEO门失败，则RCAT被否证，不得以间接角度变化补偿解释。

## C/G与收据合同

C/G均从同一F1C/GeoSat-C`training_final_only`检查点严格warm-start，严格模型键、相同class/head顺序、相同physical batch/seed/sampler、40E、新AdamW与AMP；共同clean+single-LEO forward和共同`L_base`不变。C仍写共同物理`RX×class×scene`覆盖，RCAT辅助字段为N/A/0；G只额外加入上述一个固定损失，无新模型、forward、状态或缓存。

每批收据链入不透明`base_index`（仅无该字段时回退`sig_i`）、TX、物理`rx_i`、场景、每格`n_rc`及固定effective weights，标明同物理clean/LEO对。G还记录每格`q`、零行、非有限行、`g_rc`和loss；终态要求3个场景全部28格闭合，即84格共同覆盖，且G每场景逐格闭合、至少一个`q>0`、首个正`q`批的raw辅助VJP闭合。

VJP必须证明LEO`feat_joint`和shared encoder均finite且nonzero；exact-head参数对RCAT辅助项为N/A/None-or-zero预期，绝不可伪称为非零。另行验证clean/LEO输出中的`tx_logits`仍是共同`L_base`的live exact-head输入路径。同步RX或类别置换只重命名28格，故保持等价。

## 后冻结与判定

后冻结仍执行42步、fixed400proxy、`L`-only totalized-L2 Gaussian、clean6/6、LEO18/18、fold/global overall与proxy双门6/6；所有门均为非补偿。只有全部通过时，状态可为`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`，不得自行提升为主结论。资源增量仅为每批`O(Bd)`归一与28格归约，不增加forward或GPU并发；真实VRAM、时间和性能仍待实测。

## 实现追溯

|冻结要求|实现锚点|验证证据|
|---|---|---|
|唯一totalized-L2辅助项与固定`0.02/28`|`code/cvsrffi/phase1_rcat.py:rcat_loss`|数值与空格回归|
|共同clean+single-LEO、L-only物理RX收据|`code/SSDG/train_ssdg.py:update_rcat_common_batch_sequence_receipt`接线|共同批次/metadata收据|
|C辅助N/A、G唯一加项|`add_rcat_to_loss`与C/G终态检查|冻结参数负例|
|LEO特征/shared encoder非零VJP；head辅助N/A|`rcat_aux_gradient_audit`|首个正`q`批审计|
|三场景×28格终态|`validate_rcat_terminal_receipt`|84格终态收据|
