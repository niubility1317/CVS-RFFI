# SF-TAPFT全路线研发、实验与瘦身综合报告

## 1.报告结论

本报告统一整理SF-TAPFT从设计起点、V1机制验证、R0清洁基准、R1的P0–P4容量归因、S2阶段长度、S4 Adapter rank、16个bundle独立query、ERBT-IDR/D92级联尝试，到M02 norm瘦身矩阵的全部可验证证据。

路线已经回答四个核心问题。

1. **目标域监督是否有用。**有用。最初V1在60条support的4折OOF中把balanced accuracy从60.4167%提高到89.5833%，但V1存在冻结tensor被完整state averaging改变、fold0被错误保存为final和缺少独立query三个问题，因此只能作为机制阳性。
2. **性能主要来自哪里。**主要来自目标分类头和time norm。R1中P0 head-only已经达到85.4167%，P1 head+norm达到86.1111%；继续加入R16 Adapter后OOF提高到89.5833%，但完整`t3`和更深`t2/fusion`没有继续提高。独立query进一步反转了OOF排序：P1/M02达到86.67%，超过所有P2–P4候选。
3. **C阶段和Adapter是否必要。**对当前独立query最优路线不必要。C删除后的P2-R16 OOF与完整P3-R16相同；更重要的是，完全不含Adapter和C阶段的M02取得独立query三指标共同最优。因而S3的`t3`增量替代和S4继续压缩Adapter不再是当前主线，实际Adapter rank开销已经降为0。
4. **还能怎样瘦身。**M02内部继续缩减norm后，仅训练`t3.norm`的S02把变化元素从1584降到1152，减少27.27%，support OOF提高到89.5833%，NLL降到0.424413，是当前最小support OOF候选；但它尚未经过新的独立query。把预算截断到600/300步分别节省66.78%/82.64%时间，却使NLL超过预登记上限，不能晋级。

当前最高可信科学结论是：

> 在receiver`20-1`、`leo_clear_weak`、旧6类、每类K=10的单seed独立query上，SF-TAPFT M02（持久目标head+全部time norm、无Adapter、无B/C阶段）将DA1_REG0 balanced accuracy提高到86.67%，class floor提高到60%，NLL降到0.5094。该结果是独立truth-last闭合，但不能外推到其他receiver、seed或LEO场景。

当前最小结构候选S02的状态仍是`SUPPORT_OOF_WINNER_PENDING_INDEPENDENT_QUERY`，不能替代M02的独立query结果。

## 2.证据范围与统一协议

### 2.1固定输入

|字段|值|
|---|---|
|Phase1基础checkpoint|`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|Phase2协议|`p2_min_v1`|
|数据状态|`VALIDATED_ONCE`|
|目标receiver|`20-1`|
|旧类|6类|
|K-shot|每类K=10，共60条support|
|support capsule|`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`|
|support split|`stage2b-rx20-1-seed713101-before-support-prefix`|
|数据split seed|`713101`|
|优化seed|`392002`|
|独立query|rank10–19，旧6类各10条，共60条|
|新类注册|未执行，全部实验为`REG0`|

K=10指每个旧类10个互不重复的物理support，6类共60条。所有适配与瘦身实验只读取固定received IQ support；query不参与训练、选择、阈值、温度、回滚或状态更新。独立query实验先形成prediction，再由独立scorer连接truth。

### 2.2证据层级

|层级|能证明什么|本路线对应证据|
|---|---|---|
|实现与测试|方法入口可达、权限和参数集合正确|聚焦单测、真实checkpoint no-query smoke|
|support OOF|同一60条support内的结构筛选信号|R0/R1和M02 norm瘦身|
|全support refit|最终bundle由全部60条support重新拟合|V2 clean-single bundle|
|独立query truth-last|固定bundle在未参与训练的query上表现|现有16个bundle query闭合|
|跨receiver/seed/scene确认|稳定泛化|尚未完成|

报告不把低层证据冒充高层证据。V1的89.5833%和S02的89.5833%都是support OOF；M02的86.67%才是本路线已经闭合的独立query BA。

## 3.方法起点：为什么从FSFA-v2转向SF-TAPFT

前序联合审计给出了两组相反但互补的证据。

- 无梯度FSFA-v2使用8维域偏移编码和源域嵌套方向选择，12/12个域都选择uniform/zero-adapt，平均增益为0；gradient-only pilot平均变化仅`+0.003819`。这证明低维全局域摘要无法稳定恢复类条件决策边界。
- 目标域监督参考中，冻结模型为68.75%，目标域全监督参考上界为93.75%；time-only prototype CE达到81.25%，replacement target-proxy三seed达到83.33%±5.89%。这证明60条目标support包含有效身份监督，主要问题是训练不足、决策边界冻结和参数范围过窄。

由此形成SF-TAPFT：只保留Phase1 checkpoint，使用有标签目标support进行source-free target adaptation。第一版核心损失为balanced CE、LOO prototype CE和L2-SP；模型按head/norm、time Adapter、末级时间块渐进开放，frequency/domain分支保持冻结。

## 4.V1机制验证及其问题

V1使用4折target-inner OOF，每折计划A/B/C=`500/1500/2500`步，总计18,000个optimizer step；A阶段训练目标head与time norm，B阶段加入R16 time Adapter，C阶段加入完整`t3`。

|指标|DA0_REG0|DA1_REG0|变化|
|---|---:|---:|---:|
|support OOF BA|60.4167%|89.5833%|+29.1667pp|
|NLL|4.746183|0.410615|-4.335568|
|fold方差|0.009404|0.002459|-73.85%|
|非退化fold|N/A|4/4|100%|

完整loss轨迹表明A阶段loss从1.625207降至0.456088，B阶段继续降至0.244023，C阶段只从0.244029降至0.243382。C阶段2500步的support loss边际仅0.2651%。墙钟约1小时11分59秒，GPU显存约692–702MiB，但CPU瞬时约40核，说明逐step验证、state-distance计算、完整snapshot复制和GPU–CPU同步是主要瓶颈。

V1不能晋级，原因是：

1. top-3 checkpoint对完整floating `state_dict`求平均，167个许可集合外tensor发生漂移，最大绝对偏移0.5；
2. 最终bundle保存fold0模型，只训练44条support，不是全60条support refit；
3. 没有独立query；
4. 持久目标head属于本轮放宽后的高性能诊断，不是旧“冻结原型分类边界”方案。

V1的交付状态是`DIAGNOSTIC_POSITIVE_BUT_INVALID_FOR_PROMOTION`。

## 5.R0清洁基准：1个

R0=`SF_TAPFT_V2_CLEAN_R16_T3`。它保持V1的主要结构和预算，但修复归因与最终模型生成：

- 只平均许可trainable delta与目标head，未训练state逐tensor复制基础checkpoint；
- 4折只负责OOF选择；
- 选择通过后从基础checkpoint出发，用全部60条support重新拟合；
- `fold0_as_final=false`；
- bundle严格回读，非许可参数变化为0；
- 新增只读query prediction接口，但R0训练阶段不打开query。

|指标|R0结果|
|---|---:|
|support OOF BA|89.5833%|
|相对冻结BA|+29.1667pp|
|NLL|0.410612|
|最差fold BA|83.3333%|
|最终refit步数A/B/C|203/197/225|
|变化元素|16,385|
|占bundle state|1.551%|
|独立query DA1 BA|80.00%|
|独立query floor|30.00%|
|独立query NLL|0.5984|

R0证明V1阳性并非必须依赖越界averaging，但也暴露出support OOF与最终query之间存在9.58pp差距。

## 6.R1容量归因：P0–P4全部覆盖

P0–P4是嵌套容量实验。所有候选都训练960参数目标head；下表的变化范围指encoder侧新增开放部分。

|层级|训练范围|代表行|OOF BA|最差fold|NLL|变化元素|相对前级|
|---|---|---|---:|---:|---:|---:|---|
|P0|target head only|M01|85.4167%|69.4445%|0.503753|960|head单独贡献25.0000pp|
|P1|P0+全部time norm|M02|86.1111%|77.7778%|0.436715|1,584|BA+0.6944pp，NLL-0.067038|
|P2|P1+R16 time Adapter|M09|89.5833%|83.3333%|0.410612|6,881|BA+3.4722pp，NLL-0.026103|
|P3|P2+完整`t3`|R0|89.5833%|83.3333%|0.410612|16,385|BA/NLL无增益，+9,504元素|
|P4|P3+`t2.pw/time_fuse/fuse/id_proj`|M14|89.5833%|83.3333%|0.410612|105,377|BA/NLL无增益，+88,992元素|

P0已经获得绝大多数增益，说明重新建立目标域决策边界是SF-TAPFT成功的首要因素。norm提供小而稳定的额外收益。R16 Adapter在support OOF中继续增加3.4722pp，但这一增益没有在独立query中保留：M09 query BA为83.33%，低于M02的86.67%。完整`t3`和P4高容量层既未改善OOF，也未改善query。

P4相对P2把变化元素从6,881扩大到105,377，约15.31倍，却没有可观测OOF收益。这一结果直接否定了“继续扩大时间/融合层必然提高目标域性能”的假设。

## 7.S2阶段长度：C删除、300步、500步均覆盖

|方案|行|计划A/B/C|最终选择A/B/C|OOF BA|最差fold|NLL|相对完整R0|
|---|---|---:|---:|---:|---:|---:|---|
|完整C2500参考|R0|500/1500/2500|203/197/225|89.5833%|83.3333%|0.410612|参考|
|删除C|M09|500/1500/0|203/197/0|89.5833%|83.3333%|0.410612|完全持平|
|C300|M10|500/1500/300|152/199/3|87.5000%|77.7778%|0.482631|BA-2.0833pp，NLL+0.072019|
|C500|M11|500/1500/500|157/200/3|87.5000%|77.7778%|0.462989|BA-2.0833pp，NLL+0.052377|

删除C的P2-R16与完整P3-R16完全持平，说明完整`t3`更新不是当前support OOF的必要条件。C300/C500不能被解释为严格的“只改C长度”，因为旧实现按总步数重算warmup和cosine时钟，A/B实际学习率轨迹也发生变化。它们证明的是“缩短总调度的联合版本”没有达到非劣，而不是300/500个C步本身一定有害。

后续M02完全删除B/C和Adapter后，独立query反而达到全矩阵最高值。这使S2最终结论从“C可删除”进一步推进到“当前优选只需要A阶段”。

## 8.S4 Adapter rank：R32、R16、R8、R4均覆盖

为了保持归因一致，rank比较采用P3、rho=0.5、KD=0的同范围结果。

|rank|行|OOF BA|最差fold|NLL|变化元素|独立query BA/floor/NLL|
|---:|---|---:|---:|---:|---:|---|
|R32|M04|87.5000%|77.7778%|0.445613|21,521|83.33%/60%/0.5356|
|R16|R0|89.5833%|83.3333%|0.410612|16,385|80.00%/30%/0.5984|
|R8|M12|82.6389%|63.8889%|0.513740|13,817|81.67%/50%/0.5438|
|R4|M13|87.5000%|77.7778%|0.450062|12,533|81.67%/50%/0.5483|

rank与性能不单调。R16的support OOF最好，R8最差，R4又回升。独立query上R32高于R16，但仍低于无Adapter的M02；R8/R4也没有超过M02。因而“选出最佳Adapter rank”不是最终部署结论，真正的结论是Adapter整体不如更小的head+norm结构。

设计要求的R32→R16→R8→R4覆盖已经完成，但不再继续围绕Adapter做更细rank搜索：当前主线的Adapter rank已经等于0。

## 9.R1其余重要轴：rho与KD

|候选|变化|OOF BA|NLL|独立query BA|独立queryfloor|判定|
|---|---|---:|---:|---:|---:|---|
|R0|rho0.5|89.5833%|0.410612|80.00%|30%|清洁参考|
|M07|rho0.75|89.5833%|0.368684|81.67%|50%|未超过M02|
|M08|rho1.0|90.9722%|0.365321|81.67%|50%|OOF第一，但query未保持|
|M06|KD0.1|86.1111%|0.442336|85.00%|50%|query第二，但三指标弱于M02|

M08展示了本路线最明显的OOF→query排名反转：它是support OOF最高候选，却只得到81.67%独立query BA。M06的OOF只有86.11%，query反而达到85.00%。这证明support OOF适合筛除明显失败结构，但不能替代最终bundle的独立query排序。

### 9.1R0+M01–M15完整support OOF表

|候选|profile/rank|rho/KD|最终refit A/B/C|OOF BA|最差fold|NLL|变化元素|
|---|---|---|---:|---:|---:|---:|---:|
|M08|P3/R16|1.00/0|186/191/2|90.9722%|77.7778%|0.365321|16,385|
|M07|P3/R16|0.75/0|166/200/1|89.5833%|77.7778%|0.368684|16,385|
|R0|P3/R16|0.50/0|203/197/225|89.5833%|83.3333%|0.410612|16,385|
|M09|P2/R16|0.50/0|203/197/0|89.5833%|83.3333%|0.410612|6,881|
|M14|P4/R16|0.50/0|203/197/2|89.5833%|83.3333%|0.410612|105,377|
|M05|P4/R32|0.50/0|203/170/3|87.5000%|77.7778%|0.441713|110,513|
|M03|P2/R32|0.50/0|203/170/0|87.5000%|77.7778%|0.445613|12,017|
|M04|P3/R32|0.50/0|203/170/4|87.5000%|77.7778%|0.445613|21,521|
|M13|P3/R4|0.50/0|203/288/4|87.5000%|77.7778%|0.450062|12,533|
|M11|P3/R16-C500|0.50/0|157/200/3|87.5000%|77.7778%|0.462989|16,385|
|M10|P3/R16-C300|0.50/0|152/199/3|87.5000%|77.7778%|0.482631|16,385|
|M02|P1/R16|0.50/0|327/0/0|86.1111%|77.7778%|0.436715|1,584|
|M06|P3/R16|0.50/0.10|197/255/5|86.1111%|77.7778%|0.442336|16,385|
|M01|P0/R16|0.50/0|607/0/0|85.4167%|69.4445%|0.503753|960|
|M12|P3/R8|0.50/0|203/267/3|82.6389%|63.8889%|0.513740|13,817|
|M15|P4/R8|0.50/0|203/267/2|82.6389%|63.8889%|0.513740|102,809|

## 10.16个bundle独立query：完整数据

每行使用同一60条独立query，状态均为`DA0_REG0→DA1_REG0`。新类相关指标为`N/A`。

|候选|DA0 BA|DA1 BA|ΔBA|DA0 floor|DA1 floor|Δfloor|DA0 NLL|DA1 NLL|ΔNLL|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|R0|71.67%|80.00%|+8.33pp|10%|30%|+20pp|0.9367|0.5984|-0.3383|
|M01|71.67%|81.67%|+10.00pp|10%|50%|+40pp|0.9367|0.5731|-0.3636|
|M02|71.67%|86.67%|+15.00pp|10%|60%|+50pp|0.9367|0.5094|-0.4272|
|M03|71.67%|83.33%|+11.67pp|10%|60%|+50pp|0.9367|0.5347|-0.4020|
|M04|71.67%|83.33%|+11.67pp|10%|60%|+50pp|0.9367|0.5356|-0.4011|
|M05|71.67%|83.33%|+11.67pp|10%|60%|+50pp|0.9367|0.5354|-0.4013|
|M06|71.67%|85.00%|+13.33pp|10%|50%|+40pp|0.9367|0.5367|-0.4000|
|M07|68.33%|81.67%|+13.33pp|20%|50%|+30pp|1.1763|0.5659|-0.6104|
|M08|70.00%|81.67%|+11.67pp|30%|50%|+20pp|1.3426|0.5652|-0.7774|
|M09|71.67%|83.33%|+11.67pp|10%|60%|+50pp|0.9367|0.5406|-0.3961|
|M10|71.67%|83.33%|+11.67pp|10%|60%|+50pp|0.9367|0.5445|-0.3922|
|M11|71.67%|83.33%|+11.67pp|10%|60%|+50pp|0.9367|0.5438|-0.3929|
|M12|71.67%|81.67%|+10.00pp|10%|50%|+40pp|0.9367|0.5438|-0.3928|
|M13|71.67%|81.67%|+10.00pp|10%|50%|+40pp|0.9367|0.5483|-0.3883|
|M14|71.67%|83.33%|+11.67pp|10%|60%|+50pp|0.9367|0.5408|-0.3959|
|M15|71.67%|81.67%|+10.00pp|10%|50%|+40pp|0.9367|0.5439|-0.3928|

16/16行的BA、floor和NLL方向都改善，证明SF-TAPFT的独立query收益不是单一候选偶然现象。M02是唯一三指标共同最优：BA86.67%、floor60%、NLL0.5094。逐类准确率为90/100/80/60/90/100%，困难类3从共同DA0的10%提高到60%。类4从100%降到90%，所以总体改善不等于逐类零回退。

本轮科学状态为`SINGLE_SEED_SINGLE_RECEIVER_CLEAR_QUERY_CLOSURE_POSITIVE`。

## 11.M02成为主锚后的结构结论

M02只训练目标head和9个time norm参数tensor，总变化元素1584，占bundle state约0.150%。它没有time Adapter，没有完整`t3`更新，没有`t2/fusion`高容量更新，也没有B/C阶段。

这组结果改变了原始瘦身路线：

- S1单模型已经实现：所有正式bundle都是全60条support refit单模型；
- S2阶段删除已经实现：M02为A-only；
- S3完整`t3`改轻型增量被拒绝：优选根本不训练完整`t3`；
- S4 Adapter rank继续压缩被拒绝：优选不使用Adapter；
- S5 target head压缩仍延后：head只有960元素，且P0表明它是主要性能来源。

因此，下一轮从“压缩Adapter/末级块”转为“压缩M02内部norm范围和仿射维度”。

## 12.M02 norm逐级瘦身

原计划16行，用户为加速关键闭合定向停止S01/S03/S04/S06/S07/S09/S12/S13；这些行统一为`USER_DIRECTED_PRUNED/NO_PERFORMANCE_RESULT`，不得从partial artifact推断性能。关键8行完整闭合。

|row|结构|OOF BA|最差fold|NLL|变化元素|wall-clock|判定|
|---|---|---:|---:|---:|---:|---:|---|
|S00|head+全部norm锚点|86.1111%|77.7778%|0.436715|1,584|3:46:23|锚点|
|S02|head+仅`t3.norm`|89.5833%|77.7778%|0.424413|1,152|3:48:10|PASS，最小候选|
|S05|head+仅`time_fuse.norm`|84.0278%|69.4445%|0.506162|1,056|3:44:24|BA/floor/NLL失败|
|S08|head+`t3+fuse.norm`|86.1111%|77.7778%|0.452126|1,248|3:48:40|PASS|
|S10|head+全部norm weight|86.1111%|77.7778%|0.414765|1,272|4:00:26|PASS，最低NLL|
|S11|head+全部norm bias|87.5000%|77.7778%|0.446656|1,272|3:55:45|PASS|
|S14|S00，600步上限|86.1111%|77.7778%|0.508258|1,584|1:15:13|NLL失败|
|S15|S00，300步上限|86.1111%|77.7778%|0.514836|1,584|0:39:17.91|NLL失败|

S02相对S00减少432个变化元素，即27.27%，BA提高3.4722pp，NLL改善0.012302，最差fold不下降。S05证明融合norm不能单独承担校正；S08证明在`t3.norm`上追加fuse没有BA收益；S10说明weight-only有较好校准，但比S02多120个元素且BA低3.4722pp；S11说明bias-only可以较早选中，但综合性能仍弱于S02。

S02 bundle只比S00小1152B，因为当前clean-single bundle仍保存完整模型state。参数瘦身已经实现，但delta-only封装尚未实现。

## 13.训练时间、CPU与GPU开销

|现象|证据|解释|
|---|---|---|
|参数减少但时间不降|S02比S00少27.27%变化元素，仍为3:48:10|完整backbone前向和逐step验证未减少|
|GPU显存低|关键8行峰值676–690MiB|模型/优化器不是显存瓶颈|
|CPU参与明显|V1运行时约40核瞬时占用|验证汇总、state-distance、snapshot、数据组织和同步在CPU执行|
|短步数速度显著|S14/S15节省66.78%/82.64%wall-clock|总optimizer/validation次数直接下降|
|短步数校准失败|NLL分别比S00高0.071542/0.078121|硬分类相同不代表概率可靠性相同|

S14/S15的BA和floor都与S00持平，但NLL上限为`0.436715+0.03=0.466715`；S14为0.508258，S15为0.514836，均明显超限。它们适合证明“速度上界”，不适合作为正式替代checkpoint。

真正的工程加速应独立验证：稀疏checkpoint validation、冻结前缀embedding缓存、validation tensor常驻GPU、只在候选步计算state-distance、delta-only bundle。这些优化应先证明`max_abs_logit_delta<1e-5`和BA不变，再与结构瘦身合并。

## 14.ERBT-IDR与D92 E0去RF32探索

### 14.1SF-TAPFT→ERBT-IDR M29

第一次M29-FFT96-A4级联没有产生性能结果。SF bundle使用的60条旧类support与现有ERBT Stage2-C胶囊的旧类support标签排列相同，但received IQ摘要不同。强行级联会隐含消费另一胶囊的60条target-derived state，无法证明同row K10和support/query物理互斥。

该实验在读取ERBT query之前停止，状态为`STOPPED_EARLY_PROTOCOL_MISMATCH/NO_PERFORMANCE_RESULT`。没有prediction、没有truth、没有scorer，也没有ERBT-IDR性能数字。

### 14.2旧类K10、REG0同row实现

为解决上述问题，后续实现了`stage2_sf_erbt_oldonly.py`和`run_sf_erbt_oldonly.py`：使用与SF support逐字节匹配的rank0–9作为60条support，rank10–19作为60条独立query，query NPZ不含truth，truth sidecar只由scorer打开。该实现最终首先用于现有16个SF bundle的独立query闭合。

### 14.3D92 E0去RF32

`D92-E0-NORF32`锁定`identity160+FFT96(A4)`、`rf32_used=false`，计划比较`SF_HEAD`与`SF_D92E0_NORF32`的旧类BA和floor，不注册新类。该工作只达到`LOCAL_VERIFIED`预登记；没有启动后prediction闭合或score，因此性能仍为`NO_PERFORMANCE_RESULT`。不能把D92历史独立结果或SF query结果拼接成该级联性能。

## 15.已经优化落地的实现

|实现|落地内容|验证状态|
|---|---|---|
|target-only适配核心|balanced CE、LOO prototype、L2-SP、selective KD、A/B/C解冻|已实现并真实运行|
|目标head|6×160归一化余弦head，source/target原型按rho初始化|已实现；当前高性能路线保留|
|严格输入边界|适配函数无source/query入口，审计记录全部访问开关|16/16容量行与8/8瘦身行验证|
|delta averaging|只平均许可变化delta，未训练state从基础checkpoint复制|R0以后验证，非许可变化0|
|全support refit|OOF只选结构/步数，最终从基础checkpoint用60条support重拟合|R0和后续bundle验证|
|P0–P4可训练profile|head、norm、Adapter、完整`t3`、time/fusion嵌套范围|16行真实矩阵闭合|
|阶段与rank配置|C0/C300/C500/C2500，R4/R8/R16/R32|真实矩阵闭合|
|query truth-last|prediction不含truth/role，scorer后连接同一query ID|16/16独立query闭合|
|M02精细瘦身|`norm_scope`、`norm_affine`、固定LR时钟截断|关键8行闭合|
|资源遥测|GNU time、最大RSS、GPU采样、bundle字节数|M02关键8行闭合|
|严格bundle兼容|旧M02只允许新增三字段使用默认值，未知字段拒绝|79项聚焦回归通过|

当前仍未落地或未完成科学闭合的部分：

- S02新的独立query里程碑；
- 跨receiver、跨seed、`leo_low_elev_weak`和`leo_rain_weak`确认；
- sparse validation与冻结前缀embedding缓存的数值等价实现；
- delta-only bundle；
- target head最终压缩；
- ERBT-IDR/D92 E0同row级联性能。

## 16.设计追踪终态

|设计块|状态|结论|
|---|---|---|
|R0清洁参考|verified|修复归因与全support final|
|R1 P0–P4|verified|head/norm是主要收益，深层容量不增益|
|S2 C删除/300/500|verified|C可删除；300/500旧行受调度耦合|
|S3`t3`轻增量|rejected|当前优选不训练完整`t3`|
|S4 R32/R16/R8/R4|verified后rejected继续扩展|所有rank已覆盖，但无Adapter的M02更优|
|S5 head压缩|deferred|960元素且贡献最大，最后处理|
|M02 norm范围/affine|verified|S02为最小support OOF候选|
|M02短步数|verified negative|速度显著，NLL门槛失败|
|S02独立query|deferred|最高风险剩余项|
|ERBT-IDR/D92级联|blocked by evidence, not rerun|当前无同row性能结果|

严格设计对齐方面，R0、P0–P4、阶段长度、rank、rho、KD、query闭合和M02瘦身均由真实配置与artifact支撑。M02 norm原16行实验因用户定向裁剪只闭合关键8行；其余8行必须保持`USER_DIRECTED_PRUNED/NO_PERFORMANCE_RESULT`，不能写成完整16行性能覆盖。

## 17.最终研发判断

SF-TAPFT已经从“高性能但归因不干净的V1”推进到“实现干净、独立query阳性、结构显著缩小的M02”。路线的关键发现不是某一个最大OOF数字，而是三次收敛：

1. 从无梯度域摘要收敛到目标监督梯度适配；
2. 从R32/P4高容量教师收敛到head+norm；
3. 从全部time norm进一步收敛到`t3.norm`候选。

就现有证据，M02是正式保留checkpoint，S02是下一里程碑候选。S14/S15只能作为速度边界；ERBT-IDR和D92 E0去RF32尚无合法性能结果。下一次科学动作应只做S02新的独立query，避免重复使用已经参与M02选择的rank10–19 truth。下一次工程动作应独立做数值等价加速，不能把速度优化与结构变化混在同一因果实验中。

## 18.证据索引

- [SF-TAPFT V1全面分析](stage2_sf_tapft_v1_rx20_1_targetinner_s392002_20260826_r1_comprehensive_analysis.md)
- [R0环境修复与完成报告](stage2_sf_tapft_v2_clean_r16_t3_rx20_1_s392002_20260826_r2_envfix1_report.md)
- [R1容量矩阵综合报告](stage2_sf_tapft_v2_capacity16_comprehensive_report_20260827.md)
- [16候选机器可读指标](stage2_sf_tapft_v2_capacity16_candidate_metrics_20260827.csv)
- [64折机器可读指标](stage2_sf_tapft_v2_capacity16_fold_metrics_20260827.csv)
- [现有16个bundle独立query闭合](stage2_sf_tapft_v2_existing16_queryclosure_rx20_1_s392002_20260827_r1_report.md)
- [ERBT-IDR M29协议停止记录](stage2_sf_tapft_erbt_idr_m29_rx20_1_s392002_20260826_r1_report.md)
- [D92 E0去RF32预登记](stage2_sf_d92e0_norf32_oldonly_rx20_1_s713101_20260826_r1_report.md)
- [M02 norm瘦身报告](stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827_r1_report.md)
- [SF-TAPFT V2设计追踪](sf_tapft_v2_staged_traceability_20260826.md)

## 19.交付状态

|对象|状态|最高证据|
|---|---|---|
|R0/R1容量矩阵|VERIFIED|16/16 artifact完整并分析|
|独立query矩阵|VERIFIED|16/16 prediction与truth-last score完整|
|M02 norm瘦身|VERIFIED|关键8/8分析，8行用户裁剪|
|S02最终性能|PARTIAL|support OOF胜出，独立query缺失|
|ERBT-IDR/D92级联|PARTIAL|实现/预登记存在，性能缺失|
|本综合报告|ANALYZED|正式证据逐项回读|
