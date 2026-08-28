# SF-TAPFT P1紧凑部署D0–D4工程回放完整实验报告

## 1.结论先行

本轮已经完成设计报告要求的P0工程优化和P1最小矩阵实现，并在N607上完成旧6类K=10、最大Q180的真实checkpoint实验闭合。最终状态为：

`ANALYZED_ENGINEERING_REPLAY_COMPOSITE/NO_FORMAL_PROMOTION`

推荐工作点仍是：

`D0=P0C H6 Compact/FP32 cache/delta-only`

修复后的D0在Q180上把`DA0_REG0`的130/180提升到150/180，BA从72.2222%升到83.3333%，floor从10.0000%升到56.6667%，NLL从0.870038降到0.500754，ECE-10从0.130062降到0.074766。其资源为1152个可训练/实际变化元素、5305B delta文件、10.80秒单次进程墙钟和1611424KiB最大RSS。

D4的Class-CVaR确实执行了30步并降低support尾部损失，但其180条argmax与D0完全相同，只把NLL改善0.000699、ECE改善0.000066，同时墙钟从10.80秒增至12.89秒。因此D4通过门槛但没有形成值得部署的性能收益，判定为`PASS_NO_INCREMENTAL_VALUE`。

D1、D2、D3都没有通过相对D0的完整晋级门槛：

- D1取得152/180和84.4444%BA，但floor降至53.3333%，类4相对D0回退10pp；
- D2只有148/180和82.2222%BA，BA和floor均低于D0；
- D3取得全矩阵最高154/180、85.5556%BA和63.3333%floor，但NLL为0.575789，高于D0+0.02门槛0.520754；类4回退10pp；support-only OOF温度使墙钟达到58.45秒，超过20秒上限。

由于本轮复用了历史已暴露truth的rx20-1 query，本结果只能证明实现、资源和工程回放有效，不能把D0或其他行晋级为新的科学默认。正式P1晋级仍须在新的未暴露合法capsule上确认。

## 2.数据、状态与证据边界

- 基础checkpoint：ADV3B02 CORE90；
- 阶段：Stage2-B；
- 接收机/场景：`rx20-1/leo_clear_weak`；
- support：旧6类，每类K=10，共60个独立物理样本；
- 注册：REG0，不注册新类，新类指标均为N/A；
- 协议：`p2_min_v1/VALIDATED_ONCE`；
- capsule：`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`；
- split：`stage2b-rx20-1-seed713101-before-support-prefix`；
- Q60：support pool中未用于support的rank10–19，每类10条；
- Q120：独立query分区，每类20条；
- Q180：Q60∪Q120，每类30条，opaque query ID交集为0；
- 适配阶段只读取checkpoint、冻结Phase1知识和合法support；receipt均为`query_opened=false/query_truth_opened=false/query_role_opened=false/source_opened=false`；
- D0/D4的Q60/Q120共4份prediction全部形成后才连接truth；D1/D2/D3也沿用各自先预测后评分的不可变artifact。

本报告使用四状态命名中的`DA0_REG0`与`DA1_REG0`。REG1未发生，所有新类准确率和新旧类调和指标均为N/A。

## 3.设计报告落地实现

### 3.1资源测量语义修正

`sf_tapft_deployment_benchmark.py`已把Linux当前RSS与进程生命周期峰值RSS分开：

- 当前RSS读取`/proc/self/status`中的`VmRSS`；
- 历史峰值明确命名为process lifetime max RSS；
- 常驻模式支持CUDA allocated/reserved峰值和`mem_get_info`自由显存起点/最低点/终点；
- 未实现的cold-start模式会被拒绝，不能把同进程重复测量冒充冷启动。

本次D0–D4性能运行保留GNU time的单次进程墙钟和最大RSS。没有执行3预热+10正式的独立资源benchmark，也没有形成连续GPU显存采样，因此GPU allocated/reserved峰值为`NOT_CAPTURED`，不据此声明显存优势。

### 3.2cache storage/compute/device拆分

原单一`prefix_cache_dtype`已拆为：

- `cache_storage_dtype`；
- `suffix_compute_dtype`；
- `cache_device`。

FP16/BF16 storage会在训练前一次性materialize为FP32 compute cache，消除每步FP16→FP32转换。尚未完成等价性验证的FP16/BF16 suffix计算会被明确拒绝。D0/D4使用FP32 storage/compute，cache为929520B。

### 3.3独立CompactH6Suffix

D0/D4不再让适配器持有完整checkpoint训练引用。新的`CompactH6Suffix`只复制：

- `t3`；
- `t_pool/t_proj`；
- `meta_adapter_time`；
- `fuse/meta_adapter_fusion`；
- `cls_head`；
- target head；
- H6 prefix cache。

其中只有`t3.norm.weight/bias`和target head可训练。其forward与reference suffix的logit和梯度等价测试通过，适配后再把许可状态应用回常驻模型。D0/D4均为0次完整backbone训练forward、450次cached suffix forward和1次cache构建forward。

### 3.4delta-only与原子回滚

部署路径只输出delta v2，不生成完整clean-single bundle。delta写入采用不可覆盖的原子替换；写入失败保留旧文件，加载失败会回滚模型状态。当前文件大小：

- D0/D4：5305B；
- D1：5800B；
- D2：6790B；
- D3：8211B。

raw可训练状态快照分别为4608/4992/5472/6336B。文件比历史4628B略大，来自当前delta v2元数据和序列化开销，不是可训练参数增加。

### 3.5Q2A/Q2B部署化

- D1 Q2A：训练target head、`t3.norm weight+bias`和`t2.norm weight`，共1248个元素，固定503步；
- D2 Q2B：在D1基础上增加`t1.norm weight`和`time_fuse norm weight`，共1368个元素，固定231步。

两者的训练范围超出H6只缓存`t3.norm`的契约，因此正确关闭prefix cache并走完整backbone forward，避免错误复用H6缓存。

### 3.6R1-T support-only OOF温度

D3训练全部time norm，共1584个元素、327步。部署入口先按4-fold support OOF形成只依赖support的logit，固定步拟合正温度，然后执行full-support refit：

- OOF NLL：0.665581→0.646321；
- 温度：`T=1.1981552674`；
- argmax保持：true；
- 最终有效head scale：6.6769309603。

原矩阵D3虽配置OOF温度，但旧deploy入口没有执行，已标记`METHOD_MISMATCH_NO_R1T_RESULT`；本报告只使用定点修复run的D3。

### 3.7head-only Class-CVaR

D4在H6的300步基础阶段和150步fast tail后执行100步head polish，其中最后30步加入：

`CE+0.03×Mean(Top2 class mean losses)`

Class-CVaR对类别ID置换保持同一形式，不硬编码困难类。30步CVaR值从0.466826降至0.464892，但Q180分类边界没有变化。

### 3.8HardPair停止

D0–D4统一`hard_pair_weight=0`。没有继续搜索HardPair权重，也没有引入类别ID专属分支。

## 4.真实运行中发现并修复的两个正确性问题

### 4.1Compact梯度裁剪遗漏

首次D0/D4实现中，梯度裁剪集合仍从完整`student.parameters()`和head取值。独立`CompactH6Suffix.t3.norm`不属于完整student，导致其梯度未被裁剪，改变训练轨迹。与历史P0C逐样本对比发现：

- Q60 argmax变化0条；
- Q120 argmax变化2条；
- 其中1条类3样本由正确类3变为错误类1；
- 旧D0因此只有149/180，不再满足H6方法等价。

修复后，裁剪集合直接遍历`optimizer.param_groups`的实际可训练参数。优化器参数ID此前已与唯一许可集合做精确相等校验。新增强裁剪回归使reference和compact的`t3.norm.weight/bias`及target head在`atol=1e-7`下等价。118项聚焦测试和独立P0/P1定点审查通过。

旧D0/D4被标记为`METHOD_MISMATCH_COMPACT_GRADCLIP_OMISSION/NO_VALID_P1_RESULT`，没有用于最终矩阵。

### 4.2smoke storage/compute契约漂移

r1真实checkpoint无query smoke仍把FP16 storage cache直接送入FP32 suffix，新严格契约正确拒绝该路径。r1在正式support适配前技术停止，状态为：

`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE/NO_PERFORMANCE_RESULT`

修复后smoke先一次性materialize，再执行suffix。r2真实checkpoint结果：

- FP32最大logit差：0；
- FP32最大梯度差：0；
- prediction一致：true；
- FP16 storage materialize后有限：true；
- support：60；
- `query_opened=false`；
- 状态：`SMOKE_PASS`。

## 5.最终冻结矩阵与artifact来源

|行|方法|步数|最终有效run|
|---|---|---|---|
|D0|P0C H6 Compact|300/150/70|`stage2_sf_tapft_p1_compact_gradclip_fix_rx20_1_s392002_20260828_r2`|
|D1|Q2A-Deploy|503/0/0|`stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828_r1`|
|D2|Q2B-Deploy|231/0/0|同上|
|D3|R1-T+4-fold support OOF温度|327/0/0|`stage2_sf_tapft_p1_d3_r1t_fix_rx20_1_s392002_20260828_r1`|
|D4|H6+Class-CVaR|300/150/100|与D0同r2|

最终分析没有跨row拼接指标。每行的BA、floor、NLL、逐类结果、资源和判定都来自该行自己的同一bundle与同一Q180 prediction。

## 6.Q180总体结果

共同`DA0_REG0`为130/180、BA 72.2222%、floor 10.0000%、NLL 0.870038、ECE-10 0.130062。

|行|`DA1_REG0`正确数|BA|floor|NLL|ECE-10|BA变化|floor变化|NLL变化|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|D0|150/180|83.3333%|56.6667%|0.500754|0.074766|+11.1111pp|+46.6667pp|-0.369284|
|D1|152/180|84.4444%|53.3333%|0.519486|0.062184|+12.2222pp|+43.3333pp|-0.350553|
|D2|148/180|82.2222%|53.3333%|0.511932|0.076527|+10.0000pp|+43.3333pp|-0.358107|
|D3|154/180|85.5556%|63.3333%|0.575789|0.082368|+13.3333pp|+53.3333pp|-0.294249|
|D4|150/180|83.3333%|56.6667%|0.500055|0.074701|+11.1111pp|+46.6667pp|-0.369983|

所有行相对DA0都有显著正收益，但“有域适应收益”不等于“优于D0”。D3最大化BA和floor，D4最小化NLL，D0则在精度、安全和资源之间最均衡。

## 7.各类别准确率

每类30条。

|类别|`DA0_REG0`|D0|D1|D2|D3|D4|
|---:|---:|---:|---:|---:|---:|---:|
|0|60.0000%|80.0000%|96.6667%|76.6667%|96.6667%|80.0000%|
|1|100.0000%|93.3333%|100.0000%|93.3333%|93.3333%|93.3333%|
|2|86.6667%|90.0000%|90.0000%|90.0000%|90.0000%|90.0000%|
|3|10.0000%|56.6667%|53.3333%|53.3333%|63.3333%|56.6667%|
|4|96.6667%|90.0000%|80.0000%|90.0000%|80.0000%|90.0000%|
|5|80.0000%|90.0000%|86.6667%|90.0000%|90.0000%|90.0000%|

相对D0的类别变化：

|行|类0|类1|类2|类3|类4|类5|最差变化|
|---|---:|---:|---:|---:|---:|---:|---:|
|D1|+16.6667pp|+6.6667pp|0|-3.3333pp|-10.0000pp|-3.3333pp|-10.0000pp|
|D2|-3.3333pp|0|0|-3.3333pp|0|0|-3.3333pp|
|D3|+16.6667pp|0|0|+6.6667pp|-10.0000pp|0|-10.0000pp|
|D4|0|0|0|0|0|0|0|

D1和D3都把容量换成了类0增益，但共同牺牲类4。D3还改善类3，因此绝对BA最高；不过其类4回退和NLL恶化说明边界收益并不均匀。

## 8.逐类NLL

|类别|DA0|D0|D1|D2|D3|D4|
|---:|---:|---:|---:|---:|---:|---:|
|0|1.738265|0.444777|0.307583|0.475577|0.439523|0.441803|
|1|0.329098|0.222214|0.136457|0.245732|0.291341|0.223113|
|2|0.696159|0.384456|0.361081|0.399938|0.388583|0.384388|
|3|1.309860|1.223761|1.347855|1.213076|1.249931|1.222054|
|4|0.133540|0.312601|0.492969|0.345014|0.645451|0.311762|
|5|1.013308|0.416717|0.470970|0.392254|0.439905|0.417211|

D3的主要NLL问题不是全局温度不足，而是类4真实类别概率质量明显恶化到0.645451；support-only温度只能缩放全局logit，不能恢复该局部边界。类3虽然准确率最高，NLL仍为1.249931，剩余错误仍较高置信。

## 9.混淆矩阵

行是真实类，列是预测类。

### 9.1共同DA0

|true\pred|0|1|2|3|4|5|
|---:|---:|---:|---:|---:|---:|---:|
|0|18|11|0|1|0|0|
|1|0|30|0|0|0|0|
|2|0|0|26|0|3|1|
|3|3|23|0|3|1|0|
|4|0|1|0|0|29|0|
|5|0|2|1|0|3|24|

### 9.2D0与D4

|true\pred|0|1|2|3|4|5|
|---:|---:|---:|---:|---:|---:|---:|
|0|24|6|0|0|0|0|
|1|0|28|0|2|0|0|
|2|0|0|27|0|2|1|
|3|1|11|0|17|1|0|
|4|0|1|0|2|27|0|
|5|1|0|0|0|2|27|

### 9.3D1

|true\pred|0|1|2|3|4|5|
|---:|---:|---:|---:|---:|---:|
|0|29|1|0|0|0|0|
|1|0|30|0|0|0|0|
|2|0|1|27|0|1|1|
|3|3|10|0|16|1|0|
|4|0|5|0|1|24|0|
|5|0|0|0|2|2|26|

### 9.4D2

|true\pred|0|1|2|3|4|5|
|---:|---:|---:|---:|---:|---:|
|0|23|7|0|0|0|0|
|1|0|28|0|2|0|0|
|2|0|0|27|0|2|1|
|3|2|11|0|16|1|0|
|4|0|1|0|2|27|0|
|5|1|0|0|0|2|27|

### 9.5D3

|true\pred|0|1|2|3|4|5|
|---:|---:|---:|---:|---:|---:|
|0|29|1|0|0|0|0|
|1|2|28|0|0|0|0|
|2|0|1|27|0|0|2|
|3|4|6|0|19|1|0|
|4|0|5|0|1|24|0|
|5|0|0|0|1|2|27|

D3把类3→类1从D0的11条降到6条，但类4→类1从1条增加到5条，说明局部冲突从类3部分转移到了类4。

## 10.Q60与Q120分区稳定性

|行|Q60 BA/floor/NLL|Q120 BA/floor/NLL|Q180 BA/floor/NLL|
|---|---|---|---|
|D0|86.6667%/70%/0.460847|81.6667%/50%/0.520708|83.3333%/56.6667%/0.500754|
|D1|85.0000%/50%/0.517534|84.1667%/55%/0.520461|84.4444%/53.3333%/0.519486|
|D2|86.6667%/70%/0.465563|80.0000%/45%/0.535116|82.2222%/53.3333%/0.511932|
|D3|83.3333%/60%/0.570450|86.6667%/65%/0.578459|85.5556%/63.3333%/0.575789|
|D4|86.6667%/70%/0.460481|81.6667%/50%/0.519842|83.3333%/56.6667%/0.500055|

D1在两个分区的BA最稳定，但floor仍低于D0且类4保护失败。D3在Q60低于D0、Q120高于D0，说明其绝对性能优势对query分区更敏感；Q180虽提高样本量，仍不能替代多receiver、scene和seed确认。

## 11.同row配对变化

|行|错→对|对→错|都对|都错|精确McNemar p|
|---|---:|---:|---:|---:|---:|
|D0|24|4|126|26|0.000180|
|D1|28|6|124|22|0.000195|
|D2|22|4|126|28|0.000534|
|D3|31|7|123|19|0.000116|
|D4|24|4|126|26|0.000180|

所有行相对DA0均有统计不对称的净正收益；该检验只描述当前180条同row样本，不支持跨域外推。

## 12.资源结果

|行|可训练/变化元素|步骤|完整backbone forward|cached suffix forward|cache|snapshot|delta|墙钟|最大RSS|GPU峰值|
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
|D0|1152/1152|300/150/70|0|450|929520B|4608B|5305B|10.80s|1611424KiB|NOT_CAPTURED|
|D1|1248/1248|503/0/0|503|0|0|4992B|5800B|19.67s|1783224KiB|NOT_CAPTURED|
|D2|1368/1368|231/0/0|231|0|0|5472B|6790B|12.26s|1763420KiB|NOT_CAPTURED|
|D3|1584/1584|327/0/0+4-fold OOF|327+OOF|0|0|6336B|8211B|58.45s|1809016KiB|NOT_CAPTURED|
|D4|1152/1152|300/150/100|0|450|929520B|4608B|5305B|12.89s|1648460KiB|NOT_CAPTURED|

解释：

- D0通过cache把450次完整backbone forward替换为suffix forward，是当前最快且内存最低的有效行；
- D1虽然只多96个参数，但503次完整backbone forward使时间接近20秒；
- D2用231步将时间压到12.26秒，但其BA和floor不够；
- D3的单次full-support refit本身不应需要58秒，主要额外成本来自4-fold support OOF训练；它不满足星上快速适配档；
- D4比D0多30步CVaR和30步额外head polish，增加2.09秒，未改变argmax。

GNU time的最大RSS包含Python、PyTorch、CUDA context、动态库和allocator，不等于模型tensor大小。基础模型tensor约4199312B，D0/D4 cache约929520B，继续把1152个参数再压缩少量不会显著降低1.5GB级进程RSS。

## 13.晋级门槛逐项判定

相对修复后D0：

- BA≥D0；
- floor≥D0；
- 最差类别变化≥-5pp；
- NLL≤D0+0.02=`0.520754`；
- 可训练元素≤1584；
- delta≤10KB；
- wall-clock≤20秒。

|行|BA|floor|类别保护|NLL|元素|delta|时间|综合|
|---|---|---|---|---|---|---|---|---|
|D0|PASS|PASS|PASS|PASS|PASS|PASS|PASS|`ENGINEERING_REPLAY_PASS/BASELINE`|
|D1|PASS|FAIL|FAIL|PASS|PASS|PASS|PASS|`NO_PROMOTION`|
|D2|FAIL|FAIL|PASS|PASS|PASS|PASS|PASS|`NO_PROMOTION`|
|D3|PASS|PASS|FAIL|FAIL|PASS|PASS|FAIL|`NO_PROMOTION`|
|D4|PASS|PASS|PASS|PASS|PASS|PASS|PASS|`PASS_NO_INCREMENTAL_VALUE`|

满足全部门槛的最小候选是D0。D4虽然也通过，但与D0 argmax完全相同，NLL改善仅0.000699，资源更差，不应替换D0。

## 14.发布与验证

关键实现commit：

- `656af576ae86d408ff5a206712d81524edca765c`：P0工程优化、D0–D4矩阵；
- `24493929ffce87f371bf036b219cc733b75d05eb`：D3 support-only OOF温度；
- `7ffef5703aa536ac0f3a29474eea3261ba2f8f5f`：Compact梯度裁剪修复；
- `b17c8fe18c1d7b2165819194a347d294f14b9423`：smoke cache materialize修复；
- `6bfc808be9cb089f58cd52ba4d09289d3a06cf4e`：r2预登记。

验证：

- 118项P1聚焦测试通过；
- smoke脚本编译和20项部署/benchmark测试通过；
- 独立P0/P1定点审查PASS；
- r2 release归档SHA256：`aa4972aeb62e80d5057576490c73baa905f6a997fbd01ca6a9deec77897601eb`，本地/远端一致；
- 远端编译PASS；
- 所有有效support行状态为`DEPLOY_ADAPT_COMPLETE`；
- 所有Q60/Q120 prediction receipt为`PREDICTIONS_COMPLETE`；
- 所有score为`ANALYZED`和`truth_join_after_prediction_only=true`。

## 15.尚未完成的设计项

以下项目已明确保留为后续独立工程/科学候选，不冒充本轮完成：

- 真正每次新子进程的cold-start资源基准；
- 常驻模式3预热+10正式的完整GPU allocated/reserved/free采样；
- FP16/BF16 suffix计算等价性；
- mixed cache；
- CUDA Graph；
- `torch.compile`/AOTAutograd；
- Norm标准化输入预计算；
- 冻结suffix eval语义独立消融；
- 新未暴露capsule；
- 多receiver；
- low-elevation/rain；
- 多seed；
- K=5/K=2；
- 新类注册和开放集侵入。

## 16.最终判断与下一步

当前可以保留两个概念档位，但只有一个默认：

- 最小平衡档：D0。1152元素、5305B delta、10.80秒、BA 83.3333%、floor 56.6667%、NLL 0.500754；
- 性能研究档：D3。BA 85.5556%、floor 63.3333%，但NLL、类4保护和58.45秒时延均未过门，不能部署为当前默认。

下一步不应继续在已暴露rx20 query上调整温度、步数或损失。应冻结D0及D1/D2/D3候选定义，在新的未暴露合法capsule上先做单seed、Target5或更小的可证伪确认；只有同样通过BA、floor、类别保护、NLL和资源门槛后，才进入多receiver/scene/seed扩展。

