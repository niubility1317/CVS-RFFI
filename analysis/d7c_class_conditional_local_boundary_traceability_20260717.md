# D7c类条件IQ head与局部边界组合审计

日期：2026-07-17

状态：独立审计阻断修复与本地回归完成；真实K-shot support重放尚未执行，因此本文不声称D7c独立性能。

## 0.独立审计阻断修复追踪

|ID|来源|要求|目标文件|状态|验证|说明|
|---|---|---|---|---|---|---|
|D7C-R01|`项目.md`7.1.1|每个operator feature逐样本绑定`parent_received_iq_sha256/operator_id/view_seed`，fit不得接受普通mapping或伪造随机feature|D7a、D7c、测试|verified|普通mapping、随机payload篡改、错误IQ SHA均fail closed|validated artifact由固定received IQ内部生成并密封|
|D7C-R02|`项目.md`7.2|query逐样本执行，callback不能观察其它query，模块闭包内无q-q图|D7a、D7c、测试|verified|batch-coupled extractor首行严格不变|每次callback输入batch size固定为1|
|D7C-R03|`项目.md`10.3.1|K10锁定operator/rival/beta后，K1/K5只重建prototype且K1可运行|D7a、D7c、测试|verified|K1/K5策略逐位检查并完成一次query推理|不重新执行删除法选择|
|D7C-R04|资源硬门|从base累加真实参数、query-fit与状态；超限fail closed；区分head MAC和未知端到端项|D7c、测试|verified|60k参数、query-fit/update、伪造state bytes均拒绝|端到端MAC/时延/显存保留`None`待profile|
|D7C-R05|逐类floor|联合beta门要求每个已注册类非退化|D7c、测试|verified|100组构造分数逐类检查|非floor类退化也触发全beta回退|
|D7C-R06|注册前后同row|after必须绑定并验证parent旧support lineage和逐view feature指纹|D7c、测试|verified|替换旧IQ/hash或同IQ改feature extractor均拒绝|state保存旧support lineage和binding fingerprint|
|D7C-R07|算法声明边界|不得虚称严格D7b prototype-Gram组合；明确为D7a calibrated-confusion local boundary|D7c、本文|verified|模块docstring、candidate与本文一致|保留局部margin，但不声称prototype-Gram等价|

## 1.组合边界

D7c复用D7a类条件表示，并增加calibrated-confusion local boundary；它不是严格D7b prototype-Gram组合，也不复制第二套特征空间：

1. D7a继续独占每类锁定`operator/prototype/calibration`。
2. D7c在D7a校准后的逐类分数空间中，为每类增加一个`rival_index`和一个标量`beta`。
3. 最终分数为：

   `s'_c=s_c+beta_c*(s_c-s_rival(c))`

4. Query阶段按每个物理query逐样本执行`base_state.used_operators`中的固定接收IQ变换和representation前向；callback永远只接收一行。D7c随后只做逐类索引、减法、标量乘法和加法。

不能直接把D7b的单一prototype Gram张量接到D7a后面。原因是D7a不同类别可能锁定在不同operator空间，类间prototype直接做Gram相似度没有统一表示语义。D7c因此复用D7b的“每类局部rival margin”机制，但把rival定义改为物理support删除验证中的D7a校准类混淆，不新建或重复backbone。

## 2.项目协议映射

|项目约束|D7c实现|
|---|---|
|单一clean样本进入Phase2前只能对应一个LEO星地信道状态|D7c只接收已经存在的固定`received_iq`，并逐样本验证其真实SHA-256，不生成任何新LEO信道状态|
|固定接收IQ最多3个view|由D7a validated artifact builder对每个物理样本逐个执行固定operator集合：`base/dc_rms/dc_rms_spec15`；每个binding记录父IQ哈希、operator和固定`view_seed=0`|
|view不增加K|资源审计固定声明`views_count_as_additional_k=false`、`additional_physical_samples_from_views=0`|
|support可选择，query不可拟合|rival和beta只由物理support删除验证锁定；query API没有label/truth/role/quota/graph参数，资源审计继承并核验D7a的`query_rows_used=0`和`query_updates=0`|
|逐样本、全注册类决策|Query特征提取callback的batch size固定为1，分数形状为`[N,C]`，每行独立`argmax`，没有q-q交互或batch重分配|
|无角色Oracle与类别配额|资源审计将角色、真实batch类数、类别配额和全局分配访问全部锁为`false`|
|注册后锁旧类状态|D7a旧operator/prototype/calibration与D7c旧rival/beta均逐项bitwise校验；旧rival禁止引用新类|
|轻量部署|从D7a状态累加真实训练参数和持久状态，训练参数超过50k或状态超过256KB即fail closed；适配epoch 0、dense query graph 0B|

## 3.Support-only选择审计

### Before

1. 使用D7a已经锁定的每类operator与calibration。
2. 对每类物理support做leave-two-out；不足4-shot时退化为leave-one-out，但每类必须保留至少一个拟合样本。
3. 每个删除fold只用保留support重算该类已选operator下的prototype。
4. 在held-out support上产生D7a校准类分数。
5. 每类rival取该类held-out样本上平均分最高的非本类。
6. 每类`beta`仅从`[0,0.05,0.10,0.20]`选择，要求该类准确率和总体准确率均不低于`beta=0`。
7. 合并所有逐类beta后再次执行每个已注册类及总体准确率非退化门；任一类退化即全部回退到`beta=0`。

### After

1. D7a扩展状态必须证明旧classes、class operators、prototypes与calibrations未改。
2. D7c父状态旧rival与beta保持bitwise不变，且旧rival仍只能指向旧类。
3. 只对新增类执行物理support删除验证并追加rival与beta。
4. 新类beta选择除新类删除验证非退化外，还使用完整注册old support做intrusion guard。
5. 多个新类beta合并后再次核查每个新类、每个旧类、new总体和old总体准确率；任一类退化时只把新增类beta回退到0，旧状态仍不动。

审计artifact中固定记录：

- `query_rows_used=0`
- `query_labels_used=false`
- `query_roles_used=false`
- `query_quota_used=false`
- `old_state_bitwise_locked=true/false`
- `rival_source=heldout_d7a_calibrated_class_confusion`
- 每类候选beta的同一行support指标、资格判断与最终选择
- 合并前后非退化门和回退状态

## 4.接口兼容和资源

实现文件：

- `code/cvsrffi/stage2_class_conditional_local_boundary.py`
- `tests/test_stage2_class_conditional_local_boundary.py`

D7c状态直接持有D7a状态引用，不重复保存prototype或calibration。用于打分的新增张量只有：

- 每类一个实际float32 beta：4B
- 每类一个实际int64 rival index：8B

因此D7c打分张量相对D7a固定增加`12*C`B。但正式D7c持久状态还必须保存每个物理support的`label/physical_sample_id/parent_received_iq_sha256`，以及每个合法operator view的binding fingerprint，用于注册后验证旧support lineage和特征身份未改变。`persistent_state_bytes`按实际字符串字节、索引数组、beta/rival张量和D7a真实部署状态重新计算，并在构造状态时执行256KB硬门。

此前基于未绑定support lineage状态推导的6类/11类字节表不再作为当前资源证据。每个正式候选必须从实际K-shot validated artifact报告真实状态字节；若超过256KB则fail closed，不能用声明值覆盖。

D7c不会把representation前向增加到类别数。每个物理query的前向次数等于D7a实际使用operator的去重数，范围为1至3；由于query严格逐样本执行，批量N个query的callback调用次数为`N*used_operator_count`，但单样本平均值与P95均为`used_operator_count`。

每个query的head侧MAC估计为：

`C*D+5*C`

其中`C*D`是D7a逐类prototype点积，`5*C`覆盖校准和单rival margin的标量操作。该值仅是head侧MAC，不包含固定IQ变换、representation前向和运行时开销。端到端MAC、时延与峰值显存均明确保留为`None`，必须由目标部署profile补齐后才可形成Pareto声明。

## 5.验证

执行环境：

`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`

命令：

```text
python -m pytest -q tests/test_stage2_class_conditional_iq_head.py tests/test_stage2_class_conditional_local_boundary.py tests/test_stage2_local_contrastive_boundary.py tests/test_stage2_floor_sparse_operator_fusion.py
```

结果：

```text
...................................                                      [100%]
35 passed
```

覆盖项：

- D7a状态直接复用与手工分数公式一致
- 每个物理sample/operator binding均绑定真实父IQ哈希、operator、固定view seed和feature哈希
- 普通mapping、随机feature payload篡改和错误IQ SHA均fail closed
- Query callback每次只接收一行，batch-coupled extractor不能让其它query改变首行结果
- support-only rival/beta锁定和逐类非退化门
- 注册后旧D7a/D7c状态bitwise锁定
- 注册后旧support lineage和逐view feature fingerprint保持不变
- K10锁定策略后，K1/K5只重建prototype且完成真实query推理，不重选operator/rival/beta
- 旧rival不引用新类；旧operator/prototype/calibration/rival/beta均保持锁定
- 继承真实参数、query-fit/query-update和持久状态审计；60k参数、非零query使用和伪造state bytes均fail closed
- 0epoch、无dense query graph、状态小于256KB；head MAC与未知端到端profile严格分离
- Query公开接口无label/truth/role/quota/graph
- D7b和floor-sparse相邻模块回归通过

## 6.证据边界与下一步

现有真实D7a support-lock artifact：

`automation_reports/CV-SincNet/d4a_single_observation_smoke_20260717_010128/dev_k10_new5_r2/d7a_support_lock_r2/support_lock.json`

该artifact形成于validated operator feature artifact、逐view binding fingerprint和当前持久状态口径之前。它可以作为历史开发线索，但在当前D7c绑定协议下属于`UNVERIFIED_UNDER_CURRENT_D7C_BINDING`，不能复用其中的状态字节估算或直接作为正式D7c输入。本文没有从汇总指标反推rival/beta，也没有打开query truth重新选择D7c。

下一步应在新的未评分receiver/seed或独立confirmation矩阵上，从合法LEO_weak support包直接运行D7a→D7c锁定，先密封预测artifact，再由隔离scorer评分。当前两套已评分query只能用于开发诊断标记，不能声称独立确认性能。

仍需由正式runner闭包补齐两项模块外证据：其一，validated artifact builder所调用的feature extractor/checkpoint必须由sealed Phase1 checkpoint哈希和代码allowlist固定，不能由任意外部callback替换；其二，虽然模块保证callback逐样本输入，但正式运行时还必须审计callback不读取外部query batch全局状态。两项缺失时只能做本地单元级协议验证，不能形成正式Phase2性能或部署证据。

## 7.D8b strict K10 enrollment-only真实锁定

新增runner：

- `code/scripts/run_d7c_support_only_enrollment.py`
- `tests/test_run_d7c_support_only_enrollment.py`

输入只使用D8b `receiver=1-20,seed=713201,new5`的before/after `enrollment_only`密封包：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d8b_k10_new5_enrollment`

runner CLI没有query、truth、prediction、score或scorer参数。pre-open验证逐场景每个已注册类实际可达support恰为10条，before为6×10，after为11×10；三个scenario的物理sample ID与父IQ SHA集合互斥。每个operator view仍由单一固定LEO_weak接收IQ逐样本生成，feature extraction batch size为1。

### 7.1失败证据与最小修复

- `d7c_support_only_strict_k10_v1`：显式CPU输入与sealed CUDA TorchScript权重设备不一致，在第一条support forward失败；目录为空，没有state或COMMIT。
- `d7c_support_only_strict_k10_v2`：before/after对同一旧support分别执行GPU前向，数值不保证bitwise一致，D7c在旧feature binding SHA门fail closed；没有state或COMMIT。
- v3没有放宽旧lineage/binding门。runner在同一support-only进程内按固定operator view的float32字节SHA缓存首次合法samplewise feature；after旧support复用完全相同feature，新support才执行新增前向。实际记录990次cache miss、1458次cache hit，`old_support_second_gpu_forward_required=false`。

### 7.2v3不可变artifact

输出：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d7c_support_only_strict_k10_v3`

- 状态：`SUPPORT_ONLY_D7C_LOCKED_NO_QUERY_OPEN`
- `COMMIT.json` SHA-256：`09cbb11121b7536a64ba295951723267b81120136d0f9073580149f461935aa1`
- `support_audit.json` SHA-256：`67ed7a37e15dfdd2d6487b42aa27c780a3312cc7ac37cfc1958d619730b7f02a`
- 39个输出文件全部只读；COMMIT对audit、report和state哈希引用复算一致。
- `query_package_opened/query_truth_opened/query_prediction_opened/query_score_opened/scorer_opened`全部为false。
- 12个K1/K5嵌套前缀证明全部保持operator、calibration、rival和beta锁定，只重建prototype。它们是从K10包形成的support-only结构证明，不是独立K1/K5密封包或性能证据。

### 7.3Support逐类、floor与资源结论

|scenario|阶段|类数|support overall baseline→final|support floor baseline→final|持久状态|operator数|
|---|---|---:|---:|---:|---:|---:|
|`leo_clear_weak`|before|6|0.75→0.75|0.20→0.20|30,998B|3|
|`leo_clear_weak`|after-old|6|0.85→0.85|0.60→0.60|56,805B组合状态|3|
|`leo_clear_weak`|after-new|5|0.56→0.56|0.30→0.30|同上|3|
|`leo_low_elev_weak`|before|6|0.7833→0.7833|0.50→0.50|30,971B|3|
|`leo_low_elev_weak`|after-old|6|0.8333→0.8333|0.70→0.70|56,764B组合状态|3|
|`leo_low_elev_weak`|after-new|5|0.38→0.40|0.00→0.00|同上|3|
|`leo_rain_weak`|before|6|0.7167→0.7167|0.30→0.30|30,982B|3|
|`leo_rain_weak`|after-old|6|0.8333→0.8333|0.50→0.50|56,775B组合状态|3|
|`leo_rain_weak`|after-new|5|0.32→0.38|0.00→0.00|同上|3|

51个逐类support行全部满足相对各自`beta=0`基线不退化，说明逐类门正确生效；但low-elevation和rain的after-new support floor仍为0，clear的before floor也仅0.20。该support证据不支持把D7c预登记为有希望的性能候选，更不能替代query确认。v3应保留为合法support锁定与资源证据；若主线程必须在query前二选一，D7c当前应标记`SUPPORT_FLOOR_INSUFFICIENT_NOT_SELECTED`，而不是仅凭`LOCKED`状态晋级。

端到端MAC、时延与峰值显存仍未profile，保持`None`；当前只验证0训练参数、0epoch、head侧MAC、3个固定received-IQ representation forward和小于256KB状态。

选择层否决marker：

`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d7c_support_only_strict_k10_v3\SUPPORT_FLOOR_INSUFFICIENT_NOT_SELECTED.json`

该marker明确`candidate_bound_query_generation_authorized=false`和`candidate_bound_query_package_created=false`；D7c不生成candidate-bound query包。

### 7.4验证

```text
python -m pytest -q tests/test_run_d7c_support_only_enrollment.py \
  tests/test_stage2_class_conditional_iq_head.py \
  tests/test_stage2_class_conditional_local_boundary.py \
  tests/test_stage2_local_contrastive_boundary.py
```

结果：`31 passed`。
