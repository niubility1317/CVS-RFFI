# qKNN Stage2-C effective8 LEO_weak-only正式实验报告

|字段|内容|
|---|---|
|实验ID|`qknn_ground_effective8_r16_e12_leoonly_20260715_v14`|
|时间|2026-07-15 13:44 CST|
|operator|Codex`/root`|
|当前状态|`CANDIDATE_LOCK_V1_INVALID_SOURCE_CACHE_PROVENANCE`；repair6候选锁v2已完成本地验证，待同步并重新执行source validation/candidate lock；target matrix未启动|
|基座模型|同一ADV3B02 checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|目标|在新版`LEO_weak-only`边界下，以≤50k参数、≤20epoch、≤256KB持久状态的极轻型适配器和逐样本1→3→5-view推理，完成5receiver×5seed×3场景×新类5/10/20×K=1/5/10/20正式确认|

## 协议追踪表

|ID|Source section|Requirement|实现文件|状态|本地证据|
|---|---|---|---|---|---|
|P2-LW-01|`项目.md`7.1、9|Phase2只读取已叠加允许场景的post-channel IQ|`leo_weak_cache.py`、cache builder、trainer、validator、benchmark|implemented|cache成员列表先审计，禁止`raw_iq`、clean和派生feature/logit/prototype|
|P2-LW-02|`项目.md`7.1、9|训练、验证、门限和promotion不得使用clean派生信号|trainer、validator、candidate lock|implemented|formal训练使用same-LEO teacher/reference；source holdout只读密封cache-set|
|P2-LW-03|`项目.md`7.1|逐样本保留场景、satellite seed、overlay和waveform摘要|cache schema、formal rows、formal predictions|implemented|sample ID、逐query overlay ID/post-channel IQ SHA及support/query root均进入正式证据|
|P2-LW-04|`项目.md`5、7.1|Phase1离线叠加信道，Phase2禁止重新读取ManySig/ManyTx|builder、strict loader、formal CLI|implemented|训练与验证CLI只接收cache-set；benchmark拒绝legacy raw/feature字段|
|P2-LW-05|`项目.md`8.5|query默认1-view，低置信度才请求3/5-view|benchmark、adaptive TTA|implemented|base-only路径不会预构造其余4个View；逐query记录view budget|
|P2-LW-06|`项目.md`7.3、8.5、9|禁角色Oracle、类别quota、query fit和batch图|config、candidate lock、benchmark、summarizer|implemented|所有注册类统一head和门限；正式字段全部为false|
|P2-LW-07|`项目.md`9|锁定candidate/head/TTA及K=1/5/10/20有序嵌套|plan、runner、candidate lock、summarizer|implemented|固定seed`713101–713105`、`query_per_tx=20`、25份cache spec和300份row config；逐样本prediction支持独立重算|
|P2-LW-08|`项目.md`8.5、9|报告参数、epoch、MAC、延迟、峰值显存、状态、平均/P95 View|benchmark formal rows|implemented|逐row资源字段已落地；MAC明确标注FFT/View变换未计入的边界|
|P2-LW-09|`项目.md`9|完成5×5×3×3×4正式确认及逐类、逐receiver证据|formal plan、N607 artifacts|pending|本地计划精确生成25个target cache-set、300次评估、900条场景row；尚未运行|
|P2-LW-10|`项目.md`7.1|旧v13 raw/clean命令不得启动|v13报告、v14 runner|rejected|v13目标run/log不存在；v14不复用旧命令|
|P2-LW-11|`项目.md`10.3.1|同一极轻型候选需补充Stage2-B target-old-only独立确认|Stage2-B formal plan/artifacts|pending|当前v14是Stage2-C source锁定与old/new矩阵；尚无Stage2-B正式row，不得以Stage2-C旧类边际统计替代|

追踪状态计数：implemented=8，pending=2，rejected=1。当前最高风险是P2-LW-09与P2-LW-11：Stage2-C尚无新版target矩阵结果，Stage2-B尚未生成正式独立确认row，因此不得声明达到准确率或遗忘目标。

## 方法、输入与输出

### Phase1离线cache构建

输入为ManySig/ManyTx物理IQ、预注册TX/receiver/day集合、三个允许场景和satellite seed。该阶段在Phase2边界外实际叠加`simplified_leo_residual`，输出每场景一个只含`leo_weak_iq`和sample-level provenance的NPZ，以及一个cache-set manifest。输出不含raw/clean IQ、feature、logit、FFT或prototype。

source训练cache使用ManySig TX selector index`0–5`和receiver selector index`0–5`；解析后的真实TX标签为`14-10,14-7,20-15,20-19,6-15,8-20`，真实source receiver标签为`1-1,1-19,14-7,18-2,19-2,2-1`。source validation cache覆盖receiver selector index`0–6`，其中前6个真实receiver作为source参考，真实receiver`2-19`为完整holdout。target cache以5个receiver和5个确认seed分别构建，每个cache预先包含6个target-old TX和有序嵌套的20个target-new TX，K和query切分发生在密封cache内部。

### 极轻型ADV3B02适配

ADV3B02主体冻结，只注入rank16 effective-feature LoRA到8个进入`feat_joint/z_id`的关键Linear：`t_proj`、`f_proj`、`pa_proj.0`、`fuse.0`、`cls_head.id_proj.0`、`cls_head.pa_proj.0`、`cls_head.id_gate.0`、`cls_head.joint_proj.0`。可训练参数44,048，训练12epoch，FP16 LoRA状态88,096B；部署前可合并到原Linear，新增持久动态LoRA MAC为0。

formal训练不再使用clean teacher。对同一密封LEO_weak base observation，teacher为冻结ADV3B02，student为LoRA后的base加一个轮换`rx_light5`接收侧View。训练每步只延迟构造base和一个轮换额外View，不再预构造全部5个View；CFO轮次的底层`rx_cfo3`仍同时生成正负两个CFO变换但只做2次backbone前向，这是剩余的非阻断压缩空间。损失为：SmoothL1特征保持1、cos保持2、prototype CE 0.2、same-LEO reference 22、feature margin 4.5、same-LEO margin 7.5、teacher-logit KL 0.16、双View一致性0.25、relation Gram 0.5、prototype Gram去混淆0.25、nested worst-K风险0.5。训练命令使用`leo_reference_*`参数名，不再把formal anchor写成`clean_*`。

输出为`effective8_adapter_fp16.pt`、完整epoch loss trace和训练manifest。训练manifest绑定checkpoint、adapter state和source cache-set，并明确`clean_sample_access=false`、`clean_derived_signal_access=false`；trainer、LoRA、validator、共享tensor bridge和benchmark等代码哈希由后续candidate lock统一绑定，不能误称训练manifest自身包含代码哈希。

### source holdout与candidate lock

validator唯一数据入口为`--source_cache_set`，固定调用`load_verified_leo_weak_cache_set(expected_scope="source_validation", allowed_roles={"source"})`。它以receiver 0–5构造source prototype/statistics，以receiver 6完成物理样本不重叠的校准和评估，统一锁定symmetric head、K=1/5/10/20规则和自适应TTA门限。任何gate失败返回非0并阻断candidate lock。

candidate lock固定checkpoint、adapter、promotion manifest、source validation、source statistics、source train/validation cache-set、类别切分、direct ADV3B02类映射、head、TTA门限、5个receiver、5个seed和所有相关代码哈希。执行前由runner重新推导阶段、命令数和300个cell身份；lock逐个绑定25份target cache build spec和300份Stage2 config的文件/内容SHA，benchmark前再次检查spec未漂移，并要求实际cache-set的`build_spec_sha256`与lock一致。K1/K5/K20不能重新选参。

### 自适应多View推理

每个query先执行`rx_base`一次backbone+FFT96。当source锁定门限判定低置信度时，才对同一post-channel物理观测生成`rx_shift_m2/p2`并提升到3-view；仍低置信度才生成`rx_cfo_m1e4/p1e4`并提升到5-view。决策只使用当前query分数的margin和View分歧，不使用真实old/new角色、类别数量、quota、query标签、query排序或跨query图。

base-only路径直接复用输入张量，不调用5-view构造器；shift和CFO View按需物化。跨场景ALL资源统计由全部query的`view_budget`直接重算平均、真实全局P95和1/3/5触发率，不再平均场景P95。每条正式prediction同时保存direct预测标签、逐query overlay/IQ哈希、实际locked-head状态SHA和对应formal-row内容摘要；汇总器从逐样本truth/prediction重新计算准确率、逐类最差值、遗忘率、K1 direct增益和View资源，不信任row中预填的汇总指标。

## 正式矩阵与成功门槛

执行计划由`cvs_stage2c_effective8_formal_matrix_20260715.json`生成：

|维度|取值|数量|
|---|---|---:|
|target receiver|`20-1,3-19,7-14,7-7,8-8`|5|
|确认seed|`713101–713105`|5|
|场景|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|3|
|真实新类规模|5、10、20有序嵌套TX|3|
|K|1、5、10、20有序嵌套support|4|

计划生成2个source cache-set、25个target cache-set、300次benchmark调用和900条场景级formal row。8个GPU shard合计覆盖全部命令且无重复。每个target cache只构建一次，再执行该receiver×seed下12个新类规模×K组合。所有shard完成后，`finalize`阶段先收集300个输出目录并逐文件绑定SHA，再汇总900行正式证据；缺任一cell、重复cell或CSV schema漂移都会失败关闭。

成功门槛遵循2026-07-15更新后的`项目.md`：Stage2-B与Stage2-C的K10 old_acc≥0.92、聚合min_old_class_acc≥0.88；Stage2-C新类5/10/20分别≥0.92/0.90/0.86；K5相对matched K10四项指标drop≤3pp；K1总体及逐receiver old_adaptation_gain≥0；K1相对严格direct ADV3B02总体增益≥2pp且matched 95%CI下界>0、逐receiver增益≥0。K5/K10/K20遗忘不得高于同row identity-only单qKNN。Stage2-B通过不能替代Stage2-C的新类与H门槛。

## 资源口径

正式row记录adapter参数、epoch、optimizer step、FP16状态、统一head状态、门限状态、平均/P95/worst-case backbone forward、1/3/5触发率、部署query时延、含enrollment摊销时延、峰值CUDA显存和host working set。rank16 LoRA为44,048个训练参数、88,096B FP16状态；加统一head和3个门限后的候选增量持久状态约103,796B，低于256KiB门槛。该数字不含ADV3B02基座、类别字符串或运行时缓存，不能称为卫星总模型占用。模型MAC通过实际执行的Conv1d/Linear/SincConv1d hook计数，另加support head MAC；FFT96和接收View变换运算目前只计入端到端时延，不并入该MAC数字。延迟来自服务器GPU，不等价于真实星载硬件时延；因此当前可声称“地面极轻型参数适配、星上轻量推理”，不能据此声称已经证明星上在线训练可行。

## 本地验证

根目录`E:\type10-7`不是Git仓库；Git承载面为`E:\type10-7\github_publish\CVS-RFFI-repo`。本轮本地验证均使用`ssr-gpu`解释器：

```text
python -m py_compile <trainer/validator/cache/benchmark/lock/plan/runner>
python -m pytest -q <10组聚焦测试>
结果：79 passed
git diff --check
结果：PASS
```

正式计划生成测试确认：source cache-set=2，target cache-set=25，benchmark invocation=300，formal scenario row=900，collection=1，summary=1；8个shard合计25次cache构建和300次benchmark，无重复命令。

## N607启动与结束条件

启动前必须重新执行本地只读direct preflight和live GPU/process inventory。先只启动`source_pipeline`：构建2个source cache-set→训练12epoch→source validation→candidate lock；只有state为complete且source validation PASS，才允许启动8个matrix shard。任何Traceback、OOM、nan/inf、cache/hash/receiver/class/protocol gate失败都会停止后续阶段。

### 启动前版本与同步证据

|项目|证据|
|---|---|
|Git承载面|`E:\type10-7\github_publish\CVS-RFFI-repo`，commit=`fef819bd062a`|
|N607 direct preflight|2026-07-15 14:15 CST通过；项目根、GPU可见|
|live lane inventory|2026-07-15 14:16 CST：`active_training_processes=[]`、`gpu_compute=[]`|
|GPU/磁盘|8×RTX 3090均约10MiB；项目盘可用7.6TB|
|数据/checkpoint|ManySig、ManyTx及ADV3B02 checkpoint存在|
|远端环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；Python 3.10.19、Torch 2.1.0+cu121|
|覆盖前快照|`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/qknn_ground_effective8_v14_remote_before_sync_20260715_1420`；5个既有同名文件|
|同步|14个生产代码/config/report文件按仓库相对路径同步到N607项目根；14/14 SHA256一致|
|远端验证|同一远端Python对10个生产`.py`执行`py_compile`通过|
|SSH清理|每次SSH/SCP后本地`ssh.exe=0`、N607:22 established连接=0|

同步映射为同相对路径：`code/cvsrffi/leo_weak_cache.py`、cache builder、formal trainer、validator、benchmark、candidate lock、summarizer、plan builder、collector、runner、3个正式config和本报告；本地根均为Git承载面，远端根均为`/home/szu2070436088/2510044040/CV-SincNet`。

### source pipeline精确启动约定

|字段|值|
|---|---|
|工作目录|`/home/szu2070436088/2510044040/CV-SincNet`|
|物理GPU|GPU0；以`CUDA_VISIBLE_DEVICES=0`暴露，子命令内部使用`cuda:0`|
|plan manifest|`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/protocol_plan/plan_manifest.json`|
|runner日志|`logs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/source_pipeline_runner.log`|
|step日志|`logs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/source_pipeline_steps/`|
|state|`runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/source_pipeline_state.json`|
|PID文件|`logs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/source_pipeline_runner.pid`|
|预期产物|2个source cache-set、`effective8_adapter_fp16.pt`、`training_manifest.json`、`source_validation_v2/`、`candidate_lock_v2.json`；旧`candidate_lock.json`仅保留为无效证据|

精确plan生成命令：

```bash
PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python paper_reproduction/scripts/build_cvs_stage2c_effective8_formal_plan.py --plan paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json --out_dir runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/protocol_plan --runtime_project_root /home/szu2070436088/2510044040/CV-SincNet
```

精确runner命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/code:/home/szu2070436088/2510044040/CV-SincNet nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u paper_reproduction/scripts/run_cvs_stage2c_effective8_formal_plan.py --plan_manifest runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/protocol_plan/plan_manifest.json --project_root /home/szu2070436088/2510044040/CV-SincNet --stage source_pipeline --log_dir logs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/source_pipeline_steps --state_json runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/source_pipeline_state.json > logs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/source_pipeline_runner.log 2>&1 &
```

当前仍没有新版adapter、validation、candidate lock或formal metrics；启动后也必须以state、PID/cmdline、日志和产物共同判断状态，不能把landed submit当作训练或部署成功。

## 2026-07-15 source pipeline失败关闭与protocol repair1

首次source pipeline已实际landed到N607物理GPU0，runner PID为`1531601`。source train cache构建在约13.5s完成，source validation cache构建在约15.3s完成；随后训练步骤在约2.2s内失败，错误为：

```text
ValueError: source TX set drift for leo_clear_weak: ['14-10','14-7','20-15','20-19','6-15','8-20']!=['0','1','2','3','4','5']
```

这是正确的协议失败关闭，不是性能结果。根因为plan generator把Phase1 cache builder使用的selector index`0–5`原样传给了Phase2 trainer，而严格cache manifest保存的是解析后的真实TX/receiver标签。两个source cache均已完整生成，但adapter训练、source validation、candidate lock和全部target matrix均未开始；因此没有任何新版准确率、遗忘率或资源结论。

repair1保持cache build spec中的selector index不变，仅在训练与验证命令边界显式传递真实标签：source TX=`14-10,14-7,20-15,20-19,6-15,8-20`，source train receiver=`1-1,1-19,14-7,18-2,19-2,2-1`，source validation holdout receiver=`2-19`。正式config同时封存selector到真实receiver的映射，plan validator要求6个source receiver唯一、训练与验证参考集合完全相同、holdout唯一且不重叠。该修复不改变LEO_weak样本、不引入clean、角色Oracle、类别quota或query信息，也不改变adapter、损失、K、seed、场景和target矩阵。

repair1本地验证使用`ssr-gpu`解释器：plan builder`py_compile`通过；10组相关测试`79 passed`；修复计划精确生成2个source cache-set、25个target cache-set、300次benchmark调用、900条场景row、1次collection和1次summary。修复计划保存在本地非发布artifact`local_artifacts/qknn_ground_effective8_v14_protocol_plan_repair1_20260715/`，尚未覆盖远端正式plan。已按cache builder的规范化JSON算法核对repair1与现有cache的`build_spec_sha256`：source train均为`7897de1138ee67bf0ebcb91df3ed11f4993020dc8222e9535f18a54b7e8dd2f3`，source validation均为`75a7d772aa9d39b209a9ddd2b6f310189ff43107d603d9bd8573fe6dd897256f`。原始JSON文件SHA因格式化不同而不同，但实际build spec语义哈希完全相同，故严格loader允许复用已完成cache；恢复前仍须快照远端旧plan，再同步修复代码/config和修复plan。只有source pipeline从失败步骤恢复且promotion PASS后，才允许启动target matrix。

### repair1远端同步与恢复门禁

2026-07-15 14:32 CST重新执行direct preflight与live inventory：8张RTX 3090均约10MiB，`active_training_processes=[]`、`gpu_compute=[]`。覆盖前快照为`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_repair1_before_sync_20260715_143529`，包含旧protocol plan、将覆盖的config/builder/report，以及首次失败的runner日志、`0002_train.log`和state；原cache、日志、state和run产物未删除或移动。

Git提交`e4a67c5`封存selector→真实标签修复，提交`7eacc1a`封存cache兼容性审计。config、plan builder、report和本地生成的repair1 plan压缩包已通过direct SCP同步；本地/远端SHA256分别一致为`0627c019...`、`6de1b42d...`、`f1cd8378...`、`6da35637...`，远端builder`py_compile`通过。repair1 plan已覆盖到原正式plan目录，远端runner validator确认source step=5、target cache step=25、benchmark step=300，并确认训练TX/receiver与validation holdout均为真实标签。两份source spec的规范化SHA继续与现有cache manifest逐项相等。所有SSH/SCP后均确认本地`ssh.exe=0`、N607:22 established连接=0。

恢复将复用原`source_pipeline_state.json`：步骤`0000`和`0001`的完整命令未变化且状态为complete，因此runner只从命令已经变化的`0002_train`重跑；旧失败记录与原日志已单独保存在快照的`failure_evidence/`。恢复前仍需再次执行live inventory；若出现其他任务则不启动。

## 2026-07-15 repair1恢复失败与NumPy/Torch bridge repair2

14:41 CST启动前live inventory再次确认`active_training_processes=[]`、`gpu_compute=[]`。repair1 source pipeline以物理GPU0、PID`1544400`实际landed，runner正确跳过两项已完成cache步骤并进入`0002_train`，但在首个prototype loader batch、任何epoch和optimizer step之前再次失败关闭：

```text
TypeError: expected np.ndarray (got numpy.ndarray)
```

按完整日志而非tail完成审计：runner日志8行、source train cache日志73行、source validation cache日志73行、repair1训练日志20行均已逐行读取。两个cache构建日志均完整结束，3个场景分别为6,000条训练行和8,400条验证行，`clean_sample_access=false`、`phase2_sample_view_policy=leo_weak_only_no_clean_access`；没有OOM、Killed、nan/inf或epoch/loss记录。失败位置唯一落在`SealedLeoWeakSourceDataset.__getitem__`的`torch.from_numpy`。

远端运行环境为NumPy`2.2.5`与Torch`2.1.0+cu121`。独立只读探针证明：精确`np.ndarray`身份为true，普通float32数组和密封cache的连续float32 IQ行都在`torch.from_numpy`报同一TypeError，`torch.as_tensor`/`torch.tensor(np_array)`也失败；相同内存通过`torch.frombuffer(memoryview(array), dtype=torch.float32)`精确得到正确shape/dtype。由此根因是服务器Torch 2.1与NumPy 2.x桥接不兼容，不是cache损坏、标签漂移、训练发散或方法失败。

repair2不修改共享Conda环境，避免影响其他项目；仅在formal trainer的数据桥接层新增`_float32_numpy_to_tensor`：先将IQ保证为连续float32并调用标准`from_numpy`，仅当异常精确匹配`expected np.ndarray`且对象仍为原生`np.ndarray`时，回退到同一buffer的`frombuffer`视图。该修复不复制整份cache、不改样本值，不增加模型参数、训练epoch、损失、View或backbone forward。新增回归测试模拟NumPy2/Torch2.1异常并验证shape、dtype、数值和标签；本地trainer`py_compile`通过，聚焦文件10项通过，10组正式相关回归升级为`80 passed`。candidate lock会在运行时重新绑定repair2 trainer SHA，不需要改变cache spec、K/seed或target cell身份。

repair2远端覆盖前快照为`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_repair2_before_sync_20260715_144746`，封存旧trainer、旧report、repair1 runner/train完整日志和state。Git提交为`123b69b3dbbf`；trainer和report本地/远端SHA256一致，分别为`a6fbabb58aabca185c343c43a75d883f594757926f6449547c31bf152e7fe1d9`和`fde478f9f83ea95eb531ba143cbd6a1a18f1ac5847b802a4b21eb552d902232a`，远端`py_compile`通过。随后使用正式source cache进行严格loader smoke：18,000条三场景数据加载通过，单样本shape=`(2,256)`、dtype=`float32`、buffer转换与原数组sum差`4.77e-7`，DataLoader batch shape=`(4,2,256)`，且`clean_sample_access=False`。这证明repair2在实际N607环境和实际密封cache上修复了首batch阻断点；尚未证明12epoch训练、promotion或性能目标通过。

## 2026-07-15 repair2恢复失败与LoRA device repair3

repair2 source pipeline以GPU0、PID`1549892`实际landed并越过严格cache loader和prototype构建，但在首次student backbone前向、任何epoch日志或optimizer step之前失败关闭。runner完整日志8行、训练完整日志32行已逐行读取；唯一错误为LoRA的`lora_a`权重在CPU、输入与冻结base在`cuda:0`，触发`Expected all tensors to be on the same device`。代码追踪确认ground trainer先获得已在CUDA的ADV3B02，再调用`inject_feat_joint_lora`；旧`LoRALinear`构造器新建`nn.Linear`时未继承被替换base的device/dtype。support-only历史路径之所以未暴露，是其注入后还会统一执行`model.to(device)`。

repair3在`LoRALinear`构造时让`lora_a/lora_b`显式继承`base.weight.device`和`base.weight.dtype`，保持冻结base、rank、alpha、参数量、初始化和合并语义不变。新增float64基座回归测试，若低秩层仍默认float32则前向必然失败；当前该测试确认device/dtype继承且identity初始化逐元素精确。support LoRA测试20项通过，包含formal链路的11组完整相关回归共`100 passed`，脚本`py_compile`通过。该修复不增加参数、MAC、状态、epoch、View或数据访问；candidate lock会在运行时绑定新的LoRA实现SHA。

repair3远端覆盖前快照为`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_repair3_before_sync_20260715_145448`，封存旧LoRA实现、旧report、repair2 runner/train完整日志和state。Git提交为`dfd0541c4c48`；LoRA脚本与report本地/远端SHA256一致，分别为`7877bcfd243dce0d6739a2b574e0db6fd401c8bf5ae86253bc52a4f17c0fc07b`和`4e2e9ade2e64f140fb8f0bef7fd27d18b5398d18d968bddcfc7c1748309d6565`，远端`py_compile`通过。GPU0微型CUDA探针确认base、`lora_a`、`lora_b`均位于`cuda:0`、dtype一致，identity初始化前向`max_abs_diff=0.0`。这修复了已知device阻断点，但仍不把smoke视为12epoch训练或promotion成功。

14:56 CST再次确认8张GPU均无既有任务后，repair3 source pipeline以runner PID`1553723`恢复，训练子进程PID`1553733`已实际占用GPU0约580MiB。启动后13s runner与训练子进程均存活，表明已越过selector、NumPy bridge和LoRA device三个已知阻断点；当时尚无epoch行，状态只可记为running，不能记为训练完成或source PASS。训练活动期间保持monitor-only，不再覆盖远端代码、配置或产物。

## 2026-07-15正式12epoch训练完成与validation bridge repair4

repair3训练步骤在59.87s内完成12/12epoch并返回0，生成94,054B的`effective8_adapter_fp16.pt`，SHA256=`9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b`；`training_manifest.json`为13,603B，SHA256=`15bafc5dc2ac369bb230461bc2fef23e3232d3df767e3dbd22f7da33eb77d8bc`。完整训练日志481行已逐行解析，无NaN/Inf、OOM、Killed或缺失epoch。关键曲线如下：

|指标|epoch1|epoch12|相对变化|全程解释|
|---|---:|---:|---:|---|
|total loss|1.20458|1.10122|-8.58%|非单调；最低epoch8为1.08844|
|prototype CE|1.24269|1.15447|-7.10%|整体下降|
|worst-K risk|1.39363|1.21614|-12.74%|最低epoch8为1.17478，后期有波动|
|nested K1|1.49032|1.28784|-13.59%|极少shot目标明显改善，但尚非准确率|
|nested K5/K10/K20|1.12039/1.13958/1.13071|1.02775/1.03624/1.04733|-8.27%/-9.07%/-7.38%|三档均下降|
|View consistency|0.06453|0.06090|-5.62%|缓慢下降|
|feature margin|0.03299|0.02752|-16.60%|下降|

日志中历史字段`clean`/`clean_margin`是沿用旧键名的same-LEO reference损失记录，不代表读取clean样本；同一manifest明确`clean_sample_access=false`、`clean_derived_signal_access=false`、`target_receiver_data_used_for_training=false`。可确认训练数值稳定、worst-K surrogate总体改善；不能从这些loss直接推出old/new准确率达标。

训练完成后source validation仅运行2.89s便在首个prototype batch失败，完整validation日志10行的唯一错误仍是NumPy2.2.5/Torch2.1.0的`torch.from_numpy`不兼容，尚未计算任何source准确率或promotion gate。repair4新增共享`cvsrffi.tensors.numpy_to_tensor_compat`，validator的float32 IQ、int64 label和拼接feature三类入口全部改用buffer bridge；正式benchmark原有三个NumPy入口已使用兼容helper，micro helper现委托同一实现。静态测试要求validator源码不再含`torch.from_numpy`。candidate lock新增绑定共享tensor bridge、micro helper和LoRA实现，补齐此前间接依赖未进入代码哈希的问题。已产出adapter的trainer与LoRA实现保持不变，runner恢复时会跳过complete的`0002_train`，只重跑validation。4个相关文件`py_compile`通过，聚焦23项通过，12组完整正式回归为`107 passed`。

repair4远端覆盖前快照为`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_repair4_before_sync_20260715_150657`，封存旧tensor/micro/validator/candidate-lock代码、旧report、首次validation完整日志和state。Git提交为`c6a1fb8d9864`；5个同步文件本地/远端SHA256逐项一致，远端4个Python文件`py_compile`通过。N607 GPU0实际探针对float32 IQ与int64 label分别完成buffer bridge和CPU→CUDA移动，shape、dtype、值均一致。此次未覆盖trainer、LoRA、adapter或training manifest，已完成训练证据保持原样。恢复前仍需live inventory；恢复只能从`0003_source_validation`继续。

## 2026-07-15 source validation PASS与candidate-lock repair5

repair4恢复后source validation完整执行并返回0，日志523行已逐行解析，`source_validation_pass=true`、`failed_gates=[]`，12项source门禁全部通过。主要同row结果如下；它们来自source receiver`2-19`holdout，只用于锁定head/TTA与安全promotion，不是5个target receiver的正式性能：

|候选/推理|accuracy|min class|平均backbone forward|结论|
|---|---:|---:|---:|---|
|base checkpoint fixed1|86.678%|70.498%|1|source参考|
|ground LoRA fixed1|86.794%|69.349%|1|总体较base+0.115pp，最低类下降|
|ground LoRA fixed5|86.505%|68.199%|5|source上不优于fixed1|
|ground LoRA adaptive|87.341%|71.169%|1.124|93.815% query停在1-view，6.156%到3-view，0.029%到5-view；P95=3|
|locked head K1 adaptive|87.082%|71.264%|1.112|K1 source压力门禁通过|
|locked head K5 adaptive|87.370%|71.648%|1.125|source-only锁定|
|locked head K10 adaptive|87.370%|71.264%|1.125|source-only锁定|
|locked head K20 adaptive|87.543%|70.498%|1.136|source-only锁定|

source holdout明显低于正式target-old 92%/最低类88%门槛，但source promotion gate设计为相对稳定性、无明显退化和资源/权限门禁，不是target绝对性能替代品；因此source PASS只允许生成candidate lock，不能声明目标达成。candidate lock随后8行日志失败关闭，根因不是性能：`source_validation.json`已经显式包含权威字段`clean_sample_access=false`、`clean_derived_signal_access=false`和密封cache审计，但旧lock还强制要求已被新字段取代的冗余`clean_samples_used_for_validation`，而validator未写该旧字段。

repair5让未来validator同时显式写`clean_samples_used_for_validation=false`；candidate lock不再依赖该旧冗余字段，仍硬性要求两个权威clean不可达字段、密封cache SHA/audit、全部validation gates、权限、receiver holdout、无角色Oracle、无类别quota和source-only head/nested-K锁，因此没有放宽`LEO_weak-only`协议。现有已签名validation无需篡改或覆盖即可由权威字段进入lock。聚焦validator/candidate-lock测试17项通过，12组完整正式回归保持`107 passed`。

最新用户目标把target-old门槛从95%调整为92%，并把Stage2-B纳入正式目标。按AGENTS规则先更新根`项目.md`10.3.1，再更新Git协议镜像`docs/cvs_stage2c_extreme_light_goal_20260714.md`，Git提交`f4dc1aa1fbdc`；根目录仍非Git，故该镜像是版本承载面。正式config、plan validator和summarizer现统一锁定`K10_OLD_TARGET=0.92`，其余最低类、新类、K5、K1和遗忘门槛不变。当前v14 plan必须在candidate lock恢复前重新生成并同步，旧0.95 plan不得继续用于正式汇总。

repair5代码/门槛提交为`92513ba312eb`。远端覆盖前快照为`/home/szu2070436088/2510044040/CV-SincNet/code/snapshots/qknn_ground_effective8_r16_e12_leoonly_20260715_v14_repair5_before_sync_20260715_152451`，包含旧`项目.md`、Git协议镜像、config、validator/lock/plan/summarizer、report、旧protocol plan和candidate-lock失败证据。9个同步文件本地/远端SHA256逐项一致，远端4个脚本`py_compile`通过。repair5 plan仍精确覆盖2个source cache、25个target cache、300次benchmark和900条场景row；两份source规范化spec SHA与现有cache完全相等。runner state核对显示步骤0–3均为complete且命令逐项相同，步骤4为failed且命令相同，因此恢复只会重跑candidate lock。远端config与summarizer均读出K10 old target=`0.92`；target matrix仍未启动。

## 2026-07-15 candidate source-cache provenance失败关闭与repair6

repair5 candidate lock随后实际生成，但锁内`source_leo_weak_cache_sets.source_train.path`与`source_validation.path`都指向`phase1_caches/source_validation/cache_set.json`。训练manifest本身仍正确绑定`phase1_caches/source_train/cache_set.json`；错误发生在validator构造promotion时：先复制training manifest，再用source-validation缓存的同名`source_leo_weak_cache_set_*`字段覆盖了source-train字段。旧候选锁仅逐项校验文件SHA，没有校验两个缓存必须不同或`cache_scope`必须分别为`source_train`/`source_validation`，因此错误锁通过。该锁统一标记为`INVALID_SOURCE_CACHE_PROVENANCE`，保留原文件和SHA作为失败证据，不得进入target matrix、方法排名或成功声明。已完成的12epoch adapter训练及两个原始cache未被覆盖，且target matrix从未启动。

repair6把candidate lock升级为schema`cvs_stage2c_source_candidate_lock_v2`并强制显式输入原始`training_manifest.json`。锁生成器要求promotion、source validation和实际training manifest三方训练清单SHA一致；promotion必须分别记录`source_train_leo_weak_*`与`source_validation_leo_weak_*`；锁分别从training与validation artifact读取缓存，重新校验文件SHA、`cache_scope`，并拒绝同路径或同SHA。锁内新增绑定training manifest路径/SHA，运行时verifier会重新打开并校验。validator不再用validation字段覆盖training通用字段。为避免覆盖无效证据，repair6改用新目录`source_validation_v2/`和新锁`candidate_lock_v2.json`，全部300个benchmark只引用v2锁；v1 schema会被当前verifier拒绝。

本地`ssr-gpu`验证结果：6个修改脚本/测试文件`py_compile`通过；candidate/plan/validator聚焦测试20项通过；13个正式相关测试文件共110项通过；`git diff --check`待提交前复核。repair6 plan仍精确生成2个source cache-set、25个target cache-set、300次benchmark、900条场景row、1次collection和1次summary。与repair5逐项比较：2个source cache命令完全相同，12epoch训练命令完全相同，两份source cache spec字节相同，300份Stage2 config字节相同；source validation唯一变化是输出目录切换为`source_validation_v2`，candidate命令新增`--training_manifest`并输出`candidate_lock_v2.json`。因此恢复时应跳过已完成cache与训练，只重新执行source validation和candidate lock；正确v2锁生成前matrix继续阻断。

Stage2-B只读边界审计同时确认：当前v14 Stage2-C cache的scope为`stage2_registered`且包含target-old与target-new，不能原样作为Stage2-B的`stage2_target_old`输入；Stage2-C的`old_acc_before_increment`也不能冒充Stage2-B正式结果。后续Stage2-B需要从同一密封LEO_weak缓存生成绑定父SHA的target-old只读projection，派生专属candidate lock，并运行5receiver×5seed×3场景×K=1/5/10/20共300条old-only场景row。该矩阵可复用同一checkpoint、source-only adapter、source statistics、head/TTA超参和物理old样本，但必须独立报告old-only head状态、逐样本before/after/direct/identity预测和资源门禁；Stage2-B通过不能替代Stage2-C的新类与H门槛。
