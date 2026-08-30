# WISER-RF历史D92 E0域适应问题完整解决报告

## 报告状态

本报告冻结截至2026-08-31 03:05 CST已经取得的代码、协议、N607运行和独立评分证据。技术问题已经解决：support/query包根路由正确，适配后单模态零范数不再导致P3异常退出，prediction可以在不读取truth的条件下完整落盘。科学问题尚未解决：已评分的10个正式A/B候选均未通过预注册表示门槛，完整Target125和WISER注册适应阶段没有获得启动授权。

这一区分决定了当前结论的边界：

> WISER-RF已经从“无法形成合法结果”的技术失败状态推进到“可以形成合法、truth-last、可比较的负结果”；尚不能写成性能提升方法，也不能写成完整两阶段方法已经验证成功。

## 一、任务与设计边界

### 1.1 两阶段目标必须分开

三份设计报告先后提出BiNOVA-D92、Q-BiNOVA-D92、BiSAGE-D92和WISER-RF。方法名称发生变化，但两阶段边界保持一致：

- 阶段A只使用旧类目标域support，学习接收机和星地信道导致的公共域偏移；
- 阶段B在新类support实际到达后，默认冻结阶段A域状态，再处理旧类—新类竞争、遗忘和类别尾部风险；
- query始终只用于冻结模型的逐样本推理，不能更新任何模型、原型、统计、阈值或候选选择状态。

当前WISER-RF实验属于阶段A。它只验证旧类目标域表示能否穿过冻结源头、源原型和重新拟合的old-only D92三个probe。由于所有正式候选的阶段A门槛均未通过，阶段B没有启动。这个停止是预注册科学门槛的执行结果，不是实现遗漏后仍宣称完成。

### 1.2 为什么停止“末端适配器+临时分类头”

历史实验已经证明，末端适配器和临时SF分类头可以改善REG0指标，却容易在D92重新估计均值、任务均衡协方差和LDA判别头后失去作用。真正需要改变的是D92消费的身份表征，而不是只在最终分类头上制造短暂收益。

WISER-RF因此采用渐进式identity backbone更新：

1. Stage 1开放`t3/f3`、投影、融合和分类头前的身份映射；
2. Stage 2继续开放`t2/f2`；
3. Stage 3继续开放`t1/f1`、time/frequency融合和频率统计映射；
4. 源分类器最终线性头、域分支和Sinc前端保持冻结；
5. 每个arm完成support适配后重新冻结全部参数，才允许打开无标签query。

这不是设计报告的严格全量复刻。当前版本尚未开放Sinc前端，也没有加入ASAM、SWA、多折Delta聚合或显式PCGrad。它是机制优先的阶段A最小实现，用于回答“更新完整identity卷积主干是否能产生D92可见收益”。

## 二、协议与源摘要

### 2.1 Phase2允许输入

当前实验只读取以下四类输入：

1. 匹配`p2_min_v1`、`VALIDATED_ONCE`、`capsule_id`和`split_id`的固定目标域received IQ；
2. 当前row合法旧类K10 support标签；
3. 与ADV3B02 checkpoint联合冻结的类原型和量化聚合Phase1摘要；
4. 冻结checkpoint及预登记配置。

support来自manifest的`before_enrollment`包根，query来自`before_apply`包根。两者物理ID互斥。训练审计固定记录`query_rows_used=0`；pilot先完成全部prediction，独立scorer随后按opaque query ID连接truth。

### 2.2 源摘要的实际数据量

实际Phase1摘要不是source replay，也不是样本级embedding。远端只读检查得到：

|项目|实际值|
|---|---:|
|摘要文件|`int8_domain_class_prototypes.npz`|
|摘要大小|5,363字节，约5.24 KiB|
|绑定文件大小|640字节|
|联合大小|6,003字节，约5.86 KiB|
|域数|26|
|旧类数|6|
|特征维度|160|
|主张量|`domain_class_q:int8[26,6,160]`|
|尺度|`domain_class_scale:float16[26,6]`|
|有效槽|`domain_class_mask:uint8[26,6]`|

摘要还保存域注册表、类注册表和`ADV3B02:z_id:unit_l2:160:v1`特征schema。绑定文件固定checkpoint SHA256、摘要SHA256、6个类ID和授权标识。

`项目.md`第5.3.2节已经允许int8域×类中心、int8低秩域残差方向、int8域系数、int8类半径及反量化FP16尺度；确有需要时还可加入不含BatchNorm运行状态的FP16全局逐特征location/scale。当前真实artifact只包含稠密域×类中心、尺度和mask，没有source协方差、样本级源特征、BatchNorm运行状态或global location/scale。协议能力与当前实际载荷不能混写。

## 三、WISER-RF实现

### 3.1 A/B/C的准确含义

所有非B0 arm都包含基础A目标。它们不是三套互斥网络。

|arm|训练组成|目的|
|---|---|---|
|`B0`|不做目标域适配|同row基线|
|`A`|冻结源头CE+目标LOO原型CE+L2-SP|验证全identity主干监督适配|
|`B`|A+类条件VSW|引入不可变源分布参照|
|`C`|A+冻结源头IQ反演CE|恢复源假设约束，不读取source样本|
|`ABC`|A+VSW+IQ反演|联合诊断|

默认权重为`lambda_proto=0.5`、`lambda_sp=1.0`、`lambda_vsw=0.5`和`lambda_inversion=0.25`。三阶段步数为1500/2500/4000，学习率从投影层`3e-4`递减到早期层`1e-5`。

### 3.2 双假设监督

基础A没有训练新的目标分类头。梯度来自两个固定语义：

- 冻结源分类头CE保持Phase1发射机身份假设；
- leave-one-out目标原型CE要求每个support样本由不含自身的同类原型正确识别。

L2-SP按可训练参数数量归一化，约束参数不要无界偏离Phase1状态。消融实验分别测试了`lambda_sp=0/0.1/2.0`和`lambda_proto=0`，用于判断哪种约束导致过拟合或把表示压回近恒等映射。

### 3.3 VSW与“协议冲突”

最初的协议冲突是Phase2禁止读取source样本、source cache和源BatchNorm状态，而VSW需要source→target参照。解决方案不是放宽query或source replay，而是只允许与checkpoint联合冻结、不可训练、不可追加的int8聚合摘要。WISER-RF从该摘要确定性构造类别条件虚拟源点，执行classwise sliced-Wasserstein对齐。

协议冲突已经解决，科学冲突仍然存在。VSW梯度可能把目标特征拉向源域公共结构，同时删除对当前接收机下发射机区分有用的细节。当前实现用固定加权和联合优化，没有显式投影冲突梯度。实验中弱VSW和强VSW都明显改善P1/P2，却没有稳定改善P3，正是这一科学冲突的直接证据。

### 3.4 三个表示probe

每个query只经过冻结模型前向，产生三类预测：

- P1`SOURCE_HEAD`：冻结源分类头；
- P2`SOURCE_PROTOTYPE`：冻结Phase1类中心；
- P3`OLD_D92`：仅用当前合法旧类support重新拟合D92。

P1/P2回答目标域表示是否与Phase1身份语义一致，P3回答收益能否穿过正式D92几何。晋级不接受只提高P1/P2而损害P3或类别floor。

## 四、两次技术故障及修复

### 4.1 v1：support/query包根错误

v1把support和query都路由到`before_enrollment`。历史manifest实际把support放在`before_enrollment`、query放在`before_apply`，因此8条run均无法形成合法prediction。

修复提交`b5fb479032c371d0df016c19e67b25aa3c94d600`将`_support_path`固定到`before_enrollment`，将`_query_path`固定到`before_apply`。新增测试先在旧代码上准确失败，修复后相关测试通过。v1 artifact保留，未评分、未覆盖、未重启。

### 4.2 v2：P3单模态零范数

query路径修复后，5条v2 run在P3出现：

```text
OldOnlyERBTError: feature row is degenerate
```

调用栈显示异常发生在identity块的逐行归一化。适配后的个别`z_id`为零，但同一物理样本的FFT96仍有效。可微D92使用PyTorch`F.normalize`，本来会把零identity块保持为零；独立精确D92却拒绝任一零模态，二者语义不一致。

修复提交`4e51e29b393cba723c2e79ed0d3314ed64d6369f`增加零安全单模态归一化：

- identity为零、FFT有效：identity块置零，继续由FFT块形成合法联合特征；
- FFT为零、identity有效：对称处理；
- identity和FFT同时退化：联合归一化仍抛出同一异常，禁止无信息prediction；
- NaN、Inf、错误维度和空矩阵仍被拒绝。

回归测试覆盖“零identity+有效FFT可评分”和“双模态均零必须拒绝”。独立P0/P1审查结论为`READY`。

## 五、验证与发布证据

### 5.1 本地验证

完整WISER相关测试集合包含11个测试文件，覆盖源摘要、VSW、模型反演、渐进解冻、P1/P2/P3、pilot、truth-last scorer、old-only D92、registered D92和四状态路径。实际命令：

```text
conda.exe run -n ssr-gpu python -m pytest
  tests/test_wiser_source_summary.py
  tests/test_wiser_model_inversion.py
  tests/test_stage2_wiser_scoring.py
  tests/test_stage2_wiser_runner.py
  tests/test_stage2_wiser_rf.py
  tests/test_stage2_wiser_pilot.py
  tests/test_stage2_sf_erbt_oldonly.py
  tests/test_stage2_sf_erbt_four_state.py
  tests/test_run_stage2_wiser_pilot.py
  tests/test_run_stage2_binova_d92.py
  tests/test_stage2_binova_d92.py -q
```

结果为59/59通过。

### 5.2 v3 release

|项目|证据|
|---|---|
|代码提交|`4e51e29b393cba723c2e79ed0d3314ed64d6369f`|
|release|`wiser_rf_zeromodal_suite_20260831_v3_4e51e29b.tar.gz`|
|本地/远端SHA256|`9ac739632a48d91600b41ca1eb005c7e16b8a12a6f327695001dcefffb267521`|
|远端编译|`PASS`|
|真实ADV3B02无query smoke|`PASS`|
|smoke query状态|`query_opened=false`|

release只进行了一次本地/远端归档SHA比对，没有增加成员哈希、签名或额外发布门。

## 六、实验结果

### 6.1 运行阶段演化

|版本|目的|最高已证状态|结果|
|---|---|---|---|
|v1|首轮8卡A/B/C因果矩阵|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|support/query包根错误，无性能结果|
|v2|queryfix重发|2条`ANALYZED`，6条技术失败|首次取得去原型和弱VSW合法负结果；其余run暴露单模态零范数问题|
|v3|zero-modal修复重发|6条全部`ANALYZED`|技术闭环恢复，全部正式候选未晋级|

截至本报告冻结时刻：

- GPU0 v2主ABC PID`2439930`已自然结束并复现单模态零范数异常；仅有2组partial prediction，没有`pilot_result.json`，未评分；
- GPU2 v3主ABC PID`2463277`已完成15组prediction和独立评分；
- v3短训练、无L2-SP、弱L2-SP、强L2-SP和强VSW均已独立评分；
- 去原型和弱VSW沿用已闭合v2结果，不因v3技术修复重复运行。

### 6.2 全部已评分正式A/B候选

下表每个单元格按`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`排列，变化量均为候选减同run B0。

|版本与候选|P1变化|P2变化|P3变化|P3 floor中位变化|几何比中位变化|判定|
|---|---|---|---|---:|---:|---|
|v2 A，`lambda_proto=0`|`+5.00/-0.83/+6.67pp`|`+6.67/+0.83/+5.00pp`|`+0.83/-3.33/-0.83pp`|`-5.00pp`|`+0.3423`|未通过|
|v2 B，`lambda_vsw=0.1`|`+7.50/-0.83/+6.67pp`|`+8.33/+0.83/+6.67pp`|`+0.83/-3.33/+0.83pp`|`-5.00pp`|`+0.7676`|未通过|
|v3短训练A|`+10.00/-0.83/+2.50pp`|`+10.00/+1.67/+2.50pp`|`0.00/-3.33/+0.83pp`|`-5.00pp`|`+0.5321`|未通过|
|v3短训练B|`+7.50/-5.00/+0.83pp`|`+8.33/-1.67/0.00pp`|`+1.67/-0.83/+0.83pp`|`0.00pp`|`+0.3698`|未通过|
|v3 A，`lambda_sp=0`|`+5.00/-5.00/+7.50pp`|`+3.33/-4.17/+6.67pp`|`+1.67/-8.33/+0.83pp`|`0.00pp`|`+0.9717`|未通过|
|v3 A，`lambda_sp=0.1`|`+8.33/-0.83/+1.67pp`|`+7.50/-2.50/+2.50pp`|`+3.33/-4.17/-0.83pp`|`-10.00pp`|`+0.3539`|未通过|
|v3 A，`lambda_sp=2.0`|`+8.33/-2.50/+5.83pp`|`+7.50/-1.67/+5.00pp`|`+1.67/-7.50/-0.83pp`|`-10.00pp`|`+0.9177`|未通过|
|v3 B，`lambda_vsw=1.0`|`+12.50/-3.33/+5.83pp`|`+13.33/-1.67/+3.33pp`|`+0.83/-2.50/-2.50pp`|`-5.00pp`|`+0.6260`|未通过|
|v3主pilot A|`+10.00/0.00/+6.67pp`|`+12.50/+4.17/+7.50pp`|`+1.67/-5.00/+0.83pp`|`-5.00pp`|`+0.6134`|未通过|
|v3主pilot B|`+14.17/-3.33/+5.00pp`|`+15.00/-0.83/+4.17pp`|`+3.33/-2.50/-0.83pp`|`0.00pp`|`+0.4331`|未通过|

所有run都先完成prediction并确认`truth_opened=false`，再由各自独立scorer连接truth；没有跨run使用truth调参或选择性重跑。

## 七、机制解释

### 7.1 已经解决的机制问题

零模态修复证明此前5条技术失败不是“D92不允许identity退化”的科学结论，而是精确D92与可微D92归一化语义不一致。v3在相同数据、seed和场景上完整产生prediction，排除了这层实现噪声。

源摘要方案也通过了协议和工程验证。6,003字节联合载荷足以为VSW提供26个source域、6个旧类、160维身份空间的固定分布锚点，不需要source IQ、样本级embedding或BatchNorm运行状态。

### 7.2 尚未解决的性能问题

10个正式候选呈现同一结构：P1/P2和类间/类内几何比经常提高，P3却在low-elev场景显著回退，类别floor也经常下降。最强的例子是`lambda_sp=0`：几何比中位提升`+0.9717`，但low-elev P3下降`8.33pp`。主pilot的A使三场景P2分别提高`12.50pp`、`4.17pp`和`7.50pp`，low-elev P3仍下降`5.00pp`。

这说明当前目标函数主要改善了冻结源语义和全局类间结构，却没有稳定保护D92依赖的局部类条件协方差与类别尾部。继续扩大step、seed或完整Target125不能解决这一机制缺口。

### 7.3 下一步应改变什么

下一候选不应继续只扫`lambda_sp`或`lambda_vsw`。更有价值的机制改动是：

1. 将P3旧D92的support cross-fit风险直接引入阶段A目标，而不是只在训练后probe；
2. 对P3类别floor和low-elev最差场景设置显式旧类风险约束；
3. 在A与VSW/C梯度之间加入冲突检测和投影，只保留不损害P3 support风险的更新分量；
4. 若仍使用源摘要，优先验证低秩类条件残差和类半径，不直接加入源BatchNorm状态；
5. 只有新的阶段A候选通过三场景正式门槛，才启动WISER专用阶段B和完整Target125。

## 八、设计—实现追踪表

|ID|来源要求|落点|状态|验证或原因|
|---|---|---|---|---|
|R01|阶段A与阶段B参数和目标分离|runner只执行old-only阶段A；自动门槛控制后续阶段|verified|当前Stage B未被误启动|
|R02|阶段A只用旧类target support|`run_stage2_wiser_pilot.py`|`verified`|`query_rows_used=0`、K10旧类registry|
|R03|query只读、prediction先于truth|pilot/scorer分进程|`verified`|所有已评分run记录`truth_join_after_prediction_only=true`|
|R04|更新完整identity主干而非末端adapter|`stage2_wiser_rf.py`|`verified`|t1–t3、f1–f3、投影和融合分阶段可训练；Sinc保持冻结|
|R05|不训练目标分类头|冻结源head+LOO原型|`verified`|源最终head保持冻结，无临时目标head|
|R06|冻结源假设与目标LOO双监督|`wiser_dual_supervision_loss`|`verified`|单元测试和真实smoke通过|
|R07|L2-SP保护Phase1状态|`normalized_l2sp_penalty`|`verified`|参数数归一化，0/0.1/2.0完成消融|
|R08|小型可量化源分布摘要|`wiser_source_summary.py`与`项目.md`5.3.2|`verified`|实际5,363字节int8域×类摘要|
|R09|源协方差或global normalization统计|协议允许低秩残差/半径及可选global location/scale|`deferred`|当前真实artifact未携带协方差或global location/scale|
|R10|类条件VSW|`classwise_sliced_wasserstein`|`verified`|弱/默认/强权重路径可达，性能未晋级|
|R11|C与ABC模型反演约束|`wiser_model_inversion.py`|`verified`|本地测试、真实ABC smoke和15单元主pilot评分完成|
|R12|适配后重新冻结|`train_wiser_arm`finally路径|`verified`|query前检查model.eval且无可训练参数|
|R13|P1/P2/P3表示probe和独立评分|WISER pilot/scorer|`verified`|6条v3和2条v2因果run已ANALYZED|
|R14|单模态零范数安全处理|`stage2_sf_erbt_oldonly.py`|`verified`|回归测试、P0/P1审查和v3真实prediction闭环|
|R15|A/B/C梯度冲突控制|尚无PCGrad/约束投影|`deferred`|当前固定加权和已暴露P3冲突，需新候选实现|
|R16|WISER专用阶段B注册残差|`phi_R`自动接续|`deferred`|基础registered D92存在，但WISER Stage A门槛未通过|
|R17|完整历史Target125|`125 outer/375 scene`|`blocked`|所有正式A/B候选`next_experiment_authorized=false`|
|R18|当前A/B候选性能晋级|预注册表示门槛|`rejected`|10/10候选未通过|
|R19|C/ABC完整主pilot结果|GPU2 v3|`verified`|15组prediction和独立评分完成；C/ABC诊断P3未改善|

追踪统计：`verified=14`、`implemented=0`、`deferred=3`、`rejected=1`、`blocked=1`、`pending=0`。

## 九、最终判断

当前交付是WISER-RF阶段A最小机制实现及其完整技术闭环，不是三份设计报告的严格全量parity。代码、协议、release、真实smoke、prediction、truth-last评分和Git发布链已经闭合；源摘要数据量已量化，协议权限与实际载荷已经分清。

性能结论同样明确：全identity主干适配可以提高冻结源头和源原型probe，却没有稳定提高重新拟合的old-only D92，也没有保护类别floor。Stage B和完整Target125因此不应启动。

最高风险剩余项不是工程稳定性，而是目标函数仍缺少P3原生风险和A/VSW/C梯度冲突控制。下一次方法迭代应先修复这一机制，再用同一历史outer、seed和三LEO场景做最小可证伪验证。

## 十、关键路径

- 实现：`code/cvsrffi/stage2_wiser_rf.py`
- 训练与probe：`code/cvsrffi/stage2_wiser_runner.py`
- 量化源摘要：`code/cvsrffi/wiser_source_summary.py`
- 模型反演：`code/cvsrffi/wiser_model_inversion.py`
- 精确D92零模态修复：`code/cvsrffi/stage2_sf_erbt_oldonly.py`
- launcher/scorer：`code/scripts/run_stage2_wiser_pilot.py`
- source binding：`configs/wiser_rf_adv3b02_source_binding.json`
- 原始实验总表：`docs/experiments/wiser_rf_cause_suite_20260831_v1_report.md`
