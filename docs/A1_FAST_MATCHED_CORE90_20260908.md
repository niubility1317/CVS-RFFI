# A1-Fast匹配CORE90重训实验

更新：用户随后指定使用已完成独立评分的ADV3B02 E200权重，seed392005。本r3已定向停止并保留产物，状态SUPERSEDED_BY_USER；当前续训见[A1指定权重续训](A1_FAST_SELECTED_ADV3B02_20260908.md)。下文RUNNING均为此前历史快照。

状态：LOCAL_VERIFIED，尚未启动。运行ID：`a1_fast_matched_core90_s392005_20260908_r2`。唯一launch owner：本任务主Agent。Git分支：`codex/a1-fast-original-20260908`。

## 用户范围和源码

用户要求按报告实现并发布，随后明确不使用已有checkpoint；使用CORE90必须在相同实验配置下重训。因此本轮先从零训练CORE90 E200，两个A1仅加载本轮新产物。未采用临时拟议的无锚点方案。有效配置没有历史权重路径。

原A1实际release已定位，1338个源码/脚本/JSON文件全部匹配`7504f6669fbc0a02b9b7446f463f561ecbcef6de`，解决原报告的源码身份缺口；不沿用后续72360源码。

## 冻结矩阵

|阶段|设置|GPU|预算|依赖|
|---|---|---|---|---|
|CORE90_MATCHED_FRESH|CORE90 base目标，从零，MUSE/RC4/DAOT关闭|3|E200|无checkpoint|
|A1_REFERENCE|原A1执行路径，RC4锚点保留|4|E200|本轮CORE90 final|
|A1_FAST_SEQUENTIAL|尺度及日志批量读回、逐视图身份教师、mean无用中间量跳过|5|E200|同一新CORE90|
|A1_FAST_BATCHED|GPU FP32未通过严格数值检查，暂缓|—|不启动|独立后续数值实验|

共同条件：seed392005；ManySig equalized；source RX=`1,3,4,6,8`、day=`1,2,3`；target RX=`0,2,5,7,9,10,11`、day=`0,1,2,3`；`L/U/V=0.07/0.63/0.30`，预期6300/56700/27000，训练标注比例0.1；L batch128，A1 U batch256；M规格lite_c双分支，6个source TX类，E200 final-only。CORE90共享数据/模型/seed/预算，但其base算法与后续A1的附加机制不同，不能写成两阶段算法相同。CORE90的历史target-based选择规则不沿用。

主比较固定FAST_SEQUENTIAL−REFERENCE；BATCHED因GPU数值检查未通过而暂缓，不根据target选择加速版。历史85.346%等仅为背景。单seed描述性验证，不晋级默认；没有默认允许掉点幅度。

## 实现与验证

保持学生前向、增强seed、原有效loss、L→U尺度顺序、Python float尺度、梯度控制、EMA更新和完整预算。inactive仍返回原loss/旧尺度。日志按device/dtype成组读回，保留原累加顺序。身份接口保留adapter和checkpoint结构。

nuisance零权重图仍有零梯度和日志消费者，未删除，避免改变AdamW。原版没有clean-anchor等后续组件。基础和tail验证的domain参数/卫星输入不同，未直接复用。搬运原已有pin_memory/non_blocking，不算新收益。不共享E20前缀。

确认原EMA/CosFace缓存缺陷：.data更新后，自动缓存与强制刷新输出不同。两组保留该旧行为，修复归A1-Stable，不能声称问题已解决。新增教师正确率/置信度、实际identity-only与prototype可用字段；保留原RC4梯度和optimizer-step证据。

本地验证：`conda run -n ssr-gpu python -m pytest tests/test_a1_fast.py tests/test_adv3b02_daot_stn.py -q --tb=short`；新增脚本py_compile；`check_a1_fast_execution.py --device cpu`验证真实M模型E1/E21/E161的DAOT L→U loss、梯度、AdamW、EMA、BN和RNG，PASS，教师输出CPU差值0。GPU/AMP在发布第一步和新CORE90完成后核验。CPU教师微基准不是GPU端到端加速，十项全程profile尚未闭合，实际耗时待E200。

独立P0/P1审查发现CORE90会被不适用的“后续全机制启用”门槛阻断，已仅对scratch-only依赖阶段取消该科学门槛；保留技术失败、E200完整性和checkpoint检查。这不代表CORE90科学晋级。

## 发布和产物

项目：`/home/szu2070436088/2510044040/CV-SincNet`。Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。release提交在发布后记录。

入口：`python code/scripts/run_a1_fast_matched_core90.py --project-root /home/szu2070436088/2510044040/CV-SincNet --run-id a1_fast_matched_core90_s392005_20260908_r2`。

checkpoint/指标在`runs/<run-id>/`，日志/诊断在`logs/<run-id>/`。已存在根目录即拒绝，不覆盖；每GPU最多两个compute进程，容量不足等待，不干预其他任务。CORE90非零退出、非E200、输入错误、输出冲突、smoke技术失败停止依赖链并保留产物。A1单行失败不影响其他健康行，不自动重试；低性能不停止。

预期：新CORE90 E200；两组A1 E200；全epoch CSV/JSONL、资源汇总、clean和三个LEO指标/日志。pipeline最终仅标`AWAITING_ARTIFACT_ANALYSIS`，完整性、实际激活、配对精度和资源解释需全量日志分析，不以命令结束冒充ANALYZED。

追踪：[a1_fast_final_traceability.md](../analysis/a1_fast_final_traceability.md)。

## GPU数值核验与r2发布

r1在训练前smoke失败，CORE90没有启动，产物保留。定点诊断确认逐视图identity-only在GPU FP32和AMP下z_id/logits差值均为0；仅batch合并在FP32有z_id约9.62e-5、logits约1.55e-4差值，超出原容差。未放宽门槛：r2只运行REFERENCE与FAST_SEQUENTIAL，batch实现保留为独立后续实验。总预算为新CORE90 E200加两个A1 E200。

## r2技术故障与r3修复

r2通过GPU执行等价检查，从零初始化并读取6300/56700/27000样本，但首批共享遥测引用了MUSE关闭路径未初始化的`rc4_route`，训练退出且pipeline状态FAILED，两个进程均已退出；未完成epoch，不复用其产物。已将该变量在每批入口初始化为None，RC4启用时仍由原路由覆盖，训练目标不变。新增测试直接执行真实初始化与遥测表达式，70项相关测试通过。r3使用全新目录`a1_fast_matched_core90_s392005_20260908_r3`重新从零运行，矩阵、seed和预算保持不变。

## r3发布读回：VERIFIED / RUNNING

发布代码提交：`ab8a5778bb3e2914908ede60a65fca9a10134e7d`，远端Git分支OID独立核对一致。release为`/home/szu2070436088/2510044040/CV-SincNet/releases/a1_fast_matched_ab8a5778`。传输归档SHA256=`f2355a88290459c10c1db8052b8edc061bb2bd013c01dc97000fd0e90995239a`，远端一致且compileall通过。

2026-09-08发布读回：pipeline PID3812777，CORE90 PID3813334，CWD与上述release一致。命令含`--from_scratch true`且无baseline/teacher checkpoint。日志确认`init=scratch`、L/U/V=6300/56700/27000和`[EPOCH-END] E001/200`；GPU3该训练进程约3800MiB，读回GPU利用率23%。全机当时两个唯一compute PID，本轮只使用GPU3，不干预另一个进程。

新初始化GPU检查PASS：逐视图FP32/AMP教师输出最大绝对差0，E1/E21/E161实际DAOT更新对照通过，证据为[GPU执行检查](../analysis/a1_fast_r3_gpu_execution.json)。独立审查额外执行54项关闭MUSE的实际共享遥测表达式，均无异常。本轮70项相关测试通过。

当前为CORE90_RUNNING，A1_REFERENCE和A1_FAST_SEQUENTIAL尚未启动，自动等待本轮CORE90 E200及新权重检查；尚无本轮完整精度或端到端加速结论。r1/r2失败产物均保留。追踪为8项verified、2项implemented、5项deferred；保留的原EMA/CosFace缓存缺陷是已知限制，修复与Fast执行优化分开，详见追踪表。
