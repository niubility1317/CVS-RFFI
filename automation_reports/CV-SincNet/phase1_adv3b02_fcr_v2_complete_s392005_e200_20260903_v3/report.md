# 当前状态：ANALYZED（2026-09-07）

训练、预测、独立评分及汇总已完成；高级机制缺项详见最终结果。以下预登记和启动记录原样保留，其RUNNING/ACTIVE为历史状态。监控自动化当前已不存在（删除请求读回not_found）。

# FCR-V2完整矩阵V3预登记

- 状态：`LOCAL_VERIFIED`
- run ID：`phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3`
- 代码修复提交：`0fd1a02a652b79284e4a522f6d4080ea30d965ac`
- V2失败记录提交：`ac5b437e2302d107aafa5fdb6ba837cd370bb45b`
- 分支：`codex/adv3b02-fcr-r1r8-s392005-20260903`
- 环境：N607普通账户，`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3`
- 数据：`Dataset_WigSig/ManySig.pkl`，equalized，split seed=`392005`；source receiver=`1,3,4,6,8`、day=`1,2,3`、TX=`0..5`、pool=`90000`，训练只消费`L_s=6300`和`U_s=56700`；target receiver=`0,2,5,7,9,10,11`、day=`0,1,2,3`
- 初始化checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`
- checkpoint身份：seed=`392005`，epoch=`200`，candidate=`ADV3B02_CORE90_SOFT_E200`；架构锁=`no_dac/no_stats`
- 固定基线：clean=`76.2268%`，LEO均值=`60.1397%`，四场景均值=`64.1615%`
- 矩阵：C0只评估；训练C1、C2、C3、S0、S1、S2、S3、S4、M1、M2、M3、M4、M5、M6，共14个E200 final-only行
- GPU：wave1的8行分别使用GPU0-7；wave2的6行分别使用GPU0-5；用户明确允许在现有任务上增加本矩阵
- 启动命令：`RUN_ID=phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3 ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release-root> bash code/scripts/launch_phase1_adv3b02_fcr_v2_complete_s392005_20260903.sh`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3`
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3`
- 预期artifact：14个`final.pth`、14份FCR诊断、15行四场景prediction、独立truth sidecar、15份独立score
- checkpoint选择：禁止使用target筛选，固定每行E200最后一个epoch的`final.pth`
- 评价边界：训练全部完成后才统一准备target输入；prediction先写出，独立scorer后连接truth；第二波发布不读取target结果
- 技术停止规则：仅在协议/query泄漏、错误seed/receiver/day/scenario、输出冲突、错误checkout、确定性重复异常、无prediction闭合、scorer连接错误或进程归属不清时停止精确run进程树；低性能不停止
- 机制解释边界：能力门控不能中止矩阵。M4/M6只有在真实pair、非零loss和非零梯度诊断同时成立时才能解释为机制已激活，否则报告`MECHANISM_NOT_ACTIVATED`
- V1修复：恢复基线`no_dac/no_stats`架构锁，真实checkpoint完整加载。
- V2修复：formal identity无模型路由只接受已定义的V1/V2正式schema，未知schema继续拒绝；不改变模型、损失、数据或矩阵。
- 本地验证：修复者聚焦45项及完整`phase1_fcr`183项通过；主流程独立运行V2训练集成38项通过。
- Git/release：运行提交`bb9e60ba3416dfded81f4c61444133f6dbaaf1e9`已推送且远端OID一致；release归档本地→远端SHA256均为`5689442a1762488dca531b70b1844728bddb548a8b141933675b8ea6696f5610`，远端编译通过。
- 真实checkpoint无query smoke：`loaded=195`、`skipped=0`、`incompatible_source_mature_identity=0`，seed=`392005`、epoch=`200`、candidate=`ADV3B02_CORE90_SOFT_E200`，V2 schema与`(2,6)`正式logits路由通过。

## 启动状态

- 当前状态：`RUNNING`
- 提交shell PID：`858714`；正式launcher PID/PGID：`858715`；wave1训练PID：C1=`859501`、C2=`859511`、C3=`859520`、S0=`859533`、S1=`859547`、S2=`859561`、S3=`859570`、S4=`859583`。
- 初始绑定：8个训练进程的CWD/代码路径/run root/seed/E200/final-only/checkpoint均与预登记一致，GPU0-7各绑定1行；日志均已创建并增长。
- 初始健康：C1已连续完成15个epoch；C2、C3、S0-S4均已越过V2的schema崩溃点，无`Traceback/RuntimeError/ValueError/OOM/Killed`。各FCR行在AMP初始动态缩放期出现3次有限unsafe-step跳过，之后跳过计数保持不变且CPU/GPU持续活跃，已转入有效optimizer step；完整epoch由后续监控确认。
- 监控策略：每30分钟短连接只读检查进程归属、行数/epoch、日志增长、GPU和确定性故障指纹；状态无变化时保持安静，完成、失败或需要用户处置时通知。
- 监控自动化：`fcr-v2-v3`，状态`ACTIVE`；每次由`luna_worker`执行边界明确的只读检查，禁止修改、停止、重启或清理。

---

# FCR-V2矩阵最终结果（V3运行）

核查日期：2026-09-07，N607北京时间10:31后读回。

## 结论

14个新训练行均有E200/200结束标记、final.pth和fcr_diagnostics.json；C0及14行均有predictions.json和独立score.json。每行四场景各168000条、六类，共672000条。15行计10080000条评分记录，但各行复用同一评价集，不是10080000个独立物理样本。最后M6预测文件修改时间为2026-09-07 10:30。产物状态VERIFIED；本次仅只读取回已有评分，没有重新训练或评分。

M6四场景均值最高65.2935%，M5的LEO均值最高60.4766%。相对C0，M6的clean提高3.6548个百分点、LEO均值提高0.2911个百分点。clean改善明显，LEO改善较小，单seed结果不能证明统计显著或自动升级默认方法。

重要：M4/M5/M6的transplant未激活；M6的factor未激活。不能把M6称为已完成三轴机制验证，也不能把M4−M3解释为移植效应。

## 固定条件与范围

基于预登记设计：split seed=392005；C0复用ADV3B02 E200；其他行从相同C0 checkpoint继续训练200epoch，使用最终checkpoint而非target最优轮。场景是clean及三种leo_*_weak模拟压力场景，不是实际卫星链路实测。矩阵定义见[设计](../../../docs/superpowers/specs/2026-09-03-adv3b02-fcr-v2-complete-matrix-design.md)。本报告是Phase1六类识别，不是Stage2旧类/新类注册竞争；不提供不存在的四态、遗忘率或注册收益。

## 全矩阵准确率

单位：%；Δ单位：百分点。LEO均值为三种LEO场景算术均值；四场景等权。

|行|设计项（实际激活见诊断）|clean|晴空|低仰角|雨衰|LEO均值|四场景均值|最差场景|ΔLEO对C0|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|C0|既有ADV3B02 E200|76.2268|61.3679|59.5875|59.4637|60.1397|64.1615|59.4637|0.0000|
|C1|继续训练E200，无FCR|78.0089|60.5351|58.5744|58.3679|59.1591|63.8716|58.3679|-0.9806|
|C2|FCR身份路由，辅助损失0|79.8470|61.2512|59.3167|59.0137|59.8605|64.8571|59.0137|-0.2792|
|C3|C2+self|79.6875|61.4452|59.5220|59.2220|60.0631|64.9692|59.2220|-0.0766|
|S0|shared/swap零点|79.7607|61.0732|59.1643|58.9012|59.7129|64.7249|58.9012|-0.4268|
|S1|S0+指纹shared|79.8375|61.5161|59.7595|59.4524|60.2427|65.1414|59.4524|0.1030|
|S2|S0+内容shared|79.6339|61.4530|59.6036|59.2726|60.1097|64.9908|59.2726|-0.0300|
|S3|两种shared|79.8065|61.4131|59.6363|59.3298|60.1264|65.0464|59.3298|-0.0133|
|S4|S3+true swap|79.8429|61.6500|59.8363|59.5476|60.3446|65.2192|59.5476|0.2050|
|M1|Core+eta+物理解码|79.8917|61.5774|59.8077|59.5250|60.3034|65.2004|59.5250|0.1637|
|M2|M1+cycle|79.7929|61.6655|59.8006|59.6208|60.3623|65.2199|59.6208|0.2226|
|M3|M2+necessity|79.7762|61.5524|59.6708|59.4792|60.2341|65.1196|59.4792|0.0944|
|M4|M3+定向移植（未激活）|79.7726|61.5381|59.7458|59.5482|60.2774|65.1512|59.5482|0.1377|
|M5|M4+复物理特征|79.6649|61.7482|59.9339|59.7476|60.4766|65.2737|59.7476|0.3369|
|M6|M5+三轴（未激活）|79.8815|61.7351|59.8839|59.6732|60.4308|65.2935|59.6732|0.2911|

## 同行增量与归因边界

|比较|Δclean|ΔLEO均值|Δ四场景均值|
|---|---:|---:|---:|
|C1−C0|1.7821|-0.9806|-0.2899|
|C2−C1|1.8381|0.7014|0.9856|
|C3−C2|-0.1595|0.2026|0.1121|
|S1−S0|0.0768|0.5298|0.4165|
|S2−S0|-0.1268|0.3968|0.2659|
|S3−S0|0.0458|0.4135|0.3216|
|S4−S3|0.0363|0.2183|0.1728|
|M2−M1|-0.0988|0.0589|0.0195|
|M3−M2|-0.0167|-0.1282|-0.1003|
|M5−M4|-0.1077|0.1992|0.1225|
|M6−M5|0.2167|-0.0458|0.0198|

C1额外预算使clean增加但LEO下降；C2路由效应已经贡献相当部分clean收益，不能将全额收益归因物理重构。S1优于S2，S3没有超过S1，不能声称两种shared互相增益。S4比S3有正增量。cycle小幅增加，necessity在本seed下下降。M5在LEO上优于M6；上述均是描述性结果，不能反馈本确认集调参或选模。

## 机制诊断

下表仅陈述诊断记录的非零损失和梯度证据，不等同语义可辨识性成立。

|行|实际激活辅助项|配置但未激活项|
|---|---|---|
|C1|无|无|
|C2|无|无|
|C3|self|无|
|S0|无|无|
|S1|self, shared_f|无|
|S2|self, shared_s|无|
|S3|self, shared_f, shared_s|无|
|S4|self, shared_f, shared_s, swap|无|
|M1|eta, response, self, shared_f, shared_s, swap|无|
|M2|cycle, eta, response, self, shared_f, shared_s, swap|无|
|M3|cycle, eta, need, response, self, shared_f, shared_s, swap|无|
|M4|cycle, eta, need, response, self, shared_f, shared_s, swap|transplant|
|M5|cycle, eta, need, physical, response, self, shared_f, shared_s, swap|transplant|
|M6|cycle, eta, need, physical, response, self, shared_f, shared_s, swap|factor, transplant|

M6累计nuisance配对2506240、content配对1895632、fingerprint配对0；对应覆盖率0.5000、0.3782、0。eta有效覆盖率0.5。M6的transplant/factor有效权重均为0。

诊断缺项：zf/zn/state的TX probe因训练切分没有覆盖至少两个评估类而N/A；内容probe、配对swap输出差、移植保持指标、clean/LEO梯度余弦也缺少有效观测。逐TX source诊断仅含TX0的1024条，不能冒充完整逐TX source评估。M6的drop_f_residual_gap=-0.019545，不能据此宣称指纹必要性验证通过。effective_rank=1.0也不足以证明良好分解。eta分量误差处于各自参数尺度，不混合解释为百分比准确率。

## 逐类结果

以下为M6，各类数值为准确率%。完整15行×4场景×6类见[per_class.csv](per_class.csv)。类别编号保留score中的0–5，未推测物理TX映射。

|场景|TX0|TX1|TX2|TX3|TX4|TX5|
|---|---:|---:|---:|---:|---:|---:|
|clean|94.4679|49.5607|76.0750|66.2500|99.7714|93.1643|
|leo_clear_weak|54.5893|33.3679|51.9464|62.7786|78.2964|89.4321|
|leo_low_elev_weak|54.9536|32.3357|49.7464|59.3464|74.0714|88.8500|
|leo_rain_weak|56.1786|31.3500|49.2750|57.8679|73.4250|89.9429|

## 资源

来自最终diagnostics的train_time_s及peak_vram_mb，不包括C0预训练和历史失败运行，也不是仪表级能耗或GPU忙时。

|行|训练墙时h|峰值显存MiB|
|---|---:|---:|
|C1|1.49|9733.6|
|C2|42.99|10402.8|
|C3|44.63|10438.6|
|S0|45.10|10398.9|
|S1|43.39|10438.1|
|S2|45.02|10444.6|
|S3|45.05|10435.3|
|S4|43.23|10436.6|
|M1|40.72|10437.2|
|M2|38.33|10437.2|
|M3|41.58|10438.7|
|M4|42.06|10437.2|
|M5|42.09|10437.2|
|M6|42.44|10437.2|

14行训练墙时相加558.12小时。并行执行，因此不等于从启动到结束的自然时间。C2及多数FCR行约38–45小时，C1仅1.49小时，计算成本不可忽略。

## 证据与状态

远端run：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_v2_complete_s392005_e200_20260903_v3。

每行证据：train.log、final.pth、fcr_diagnostics.json、target_prediction/predictions.json、target_prediction/score.json（C0无本轮训练）。[scores.json](scores.json)保留全部原评分字段；[diagnostics.json](diagnostics.json)保留14行完整诊断；[summary.csv](summary.csv)是计算汇总。此次没有复制约2.7GB的全预测到本地。

结论：评分与结果汇总完成；完整物理因子分解/三轴机制的科学证明未完成，不自动推广为默认。监控已到终点，无需继续轮询。
