# D85地面radius v2原型研发报告

## 实验身份与目标

|字段|值|
|---|---|
|实验ID|`D85_ADV3B02_GROUND_RADIUS_V2`|
|时间|2026-07-20 CST|
|执行者|Codex|
|目标|从ADV3B02 Phase1合法labeled-train流中生成只读int8中心、rank-3跨域残差和p90类内半径，使Phase2能按统一公式区分“可靠地面先验”和“宽散布地面先验”，避免D83/D84只加载中心却无法改善离散预测的问题|
|比较目标|D81地面中心Cauchy先验、D83逐cell精度加载、D84跨类共识中心|
|当前状态|`COMPLETED_DIAGNOSTIC_PERFORMANCE_NEUTRAL_EFFICIENCY_POSITIVE_NOT_PROMOTABLE`|

## 机制、假设与停止条件

主要差异不是继续增强地面中心融合强度，而是给每个合法domain×class中心附带量化p90余弦半径和rank-3跨域残差。Phase2后续统一使用标准化偏差

`e(c,d)=(1-cos(p_target(c),p_ground(c,d)))/(r_ground(c,d)+r_target(c)+epsilon)`

估计地面分量可靠度，再将可靠度用于弱正则、尺度校准或support-only闭式更新；所有类使用同一公式，不按TX/class ID分支。地面原型只读，不直接覆盖target-old或target-new原型。

预期可观察结果：D85真实组件应具有非退化的逐cell半径分布和rank-3残差，且持久化状态仅包含严格allowlist内int8数组与FP16尺度。若半径全部饱和/塌缩、active cell不足、双遍流hash不一致、checkpoint/WiSig/class binding任一哈希不匹配，立即失败关闭，不进入Stage2窄筛。若后续D85 Stage2锁定query相对D81无正向离散预测变化，则标记性能中性或负结果，不扩展到seed2/125。

最小验证矩阵：先生成一个固定Phase1组件并审计全部domain×class几何；然后仅在预登记development seed713101、K10/new5的105行合法单LEO弱观测上，与D81同row比较old-before/old-after/new/H/forgetting、全部逐类结果、混淆和资源。该组件未完成外部固定权威联合签名之前只能作为研发诊断输入，不能声称正式Phase2资格。

## 数据与协议边界

|项目|值|
|---|---|
|Phase1 checkpoint|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|WiSig输入|`Dataset_WigSig/ManySig.pkl`|
|WiSig SHA256|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|
|Phase1比例约束|当前协议锁定source全池`0.07/0.63/0.30`，`rho_label=0.07/(0.07+0.63)=0.1`|
|特征|逐样本L2归一化`z_id160`|
|统计|双遍有界流式：第一遍中心sum/count；第二遍4096-bin直方图估计p90余弦距离|
|持久化|仅严格3文件v2组件；不保留样本特征、count、source路径或FP32中心|
|Phase2 target/query访问|无|
|正式性边界|独立组件固定为`PENDING_OUTER_JOINT_SEAL`和`formal_phase2_eligible=false`；只有外部权威签发联合bundle后才可正式使用|

## 本地实现与验证

现有实现已覆盖所需最小链路，无需重写治理：

|文件|作用|
|---|---|
|`code/scripts/export_adv3b02_center_lowrank_radius_component.py`|真实checkpoint/WiSig入口、双遍有序流hash、严格输出|
|`code/cvsrffi/phase1_geometry_streaming.py`|有界内存中心和p90半径统计|
|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`|int8中心、rank-3残差、int8半径codec与验证|
|`code/cvsrffi/phase1_adv3b02_deployment_bundle.py`|runtime、组件、binding、lock的联合内容根与外部签名请求|
|`configs/phase1_d85_adv3b02_center_lowrank_radius.json`|本次生成参数和协议边界的哈希绑定配置|

验证命令：

`python -m pytest tests/test_phase1_geometry_streaming.py tests/test_phase1_center_lowrank_prototype_bundle.py tests/test_export_adv3b02_center_lowrank_radius_component.py tests/test_phase1_adv3b02_deployment_bundle.py -q`

初版结果为38项通过；修复历史split兼容漂移后为39项通过。pytest退出后的Windows临时目录清理出现`PermissionError`告警，但测试进程退出码为0且不影响项目验证。

## N607预检与运行计划

2026-07-20 06:51 CST执行`tools\n607_ssh_preflight.ps1`，直连成功，项目根可见，8张RTX3090均为0%利用率和10MiB显存。随后只读核查显示没有用户训练Python进程和GPU compute app；checkpoint为8,582,116B且哈希匹配，ManySig为2,359,341,461B且哈希匹配。远端尚缺v2导出器、流式几何和codec文件，因此必须先完成本地Git提交和精确SCP同步。

计划远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`

计划输出目录：`runs/d85_ground_radius_v2_20260720/adv3b02_component_retry1`

计划日志：`logs/d85_ground_radius_v2_20260720/export_retry1.log`

计划GPU：GPU0；该任务是单次Phase1离线推理和有界统计，不启动训练。

计划环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

计划命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/export_adv3b02_center_lowrank_radius_component.py \
  --checkpoint runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
  --wisig-pkl Dataset_WigSig/ManySig.pkl \
  --output runs/d85_ground_radius_v2_20260720/adv3b02_component_retry1 \
  --device cuda:0 \
  --expected-checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --expected-wisig-sha256 2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f \
  --expected-class-handle-binding-sha256 76735ae6d9b2d7e58f683635ca2644e00fbd27a515246aab9d47488c1ab5111f \
  --generation-config configs/phase1_d85_adv3b02_center_lowrank_radius.json \
  --expected-generation-config-sha256 108f5cdc191f4d7b55b8b453d93f456092ff886f919e390889a8dcb0dedc186a \
  --expected-generation-code-sha256 08efb45d6a1f3716f6073f846d1577d8d2d82838ea69f08204697216cb320042 \
  --batch-size 512 --num-workers 0 --min-samples-per-cell 2 --radius-histogram-bins 4096
```

运行前再次核查GPU进程；输出目录必须不存在或为空。启动后记录PID、GPU、精确命令、stdout/stderr、输出成员及SHA256。完成后必须解析组件manifest和NPZ全部形状、active cell、半径/残差分布、量化误差与状态字节，再决定D85 Stage2算法。

## 同步与版本状态

根目录`E:\type10-7`不是Git仓库；本报告及配置先写入Git工作树`E:\type10-7\code\snapshots\d81wt`，提交后再精确镜像到主发布仓库和根目录报告承载面。

本地哈希与计划SCP映射：

|本地文件|SHA256|远端目标|
|---|---|---|
|`code/scripts/export_adv3b02_center_lowrank_radius_component.py`|`08efb45d6a1f3716f6073f846d1577d8d2d82838ea69f08204697216cb320042`|`code/scripts/export_adv3b02_center_lowrank_radius_component.py`|
|`code/cvsrffi/phase1_geometry_streaming.py`|`f7eb4e5950ecaccc5fbecb25dab8d955e747d5384990ecc63100b013d7d28bf0`|`code/cvsrffi/phase1_geometry_streaming.py`|
|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`|`7bd410108129bbe8096e2b2c49180877adcc5160f9fc980eb1da404da5d5086c`|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`|
|`configs/phase1_d85_adv3b02_center_lowrank_radius.json`|`108f5cdc191f4d7b55b8b453d93f456092ff886f919e390889a8dcb0dedc186a`|`configs/phase1_d85_adv3b02_center_lowrank_radius.json`|

## 远端尝试0：协议失败关闭

|字段|结果|
|---|---|
|PID|728539|
|启动时间|2026-07-20 06:56:01 CST|
|状态|`FAILED_CLOSED_PROTOCOL_SPLIT_DRIFT_NO_PERFORMANCE_RESULT`|
|完成阶段|checkpoint和ManySig哈希验证通过；在构建source split时停止|
|错误|历史checkpoint参数`0.10/0.70/0.20`按当前定义得到`rho_label=0.125000>0.1`|
|输出组件|无|
|Phase2 target/query访问|无|
|性能指标|未进入组件生成或Stage2评估，old/new/H/forgetting均为`not_run`|
|资源表现|未进入GPU模型遍历；没有持久化组件；日志保留于`logs/d85_ground_radius_v2_20260720/export.log`|
|缺陷|初版导出器错误继承了历史checkpoint的数据split参数，无法生成符合最新项目协议的新聚合知识|

修复策略：checkpoint只贡献冻结模型状态、source registry和其他模型参数；新生成的Phase1聚合组件强制采用当前`项目.md`规定的source全池`0.07/0.63/0.30`划分。新增单测证明历史`0.10/0.70/0.20`会被覆盖且精确得到`rho_label=0.1`，不放宽任何阈值。

## 完成结果

### 真实v2组件生成结果

远端retry1于2026-07-20 07:00:41 CST启动，PID=`731053`，在GPU0完成真实ADV3B02 checkpoint与ManySig全流程。输出状态为`PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`，未伪造或绕过外部Ed25519签名。

|产物|大小|SHA256|
|---|---:|---|
|`int8_domain_class_center_lowrank_residual_radius_v2.npz`|5,854B|`1ac2424fee2ef804d83d7c8faca8d27c7c0267c0d9d7c8b97af0cf053bfb4ea6`|
|`manifest.json`|5,740B|`6990ce8c274d6f46c9a86f7272a76960b4a0055e06ae75e14f83e0b70595f112`|
|`manifest.sha256`|80B|`ec49673a964f490920b19302badee85e58f1ac70eda3e61641737c1bda020981`|
|retry1日志|—|`0c9072dcd9416fda0dd85493ed71dfcfcf06d75c305cd1f04222d574bd471471`|

预签内容根=`098badd1e82c05c1029cb02c024fe7d3c433488e8ab22e5c6e2ba0516b8d0055`，generation proof=`f28e4645191edba5b9000c231728a1569108269e190875bacfd2a03f2ef8d45a`，双遍stream hash=`fc7990be25dfb8a30c2b3ecbf2b2f6dd2310c9a0913625a73d21527ec97bc504`。严格loader目录只含上述3个组件文件；将日志混入组件目录时allowlist按设计失败关闭。

### 组件几何与压缩性能

|指标|结果|
|---|---:|
|domain×class有效cell|14×6=84，84/84有效|
|center/core形状|`[6,160]`|
|rank-3 basis/coeff形状|`[6,3,160]`/`[13,6,3]`|
|radius形状|`[14,6]`|
|radius min/p10/median/mean/p90/max|0.000331/0.000492/0.001323/0.004251/0.006833/0.100103|
|radius与domain drift Pearson|0.688886|
|rank-3重构cosine mean/min|0.999819/0.999368|
|rank-3角误差mean/max|0.9786°/2.0376°|
|rank-3 RMSE|0.001506|
|radius量化绝对误差mean/max|0.0000594/0.0002999|
|数值数组/逻辑部署/序列化大小|4,374B/5,816B/5,854B|
|query额外MAC/状态|0/0|

逐地面旧类radius中，`6-15`均值/最大值为0.009064/0.100103，`20-19`为0.007984/0.041997，显著宽于其余类；这证明真实半径不是常数或退化占位，并能识别高散布地面cell。

## D85 Stage2算法

D85先按D84方法从14个地面domain的6个旧类中心提取类无关共识模板，再用真实v2 p90半径作参数自由的可靠度校准：

`q_d=median_d(median_c(r_d,c))/(median_d(median_c(r_d,c))+median_c(r_d,c))`

`w_d=normalize(w_d,D84×q_d)`

校准后radius可靠度min/mean/max=`0.12866/0.45352/0.56766`，最终模板权重min/max=`0.02011/0.20414`。地面类中心在形成类无关domain模板后被丢弃；目标旧类和新类都只使用自身support，以相同Cauchy稳健中心公式平移。无rank/weight/hyperparameter扫描，无query/clean/source/role/quota访问，无新类地面身份映射。

## 完整105行实验结果

本地attempt1于2026-07-20 07:16:10 CST启动，124.61秒完成105/105行、7候选×15 outer rows；`stderr=0B`，完整日志无Traceback/OOM/NaN/Inf。首次本地启动因`Start-Process`漏传脚本路径立即退出，未进入方法、未创建结果目录，保留为启动封装证据，不计实验版本。

### 七候选联合指标

|candidate|B旧类注册前|A旧类注册后|N seen-new|H|F遗忘|J|min class B/A/N|mean-row floor B/A/N|old→new/new→old/new→new|判定|
|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
|D42-USLDA-INT8|92.78%|82.78%|84.67%|82.94%|10.00%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|22/8/15|D85目标，性能中性、效率正|
|D42-USLDA-FP32-MATCHED|92.78%|82.78%|84.67%|82.94%|10.00%|26.67%|80.00/53.33/73.33%|73.33/50.00/46.67%|22/8/15|与INT8全部argmax一致|
|B3_SINGLE_IQ_DIAG_FFTRF|87.78%|75.56%|72.67%|73.35%|12.22%|23.33%|80.00/60.00/40.00%|53.33/33.33/36.67%|33/22/19|诊断baseline|
|D42-D40-HNBR-INT8-NEGATIVE|85.56%|85.00%|15.33%|25.16%|0.56%|0%|66.67/63.33/0%|40.00/40.00/0%|2/0/0|新类不可达|
|D42-D41-BEC-INT8-NEGATIVE|86.11%|20.56%|78.67%|31.50%|65.56%|0%|76.67/0/36.67%|46.67/0/26.67%|142/0/32|旧类崩溃|
|D42-PROTOnet-CDA-ZID160|71.11%|48.33%|52.67%|48.97%|22.78%|0%|33.33/13.33/3.33%|13.33/0/0%|0/0/0|弱baseline|
|Z0_SUPPORT_ONLY|71.11%|48.33%|52.67%|48.97%|22.78%|0%|33.33/13.33/3.33%|13.33/0/0%|0/0/0|selection fallback|

unknown拒识、coverage、rollback、defer不属于该锁定K10/new5已注册类筛选的输出，均记为`N/A`，不以缺失指标主张部署性能。

### D85逐场景指标

|场景|B|A|N|H|F|J|min class B/A/N|mean-row floor B/A/N|混淆old→new/new→old/new→new|
|---|---:|---:|---:|---:|---:|---:|---|---|---|
|clear|98.33%|91.67%|98.00%|94.441%|6.67%|50.00%|90/70/90%|90/60/90%|2/1/0|
|low-elev|91.67%|80.00%|76.00%|76.922%|11.67%|20.00%|80/60/50%|70/60/20%|7/5/7|
|rain|88.33%|76.67%|80.00%|77.447%|11.67%|10.00%|60/30/70%|60/30/30%|13/2/8|

### D85目标候选15个outer row

|scene/fold|B|A|N|H|F|J|floor B/A/N|混淆o→n/n→o/n→n|
|---|---:|---:|---:|---:|---:|---:|---|---|
|clear/0|100.00%|100.00%|90.00%|94.74%|0%|50%|100/100/50%|0/1/0|
|clear/1|100.00%|83.33%|100.00%|90.91%|16.67%|0%|100/0/100%|0/0/0|
|clear/2|91.67%|83.33%|100.00%|90.91%|8.33%|50%|50/50/100%|1/0/0|
|clear/3|100.00%|100.00%|100.00%|100.00%|0%|100%|100/100/100%|0/0/0|
|clear/4|100.00%|91.67%|100.00%|95.65%|8.33%|50%|100/50/100%|1/0/0|
|low/0|100.00%|75.00%|80.00%|77.42%|25.00%|50%|100/50/50%|3/1/1|
|low/1|83.33%|58.33%|70.00%|63.64%|25.00%|0%|50/50/0%|1/0/3|
|low/2|83.33%|91.67%|70.00%|79.38%|-8.33%|0%|50/50/0%|0/2/1|
|low/3|100.00%|100.00%|70.00%|82.35%|0%|0%|100/100/0%|0/1/2|
|low/4|91.67%|75.00%|90.00%|81.82%|16.67%|50%|50/50/50%|3/1/0|
|rain/0|83.33%|83.33%|60.00%|69.77%|0%|0%|50/50/0%|2/0/4|
|rain/1|100.00%|66.67%|90.00%|76.60%|33.33%|0%|100/0/50%|4/1/0|
|rain/2|91.67%|83.33%|80.00%|81.63%|8.33%|50%|50/50/50%|1/0/2|
|rain/3|83.33%|75.00%|90.00%|81.82%|8.33%|0%|50/0/50%|3/0/1|
|rain/4|83.33%|75.00%|80.00%|77.42%|8.33%|0%|50/50/0%|3/1/1|

### 逐类表现

|TX|角色|B|A/N|主要缺陷|
|---|---|---:|---:|---|
|14-10|旧类|96.67%|93.33%|轻微遗忘|
|14-7|旧类|80.00%|53.33%|总体最弱旧类，注册后下降26.67pp|
|20-15|旧类|96.67%|90.00%|轻微遗忘|
|20-19|旧类|93.33%|93.33%|稳定|
|6-15|旧类|93.33%|73.33%|注册后下降20pp|
|8-20|旧类|96.67%|93.33%|轻微遗忘|
|1-16|新类|—|93.33%|良好|
|1-18|新类|—|73.33%|最弱新类|
|18-10|新类|—|90.00%|良好|
|14-11|新类|—|76.67%|偏弱|
|8-3|新类|—|90.00%|良好|

### 训练、量化与连续机制

- 20步trace×15 rows，共300条；loss min/mean/max=`0.07560/0.30719/1.11738`，support accuracy=`89.58/98.77/100%`。
- FP32/INT8 before、support、outer argmax变化均为0，margin sign flip=0；score绝对误差min/mean/max=`0.000517/0.000780/0.001340`。
- 实际执行1,080个D62 component fits和2,160次support-center transform。中心平移L2总体`0.001403..0.063294`，有效样本数最低6.8635；类内残差误差≤`2.78e-17`，FFT96/RF32误差0。
- D85不是D84的数值恒等副本：D85相对D84的before系数/截距最大绝对差在15行上的min/mean/max=`0.000880/0.191793/2.358527`，且1/15 before接纳mask发生变化；final差异=`0.002260/0.003901/0.005835`。但15/15 outer prediction hash仍完全相同，说明半径只改变了连续解和个别内层选择，没有跨越最终分类边界。

## 性能与效率对比

|项目|D84|D85|D85变化|
|---|---:|---:|---:|
|B/A/N/H/F/J|92.78/82.78/84.67/82.94/10.00/26.67%|相同|0pp|
|15-row prediction hash变化|—|0/15|无离散纠错|
|ground组件逻辑状态|25,428B|5,816B|-77.13%|
|组件含总持久状态|34,011B|14,399B|-57.66%|
|ground统计MAC|179,200|216,724|+20.94%|
|总新增适配MAC|22,069,760|22,107,284|+0.17%|
|总适配MAC|24,913,293,730|24,913,331,254|+0.00015%|
|query MAC/额外query MAC|6,624/0|6,624/0|不变|
|params/epochs/steps|2,016/20/20|2,016/20/20|不变|
|peak CUDA|22,886,912B|22,886,912B|不变|

D85在近乎不增加总计算的前提下，把真实地面组件和总持久状态分别压缩77.13%和57.66%，并首次把可审计p90半径用于domain可靠度；这是明确的状态效率改进。但它没有改善任何离散性能指标，因此不能进入seed2或完整125。

## 缺陷、门控与结论

D85相对项目K10/new5目标仍缺`A=9.22pp`、`minA=34.67pp`、`N=7.33pp`；rain旧类after最低30%，low-elev新类最低50%，总体最弱旧类14-7仅53.33%。`before_gate=false`、`after_old_gate=false`、`new_gate=false`、`pairwise_gate=false`、`joint_gate=false`；selection回退`Z0_SUPPORT_ONLY`。结论为`COMPLETED_DIAGNOSTIC_PERFORMANCE_NEUTRAL_EFFICIENCY_POSITIVE_NOT_PROMOTABLE`。

根因不是“没有使用地面压缩原型”：D85真实加载84个v2 cell、重构14个domain模板并用全部84个p90半径校准。问题是当前用法把地面半径压缩成单一domain权重，仍然只产生全类共享平移；它能判定哪个地面域更可靠，却不能回答目标support中的哪一个旧类会被新类挤压，也没有合法的新类到地面旧类身份对应。继续扫描权重或放大平移既缺乏新信息，也容易伤害类对称性和floor。

下一路线应保留v2组件的高效性，但把用途从“生成共享平移方向”改为“support-only风险约束”：利用地面rank-3残差张成的类无关域扰动子空间，对目标support估计每类沿该子空间的脆弱度，再以统一闭式公式限制旧/新全部注册类的margin收缩；地面中心只定义扰动坐标系，target support决定每类保护量。该方向不需要地面到目标类身份映射，也不需要query或角色Oracle，并有机会直接作用于14-7与6-15的注册后边界。

## 协议闭包

|边界|结果|
|---|---|
|输入观测|锁定`LEO_weak`单观测，support/query物理ID不交|
|query打开/用于fit|`false`/0行|
|clean/source样本访问|0/0|
|query truth/role Oracle|0/0|
|class quota/真实batch class count/global reassignment|0/0/0|
|新旧类公式|同式，无role-specific branch|
|地面组件更新|0，进出NPZ与manifest逐位一致|
|正式资格|`false`，缺外部权威联合签名|
|部署状态|单一INT8 affine head，query额外状态/MAC=0/0|

## 证据哈希

- `training_log.jsonl`：18,394,104B，SHA256=`b0e86081eb72ab1b123ddab0f035705f2ff6f681e3012004ff9e1eb791f88ac3`。
- `RECEIPT.json`：SHA256=`152e0a62c323fea39b791b375928b06315c672f90e3f3cad6620253086f9384c`。
- `D85_PROBE_METADATA.json`：SHA256=`688d97d4f05a56ec4940a06df96a5955d1d408e96c5766de7f04b755ad7a6d02`。
- `d85_full_performance_summary.json`：82,282B，SHA256=`789dae9bc9adf5094577958b968d7325ec5fa97cb7ce3dbac90ccc2c131a5673`。

## D83–D85三轮强制技术复盘

复盘时间为2026-07-20。已重新读取active goal与`项目.md`，刷新conversation index至1,008条，并检索地面压缩原型、radius、遗忘和floor；重新核对D83/D84/D85报告及三份完整105行日志，日志SHA分别为`a1771d71...5d5d24`、`679758e1...88fb3`、`b0e86081...f88ac3`。

|轮次|单一机制|有效经验|决定性缺陷|停止项|
|---|---|---|---|---|
|D83|逐cell地面精度进入rank-14共享协方差loading|证明地面谱能稳定改变连续分数|15/15预测不变，额外114.26M MAC无收益|协方差loading强度/rank扫描|
|D84|跨6个地面类提取14个类无关domain漂移模板，以Cauchy样本权重修正target中心|相同离散性能下，ground相关MAC较D83降80.7%|仍是全类共享平移，缺少纠错方向|继续改全局共识权重或放大平移|
|D85|真实v2 rank-3残差+p90 radius校准domain模板|组件状态降77.13%，radius非退化且与domain drift相关|1个before接纳mask改变，但15/15最终预测仍不变|radius只压成单一domain权重、参数扫描|

三轮共同结论：地面原型最有价值的信息不是旧类绝对中心，而是“合法source域内，特征沿哪些方向会漂移、漂移多宽”。D83把它变成统一协方差，D84/D85把它变成统一中心平移，都在最终类边界前被吸收。下一轮必须让地面信息直接约束support形成的分类margin，同时对旧类和新类使用同一公式；否则只能继续得到连续变化而无离散纠错。

协议复核通过：下一候选仍复用相同`VALIDATED_ONCE/p2_min_v1`D18数据；地面v2组件只读，单物理样本单LEO观测不变。基于固定received IQ的feature counterfactual只能作为同一support样本的数学扰动正则，不增加K、不进入support-held划分计数；query不生成扰动、不更新状态。不得使用ground→target-new身份映射、old/new角色、query truth、class quota、clean/source样本或跨query图。

第四轮锁定为D86`ground_radius_counterfactual_consistency`：从v2组件重构14个类无关domain漂移向量，并以对应跨类p90半径给每个方向确定唯一、无扫描的扰动幅度。对每个target support向量`z_i`构造对称数学扰动`z_i±sqrt(2r_d)u_d`，但仍把整组视为同一physical sample；在inner leave-one-physical-rank范围内最小化原support分类损失与对称扰动logit一致性，随后编译成单一INT8 affine head。对称正负扰动的一阶均值为0，不引入D78未中心化的类偏置；所有注册类与所有support样本同式，新类也直接受益。

D86相对D85的唯一主要变化是：ground radius不再决定“相信哪个地面domain并移动target中心”，而是定义“目标分类器必须抵抗多大的合法source域方向扰动”。预期信号是outer prediction至少1/15改变，且相对D81/D85不降低N、min-N或任一场景floor，同时改善A、F、min-A或混淆；若仍0/15变化，或重现D78/D79的old提升换new下降，立即淘汰，不进行seed2/125。计算目标为ground预处理<0.5M MAC、额外训练<5% D62、query额外MAC/state=0，持久状态保持14,399B以内。
