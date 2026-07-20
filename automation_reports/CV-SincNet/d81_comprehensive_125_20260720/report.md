# D81全面125稳定性实验报告

## 最终结论

- 实验状态：`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。
- 技术完成：D81最终权威运行`retry9`完成125/125个job，8/8分片`PASS`，技术失败0；375个场景对、750个before/after场景状态指标、10,125条逐TX指标全部落盘。
- 性能结论：D81未达到任何K10绝对性能门，K5相对K10退化超过3pp，K1所有receiver的旧类适应增益均为负；不能认定为本项目最强版本，也不能进入正式确认。
- 声明边界：`claim_scope=development_only_not_formal_confirmation`，`formal_launch_authority=false`，地面组件仍为`UNVERIFIED/PHASE1_COMPONENT_PENDING_OUTER_JOINT_SEAL`。本批125是5个锁定slice的稳定性screen，不是当前目标要求的`K={1,5,10,20}×new={2,5,10,20}`完整独立确认矩阵。
- matched缺口：同批direct ADV3B02列为`MISSING_NOT_RUN`，因此不能计算D81相对direct ADV3B02的K1增益与配对置信区间；不得用历史跨row最高值代替。

## 实验登记与矩阵

- 实验ID：`d81_comprehensive_125_20260720`。
- 操作者：Codex。
- 方法：`d81_ground_nuisance_cauchy_center`。
- Phase1底座：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`，SHA-256=`2699eedc...`。
- 数据：复用D18命名的`VALIDATED_ONCE`received-IQ母缓存；D18只表示数据缓存来源，不执行D18算法，也不读取D18预测。
- receiver：`20-1,3-19,7-14,7-7,8-8`。
- confirmation seed：`713102,713103,713104,713105,713106`。
- slice：`K10/new5,K10/new10,K10/new20,K5/new20,K1/new20`。
- 每个job内部场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- 每个TX每场景20个固定received-IQ query；before/after预测先只读密封，truth随后由独立scorer连接。
- 本批没有unknown类、defer、rollback或coverage选择机制；这些字段均为`N/A`，不能从本批推导unknown性能。

## 正式结果文件

- [125行同row指标](artifacts/retry9/summary/row_metrics.csv)：每行保留receiver、seed、K、new数、before/after旧类、旧类floor、新类、H、遗忘和artifact哈希。
- [375行逐场景指标](artifacts/retry9/summary/scenario_metrics.csv)。
- [10,125条逐TX指标](artifacts/retry9/summary/per_tx_metrics.csv)。
- [25行receiver×slice汇总](artifacts/retry9/summary/receiver_metrics.csv)。
- [机器可读summary](artifacts/retry9/summary/summary.json)与[gates](artifacts/retry9/summary/gates.json)。
- [125矩阵manifest](artifacts/retry9/matrix_manifest.json)、[8分片事件](artifacts/retry9/events)、[250个job日志](artifacts/retry9/job_logs)和[launcher日志](artifacts/retry9/launcher_logs)。

## 总体性能

总体均值混合了难度不同的5个slice，只用于描述分布，不能替代逐slice门槛。

|指标|125行均值|中位数|最小|最大|标准差|
|---|---:|---:|---:|---:|---:|
|注册前旧类准确率B|81.55%|84.17%|47.78%|98.61%|11.06pp|
|注册后旧类准确率C|64.40%|66.67%|26.94%|89.17%|14.13pp|
|注册前旧类floor|59.88%|63.33%|10.00%|93.33%|19.34pp|
|注册后旧类floor|35.20%|35.00%|3.33%|75.00%|16.46pp|
|seen-new准确率|59.11%|68.00%|12.67%|84.67%|20.10pp|
|H_old_new|61.09%|68.68%|17.40%|85.71%|17.45pp|
|平均遗忘|17.15pp|16.94pp|3.89pp|32.22pp|5.73pp|

## 按slice性能

|K/new|行数|B旧类|C旧类|C旧类floor|seen-new|平均H|聚合门槛H|遗忘|最差row H|最差row floor|判定|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|K10/new5|25|86.11%|76.32%|50.67%|73.61%|74.61%|74.47%|9.79pp|48.86%|28.33%|FAIL|
|K10/new10|25|86.11%|71.53%|42.27%|66.69%|68.81%|68.70%|14.58pp|48.29%|20.00%|FAIL|
|K10/new20|25|86.11%|68.71%|38.07%|68.80%|68.59%|68.49%|17.40pp|50.41%|20.00%|FAIL|
|K5/new20|25|81.27%|61.40%|30.80%|59.29%|60.03%|—|19.87pp|42.36%|11.67%|FAIL|
|K1/new20|25|68.14%|44.03%|14.20%|27.15%|33.41%|—|24.11pp|17.40%|3.33%|FAIL|

K5/new20相对matched K10/new20的均值下降为：C旧类7.31pp、C旧类floor7.27pp、seen-new9.51pp、平均H8.56pp，均超过3pp约束。K1/new20总体旧类适应增益为-24.11pp，5个receiver分别为`20-1:-28.44pp,3-19:-20.39pp,7-14:-27.44pp,7-7:-23.56pp,8-8:-20.72pp`。

## K10目标门

|new数|C旧类实测/目标|逐类旧类floor实测/目标|seen-new实测/目标|联合通过row|结论|
|---:|---:|---:|---:|---:|---|
|5|76.32%/92%|54.93%/88%|73.61%/92%|0/25|FAIL|
|10|71.53%/92%|46.53%/88%|66.69%/90%|0/25|FAIL|
|20|68.71%/92%|43.13%/88%|68.80%/86%|0/25|FAIL|

这里的逐类旧类floor采用75个receiver×seed×scenario单元聚合后的最弱旧类准确率，与`gates.json`一致；上节row floor是每个receiver/seed先跨场景汇总后再取旧类最小值，二者统计层级不同。

## 按receiver、场景与seed表现

|receiver|B旧类|C旧类|C floor|seen-new|H|遗忘|最差row H|
|---|---:|---:|---:|---:|---:|---:|---:|
|20-1|83.58%|65.48%|41.73%|61.09%|63.06%|18.10pp|28.28%|
|3-19|67.30%|51.09%|21.00%|38.16%|43.38%|16.21pp|17.40%|
|7-14|83.99%|65.60%|32.60%|65.22%|64.77%|18.39pp|36.69%|
|7-7|90.82%|74.29%|45.53%|65.49%|69.27%|16.53pp|37.80%|
|8-8|82.06%|65.54%|35.13%|65.60%|64.99%|16.51pp|36.22%|

receiver差异远大于seed差异：`3-19`是主要系统性短板，H仅43.38%、C floor仅21.00%；`7-7`最好，但仍未达到门槛。

|场景|B旧类|C旧类|C floor|seen-new|H|遗忘|
|---|---:|---:|---:|---:|---:|---:|
|leo_clear_weak|85.07%|70.16%|38.60%|65.02%|66.90%|14.91pp|
|leo_low_elev_weak|80.35%|62.51%|29.44%|56.47%|58.57%|17.85pp|
|leo_rain_weak|79.22%|60.53%|26.28%|55.84%|57.30%|18.69pp|

|seed|B旧类|C旧类|C floor|seen-new|H|遗忘|
|---:|---:|---:|---:|---:|---:|---:|
|713102|82.81%|65.40%|36.53%|59.96%|62.04%|17.41pp|
|713103|81.06%|64.76%|35.47%|58.72%|61.07%|16.30pp|
|713104|80.79%|63.70%|35.13%|58.86%|60.58%|17.09pp|
|713105|81.08%|63.42%|33.13%|59.07%|60.70%|17.66pp|
|713106|82.01%|64.72%|35.73%|58.94%|61.07%|17.29pp|

seed均值较稳定，说明主要问题不是随机种子，而是K、注册类规模、receiver和弱场景。

## 同row最佳、最差与最大遗忘

|row|B旧类|C旧类|B floor|C floor|seen-new|H|遗忘|表现|
|---|---:|---:|---:|---:|---:|---:|---:|---|
|`rx_7_7__seed_713104__k_10__new_5`|96.39%|88.61%|90.00%|58.33%|83.00%|85.71%|7.78pp|全125最高H，但C旧类、floor和new5仍均未过门|
|`rx_3_19__seed_713105__k_1__new_20`|48.89%|27.78%|20.00%|6.67%|12.67%|17.40%|21.11pp|全125最低H|
|`rx_7_14__seed_713102__k_1__new_20`|87.22%|55.00%|66.67%|5.00%|31.67%|40.19%|32.22pp|全125最大遗忘|

不得把最高H=85.71%与其他row的最高旧类、新类或floor拼接成虚构“最强性能”。完整同row上下文见`row_metrics.csv`。

## 逐旧类表现

下表为75个receiver×seed×scenario单元的注册后旧类准确率均值。

|K/new|14-10|14-7|20-15|20-19|6-15|8-20|
|---|---:|---:|---:|---:|---:|---:|
|K10/new5|75.33%|54.93%|94.93%|61.20%|72.93%|98.60%|
|K10/new10|71.53%|46.53%|93.87%|53.27%|65.93%|98.07%|
|K10/new20|68.67%|43.13%|92.13%|50.67%|61.00%|96.67%|
|K5/new20|59.40%|37.47%|86.87%|40.87%|50.53%|93.27%|
|K1/new20|33.13%|36.67%|51.20%|22.33%|32.60%|88.27%|

旧类不是均匀遗忘：`8-20`长期稳定，`20-19`和`14-7`最脆弱。通用floor约束失败并非少量seed偶然波动，而是持续的类级不均衡。

## 地面压缩原型的实际使用与遗忘解释

- 每个job确实读取84个地面压缩输入，地面逻辑状态25,428B，只读且`ground_component_update_access=false`。
- D81不是把地面旧类原型直接作为分类器锚点；它从地面int8聚合构造固定干扰谱/子空间，用target support在该子空间中的残差能量计算Cauchy权重，再平移每个target类中心。query不参与拟合，地面组件不在线更新。
- K10每job执行132次有效ground fit和264次support中心变换；K5执行72/144次。K10/new20注册后中心最大位移的场景均值为0.05281，最大0.06541；K5/new20均值0.05805，最大0.07411，说明地面组件确实改变了支持中心。
- K1按锁定边界严格恒等：有效ground fit=0，18次变换的中心位移全部为0，effective sample size=1。单样本没有类内残差，无法估计“哪个样本更像地面干扰”，因此Cauchy权重无法发挥作用。
- 这解释了为何“读取了地面压缩信息”仍不能防住遗忘：D81只用地面信息做支持样本可靠性加权/中心平移，没有把地面旧类知识作为old-logit锚、旧类均值先验或协方差约束；K1更是完全无作用。K1平均遗忘24.11pp、C旧类floor14.20%不是未加载地面组件，而是当前利用方式的机制上限。

## 量化与资源审计

|项目|实测|上限/要求|判定|
|---|---:|---:|---|
|trainable参数|2,016|≤80,000|PASS|
|adaptation epochs|20|≤30|PASS|
|optimizer steps|20|≤50|PASS|
|持久状态峰值|34,011–43,931B|≤262,144B|PASS|
|地面组件状态|25,428B|计入持久状态|PASS|
|dense query graph|0B|必须为0|PASS|
|query-dependent batch optimization|false|必须为false|PASS|
|query backbone forwards/sample|1|逐样本一次|PASS|
|FFT提取/query|1|固定received IQ单视图|PASS|
|score矩阵时延/query|0.00697–0.01422ms|诊断值，无单独门|PASS|
|metric-fit CUDA峰值|20.42–20.95MiB|记录值|PASS|
|enrollment CUDA峰值|155.94–156.21MiB|记录值|PASS|

375个场景fit全部满足`formal_target_vectors_int8_no_fp32_sidecar=true`，驻留FP32 target系数数为0；before状态的int8/FP32 support argmax变化均为0。final状态仅2/375个fit各出现1个support argmax量化变化，均位于`rx_7_14/seed713106/K10/new20`的`leo_low_elev_weak`和`leo_rain_weak`，需要作为量化生命周期异常保留，但不是本批性能失败的主要规模来源。

## GPU/CPU效率修复

- retry8验证了GPU设备绑定正确，但D81的Torch骨干/metric fit之后仍有锁定的`sklearn/numpy`LDA与OOF线性代数在CPU执行。未设线程上限时，单个D81进程创建178个线程，CPU约321%，非自愿上下文切换约470万次，GPU在CPU阶段空闲。
- 修复提交为worktree`d40ed9da`、Git承载分支`b104fd38`：每个row子进程强制`OMP/MKL/OpenBLAS/NumExpr/BLIS=2`，不改变D81数学、sklearn求解器或逐样本query规则。
- 修复后单个子进程总线程数降至8、CPU约135%，现场GPU利用率采样达到47%/18%；同一K1/new20行的六项核心指标与retry8逐项一致。
- retry9总makespan为806.56s；单job平均49.22s、中位49.44s、范围33.53–65.12s。K1/new20平均35.71s，K10/new20平均60.83s。

因此该实验是“GPU骨干与metric fit+CPU精确LDA/OOF”的混合执行，不是纯GPU；问题已从无界CPU线程爆炸修复为受控混合执行。

## 协议、覆盖与日志审计

- `protocol_schema=p2_min_v1`；无clean/raw/source访问，无query truth/role Oracle、真实batch类数、类配额、global reassignment或query拟合。
- 125个stdout全部包含`DEVELOPMENT_ROW_COMPLETE`；125个stderr均为空。
- 8个事件文件精确包含125次`JOB_START`和125次`JOB_COMPLETE`，无`JOB_FAILED`。
- 125个before/after预测artifact均与COMMIT、execution receipt、score SHA闭合；3个场景token互不重叠且并集精确覆盖truth。
- 75个K1/K10 receiver×seed×scenario配对全部通过rank0 support received-IQ哈希、query received-IQ哈希和只读truth物理ID一致性审计。
- matrix manifest SHA-256=`3fa63135de0bff0bfed1725c143f1e05248b689e3ea5827c3ece141e29043a73`。

## 执行历史与缺陷

|运行|结果|缺陷/处理|性能证据|
|---|---|---|---|
|initial|125/125技术失败|旧loader入口|无|
|retry1|技术失败|D81 evaluator仍导入旧loader|无|
|retry2|技术失败|隔离闭包缺少D45 probe依赖|无|
|retry3|技术失败|共享row pipeline被其他任务写入D62候选；改用D81独立源码快照|无|
|retry4|技术失败|snapshot launcher解析到错误`runs/code`路径|无|
|retry5|技术失败|远端sklearn1.7.0与旧1.7.2硬锁不兼容|无|
|retry6|技术失败|NumPy2/Torch2.1`from_numpy`ABI问题；改为buffer bridge|无|
|retry7|技术失败|预测发布目录未预创建|无|
|retry8|1行完成后主动中断|发现178线程CPU膨胀；保留全部产物并精确终止D81进程|1行，仅作retry9数值一致性核验|
|retry9|125/125完成|线程上限修复后8/8 shard PASS|本报告唯一权威性能结果|

汇总器随后修复了三处旧审计假设：场景token应为互斥分区并在并集上覆盖truth；显式一致的diagnostic-only preopen状态可用于开发汇总但不得升级为formal；单场景query包应是truth非空子集。对应提交为`ae3e971a/ec9889e2/3242e6ed`，Git承载分支为`f2982a01/be09707d/752a9877`。

## 复现信息

- 远端源码快照：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_source_snapshot_retry4_20260720`。
- 权威输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/d81_comprehensive_125_v2auth_retry9_20260720`。
- launcher日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/d81_comprehensive_125_v2auth_retry9_20260720`。
- 8个launcher PID：`1000172,1000175,1000178,1000181,1000184,1000187,1000190,1000193`，GPU0–7一一对应。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- D81 launcher SHA-256=`a96ecc9ba48f77d40f49e7c74c27df508c2c9855f5503114abe4520075b53cf7`。
- 最终summarizer SHA-256=`b7f27604df8da47375906294dc2a1f83860b09b3cd88ab11dc60432a31cf6978`。

启动命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
PYTHONPATH=<snapshot>:<project-root> \
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u \
<snapshot>/scripts/run_d81_125_stability.py \
  --cache-root <D18-VALIDATED_ONCE-cache-matrix> \
  --authority-root <D81-authority-final> \
  --phase1-checkpoint <ADV3B02-final> \
  --sealed-runtime <sealed-feature-runtime> \
  --method-lock <method-lock> \
  --output-root <retry9-output> \
  --ground-component-dir <ground-int8-component> \
  --ground-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c \
  --cpu-threads 2 --shard-index <0..7> --shard-count 8 --device cuda:0
```

## 最终判定与后续边界

D81资源、量化、逐样本部署和数据协议边界均通过，但性能存在三个结构性缺陷：K1地面稳健中心严格恒等、旧类遗忘随注册类数增加而扩大、receiver/旧类不均衡严重。因此D81不得晋升，不得称为最强版本；正式记录为`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`。

若继续研发，下一候选必须在不引入clean/query/role/quota权限的前提下，把地面压缩知识从“仅支持可靠性加权”提升为可作用于K1的类无关旧类先验或约束，同时仍保持target-old/new最终int8、无FP32 sidecar和相同资源上限。该建议不是本批已实现结果。
