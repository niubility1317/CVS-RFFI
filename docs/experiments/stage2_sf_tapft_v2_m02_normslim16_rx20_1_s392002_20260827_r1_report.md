# SF-TAPFT M02 norm逐级瘦身16行实验

## 预登记

- run ID：`stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827_r1`
- 状态：`ANALYZED`
- 固定提交：`9c9ae29a65cb019d6a6fd30c613ecc8b470f8cc8`
- 目标：以独立query最优M02（完整target head+全部time norm）为锚，先缩减norm范围、norm affine和训练步数；本轮不压缩target head，不重新引入Adapter、完整`t3`或B/C阶段。
- 数据：`p2_min_v1/VALIDATED_ONCE`；capsule=`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`；split=`stage2b-rx20-1-seed713101-before-support-prefix`；receiver=`20-1`；旧6类K=10，共60条support。
- query边界：所有16行只运行4折support-inner OOF和全60条support refit；不读取query、query truth或query role，不重复使用已经参与M02筛选的rank10–19 truth。
- checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 优化seed：`392002`；4折定义、target head、rho=`0.5`、loss和学习率均与M02一致。

## 16行矩阵与GPU

|GPU|row|候选|唯一变化|
|---:|---|---|---|
|0|S00|M02锚点|完整复现`head+all norm`|
|0|S01|head-only|删除全部norm更新|
|1|S02|t3 norm|只训练`t3.norm`|
|1|S03|t2+t3 norm|只训练`t2/t3.norm`|
|2|S04|backbone norm|训练`t1/t2/t3.norm`，删除`time_fuse norm`|
|2|S05|fuse norm|只训练`time_fuse.1`|
|3|S06|t1 norm|只训练`t1.norm`|
|3|S07|t2 norm|只训练`t2.norm`|
|4|S08|t3+fuse norm|训练`t3.norm+time_fuse.1`|
|4|S09|t2+t3+fuse norm|删除`t1.norm`|
|5|S10|all norm weight|全部norm只训练weight|
|5|S11|all norm bias|全部norm只训练bias|
|6|S12|late norm weight|`t3+fuse`只训练weight|
|6|S13|late norm bias|`t3+fuse`只训练bias|
|7|S14|600步|固定4500步LR时钟，截断至600步|
|7|S15|300步|固定4500步LR时钟，截断至300步|

## 命令与路径

- N607环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v2_m02_normslim16_20260827_9c9ae29a/checkout`。
- 矩阵：`configs/stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827.json`。
- 命令模板：`CUDA_VISIBLE_DEVICES=<gpu> python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix <matrix> --row-id <Sxx> --output-dir <run-root>/<Sxx> --device cuda:0 --folds 4`。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827_r1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_v2_m02_normslim16_rx20_1_s392002_20260827_r1`。
- release归档计划：本地Git归档上传到`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_v2_m02_normslim16_20260827_9c9ae29a/release.zip`，只比较该归档一次本地/远端SHA。

## 发布与启动证据

- release归档本地/远端SHA256均为`24e1ae0be752c6b5289b19d5f7dda34b586aefa4a892bda84025a1ebb3cc6f72`，远端编译通过。
- 真实checkpoint无query smoke：`SMOKE_PASS`；`query_opened=false`、`source_opened=false`、support物理样本数60、更新参数张量数2。
- 2026-08-27正式启动16/16行；wrapper PID为107874–107903的16个已登记离散进程，GPU0–7均严格2个训练进程，单进程初始显存约652–670MiB。
- 启动只读回查确认所有命令均使用冻结checkout、矩阵、row ID和各自不可覆盖输出目录；launch receipt和GPU采样器已生成。

## 判定与停止规则

- 相对S00/M02的结构非劣门槛：`BA_new≥BA_M02-0.5pp`、`floor_new≥floor_M02`、`NLL_new≤NLL_M02+0.03`。由于60条support的分辨率较粗，BA实际上需要基本持平。
- 同时报告：可训练参数数、实际变化元素、bundle大小、wall-clock、最大RSS和GPU显存采样峰值。
- 通过门槛后按“参数最少→步数最少→NLL最低”确定唯一最小候选；本轮不接入query truth。
- 技术停止仅限协议/query泄漏、错误checkpoint/capsule/split/K、错误checkout、输出碰撞、不能产生selection/bundle、确定性重复异常或进程归属不清；不得因低性能停止。

## 本地验证与审查

- TDD：先观察14项能力负测和矩阵模块缺失负测失败，再实现最小功能。
- 聚焦回归：79项通过；Python编译通过；CLI help通过；16行严格解析和GPU每卡2行检查通过。
- P0/P1定点审查：发现新增字段会拒绝旧M02bundle；已修复为仅允许`norm_scope/norm_affine/scheduler_reference_steps`以默认值缺省，未知字段继续拒绝。修复负测与query闭合回归通过；未发现其他会使本次真实实验跑错、越权、覆盖输出、不能启动或不能产生合法selection的问题。

## 预期artifact

每行原计划生成`selection.json`、`sf_tapft_clean_single_bundle.pt`、完整stdout/stderr日志和`runtime_time.txt`；矩阵级生成launch receipt、GPU采样记录和完成汇总。用户定向裁剪后，终态集合改为8个关键行完整artifact加8个`USER_DIRECTED_PRUNED/NO_PERFORMANCE_RESULT`行；关键8/8解析并完成同row分析后进入`ANALYZED`。

## 完成结果

### 运行中快照

- 2026-08-27 12:48 CST：S15（300步上限）率先闭合；其余15行继续运行，异常指纹为0。
- S15 support OOF：BA=`0.8611111939`，fold floor=`0.7777778506`，NLL=`0.5148361325`，4折均不退化；最终全support refit采用203步。
- S15资源：可训练参数元素1584，实际变化元素1584，bundle 4,292,510B，wall-clock 39分17.91秒，最大RSS 1,829,304KiB；exit status=0。
- S15权限：`query_opened/query_truth_opened/query_role_opened/source_opened/target_eval_opened`全部为`false`，`nonpermitted_changed_names=[]`，support_count=60。
- 4500步结构行会对4个OOF fold分别完整执行4500步，再进行一次选中步数的全support refit；不存在训练早停。因此本轮预计还需约8–9小时。该估计不构成结果或停止条件。
- S15必须等待同run S00锚点后才能判定非劣，目前保持`PENDING_ANCHOR`。

### 用户定向优先级裁剪

- 2026-08-27 14:32 CST，用户明确要求停止不必要实验、加速重要训练。停止前只读回查确认S14/S15已完整闭合，其余14行均属于本run、冻结checkout和各自row/output root。
- 保留关键集合：S00锚点、S02`t3-only`、S05`fuse-only`、S08`t3+fuse`、S10全部norm weight-only、S11全部norm bias-only，以及已完成的S14/S15。
- 定点停止：S01、S03、S04、S06、S07、S09、S12、S13；状态统一为`USER_DIRECTED_PRUNED/NO_PERFORMANCE_RESULT`，不得从partial状态推断性能，也不自动恢复或补跑。
- 仅向上述8行对应的child PID和wrapper PID发送`TERM`；独立读回确认16个目标PID均已退出，S00/S02/S05/S08/S10/S11仍按原PID运行。
- 裁剪后活跃训练由14行降至6行：GPU0/1/2/4各1行，GPU5保留weight/bias两项正交实验；GPU3/6/7释放。所有selection、bundle、日志、运行时和partial输出均保留，未删除或覆盖任何artifact。
- 最终闭合条件随用户优先级调整为关键集合8/8，而非原始16/16；原预登记指标门槛和S00同run锚点不变。

### 最终artifact与协议闭合

- 关键集合S00/S02/S05/S08/S10/S11/S14/S15均生成`selection.json`、`sf_tapft_clean_single_bundle.pt`、完整stdout和GNU time记录；8/8进程exit status均为0。
- 完整读取8份stdout，共172,923B；每份为一条完整JSON记录。未发现Traceback、RuntimeError、OOM、Killed、NaN/Inf或协议违规标志。
- 8行均保持`p2_min_v1/VALIDATED_ONCE`、同一capsule/split、support_count=60、每类10条；`query_opened/query_truth_opened/query_role_opened/source_opened/target_eval_opened`全部为`false`，`nonpermitted_changed_names=[]`。
- 解析136,364条GPU进程采样。单行GPU显存峰值为676–690MiB；最大RSS为1,793,920–1,842,404KiB。bundle均可由`torch.load(weights_only=True)`完整读取。

### support-inner OOF结果

|row|候选|BA|相对S00|fold floor|NLL|相对S00|选中A步数|可训练/变化元素|门槛|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|S00|完整M02锚点|86.1111%|0.0000pp|77.7778%|0.436715|0.000000|327|1584/1584|锚点|
|S02|仅`t3.norm`|89.5833%|+3.4722pp|77.7778%|0.424413|-0.012302|412|1152/1152|PASS|
|S05|仅`time_fuse.norm`|84.0278%|-2.0833pp|69.4445%|0.506162|+0.069446|244|1056/1056|FAIL：BA/floor/NLL|
|S08|`t3+fuse.norm`|86.1111%|0.0000pp|77.7778%|0.452126|+0.015411|834|1248/1248|PASS|
|S10|全部norm仅weight|86.1111%|0.0000pp|77.7778%|0.414765|-0.021950|1094|1272/1272|PASS|
|S11|全部norm仅bias|87.5000%|+1.3889pp|77.7778%|0.446656|+0.009941|219|1272/1272|PASS|
|S14|600步上限|86.1111%|0.0000pp|77.7778%|0.508258|+0.071542|203|1584/1584|FAIL：NLL|
|S15|300步上限|86.1111%|0.0000pp|77.7778%|0.514836|+0.078121|203|1584/1584|FAIL：NLL|

S02四折BA为100.0000%、77.7778%、86.1111%、94.4445%，相比S00的94.4445%、77.7778%、77.7778%、94.4445%，提升集中在fold0和fold2，最差fold没有下降。它不是由单折峰值拼接出的虚构最优，而是同一S02 row的聚合结果。

### 资源结果

|row|bundle|wall-clock|最大RSS|GPU峰值|
|---|---:|---:|---:|---:|
|S00|4,292,510B|3:46:23|1,823,160KiB|690MiB|
|S02|4,291,358B|3:48:10|1,793,920KiB|676MiB|
|S05|4,291,422B|3:44:24|1,819,132KiB|690MiB|
|S08|4,291,806B|3:48:40|1,833,000KiB|690MiB|
|S10|4,291,742B|4:00:26|1,827,708KiB|690MiB|
|S11|4,291,742B|3:55:45|1,842,404KiB|690MiB|
|S14|4,292,510B|1:15:13|1,817,812KiB|690MiB|
|S15|4,292,510B|0:39:17.91|1,829,304KiB|690MiB|

S02把可训练及实际变化元素从1584降至1152，减少432个，即27.27%；其bundle只减少1152B，因为当前clean-single bundle仍保存完整模型state，参数瘦身尚未转化为delta-only封装。S02的4500步OOF训练时间与S00基本相同（增加1分47秒），说明本轮参数裁剪降低了持久状态和优化器规模，却没有解决完整骨干前向、每步验证和GPU–CPU同步开销。

S14和S15相对S00分别缩短66.78%和82.64%wall-clock，但NLL分别恶化0.071542和0.078121，超过允许的+0.03。短步数保住了离散准确率与floor，却没有保住概率校准，因此不能晋级。

### 结论与下一步

- 唯一最小非劣候选为S02：`target head+仅t3.norm`。它同时满足BA、floor和NLL门槛，并在所有通过行中具有最少变化元素1152。
- 机制证据指向`t3.norm`是本目标域的主要有效校正位置。只保留融合norm的S05三项门槛全部失败；在`t3.norm`上追加fuse的S08没有带来BA收益，且NLL比S02高0.027713。
- S10取得最低NLL，但多使用120个变化元素、选中步数为1094，且BA低于S02 3.4722pp，因此不覆盖S02的最小候选地位。S11以219步取得87.5000%BA，但参数仍多120个、NLL也高0.022242。
- 本轮结论严格限于同一60条support的4折OOF筛选，未读取独立query。S02当前状态为`SUPPORT_OOF_WINNER_PENDING_INDEPENDENT_QUERY`，不能直接替代上一轮M02的独立query结论。
- 下一科学动作应使用新的、未参与本轮筛选的独立query里程碑验证S02；工程加速则应另行验证稀疏checkpoint validation、冻结前缀embedding缓存和delta-only bundle，不与本轮结构结论混写。
