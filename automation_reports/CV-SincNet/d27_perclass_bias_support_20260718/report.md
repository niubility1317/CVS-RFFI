# D27逐新类安全bias support-only实验

## 启动前记录

- experiment ID：`d27_perclass_bias_20260718/support_screen_v1`；日期：2026-07-18；operator：Codex；状态：`READY_FOR_N607_SUPPORT_SCREEN`。
- 目标：解决D26单一new-group bias在“旧类崩塌”和“新类崩塌”之间无可行点的问题，以每个新类1个独立安全bias同时优化旧类遗忘和新类floor。
- 比较：Z0、B3诊断、C0、D27-A `15+0`、D27-B `15+10`、D27-C `15+15`；6候选×3个LEO_weak场景×5个held-rank fold=90行。
- 数据与工作点：receiver `20-1`、开发seed `713101`、K=10、6个old+5个seen-new；直接复用D25/D26同一sealed enrollment-only LEO_weak support，不新增数据或信道view，query不打开。

## 方法锁

- 288D为同一唯一接收IQ的`z160+FFT96+RF32`拼接；每个物理support仍只有一个LEO_weak观测、一个support行，K不变。
- Stage2-B保持shared 288D diagonal+6个旧类weight，全批次15步；Stage2-C只训练新suffix 0/10/15步，旧weight、shared diagonal和旧raw score列冻结。
- 对每个新类`j`独立计算旧support安全上界：`b_j^safe=min_{i∈G_old}(winning_old_score_i-new_score_ij)-1e-4`。K>1只在`b_j^safe+[0,-0.5,-1,-2,-4]`内按固定类序做一次support LOO坐标选择；K=1直接使用cap，不伪造LOO。
- 选择目标词典序为`min_new_class_LOO→overall_LOO→worst_margin`；每个候选bias向量都必须保留所有Stage2-B old-only正确support行和逐旧类准确率，否则fail closed。
- 推理仍对全部注册类逐样本一次argmax，不读取query角色、类别数量、quota、排序或全局assignment。

## 本地版本与验证

- Git仓库：`E:/type10-7/github_publish/CVS-RFFI-repo`；根目录不是Git仓库，本报告根目录与Git镜像同步。
- D27核心提交：`67b9d2275782339e0ac07800652b997adbcca534`；runner提交：`00e89bb2`。
- runner SHA256：`9bb0deff5fa896da54947a7505eceb47e03a9d05d1a0b3d31490df36d0d9fd6b`；核心SHA256：`553d6361a728490c26963944df8353f1bc64bf1540b2ab6709f2f25bedd6f1ff`；launcher SHA256：`f67cbd548ff8d7c5082de1480da8e8c25976fc6e76214d9716474c5fee4b2f09`。
- 65项D27/D26/D25/C3相邻回归PASS；`py_compile`、`bash -n`、`git diff --check`PASS。
- 资源锁：正式档≤80,000活动参数、≤30epoch/step、≤256KB状态、无dense query图。D27 5类bias仅20B FP32预测状态且不参与梯度；20新类构造测试仍低于状态上限。

## N607计划

- 远端根：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU优先0。
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d27_perclass_bias_20260718/output/support_screen_v1`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d27_perclass_bias_20260718/support_screen_v1.log`。
- 精确命令：`cd /home/szu2070436088/2510044040/CV-SincNet && D27_GPU=0 bash code/scripts/launch_d27_perclass_bias_support_20260718.sh`。
- 只同步runner、D27核心和launcher；不覆盖远端实际FFT96/RF32 operator。启动前重新做live inventory、远端SHA、`py_compile`、`bash -n`和output不存在门。
- 03:57 CST直连preflight PASS；8张RTX3090均0%利用、约10MiB显存。live inventory为`gpu_compute=[]`、`active_training_processes=[]`、`unknown_training_active=false`，允许使用GPU0。
- 已同步runner、D27核心和launcher；远端SHA、`py_compile`、`bash -n`、实际FFT96/RF32 operator SHA及output不存在门均PASS；同步/验证后本地SSH/TCP22连接为0。

## 判定与风险

- 晋级同时要求15fold旧support非退化、逐场景old/new pooled floor相对C0均提升≥10pp、任一旧/新类相对C0下降不超过10pp、H与forgetting不劣于C0；B3仅性能参考。
- 主要风险：独立安全cap仍可能对`cls_09f8/cls_f608`过严，或一次坐标搜索在LOO上乐观。若D27仍失败，依据失败分解再决定小幅support保护松弛或双原型，不继续盲扫全局bias/epoch。
- 当前仍是development support-only筛选，`formal_metric_claim_allowed=false`；正路线才进入joint bundle method lock及正式独立确认矩阵。

## v1完成状态

- PID`3634239`正常退出，耗时15.897秒；90/90行与6件artifact齐全，`query_opened=false`，完整日志无NaN/Inf/OOM/Traceback。
- 最终选择C0，`selected_positive_route=false`。D27三臂均通过15/15fold及full-K10旧support非退化，但未通过双floor/逐类安全门。
- artifact哈希：training=`f9a18e7c6d6232e791b9bb7676724c7f528e4805239e760abd0a4a49a8dad031`、selection=`534acd1217bf3c14d297ea8060e0dba3223803fcc5332f5f5fbfa13c23332cba`、support=`d1516ba2e62405fffb3044d5c6576ab8ac915a44747afc7b110ad606f6088be1`、resource=`2722fdde8029c29a7e5b5d8581774070a5b5c2362d7ab78db88568dc075cdec6`、geometry=`9f840af50d7645140a7bd918bfa0952cbce5b959a8e5e5d8fae0726b0c66bcd8`、receipt=`8340fc52ac21e27250a166ef3f32b20fc130921a9857f4c0322d6a9566b22ae8`。
- 完整本地产物：`E:/type10-7/automation_reports/CV-SincNet/d27_perclass_bias_support_20260718/remote_output_v1`；下载哈希与receipt闭环，SSH/TCP22连接为0。

## 候选联合结果

|候选|注册前old|注册后old|seen-new|H|forgetting|fit-old非退化|判定|
|---|---:|---:|---:|---:|---:|---:|---|
|Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|N/A|control|
|B3诊断|86.67%|73.33%|73.33%|72.65%|13.33pp|N/A|60step参考|
|C0|71.67%|50.56%|54.00%|50.35%|21.11pp|N/A|最终回退|
|D27-A 15+0|80.00%|67.78%|44.67%|51.68%|12.22pp|15/15|H优于C0，floor门失败|
|D27-B 15+10|80.00%|67.22%|47.33%|52.82%|12.78pp|15/15|本轮联合最好，floor门失败|
|D27-C 15+15|80.00%|66.11%|47.33%|52.49%|13.89pp|15/15|额外5步无收益|

D27-B相对D26-v2把seen-new从8.00%恢复到47.33%、H从13.38%恢复到52.82%，同时旧类仍为67.22%；相对D26-v1把old从23.89%提高到67.22%，仅牺牲23.34pp new。逐类bias确实打开了单标量bias不存在的中间可行域，但仍未达到晋级floor。

## 逐场景

|候选|场景|注册前old|注册后old|new|H|forgetting|
|---|---|---:|---:|---:|---:|---:|
|D27-A|clear|81.67%|73.33%|44.00%|52.95%|8.33pp|
|D27-A|low-elev|75.00%|66.67%|52.00%|56.90%|8.33pp|
|D27-A|rain|83.33%|63.33%|38.00%|45.19%|20.00pp|
|D27-B|clear|81.67%|75.00%|52.00%|58.12%|6.67pp|
|D27-B|low-elev|75.00%|61.67%|54.00%|56.31%|13.33pp|
|D27-B|rain|83.33%|65.00%|36.00%|44.03%|18.33pp|
|D27-C|clear|81.67%|70.00%|52.00%|57.06%|11.67pp|
|D27-C|low-elev|75.00%|61.67%|56.00%|57.82%|13.33pp|
|D27-C|rain|83.33%|66.67%|34.00%|42.60%|16.67pp|

rain仍是联合最难场景；10步是当前较优Stage2-C点，15步没有稳定增益。

## 逐类与bias诊断

D27-B注册后旧类为`14-10=56.67%`、`14-7=66.67%`、`20-15=80.00%`、`20-19=50.00%`、`6-15=63.33%`、`8-20=86.67%`，old floor=50.00%。新类为`09f8=13.33%`、`1c2a=76.67%`、`b8fb=63.33%`、`d3af=66.67%`、`f608=16.67%`，new floor=13.33%。相对C0，弱新类floor从3.33%/6.67%升至13.33%/16.67%，但易新类和旧类总体仍不足。

- D27-B/C新类support LOO总体为77.67%/77.50%，LOO最差类为43.33%/42.50%；held new仅47.33%，存在明显LOO→held乐观差。
- D27-B/C的5个bias在15fold中几乎全部选择各自安全cap；25次坐标候选没有形成负offset收益。当前瓶颈不再是网格范围，而是只用静态bias无法处理逐样本old/new证据变化。
- D27-A闭式新权重的cap更负，LOO总体37.50%、最差类1.67%；新suffix训练10步是必要的，但15步没有继续收益。

## 完整loss与资源

- Stage2-B平均loss`0.554833→0.066382`，fit-old训练acc/floor=100%/100%。
- Stage2-C D27-B/C分别`0.727749→0.256986/0.222867`，新support训练acc=96.0%/96.67%、floor=83.33%/86.67%。没有收敛故障。
- 失败来自support LOO泛化与静态score校准，不是训练步数不足。

|候选|峰值参数|总step|状态|MAC/query|bias选择MAC|适配+注册|CPU head|
|---|---:|---:|---:|---:|---:|---:|---:|
|D27-A|2,016|15|30.86KB|3,456|262,080|50.29ms|0.0659ms|
|D27-B|2,016|25|30.91KB|3,456|262,080|67.75ms|0.0657ms|
|D27-C|2,016|30|30.88KB|3,456|262,080|89.97ms|0.0800ms|

每个query仅增加5个bias加法，score MAC仍为identity-only单qKNN的19.64%；5类bias向量仅20B。D27-B比B3少41.67%活动参数、少58.33% optimizer steps，资源目标PASS。

## 结论与下一研发方向

D27不晋级正式路线，但保留D27-B作为当前≤30step最优结构：它首次同时做到H高于C0、forgetting低于C0、old/new弱类floor均高于C0。下一步不继续增加epoch或扩大静态bias网格；需要用注册support交叉拟合一个极小逐样本evidence gate，根据当前样本的`max-new−max-old`、old/new组内margin等部署时可得证据动态校正new score，再对全部注册类一次argmax。该gate只能使用support标签拟合，query仍不可达，不得读取真实old/new角色或batch quota。
