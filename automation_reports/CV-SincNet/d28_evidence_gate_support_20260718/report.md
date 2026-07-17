# D28逐样本证据门support-only实验

## 启动前记录

- experiment ID：`d28_evidence_gate_20260718/support_screen_v1`；日期：2026-07-18；operator：Codex；状态：`LOCAL_IMPLEMENTATION_IN_PROGRESS`。
- 目标：在D27-B逐新类静态安全bias上增加极小逐样本old/new证据门，优先同时提升旧类遗忘保护、新类总体与弱类floor，不扩展288D表征、原型数量或optimizer step。
- 数据工作点：receiver `20-1`、开发seed `713101`、K=10、6个old+5个seen-new、3个LEO_weak场景×5个held-rank fold；直接复用同一sealed enrollment-only support，query不打开。
- D27诊断依据：D27-B held old/new/H为67.22%/47.33%/52.82%，新support LOO约77.67%而held new仅47.33%，且75/75个逐类bias选择停在安全cap，说明静态bias不能表达逐样本证据强弱。

## 方法锁

- 基础头固定为D27-B：同一唯一接收IQ的`z160+FFT96+RF32`拼接为288D；Stage2-B 15步，Stage2-C 10步；旧weight、shared diagonal与D27逐新类bias均不由gate更新。
- 对每个support score row独立提取`E5=[n1-o1,n1-n2,o1-o2,dN-dO,(n1-dN)-(o1-dO)]`，其中`dG`为组内top-3均值；不读取query角色、truth、quota、batch class count、顺序或全局assignment。
- support标签只用于按shot-rank构造5-fold gate层cross-fit，闭式class-balanced ridge在预注册`lambda={0.1,1,10}`中选择；gate增加0 optimizer step。
- 冻结gate对每个样本计算`Delta=clip(alpha*q,-delta,delta)`并等量加到所有new列，old列bitwise不变、new-new排序不变，随后对全部注册类一次argmax。
- K=1不伪造cross-fit/LOO，gate必须禁用并退化为D27-B。
- fail closed：fold缺组、非有限、条件数>1e6、权重范数>8、有效特征方差不足或OOF old/new/floor门失败均回退D27-B。

## 资源与协议边界

- 预计D27-B 2,016活动参数、25step、约31KB状态、3,456MAC/query；gate增加6个ridge系数和10个标准化标量、少于512B数值状态，score校正约6MAC/query并含5次减法、5次除法和new列加法，总step仍25。
- 每个物理support只对应一个已经叠加的LEO_weak IQ观测；FFT96/RF32/z160只是该固定IQ的确定性数学表征，不生成额外support行，不改变K。
- `query_opened=false`；query等同测试集，只能由封存预测后的独立scorer使用。当前仍是development support-only筛选，不允许正式性能或部署声明。

## N607计划

- 远端根：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU在启动前按live inventory分配。
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d28_evidence_gate_20260718/output/support_screen_v1`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d28_evidence_gate_20260718/support_screen_v1.log`。
- 启动前要求：本地`ssr-gpu`测试、Git diff/SHA闭包、直连preflight、GPU/process inventory、远端SHA、`py_compile`、`bash -n`及output不存在门全部PASS。

## 本地实现与验证

- 新增独立核心`stage2_support_evidence_gate.py`；runner新增`d28_v1_evidence_gate`固定6候选：Z0、B3、C0、D28-A无gate的D27-B、D28-B E5/delta1、D28-C E5/delta2，共90行。
- OOF不是只拟合old/new二元标签：每个lambda都把OOF校正实际作用于D27全类score，再检查old总体、每个旧类、new总体、new class floor均不下降且至少一个新类指标严格提升；无安全lambda、退化方差、病态矩阵或权重超限均封存为disabled passthrough。
- `ssr-gpu`相邻73项测试PASS；覆盖K1精确透传、K=5合法交叉拟合、逐行独立、old列bitwise不变、new-new顺序不变、协议API、资源上限、D28真实runner fold及D25/D26/D27历史回归。`py_compile`与`git diff --check`PASS。
- Git实现提交：`d6e270993424c0e3ece37e7d5a5933e8bc350bca`；runner SHA256=`fe654bfd5f29ca675297af3144e9127c1227a93778e026634d4c99da2bbbcc96`；D28核心SHA256=`d18110786598c05d120fec4c278bf31de555a7b19fb00d3780d5a6c5f3e52e5f`；D27核心SHA256=`553d6361a728490c26963944df8353f1bc64bf1540b2ab6709f2f25bedd6f1ff`；launcher SHA256=`6faaf1feb282eff7a3999e28902cb0e01f12420b4860d4a4c5e19071aa8a8cbb`。
- 本地`stage2_diag_cosine_exploration.py`有不属于本轮的未提交修改，因此不覆盖远端；launcher继续锁定D27已验证的远端operator SHA256=`14ec919395f9bf9f13214c677b1a3d640764214668d1d00e9109f5b149ec41ca`。
- 精确远端命令：`cd /home/szu2070436088/2510044040/CV-SincNet && D28_GPU=<live-selected> bash code/scripts/launch_d28_evidence_gate_support_20260718.sh`。
- 04:23 CST直连preflight PASS：server time、project root和8张RTX3090可见，均0%利用、约10MiB显存；live inventory为`gpu_compute=[]`、`active_training_processes=[]`、`unknown_training_active=false`。选择GPU0，精确命令锁定为`cd /home/szu2070436088/2510044040/CV-SincNet && D28_GPU=0 bash code/scripts/launch_d28_evidence_gate_support_20260718.sh`。
- 已同步runner→`code/scripts/run_d25_support_only_concat.py`、D28核心→`code/cvsrffi/stage2_support_evidence_gate.py`、launcher→`code/scripts/launch_d28_evidence_gate_support_20260718.sh`；远端5项SHA与方法锁一致，`py_compile`、`bash -n`、output不存在门均PASS。同步与验证后本地`ssh.exe`及N607/bridge TCP22连接均为0。
- 04:25 CST使用GPU0启动，PID=`3645546`；log=`/home/szu2070436088/2510044040/CV-SincNet/logs/d28_evidence_gate_20260718/support_screen_v1.log`，output=`/home/szu2070436088/2510044040/CV-SincNet/runs/d28_evidence_gate_20260718/output/support_screen_v1`。首次短探针显示进程仍正常运行，未作干预；每次SSH后本地连接均清零。
- v1在16.946秒完成90/90行，D28-B/C均因OOF identity安全门15/15fold禁用并精确透传D27-B。分析发现v1的D28组合resource字段`batch1_head_latency_*`只计gate校正、漏计D27 score；同时把完整OOF审计JSON误计入星上predictor state。性能、协议、MAC与loss不受影响，但延迟/状态Pareto字段不完整。现已修复为逐样本计时`D27 score+gate correction`，并将完整OOF trace保留为外部证据、predictor只计系数/标准化/32B头；73项回归再次PASS。最终v2 runner SHA256=`685c25a34f172c17c334a10d9c45284a0fc9f0955d9fbbc294a25d85e80d64e5`，D28核心SHA256=`dd9f06bae0e8c6137fae8ebd2e14b2d0d2d33765a15815036af0ceaeb1c1db0a`，launcher SHA256=`e44dfb3cf8b6f3632ad86dcd92d9d8f63183c8ee312dc51b2254fba41dbf1d8b`；将用独立`support_screen_v2`重跑，不覆盖v1证据。
- 04:36 CST v2远端SHA、`py_compile`、`bash -n`及output不存在门PASS；GPU0启动PID=`3652104`，log/output分别为`support_screen_v2.log`与`output/support_screen_v2`。

## 完成后补充

## v2完成状态与证据边界

- v2 PID`3652104`正常退出，耗时18.085秒；90/90行、6件artifact和完整日志齐全。v1→v2性能及gate决策逐行精确重放，只有修正后的资源/源码闭包不同。
- receiver=`20-1`、seed=`713101`、K=10；每个场景110个support行=6个old TX×10+5个seen-new handle×10。old物理TX为`14-10/14-7/20-15/20-19/6-15/8-20`；new注册handle为`09f8/1c2a/b8fb/d3af/f608`。每场景110个`physical_sample_id`、parent IQ及overlay token唯一，三场景两两不重叠。
- query清单为空：`query_opened=false`、query rows/labels=0；当前指标只是support内部leave-two-rank held结果，不是正式query或独立确认性能。Phase1组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，`formal_launch_authority=false`。
- v2 artifact SHA256：training=`cb429436e80004bc98db70dd008d956e0d6b4e0d6fe805cfb177e94adcc5a1f2`、selection=`990e26fd4842ce33a31b2a92f0878ced8d0b1491e8cf3899a3c1884748f80713`、support=`acb5129e3cc8c72b5d00cfa70e9741063184d9644b40ca92054c9803593d1e69`、resource=`8902d2206198ae1dfecb8cce06026c43c9905bf8b990f446c9a6890ef2a7598d`、geometry=`f3dfcdae3ef29ee9393fff6d5d8c08437662b2e1c0ddc6ce871b1ba090fdb3c7`、receipt=`ca52aeba76d8f757a9bdb9b4a805ffd20db87408f20dc48d2f639a88df5a1f64`。
- 本地产物：`E:/type10-7/automation_reports/CV-SincNet/d28_evidence_gate_support_20260718/remote_output_v2`；所有JSON递归有限、无NaN/Inf，stdout无Traceback/OOM/Killed/Exception。下载后`ssh.exe`与N607/bridge TCP22连接为0。

## 候选联合结果

|候选|注册前old|注册后old|seen-new|H|forgetting|gate启用|判定|
|---|---:|---:|---:|---:|---:|---:|---|
|Z0|71.11%|48.33%|52.67%|48.97%|22.78pp|N/A|control|
|B3诊断|86.67%|73.33%|73.33%|72.65%|13.33pp|N/A|60step性能参考|
|C0|71.67%|50.56%|54.00%|50.35%|21.11pp|N/A|最终回退|
|D28-A=D27-B|80.00%|67.22%|47.33%|52.82%|12.78pp|无gate|本轮基线|
|D28-B E5/δ1|80.00%|67.22%|47.33%|52.82%|12.78pp|0/15|安全透传D27-B|
|D28-C E5/δ2|80.00%|67.22%|47.33%|52.82%|12.78pp|0/15|安全透传D27-B|

D28-B/C没有获得held增益，不是因为执行错误，而是15/15外层fold和3/3 full-K10场景均没有任何lambda通过OOF身份安全门；自动选择正确回退C0，`selected_positive_route=false`。

## 逐场景与逐类floor

|场景|注册前old|注册后old|new|H|forgetting|old floor|new floor|
|---|---:|---:|---:|---:|---:|---:|---:|
|clear|81.67%|75.00%|52.00%|58.12%|6.67pp|50.00%|20.00%|
|low-elev|75.00%|61.67%|54.00%|56.31%|13.33pp|40.00%|0.00%|
|rain|83.33%|65.00%|36.00%|44.03%|18.33pp|50.00%|0.00%|

注册后old逐类为`14-10=56.67%`、`14-7=66.67%`、`20-15=80.00%`、`20-19=50.00%`、`6-15=63.33%`、`8-20=86.67%`。seen-new逐类为`09f8=13.33%`、`1c2a=76.67%`、`b8fb=63.33%`、`d3af=66.67%`、`f608=16.67%`；关键失败仍是low-elev的`09f8=0%`和rain的`f608=0%`。因此下一轮必须同时处理旧类域校正和弱新类floor，不能只调全局old/new平衡。

## gate失败机理与完整loss

- 三个lambda`0.1/1/10`的角色balanced accuracy约81.26%/80.96%/80.92%，说明E5确实含old/new证据；但δ1下平均分别造成old`-12.22/-12.22/-11.94pp`，只换来new`+6.83/+6.83/+6.17pp`和new floor`+8.33/+8.33/+7.50pp`。δ2因实际校正未触及更大clip，与δ1完全相同。
- D27-B训练old support为100%，共同平移所有new列时，只要帮助弱新类越过边界，就会让部分旧类support被新类翻转；硬旧类非退化门正确阻断了这条Pareto交换。下一步不应放松安全门或继续扫δ/λ，而应把释放限制在“远离旧类安全域、接近某个新类”的逐类区域。
- Stage2-B 15fold平均loss`0.554833→0.066382`，support准确率达到100%；Stage2-C `0.727749→0.256986`，无发散。失败属于support→held的边界泛化与类条件几何问题，不是梯度步数或数值稳定性问题。

## 修正后的资源Pareto

|候选|活动参数|step|predictor state|MAC/query|适配+注册|组合head延迟|
|---|---:|---:|---:|---:|---:|---:|
|D28-A=D27-B|2,016|25|约30.91KB|3,456|67.27ms|0.0679ms|
|D28-B/C禁用gate|2,016|25|约30.94KB|3,456|约75.9ms|约0.0757ms|
|gate若启用上界|2,022|25|D27状态+96B|3,462+少量标量运算|闭式OOF|逐行、无dense图|

- 禁用gate的predictor只比D27-B增加32B启用/config头；每场景约18.5KB完整OOF trace作为外部自动化证据保存，不进入星上predictor state。
- D28-A的query MAC为identity-only单qKNN的19.64%；相对B3，活动参数少41.67%、optimizer step少58.33%、适配MAC少61.34%，但predictor state约为B3的2.11倍且性能明显更低，尚未满足“全面优于三种对比方法”的最终目标。

## 结论与D29决定

D28是有效负筛选：它证明逐样本E5角色证据存在，但“所有new列共同平移”没有旧类零退化的可行点。D29转向极轻类条件安全释放：只对当前新类候选做受旧类安全域上界约束的正向残差，并用新类support亲和度决定释放；弱新类补偿仍受相同旧类上界约束。K=1继续精确退化D27-B，query接口仍不得接收标签、角色、quota或batch统计。
