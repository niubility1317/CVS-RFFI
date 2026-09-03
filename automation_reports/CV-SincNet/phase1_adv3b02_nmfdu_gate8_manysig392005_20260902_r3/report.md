# ADV3B02-NMFDU Gate8 ManySig392005 r3实验预登记

- run ID：`phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3`
- 修复代码提交：`aa0eaf4ba3a63c88cae6147e542bb0d6b69e36e9`
- 失败前序：r1为CUDA Half线性求解技术失败，r2为非MUSE路径RC4遥测未初始化技术失败；二者均为`NO_PERFORMANCE_RESULT`
- 候选矩阵：E1=`equal`、E2=`i_only`、E3=`i_d`、E4=`i_d_s`、E5=`physical_fixed`、E6=`physical_full`、E7=`full_no_null`、E8=`full`
- 八行均使用`physical_gate_variant=nmfdu_v1`，不包含ADV3B02基线对比
- 数据：ManySig equalized=`1`；source RX=`1,3,4,6,8`、day=`1,2,3`；target RX=`0,2,5,7,9,10,11`、day=`0,1,2,3`；TX=`0–5`
- source协议：pool=`90000`，`L_s/U_s/V=6300/56700/27000`；split/train seed=`392005`
- target评估：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`，每场景168000；target不参与训练或选模
- 训练：epochs=`200`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`、`checkpoint_selection=final_only`
- release归档：`E:\type10-7\local_artifacts\releases\adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b.tar.gz`
- release SHA256：`6d89477c7374c048d808521b8e1528eac38c6ba71e5474ad8faaff36f6afa4e0`
- release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b`
- N607 Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3/{E1,E2,E3,E4,E5,E6,E7,E8}`
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3/{E1,E2,E3,E4,E5,E6,E7,E8}.out`
- GPU：E1–E8分别使用GPU0–7；用户已授权本实验谱系的`MAX_ACTIVE_PER_GPU=4`
- 启动命令：`env ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl RUN_ID=phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3 RUNS_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_nmfdu_gate8_manysig392005_20260902_r3 MAX_ACTIVE_PER_GPU=4 bash /home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_nmfdu_gate8_manysig392005_aa0eaf4b/code/scripts/launch_phase1_adv3b02_nmfdu_gate8_manysig392005_20260902.sh`
- 停止规则：仅在数据/query越权、错误split/RX/day/seed/场景、输出覆盖、错误checkout、进程归属不清、无prediction闭合或至少两行出现同一确定性系统异常时停止；不得因低性能停止
- 预期artifact：每行最终checkpoint、训练日志、clean及三个LEO弱场景独立结果、prediction与独立scorer同row指标
- 本地验证：CUDA autocast回归、非MUSE零RC4遥测回归、关键模块`py_compile`通过；NMFDU+FastTrust聚焦套件`119 passed`；两次修复各自的一次独立P0/P1定点审查均`PASS`
- N607归档独立读回：远端SHA256=`6d89477c7374c048d808521b8e1528eac38c6ba71e5474ad8faaff36f6afa4e0`，与本地归档一致；新release关键Python模块编译和launcher原生`bash -n`通过
- N607真实checkpoint无query smoke：`PASS`；严格加载195个state tensor，missing/unexpected均为0；初始化52个NMFDU state key，得到23个有限非零梯度；source RX/day/equalized/split seed匹配，query truth和Phase2访问均为`false`
- N607定点技术smoke：`NMFDU_R3_N607_TECHNICAL_SMOKE_PASS`；CUDA autocast下Fisher/相位线性代数保持FP32且有限，非MUSE空route返回完整零RC4遥测
- 启动前GPU计算进程数：GPU0–7=`1/2/2/3/2/3/3/2`；加入E1–E8后预计=`2/3/3/4/3/4/4/3`，不超过用户授权的每GPU最多4个训练进程
- 当前状态：`RUNNING`

## 启动读回

- N607启动时间：`2026-09-02 01:30:49 +0800`；dispatcher PID=`3921425`
- 行PID/GPU：E1=`3921488/GPU0`、E2=`3921485/GPU1`、E3=`3921500/GPU2`、E4=`3921503/GPU3`、E5=`3921497/GPU4`、E6=`3921494/GPU5`、E7=`3921506/GPU6`、E8=`3921491/GPU7`
- 8行均读回`[NMFDU-LANDED]`；CWD=`/home/szu2070436088`，cmdline均绑定新release、对应不可覆盖run root、ManySig路径、split seed=`392005`及各自NMFDU消融模式
- GPU读回包含上述8个PID；启动后GPU0–7计算进程数=`2/3/3/4/3/4/4/3`，未超过用户授权上限
- 8份日志均已产生并完整打印配置、数据协议、训练和遥测路径；延后检查时8个进程持续运行且CPU/GPU有活动，仍处于首个epoch前的数据初始化阶段
- 异常扫描：未发现`Traceback`、`RuntimeError`、`UnboundLocalError`或OOM；尚无性能结果，不作科学结论

## 2026-09-02 10:09运行中阶段性分析

### 结论与交付状态

- 实验尚未跑完，最高可信状态仍为`RUNNING`，不能标记为`ARTIFACTS_COMPLETE`或`ANALYZED`。
- E1–E8的8个训练进程均存活，进程状态为`Sl/Rl`，CPU和GPU持续活动；本次只读检查未停止、重启或热修改任何进程。
- 当前每行仅有连续增长的`metrics_epoch.jsonl/csv`；尚无最终`.pth`checkpoint、target clean/三个LEO弱场景结果、prediction或独立scorer结果。
- 因此，本节所有准确率均为训练过程中的source validation诊断值，不是target test结果，不能据此作最终排名、晋级或设计有效性结论。

|交付层级|当前状态|证据|
|---|---|---|
|训练进程|`RUNNING`|8/8个绑定PID存活且日志增长|
|训练指标|部分完成|8行共351条连续epoch记录，JSONL与CSV计数一致|
|最终checkpoint|未完成|快照中8行均不存在最终checkpoint|
|target四场景评估|未开始|clean及3个`leo_*_weak`结果均不存在|
|prediction与独立评分|未开始|prediction、truth-last scorer结果均不存在|

### 数据与协议状态

- 实际source配置保持为ManySig equalized=`true`、RX=`[1,3,4,6,8]`、day=`[1,2,3]`、TX=`[0,1,2,3,4,5]`、split seed=`392005`。
- source pool=`90000`，训练实际使用`L_s=6300`、`U_s=56700`、统一验证集`V=27000`。用户给出的`V_cal=13500`和`V_select=13500`在当前`项目.md`协议下合并为一个不拆分的30%验证集，样本总量未改变。
- target保持RX=`[0,2,5,7,9,10,11]`、day=`[0,1,2,3]`，每场景168000条，场景为clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；当前训练日志没有target指标，未发现target参与训练或选模的证据。
- 8行均为用户指定的NMFDU消融，不含ADV3B02基线对比。

### 运行进度与各行最新数据

快照时间为`2026-09-02 10:09 +0800`。各行完成epoch不同，下面只能用于健康检查，不能直接横向排名。

|行|模式|最新epoch|进度|train loss|train TX Acc(%)|source clean Acc(%)|source LEO mean(%)|source LEO floor(%)|source HMean(%)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|E1|equal|57|28.5%|8.7573|63.6958|97.0037|54.5025|53.2519|68.7579|
|E2|i_only|44|22.0%|8.6899|63.4247|96.4185|51.0728|49.6889|65.5809|
|E3|i_d|43|21.5%|7.7301|68.9892|96.7296|53.7617|52.3593|67.9419|
|E4|i_d_s|36|18.0%|10.3294|53.3163|96.3222|51.4642|50.3370|66.1203|
|E5|physical_fixed|48|24.0%|8.4333|64.7800|96.2852|51.5617|50.3926|66.1594|
|E6|physical_full|41|20.5%|8.5735|63.7915|96.5519|49.9617|48.9370|64.9529|
|E7|full_no_null|38|19.0%|8.2624|64.3973|96.2259|52.8716|51.8000|67.3464|
|E8|full|44|22.0%|8.7140|63.4726|96.4333|51.1383|49.7741|65.6585|

从epoch1到各行最新epoch，train loss下降8.42–11.02，source clean Acc提高56.72–57.49个百分点，source LEO floor提高22.36–26.69个百分点，说明主训练过程正在收敛。E2、E5、E6、E8的最新source HMean比各自过程最优低2.20–2.77个百分点，属于目前可见的验证波动；没有持续发散或训练崩溃证据。

### 共同epoch公平横比

当前所有行共同完成的最新节点是epoch36。此时8行source HMean范围为65.9530%–66.1203%，极差仅0.1673个百分点；各行几乎相同符合NMFDU Stage1统一路由的预期，不能用这一节点宣称某个消融领先。

|行|模式|train loss|train TX Acc(%)|source clean Acc(%)|source LEO mean(%)|source LEO floor(%)|source HMean(%)|
|---|---|---:|---:|---:|---:|---:|---:|
|E1|equal|10.3291|53.3482|96.2963|51.4284|50.1852|65.9831|
|E2|i_only|10.3202|53.2685|96.2778|51.4765|50.2111|66.0011|
|E3|i_d|10.3166|53.3482|96.3037|51.4593|50.3074|66.0903|
|E4|i_d_s|10.3294|53.3163|96.3222|51.4642|50.3370|66.1203|
|E5|physical_fixed|10.3042|53.3642|96.3185|51.3802|50.1444|65.9530|
|E6|physical_full|10.2982|53.3642|96.2778|51.4420|50.2370|66.0235|
|E7|full_no_null|10.2870|53.4439|96.2741|51.4494|50.2370|66.0226|
|E8|full|10.3142|53.4120|96.2926|51.4259|50.1852|65.9822|

`source HMean`是source clean Acc与三个source LEO验证场景中最低准确率`source LEO floor`的调和平均，不是target HMean。

### 损失函数与NMFDU状态诊断

- 当前8行都处于NMFDU Stage1，尽管外围调度字段显示`S2_stabilize_aux`。Stage1持续到epoch80，当前可见NMFDU entropy均为`1.609438≈ln(5)`、null mean=`0`、`q_labeled=1`、`q_unlabeled=0`。
- 这表示5个物理分支当前采用等权稳定训练；`route/phys/fused_pair/branch_pair/null_cal/balance`损失均为0，仅`branch_aux`工作。最新`branch_aux`为10.59–10.97，乘权重0.2后得到NMFDU总项2.12–2.19，数值关系一致。
- 因此，Fisher可辨识性门控、物理路由和null拒绝目前尚未进入真正发挥差异的阶段。当前8行接近不是实现失效证据，而是预定Stage1行为；必须至少跨过epoch80后再检查门控分化。
- 以E1最新epoch为例，总loss=`8.7573`。主要原始项包括TX分类=`3.7787`、domain=`1.1803`、adversarial=`2.5243`、group CE=`5.5544`、NMFDU总项=`2.1226`；总loss使用各项预设权重后的贡献求和，因此不能把这些原始项直接相加。
- `train_loss_sat_cls_labeled=0`当前是预期行为：`concat_sat_ce_only`的卫星分类目标从epoch80才启动，届时会增加一次卫星视图前向。`train_loss_unlabeled=0`和`q_unlabeled=0`也符合epoch131才进入伪标签阶段的调度。
- 日志文本中的`nan`仅出现在尚未激活的诊断字段，例如`sat_cos`、`DM-ACCEPT p95`、`proxy_vaccept`和尚未执行的target joint指标；结构化JSON将其记录为`null`。351条结构化记录中没有数值型NaN/Inf，不能把这些占位符解释为loss已经发散。

### 数值稳定性异常

- 8行在epoch`1,3,6,7,14,15,19,22,32`均触发过`train_skipped_nonfinite_grad`，单epoch最高跳过比例为`0.102041`，约等于49个step中跳过5个。
- `train_skipped_nonfinite_loss`始终为0；最新epoch的`train_optimizer_step_applied=1`。从epoch33到当前快照没有再次出现非有限梯度跳过。
- 判定：这是需要保留在最终报告中的早期数值稳定性异常，但保护逻辑已将异常梯度隔离，当前没有持续复发、loss非有限、OOM或进程退出证据。故实验可以继续运行，不能表述为“完全没有异常”，也不构成当前技术停止条件。

### 为什么耗时长

1. 每个epoch都执行重型source验证。验证集有27000条，且每行每epoch评估clean加3个LEO弱场景，相当于108000个验证样本视图；8行×200epoch合计约1.728亿个source验证视图，这是主要固定开销。
2. GPU存在并发竞争。实验启动后GPU0–7计算进程数为`2/3/3/4/3/4/4/3`。负载较轻的E1近期epoch中位耗时约405秒，其余共享更重的GPU约634–649秒，慢约58%–60%。
3. NMFDU同时训练5个物理分支并计算Fisher相关统计，本身比单分支模型更重。
4. epoch80后，`concat_sat_ce_only`开始卫星视图第二次前向；epoch131后伪标签阶段还会为每个labeled step处理unlabeled batch。因此后半程不会维持当前Stage1的单位epoch速度。

### 修正后的预计完成时间

此前31–35小时是按当前Stage1速度作线性外推，忽略了epoch80后的卫星第二前向和epoch131后的伪标签阶段，属于低估。按当前最慢行和现有GPU竞争作分段估计：

- 跑到epoch80：约7.5–8小时；
- epoch80–130：约12.5–15小时；
- epoch131–200：约25–31.5小时；
- 最终target clean+3个LEO场景评估、prediction与独立评分：约2–6小时。

从本快照起，完整闭合的较现实估计为约47–61小时，即香港时间约`2026-09-04 09:00–23:00`完成。该估计假设GPU竞争和单epoch耗时大体稳定；若同卡其他任务提前结束，时间会缩短。当前不建议为了提速停止或改配，因为那会破坏已冻结实验的同row可比性。

### 阶段性数据文件

- `live_summary_20260902_1009.json`：机器可读的状态、完整性、异常和当前摘要。
- `live_rows_20260902_1009.csv`：每行最新指标、过程最优、资源与artifact闭合字段。
- `live_curves_20260902_1009.csv`：8行共351条逐epoch曲线数据。
- `common_epoch_36_20260902_1009.csv`：共同epoch36的公平横比数据。

最终详细实验报告将在8行完成训练、四场景评估、prediction和独立评分后追加；当前阶段不作候选晋级判断。

## 2026-09-03 09:55系统技术失败报告

### 最终状态判定

本次8行矩阵没有跑完，整体判定为`FAILED`。E2、E3、E4、E5、E6、E8在epoch80后因同一确定性AMP异常退出；E1和E7仍在运行。当前没有任何一行形成最终checkpoint、target clean与三个LEO弱场景评估、prediction或独立评分，因此本run不能进入`ARTIFACTS_COMPLETE`或`ANALYZED`。

|对象|判定|最高已证状态|证据|缺口|
|---|---|---|---|---|
|E1 equal|`PARTIAL`|`RUNNING`|PID`3921488`存活，完成epoch163|epoch200、最终checkpoint、四场景评估与评分|
|E2 i_only|`FAILED`|`RUNNING→STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|完成epoch80后AMP BCE异常|未完成epoch81–200及全部最终artifact|
|E3 i_d|`FAILED`|`RUNNING→STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|同一异常|同上|
|E4 i_d_s|`FAILED`|`RUNNING→STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|同一异常|同上|
|E5 physical_fixed|`FAILED`|`RUNNING→STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|同一异常|同上|
|E6 physical_full|`FAILED`|`RUNNING→STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|同一异常|同上|
|E7 full_no_null|`PARTIAL`|`RUNNING`|PID`3921506`存活，完成epoch133|epoch200、最终checkpoint、四场景评估与评分|
|E8 full|`FAILED`|`RUNNING→STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE`|同一异常|未完成epoch81–200及全部最终artifact|
|整体矩阵|`FAILED`|6行技术失败、2行运行中|8行均只有`metrics_epoch.csv/jsonl`|完整8行同row结果不存在|

### 故障指纹与根因

6行日志在同一位置出现：

```text
RuntimeError: torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.
```

完整traceback将异常定位到release提交`aa0eaf4ba3a63c88cae6147e542bb0d6b69e36e9`中的`code/cvsrffi/nmfdu_training.py:363`：

```python
null_cal = F.binary_cross_entropy(
    diagnostics["null_weight"].clamp(1e-6, 1.0 - 1e-6),
    null_target,
)
```

epoch80是卫星辅助CE开始生效的第一个完整epoch；epoch80结束后，NMFDU从Stage1切换到启用路由与null校准的Stage2。除`equal`和`full_no_null`外，其余6种模式都会进入null校准分支。该分支在CUDA autocast仍开启时调用概率形式的`binary_cross_entropy`，PyTorch明确拒绝这一组合，所以6行在第一个Stage2 batch上确定性退出。

E1没有进入可学习null校准，E7显式禁用null，因此两行绕过该调用并继续训练。这个分流与代码条件、异常行集合和epoch边界完全一致，根因已经闭合。

设计忠实的最小修复不是把已经softmax后的`null_weight`直接误当logit传给`binary_cross_entropy_with_logits`。应保持现有概率校准语义，将`null_weight`与`null_target`转为FP32，并在局部关闭autocast后计算BCE；同时增加一次“Stage2+null启用+CUDA autocast”的回归测试。此次请求是状态核验和报告，本轮未修改代码、未停止仍在运行的E1/E7，也未原地复用失败输出目录。

### 数据完整性

- 快照覆盖E1–E8全部8份stdout日志、8份`metrics_epoch.jsonl`和8份`metrics_epoch.csv`。
- 共解析776条epoch记录；每行epoch从1连续到最新epoch，JSONL与CSV记录数一致。
- 结构化记录中数值型NaN/Inf计数为0，日志UTF-8解码无替换字符。
- 6个失败行各有且仅有1个Traceback和1个相同RuntimeError；E1、E7没有Traceback、RuntimeError、OOM或Killed。
- 每行run目录都只有`metrics_epoch.csv`和`metrics_epoch.jsonl`。8行均无`final_ssdg.pth`，无target clean结果，无`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`独立结果，无prediction和独立score。

### 数据协议核对

本次实际参数仍是ManySig equalized=`true`、source RX=`[1,3,4,6,8]`、source day=`[1,2,3]`、TX=`[0,1,2,3,4,5]`、split seed=`392005`；source使用`L_s/U_s/V=6300/56700/27000`。target RX=`[0,2,5,7,9,10,11]`、day=`[0,1,2,3]`以及clean和三个LEO弱场景只登记为最终评估目标。当前artifact中没有target结果，也没有target参与训练、选模或反馈的运行证据。

### 共同epoch80公平对比

所有8行都完整结束了epoch80，因此epoch80是当前最后一个可作同进度比较的节点。该节点仍是NMFDU Stage1；Fisher路由和null校准尚未在结构化epoch结果中产生有效差异。

|行|模式|train loss|train TX Acc(%)|source clean Acc(%)|source LEO mean(%)|source LEO floor(%)|source HMean(%)|sat CE raw|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|E1|equal|13.6518|66.2468|96.6778|53.4914|52.6444|68.1687|7.8832|
|E2|i_only|13.6420|66.1511|96.6815|53.6160|52.6556|68.1789|7.8791|
|E3|i_d|13.6289|66.3265|96.6593|53.5321|52.5815|68.1113|7.8777|
|E4|i_d_s|13.6456|66.1990|96.6704|53.6012|52.6741|68.1917|7.8836|
|E5|physical_fixed|13.6317|66.1990|96.6630|53.5938|52.6222|68.1463|7.8815|
|E6|physical_full|13.6623|65.9439|96.7148|53.6519|52.7630|68.2772|7.8925|
|E7|full_no_null|13.6401|66.3265|96.6667|53.5420|52.6185|68.1442|7.8844|
|E8|full|13.6436|66.3903|96.6667|53.6309|52.6296|68.1535|7.8786|

epoch80的source HMean范围为68.1113%–68.2772%，极差0.1659个百分点。8行几乎重合，符合Stage1等权分支稳定训练；这组数值不能证明任一Fisher门控消融优于其他行。

### 两个存活行的最新诊断

|行|模式|最新epoch|train loss|train TX Acc(%)|source clean Acc(%)|source LEO mean(%)|source LEO floor(%)|source HMean(%)|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|E1|equal|163|9.4932|80.7079|98.0963|81.8358|80.5370|88.4536|
|E7|full_no_null|133|13.2912|67.3948|97.5704|58.3975|56.8370|71.8309|

两行处于不同模式和不同epoch，不能把88.45%与71.83%作为公平排名。它们仍只是source validation诊断，不是target test性能。

E1的equal消融始终保持NMFDU Stage1式等权路由：entropy=`1.609438≈ln(5)`、null mean=`0`、route/phys/balance损失均为0。E7已经进入NMFDU Stage3：entropy=`1.282402`、route loss=`1.490259`、phys loss=`0.008814`、balance loss=`0.019596`，说明非null Fisher路由已经实际激活；null loss按`full_no_null`定义保持0。

卫星辅助CE均从epoch80开始，E1与E7最新raw sat CE分别为4.9924和8.1942。伪标签路径均从epoch131开始，最新unlabeled loss约为`1×10^-6`和`9×10^-6`，`q_unlabeled`分别为0.4490和0.3673。调度实际落地与预登记一致。

### 数值稳定性更新

上一份10:09快照只能证明epoch33之前的情况；新增数据推翻了“epoch33后未复发”的时限外推。8行都在epoch74再次记录非有限梯度跳过，E1还在epoch122和137再次出现。所有行`train_skipped_nonfinite_loss=0`，结构化loss均为有限值，后续optimizer仍有更新。它们没有直接导致本次6行退出，但说明梯度稳定性问题不是只发生在训练初期，后续修复验证应保留这一独立诊断。

### 时间与下一步边界

从09:55快照按最近5个epoch中位耗时估计，E1剩余37个epoch约需9.8小时，E7剩余67个epoch约需16.4小时；这只估计两条存活训练，不包含最终四场景评估。由于其余6行已经确定性失败，原先“2026-09-04完成完整矩阵”的预计失效，当前无法给出本run的8行完整完成时间。

若继续该实验谱系，必须先在本地Git工作树修复AMP BCE、加入针对性回归测试并完成限定P0/P1复审，然后以新commit、新release和全新不可覆盖run ID重跑冻结矩阵。r3的6个失败输出必须保留，不能热补丁、原地续跑或覆盖。

### 本次数据文件

- `status_summary_20260903_0955.json`：整体状态、8行完整摘要、artifact闭合状态与故障事件。
- `row_summary_20260903_0955.csv`：每行最新/最优source指标、阶段损失、异常计数和artifact清单。
- `epoch_curves_20260903_0955.csv`：8行共776条逐epoch结构化曲线。
- `common_epoch_80_20260903_0955.csv`：最后共同epoch80的同进度比较。
- `failure_events_20260903_0955.csv`：6行同指纹故障位置、触发边界和checkpoint状态。
- 完整日志与原始metrics只读快照保存在`E:\type10-7\local_artifacts\nmfdu_r3_analysis_20260903_0955`，不作为Git仓库中的大文件交付。
