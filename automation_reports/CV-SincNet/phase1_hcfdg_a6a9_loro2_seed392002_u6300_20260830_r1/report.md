# HCF-DG V2 A6–A9 Phase1深层域泛化实验报告

## 1.报告状态与结论边界

- Run ID：`phase1_hcfdg_a6a9_loro2_seed392002_u6300_20260830_r1`
- 当前最高状态：`LOCAL_VERIFIED`。实现、聚焦测试、真实checkpoint无query smoke和独立P0/P1复审已完成；N607正式8行训练尚未在本报告当前版本中产生性能结果。
- 实现提交：`87526d9d3c39e7e05c2c616fa65bc084ee321841`，分支`codex/phase1-hcfdg-20260830`，远端OID已独立核对一致。
- 科学范围：仅Phase1 source-only域泛化。训练、选模、调参和重跑不得访问Phase2 capsule、目标接收机、support、query、truth、目标prototype或目标域统计。
- 本轮目标：在A0–A5已冻结结构证据上实现并验证HCF-DG V2的反事实环境传输、课程式环境交换、分层HDRO和内容条件LODO原型。
- 用户当前明确覆盖原三seed方案：正式A6–A9只使用`seed=392002`；保留fold1和fold8，因此矩阵为`4候选×2 folds×1 seed=8行`。
- 正式训练只在N607普通账户执行。本地只用于代码测试和真实checkpoint无query smoke，不把本地smoke当作正式训练或性能结果。

## 2.为什么从A4/A5继续到A6–A9

A0–A5正式快速筛选已在N607完成36/36行，全部为`ARTIFACTS_COMPLETE`，并完整解析144,000条update记录、144个严格评估JSON和36个final checkpoint。6行均值结果如下：

|候选|Clean|LEO clear|LEO low-elev|LEO rain|LEO mean|LEO floor|解释|
|---|---:|---:|---:|---:|---:|---:|---|
|A0|57.20|33.35|33.15|33.49|33.33|6.74|ADV3B02闭集参数控制|
|A1|57.77|34.24|34.54|34.66|34.48|6.27|单参数量控制|
|A2|60.10|35.96|35.69|35.91|35.85|8.56|单identity主干+环境因子化|
|A3|60.04|36.05|35.89|36.20|36.05|9.39|加入TX×环境矩形batch|
|A4|60.63|36.19|36.22|36.44|36.29|8.53|加入普通LODO，平均性能冠军|
|A5|59.63|36.24|36.12|36.43|36.26|7.70|加入rank-4公共—特定头，未形成边际收益|

A4相对A0把Clean提高3.43pp、LEO mean提高2.96pp，但fold8的class floor仍明显低于fold1。A6–A9因此不再继续堆叠旧ADV3B02 loss soup，而是针对“环境替换是否保持TX身份”“环境尾部风险”“内容错配原型”三个机制问题进行深层外推。

## 3.本轮实现内容

### 3.1单identity主干与反事实输出

- 保留一个`lite_d`identity backbone，输出160D`z_id`；没有创建第二套身份主干。
- 环境编码器由receiver、day、channel三个16D分量组成48D`z_env`，物理环境键只进入环境分支。
- 新增`HCFDGCounterfactualOutput`，保存反事实公共头logits、反事实`z_id`、环境预测、目标环境、传输前后融合特征、TX标签、源/目标配对索引和swap模式。
- 反事实传输发生在融合特征层，经有界低秩变换后仍使用同一个公共身份头；训练不会再跑一次identity backbone。
- 配对约束为same-TX，目标环境因子必须与源因子不同；模型不能把目标环境标签作为预测输入。

### 3.2课程式反事实环境交换

- A6：Stage3和Stage4均使用same-TX`receiver_swap`，作为单因素反事实基线。
- A7–A9：Stage3按固定update区间依次使用receiver、day、channel交换：
  - update4001–4567：`receiver_swap`；
  - update4568–5134：`day_swap`；
  - update5135–5700：`channel_swap`；
  - update5701–6300：`joint_swap`。
- 为消除正式长行中的随机中断风险，trainer在identity前向前检查当前batch是否存在same-TX异因子配对；若不存在，最多重采样64个source-only矩形batch。被拒batch不执行identity前向、不执行optimizer update。64次仍无合法配对时fail closed，不静默改变方法。

### 3.3分层HDRO

- A8和A9启用hierarchical DRO，损失权重固定为`0.10`。
- 风险组由可用的TX、receiver、day、channel及其层级组合构造；先估计细粒度组风险，再向父组收缩，重点抑制少数TX×环境单元的尾部崩塌。
- A6和A7关闭HDRO，确保A8相对A7的差异只来自分层尾部风险机制。

### 3.4内容条件LODO原型

- 仅A9启用content-conditioned LODO。
- 从当前固定IQ构造26D、长度无关、绝对增益归一化的内容键：4个幅度统计、4个短时滞自相关、8个粗频谱能量、4个一/二阶差分统计、6个局部能量及变化统计。
- LODO原型只在source support内按内容距离加权；若某类缺少合法近邻，则回退到普通类原型，不跨域打开目标信息，也不强行制造配对。

### 3.5单前向星地信道增强

- 按用户要求采用报告方案，不使用ADV3B02/ADV3B03的clean+satellite拼接双前向训练。
- 每个96样本主batch固定29个样本替换为`mixed_orbit`接收IQ，其余67个保持clean；batch大小不扩张，整个batch只执行一次共享identity backbone前向。
- 增强器同时产生channel/scenario、CFO、phase noise、SNR、multipath和elevation bin，供环境分支监督；clean样本使用独立clean channel标签。
- 每4个主更新追加一次仅环境分支的source-only辅助损失，但仍合并进该次optimizer step，不增加identity前向，也不增加正式6300 update计数。

### 3.6损失与阶段门控

冻结总损失权重为：identity CE=`1.00`、LODO=`0.40`、counterfactual=`0.15`、HDRO=`0.10`、CSD=`0.15`、factor auxiliary=`0.05`。各机制不是从update1全部打开，而是按阶段门控：

|阶段|update|主训练内容|
|---|---:|---|
|Stage0|1–700|环境预训练；TX身份不可见，不执行identity主干|
|Stage1|701–1900|identity CE+CSD，建立公共身份路径|
|Stage2|1901–4000|加入LODO、环境因子辅助；A9启用内容条件原型|
|Stage3|4001–5700|加入反事实损失；A8/A9同时启用HDRO|
|Stage4|5701–6300|联合环境swap，GRL降至0.01做末段稳定化|

Stage2/3的GRL强度为`0.05`，Stage4为`0.01`，Stage0/1为0。训练进度50%处，即update3150，冻结Sinc前端和第一个时域块。优化器为AdamW，backbone初始LR=`1e-4`，新头初始LR=`3e-4`，前5%线性warm-up后cosine下降至约`1e-6`。

## 4.数据、划分与source-only选模

- 数据文件：`Dataset_WigSig/ManySig.pkl`。
- source receiver全集：`1,3,4,6,8`。
- 训练日期：`day1/day2/day3`；day4不进入本次训练或source LORO选模。
- source角色比例：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`。
- `fold1`：heldout receiver=`1`，训练receiver=`3,4,6,8`。
- `fold8`：heldout receiver=`8`，训练receiver=`1,3,4,6`。
- 每行训练时严格排除heldout receiver；最终checkpoint只在同一source heldout receiver的day1/2/3上零适配评估。
- `U_s`仅用于廉价环境学习，不用于身份伪标签。
- source-only晋级证据为同row的Clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`、LEO mean、LEO floor和资源遥测；不得根据任何目标接收机结果反馈选种、调参、重训或选择性重跑。

## 5.正式实验矩阵

|Row|Candidate|Heldout fold|训练receiver|训练day|Seed|Updates|GPU|
|---|---|---:|---|---|---:|---:|---:|
|1|A6|1|3,4,6,8|1,2,3|392002|6300|0|
|2|A6|8|1,3,4,6|1,2,3|392002|6300|1|
|3|A7|1|3,4,6,8|1,2,3|392002|6300|2|
|4|A7|8|1,3,4,6|1,2,3|392002|6300|3|
|5|A8|1|3,4,6,8|1,2,3|392002|6300|4|
|6|A8|8|1,3,4,6|1,2,3|392002|6300|5|
|7|A9|1|3,4,6,8|1,2,3|392002|6300|6|
|8|A9|8|1,3,4,6|1,2,3|392002|6300|7|

每张GPU仅运行一个本run正式行，低于项目允许的每卡最多两个训练实验。dispatcher只负责这8行，不得接管或影响无关进程。

## 6.本地验证与独立审查

- HCF-DG聚焦测试：144/144通过，覆盖model、loss、trainer、launcher、阶段边界、formal矩阵、反事实配对和严格失败路径。
- Python编译检查：`model.py`、`losses.py`、`trainer.py`和正式launcher全部通过。
- `git diff --check`通过。
- 本地A9真实checkpoint无query smoke：4个update分别穿过Stage0/1/2/3，生成5,453,517字节final checkpoint；clean和三种LEO评估均为`checkpoint_load_strict=true`，终态`ARTIFACTS_COMPLETE`。该smoke只证明代码路径和artifact闭合，不作为正式性能证据。
- 独立初审唯一P1：day/channel课程在随机batch无same-TX异因子配对时可能抛`ValueError`并中断6300步正式行。
- 修复后定点复审：Luna结论`PASS`。确认重采样发生在identity主干前、尊重`valid_tx_mask`、所有V2非Stage0更新经过该路径、未引入query/target访问，未发现新的直接P0/P1。

## 7.N607发布与正式命令

- 账户：N607普通账户`szu2070436088`；禁止使用管理员账户。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_87526d9d`。
- 正式run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hcfdg_a6a9_loro2_seed392002_u6300_20260830_r1`。
- dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hcfdg_a6a9_loro2_seed392002_u6300_20260830_r1.dispatcher.log`。
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。

正式命令冻结为：

```bash
nohup /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_87526d9d/code/scripts/launch_phase1_hcfdg_matrix_20260830.py --formal --run-id phase1_hcfdg_a6a9_loro2_seed392002_u6300_20260830_r1 --stage deep --seeds 392002 --folds 1,8 --gpus 0,1,2,3,4,5,6,7 --code-root /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hcfdg_87526d9d --python /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python --run-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hcfdg_a6a9_loro2_seed392002_u6300_20260830_r1 --wisig-pkl /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl
```

## 8.预期artifact与完成定义

每一行必须保存：

- `metrics.csv`和`metrics.jsonl`，连续记录update1–6300及stage、loss、LR、margin、吞吐、显存和耗时；
- `final_hcfdg.pt`，含candidate、fold、seed、source split、精确update、runtime重建参数和公共推理头；
- clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`四个独立评估JSON及log；
- 严格checkpoint重建字段，要求missing、unexpected和shape mismatch均为空；
- `ARTIFACTS_COMPLETE.json`或精确`TECHNICAL_FAILURE.json`。

只有8/8行训练达到6300 updates、final checkpoint可严格重建、四场景评估齐全，才能将run标记为`ARTIFACTS_COMPLETE`；完整解析全部日志与artifact并形成同row比较后才是`ANALYZED`。训练进程自然退出但缺少任一评估，不等于实验完成。

## 9.停止规则与科学判断

只允许因以下预登记系统技术失败停止对应run：错误candidate/fold/receiver/day/seed/update，数据越权，错误release或checkout，输出冲突，命令无法运行，进程归属不清，无法产生final checkpoint/四场景artifact，或同一确定性pre-prediction异常至少在两行复现。停止前必须精确绑定该run的dispatcher和worker进程树并保留partial artifact。

低性能、负收益、fold8困难或中间准确率波动不是技术失败，不得停止、重启、热补丁或选择性重跑。正式结果返回后，仅按source-only同row证据解释A6→A7→A8→A9的机制增量；本轮单seed结果只能作为深层机制筛选，不能冒充多seed稳定性结论、目标接收机结论或Phase2性能。

## 10.结果表（正式完成后回填）

|Row|Clean|Clean floor|LEO clear|LEO low-elev|LEO rain|LEO mean|LEO floor|GPU-h|状态|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|A6-F1-S392002|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|LOCAL_VERIFIED|
|A6-F8-S392002|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|LOCAL_VERIFIED|
|A7-F1-S392002|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|LOCAL_VERIFIED|
|A7-F8-S392002|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|LOCAL_VERIFIED|
|A8-F1-S392002|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|LOCAL_VERIFIED|
|A8-F8-S392002|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|LOCAL_VERIFIED|
|A9-F1-S392002|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|LOCAL_VERIFIED|
|A9-F8-S392002|待运行|待运行|待运行|待运行|待运行|待运行|待运行|待运行|LOCAL_VERIFIED|
