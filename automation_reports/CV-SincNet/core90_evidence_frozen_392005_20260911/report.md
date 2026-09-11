# CORE90证据头复验与N607发布

日期：2026-09-11。Run ID：`core90_evidence_frozen_392005_20260911`。唯一launch owner：本任务主Agent。

用户授权：再次核对设计报告和已接受计划，验收通过后发布实验；seed固定392005；允许突破每GPU两个实验的限制。保护现有健康进程和全部历史产物。

## 验收结论

代码/发布正确性验收通过，适用于“从零H0→同一冻结表示H1–H4”的第一阶段。设计追踪的12项均有实现及技术验证；R04/R06的物理含义仍是未验证receiver proxy，R10的H5代码已测但正式实验按原计划推迟；不把这三项解释为真实物理机制或H5必要性已证实。此处是验收边界，不是以实验性能作启动门槛。

复查修复：

1. H4输入按TX排序时可能产生单类别episode，查询数非零但CE恒零。现按seed确定的类内洗牌和类间交错消费全部L_s记录，loss和物理ID同步排列，单类别episode不计有效查询，激活检查要求有效竞争查询。
2. “模型不匹配”改为全部已注册类都超出条件一致性范围，而不是只检查最高logit类。
3. 域外support与query使用相同额外方差，避免clamp后虚增support精度。
4. 实际数据契约增加并校验raw/equalized、seed、TX映射、source/target日期和比例；equalized通过`eq_list[eq_i]`核对物理映射。
5. H0训练内部目标评估显式委托本次runner；所有候选source拟合和V校准完成、模型冻结之后，才执行目标预测。全部候选预测先保存，再由独立scorer连接truth。
6. 增加真实source-only评估、三种强基线冻结参数导出、目标四场景预测、按RX/day/TX及RX×TX评分、risk-coverage、配对rescue/harm与资源记录。

独立P0/P1审查分别覆盖数学/判决、生产路径激活、发布顺序/进程归属，发现项已定点修复和复验，无未解决发布阻断。最终聚焦验收31项及相邻终态回归113项通过，合计144项、0失败、0跳过；证据：`local_artifacts/core90_release_acceptance_392005.xml`和`local_artifacts/core90_release_terminal_regression_392005.xml`。此外本机H4的B256/C6/D160前反向约5.85秒、峰值CUDA分配2347MiB，只作为合成技术容量证据。

## 冻结矩阵和预算

|顺序|候选|数据与训练|预算|GPU|
|---|---|---|---|---|
|1|H0当前协议CORE90重建|按本次契约从零，不加载历史checkpoint或外部teacher|200epochs，batch128，原CORE90方法日程|物理GPU2|
|2|H0评估、H1、H2、H3、H4|共享上述同一个合规H0；H1–H4冻结全部backbone，仅L_s拟合头|H1–H4各20epochs、batch64、lr0.001；所有seed392005|依次绑定GPU0/1/2/3/4，并行source阶段|
|2|cosine、共享对角Gaussian、linear-ridge|H0相同特征，由L_s闭式拟合；只冻结分类器weight/bias|固定默认ridge0.01，无目标调参|随H0 source评估生成|
|3|上述8个固定候选|全部source校准冻结后，预测目标clean及三LEO弱场景|每个物理样本、每个Phase1场景固定seed；batch64|最多5个新增进程，GPU0–4分两批|
|4|独立评分|8候选×4场景预测全部保存后连接truth|一次评分，结果不回流选择/调参/重跑|CPU|

H5本次不启动：原报告和计划要求先由source H3/H4错误分析证明非线性残差必要性。联合训练也不跳过冻结比较阶段。H0阶段新证据头未启用是明确对照；H1–H4在后续阶段排队，不能把它们写成已经运行。

H4本次的正式目标评估属于固定类别Phase1读出比较；其source support episode真实训练，未见TX注册数学/接口已测试，但本轮目标评估不构成未见TX或target K-shot注册证据。后续Stage2需使用其合法capsule和注册类设计，不能把本轮固定类别重命名为新类。

## 数据契约和checkpoint来源

数据：N607的`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，equalized=1。source RX=`[1,3,4,6,8]`、day=`[1,2,3]`；target RX=`[0,2,5,7,9,10,11]`。现有day/RX builder的heldout-day参数为`0`，其`seen_day_unseen_rx`及`unseen_day_unseen_rx`联合覆盖target day0–3，不能把所有日期同时传作heldout而清空训练日期。

角色比例L_s/U_s/V=0.07/0.63/0.30，source监督训练比例为0.1；单一V，不拆V_cal/V_select。实际物理ID、TX映射和数量由本次loader产生并写入`data_contract.json`/`data_summary.json`，不凭预估数启动。H0argv为from_scratch，无外部teacher；同run EMA保留。H1–H4只加载本次H0的地面安全导出，并严格比较同一契约及上游/选模记录。旧joint_safe产物不复用。

模型选择为固定最终轮次；target在所有source拟合/校准结束后才访问。quality只由L_s受控退化拟合，state为接收统计代理；V仅作显式校准及描述性source诊断，不能更新模型参数/normalization。source校准与同V评价不视为独立风险保证。

## 环境、命令及产物

已验证普通身份`szu2070436088@dell-DSS8440`，使用项目固定direct SSH配置，不使用管理员或bridge。远端解释器`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，Torch2.1.0+cu121、NumPy2.2.5；数据转换已有NumPy C-API兼容fallback，最终以远端真实source checkpoint smoke为准。

本地Git分支`codex/core90-evidence-head-20260911`。发布代码commit与release目录在落地记录补充；归档只传本次Git的`code/`，一次SHA比较和一次远端编译。数据及旧checkpoint不传输。

预登记执行命令（`RELEASE`替换为实际不可变目录）：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u RELEASE/code/scripts/run_core90_evidence_experiment.py --root /home/szu2070436088/2510044040/CV-SincNet/runs/core90_evidence_frozen_392005_20260911 --dataset /home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl --seed 392005 --train-gpu 2 --head-gpus 0,1,2,3,4
```

CWD为RELEASE。根目录排他创建；实际子进程argv、PID、GPU、log位置保存在`commands.jsonl`。状态保存在`status.json`。预期产物：H0/final_ssdg.pth、各候选ground_checkpoint.pt/deployment.pt、source校准/探针/强基线、各候选四场景tensor-only预测、独立metrics及comparison.json。记录seed、覆盖、floor、NLL/Brier/ECE、risk-coverage、rescue/harm、拟合/预测时间和存储量；不以单seed晋级。

## 技术停止与恢复

只因本run的协议越界、契约不一致、无法执行、数值失败、输出冲突、缺少合法预测闭合或明确系统异常阻断后续阶段；低准确率、负收益不停止。无自动重跑、无旧checkpoint回退。失败保留本root所有产物和traceback，不影响其他任务；若必须终止，只针对核实PID/CWD/argv的本run进程。相同异常不盲目重复发布。

状态推进：LOCAL_VERIFIED→LANDED→RUNNING→ARTIFACTS_COMPLETE→ANALYZED。首批启动后核实PID/PPID/CWD/argv/GPU/log增长。E1只证明开始训练，E200及预测/独立评分完成前不宣称实验完成。

## 落地与启动读回：VERIFIED

- 发布代码：`edca15a6d79187872190a5008a58250a8da9d015`；release为`/home/szu2070436088/2510044040/CV-SincNet/releases/core90_evidence_392005_edca15a6`。本地/远端归档SHA256均为`05d2e03965932508f554b41c196e89f2bbb56f477b05108911fe926e7650e706`，远端compileall通过。
- 普通账户独占dispatcher由`f04a121b`记录。Coordinator PID=1342406，H0训练PID=1343028、PPID=1342406；`/proc/1343028/cwd`为上述release，`CUDA_VISIBLE_DEVICES=2`，GPU进程表确认约3046MiB。本次没有停止旧任务。
- `data_summary.json`实际数量：L_s=6300、U_s=56700、V=27000、target=168000，source/target接收机及day均与预登记一致。启动metrics实际seed=392005、from_scratch=true、baseline_ckpt为空；外部teacher为none。单一V的两个历史兼容名称指向同一组27000物理记录，不是额外划分。
- source checkpoint保存/重载/前向smoke通过，日志明确其不作为正式训练初始化。H0正式状态`H0_TRAINING`，已完成E2/200；日志由6200字节增长至17445字节。E1/E2实际train_loss分别约16.9309/16.3173，未发生nonfinite loss；E1有1/49个batch的非有限梯度跳过，E2为0，训练继续且无系统失败。保留该现象，不把“成功启动”写成长期数值稳定证明。
- N607环境没有pytest，未安装或修改环境；改用本地版本`60d7bf61`的独立、无需pytest的source-only检查脚本。单文件SHA核对后运行，`remote_head_runtime.json`读回PASS：H1/H2/H3/H4实际前向及反向梯度有限，H4有效support查询12，冻结骨干未更新。仅24个L_s样本、独立scratch技术夹具，未修改正式训练权重或初始化。
- H1–H4正式拟合仍等待H0 E200；H5保持未启动。当前是RUNNING，不是ARTIFACTS_COMPLETE，也没有目标确认指标。

随机种子说明：本次实验/数据划分/模型及冻结头seed为392005；CORE90内部固定的增强子流常量仍按历史方法保留（启动日志`concat_sat_seed=2027`），不是加载其他seed模型。目标确认的逐物理样本信道子流由本次392005及scene/opaque ID确定，对全部候选相同。

本地启动证据：`data_summary.json`、`remote_head_runtime.json`和`startup_metrics_epoch.jsonl`。后者是启动时E1–E2快照，完整日志继续保存在远端run/H0。后续结论须读取完整可用产物，不能由此快照推断最终性能。


## 完整结果：2026-09-11

全部32组目标预测与独立评分于12:09完成，状态ARTIFACTS_COMPLETE。H0完整200轮日志已读取，17份执行日志均无Python traceback。详细数据见[detailed_results.md](detailed_results.md)、[metrics.csv](metrics.csv)及[grouped_metrics.csv](grouped_metrics.csv)。全部2784个分组的计数、加权准确率和相对H0配对净变化已交叉核对。H1–H4四场景闭集准确率均低于H0；cosine仅小幅描述性改善。单seed不晋级，不回流目标调参。H5保持未启动。
