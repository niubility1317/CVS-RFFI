# CVS_META_ADAPTER_TRI_R4_V1 Task12本地验证报告

- run ID：`phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1`
- 更新时间：2026-08-25（Asia/Hong_Kong）
- 状态：`LOCAL_VERIFIED`
- 当前代码提交：`b0cbbda9e1943df930bda47ef0064b6112b80fde`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 本地工作目录：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\meta-adapter-tri-r4-v1-20260824`

## 结论

Task12本地软件链、真实ADV3B02 Core90 checkpoint迁移smoke和唯一一次P0/P1审查已经闭合，可以进入N607发布。该结论只证明代码、协议边界、真实反向传播、资源预算和artifact链可运行，不包含Phase1训练结果，也不包含DA0_REG0与DA1_REG0的性能收益。当前没有可报告的正向域适应收益。

## 冻结候选与数据边界

|候选|训练方式|步长状态|作用|
|---|---|---|---|
|P0|冻结base、step0评估|不训练|同一checkpoint控制|
|P1|随机初始化adapter|固定|随机adapter控制|
|P2|source监督adapter|固定|普通监督更新控制|
|P3|FOMAML|固定|少步元学习|
|P4|FOMAML+Meta-SGD|模块级可学习|主候选|

Phase1训练与选择只使用source角色`L_s/U_s/V_cal/V_select`。训练日固定为0、1；最终clean和三个LEO弱场景评估使用独立held-out test日2、3，并逐`physical_sample_id`验证与四个训练/选择角色无交集。每个候选分别输出`clean`、`leo_clear_weak`、`leo_low_elev_weak`和`leo_rain_weak`结果。

Phase2保持`p2_min_v1/VALIDATED_ONCE`边界，只允许合法target support进入3步更新。query在适配冻结后才打开，不更新状态；source、clean、query truth和query role均不进入适配。runner在打开query前要求外部`class_ids/prototypes`与strict bundle中封存值逐项一致。

## Fix Round1修改

- Phase1入口新增P0～P4真实执行分支；P1～P4使用独立不可覆盖子目录，launcher完成跨候选汇总选择。
- 单候选未通过clean或`Y_guard`门槛时写`SCIENTIFIC_FAILURE_NO_PROMOTION`并返回0，后续候选继续运行。
- 冻结原型改为仅从`L_s`真实encoder embedding按注册类计算均值；每类必须有样本、有限且非零。同一原型写入carrier、strict bundle和`frozen_prototypes.npz`。
- source选择新增`Y_guard`逐类floor；空`Y_guard`或缺少clean/guard证据不能默认通过。
- Target5固定为1 receiver×5 operating points×3场景=15行；Target25固定为5×5×3=75行。scorer精确检查笛卡尔积、单seed、K-shot与operating point一致性。
- no-query smoke新增`base_init/meta_bundle`两种明确状态，未改变runner config allowlist。

## 本地验证证据

1. Task12完整聚焦集合：230项测试，全部通过。
2. 邻近回归：31项测试，全部通过。
3. 生产模块和launcher静态编译：通过。
4. `git diff --check`：通过。
5. Fix Round1定点复审测试：9项通过。
6. 已知警告仅为既有`torch.cuda.amp.autocast`弃用警告，不影响本次结果。

聚焦集合覆盖episode物理隔离、三站点rank-4 adapter、严格checkpoint、objectives、inner/outer loop、P0～P4入口、Phase2 support-only适配、same-row runner、truth-last scorer和深冻结handoff。邻近集合覆盖原meta-SSL、ADV3B02 CRRA及structured late-block链。

## 真实checkpoint无query smoke

- checkpoint来源：`N607:/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 本地只读副本：`local_artifacts/task12_real_checkpoint_smoke/best_joint_safe_ssdg.pth`
- receipt状态：`REAL_BASE_CHECKPOINT_ADAPTER_INIT_NO_QUERY_SMOKE_PASS`
- legacy迁移：`rank0_legacy_shell`
- strict load：`true`
- backward次数：3
- 可训练参数：8,670/1,058,341，比例0.0081920666（0.8192%）
- query打开：`false`
- source打开：`false`
- query状态更新次数：0
- 性能结果：`null`

修复后smoke再次通过，证明新增bundle原型绑定不会拒绝合法链。该smoke使用固定技术support与同值冻结原型，不读取query，不用于估计准确率。

## 唯一P0/P1审查

首次审查：P0=NONE，P1=4项，结论为不可发布。四项分别是P0～P4/四场景链缺失、全零且未绑定的原型、错误的guard floor，以及Target5/25行数错误。

Fix Round1定点复审：

- P1-1：`RESOLVED`
- P1-2：`RESOLVED`
- P1-3：`RESOLVED`
- P1-4：`RESOLVED`
- N607发布结论：可以发布
- NONBLOCKING：200步复用固定4个训练episode，保留为后续覆盖度优化，不阻断首轮screen。
- `REJECTED_EXTRA_GATE`：NONE

## Git与交付状态

- Fix Round1提交：`b0cbbda9e1943df930bda47ef0064b6112b80fde`
- 自动push：成功
- 独立远端OID回读：与本地`HEAD`一致
- 工作树：clean
- 真实checkpoint、临时bundle、support、prototype和smoke receipt保留在本地artifact目录，不进入Git。

## 下一步

Task13将以当前`LOCAL_VERIFIED`代码为基础，写入最小N607预登记字段，执行只读资源/路径preflight，制作一个release归档并只进行一次本地到远端SHA比较和一次远端编译。随后顺序发布P1～P4，P0由每个候选训练前的冻结step0评估产生。启动后只做一次PID/CWD/cmdline/GPU/log增长核对，状态只能写`RUNNING`，不能提前写性能收益。


## Task13 N607最小预登记

- 预登记状态：`LOCAL_VERIFIED/NOT_LAUNCHED`
- release Git提交：`fe886b4e2fa4ed40aadeb617c5cdd4e50460f842`
- 候选矩阵：P0冻结control；顺序运行P1随机adapter、P2监督adapter、P3 FOMAML固定LR、P4 FOMAML+Meta-SGD。
- N607账户：普通`N607`用户`szu2070436088`
- GPU：0；preflight时GPU0显存1/24576MiB、利用率0%，每GPU并发训练数计划为1。
- release归档本地路径：`E:\type10-7\release_archives\phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1_fe886b4e.tar.gz`
- release归档远端路径：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1_fe886b4e.tar.gz`
- 远端CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1/checkout`
- 输入checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 输入WiSig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- output root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1`
- stdout日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1.out`
- Python：`/home/szu2070436088/.conda/envs/ssr-gpu/bin/python`
- 启动命令：`/home/szu2070436088/.conda/envs/ssr-gpu/bin/python code/scripts/launch_phase1_adv3b02_meta_adapter_tri_r4_v1.py --config configs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824.json --output-root /home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_meta_adapter_tri_r4_v1_s392002_20260824_r1 --python /home/szu2070436088/.conda/envs/ssr-gpu/bin/python --gpu 0`
- expected artifacts：每个P1～P4子目录中的`logs.jsonl`、`metrics.csv`、`selected_meta_bundle.pt`、`source_adaptation_curve.json`、`run_summary.json`、`p0_control_evaluation.json`、`final_checkpoint_evaluation.json`、`frozen_prototypes.npz`，以及矩阵级`candidate_matrix_summary.json`。
- 技术停止规则：仅当协议越权、错误checkout/output root、输出覆盖、无法产生规定artifact、launcher-wide故障，或至少两个候选出现相同确定性pre-artifact异常时停止该run；不得因低准确率停止。
- 当前没有启动PID、没有`RUNNING`状态，也没有性能结果。

