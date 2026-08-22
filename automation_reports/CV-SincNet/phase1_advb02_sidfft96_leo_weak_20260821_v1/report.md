# ADV3B02 SID-FFT96 Phase1实验报告

## 当前结论

- 状态：`REPAIRING_P0_NPZ_BOUNDARY`；S0已完成，完整独立release v2已验证；P0第二次启动完成统计后在NPZ写入边界技术失败，定点修复后仅重发P0，S1–S3继续等待合法mask。
- 本轮仅发布前端频谱可辨识性路线P0与S0–S3单seed矩阵；尚无N607性能结果，不声明优于CORE90。
- 基座固定为`ADV3B02_CORE90_SOFT_E200`，不修改Phase2/Phase3，不使用target/query，不与NTRS或CRRA组合。

## 最小预登记

- run ID：`phase1_advb02_sidfft96_leo_weak_20260821_v1`
- 实现提交：`c71dd4da67154d4210834e78c2b3ec68ac2866ce`
- 发布提交：`9a7172029a9182e1cec3be7ec49d778ad2e77f3a`
- Git分支：`codex/advb02-ntrs-v2-recovery-20260820`
- 单seed：`392002`
- Phase1源角色：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`
- source/target接收机：`R_s∩R_t=∅`
- 训练场景：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- Core90卫星策略：`concat_sat_ce_only=true`、`lambda_sat_cls=0.68`、`lambda_sat_cons=0`、卫星监督E80开始；E1–40仅clear且`p=0.30`，E41–90为low_elev+rain且`p=0.60`，E91–200为三场景且`p=0.80`。
- 最终评测：clean及三种LEO_WEAK逐场景，禁止只用聚合均值替代逐场景结果。

## 实验矩阵

|row|候选|输入掩码|训练策略|用途|
|---|---|---|---|---|
|P0|`P0_SPECTRAL_AUDIT`|无|只遍历`L_s`，统计TX散度、RX/day/LEO散度与噪声|生成固定center/phase/SID掩码|
|S0|`S0_FROZEN_CORE90`|无|不训练，仅冻结checkpoint评测|同checkpoint基线|
|S1|`S1_CENTER`|P0固定中心带|冻结ADV3B02，仅训练零初始化SID投影|中心频带对照|
|S2|`S2_PHASE`|P0固定相位带|冻结ADV3B02，仅训练零初始化SID投影|相位证据对照|
|S3|`S3_SIDFFT96`|P0多频带J-score掩码|冻结ADV3B02，仅训练零初始化SID投影|完整96维幅相SID候选|

## 环境、输入与输出

- 本地CWD：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\advb02-ntrs-leo-weak-20260820`
- N607项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 源数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 基座checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_advb02_sidfft96_leo_weak_20260821_v1`
- log root：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_advb02_sidfft96_leo_weak_20260821_v1`
- launcher：`code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh`

## 精确命令

矩阵dry-run：

```bash
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --dry-run
```

独立row发布形式：

```bash
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=P0
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=S0
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=S1
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=S2
bash code/scripts/launch_phase1_advb02_sidfft96_leo_weak_20260821.sh --only=S3
```

GPU由发布时只读preflight后通过`GPU_P0/GPU_S0/GPU_S1/GPU_S2/GPU_S3`记录；每GPU训练进程不超过2个。

2026-08-22实际分配：`GPU_P0=0`、`GPU_S0=1`、`GPU_S1=2`、`GPU_S2=4`、`GPU_S3=5`。P0与S0先行；S1–S3仅在P0的`sid_mask.npz`完整后启动。

## 本地验证与独立审查

- 聚焦测试：SID相关21项及直接稳定性测试4项，共25项通过；`py_compile`、launcher`bash -n`和`git diff --check`通过。
- 修复后P0真实小样本：只读取`L_s`，8个物理样本形成clean及3种LEO_WEAK共32个source视图，`query_input_count=0`、`target_input_count=0`，状态`VERIFIED`。
- 真实ADV3B02 checkpoint无query smoke：仅允许5个`sid_fft96.*`新键缺失，无unexpected key；`raw_sid_logit_max_abs=0`、`raw_sid_z_max_abs=0`；成熟路径可训练参数为0，SID可训练参数为41,280，梯度仅进入4个`sid_fft96.projector.*`参数。
- 唯一一次独立P0/P1审查初次发现：跨域散度未按TX条件化，可能在域间TX构成不均衡时污染`J_b`。
- 修复：每个TX内部计算跨(receiver,day,LEO view)散度，再对TX等权聚合；新增不平衡fixture。旧实现该fixture得到错误`domain_scatter=6.25`并失败，修复后通过。
- 原审查员定点复审：原P1已闭合，无残留机制，结论`READY`。未增加第二次全量审查。

## N607只读资源预检

- 2026-08-21 22:29–22:31（Asia/Hong_Kong）直连普通账号成功；服务器`dell-DSS8440`、8张RTX 3090、数据与项目控制面可见。
- 每张GPU当前均有2个计算进程，利用率约83%–99%，已达到默认上限。不得启动S1–S3，也不得终止、修改或挤占现有Phase1任务。
- release仍可按最小流程落地并完成编译/dry-run；矩阵发布状态在资源释放前标记为`READY_QUEUED`，不伪报`RUNNING`。

## N607发布结果

- 本地归档：`E:\type10-7\local_artifacts\releases\phase1_advb02_sidfft96_leo_weak_20260821_v1\phase1_advb02_sidfft96_9a717202.tar.gz`
- 远端归档：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_9a717202.tar.gz`
- 不可覆盖release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_9a717202`
- 唯一归档SHA-256：本地与远端均为`1a3281933b2879ef3077b0af386e39d976f881f7b04594aef3a3d1ca59774305`，比较结果`VERIFIED`。
- 远端解压文件数：16；远端关键Python模块编译、launcher语法检查和完整矩阵dry-run均为`VERIFIED`。
- dry-run回读固定矩阵`P0,S0,S1,S2,S3`、seed`392002`、比例`0.07/0.63/0.15/0.15`及三种LEO_WEAK场景。
- 当前8张GPU各有2个既有计算进程，P0/S0/S1/S2/S3均未启动；未创建PID、未占用额外GPU、未修改现有run。合法空槽出现后，按预登记独立row命令启动并执行一次PID/CWD/cmdline/GPU/log增长检查。

## 启动与技术修复记录

- 2026-08-22 11:41:31启动P0与S0：P0 launcher PID`841018`绑定GPU0，S0 launcher PID`841019`、子进程PID`841047`绑定GPU1；CWD均为不可变release目录。
- S0启动回读：launcher与子进程存活，GPU1显存约2,030MiB、利用率33%，`S0_FROZEN_CORE90.out`已创建。
- P0首次启动：PID退出、GPU0未占用、P0输出目录未创建；完整日志唯一异常为`ModuleNotFoundError: cvsrffi.spectral_identifiability`，无数据读取、无部分mask、无输出覆盖。
- 根因：changed-files release包含新增`cvsrffi/spectral_identifiability.py`与`SSDG/train_ssdg.py`，但缺少两个常规包的`__init__.py`，主工程包遮蔽release overlay。
- 最小修复：`code/cvsrffi/__init__.py`与`code/SSDG/__init__.py`使用标准`pkgutil.extend_path`，允许release新增模块与主工程既有模块联合发现；不改变算法、数据或训练配置。
- TDD证据：修复前`cvsrffi`与`SSDG`两个overlay导入用例均失败；修复后2项通过，SID聚焦21项继续通过，`py_compile`与`git diff --check`通过。
- 远端overlay导入进一步发现：release中的新`train_ssdg.py`要求`DEFAULT_CRRA_CHANNEL_FAMILY`，而N607旧主工程`training_controls.py`不含该符号。由此确认根因是分支级依赖闭合，不应继续逐文件补洞。
- 发布修复：保留失败release与全部日志，创建当前Git提交的完整独立release v2，使代码依赖不再回退到N607旧主工程；不改变数据、checkpoint、run ID、row、GPU、seed或科学配置。
- 完整独立release v2：本地归档`E:\type10-7\local_artifacts\releases\phase1_advb02_sidfft96_leo_weak_20260821_v2\phase1_advb02_sidfft96_full_3d29039d.tar.gz`，远端release`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_advb02_sidfft96_full_3d29039d`；归档本地/远端SHA-256均为`1f953f8f518a11957f130977cc0102f4e8deefd351df728bfc7faa6b9188678c`，4,230个文件，远端编译、导入及dry-run均为`VERIFIED`。
- 只允许重发首次技术失败且无artifact的P0；S0不重启，S1–S3继续等待P0合法mask。
- P0在release v2完成全部统计并写出JSON/CSV后，于`np.savez_compressed`失败：`center_mask`和`phase_mask`由PyTorch`.numpy()`产生，表面类型为`numpy.ndarray`，但其类对象与NumPy2.2.5当前`np.ndarray`不同；与当前NumPy数组共同进入`__array_function__`分派时触发`TypeError`。未产生合法`sid_mask.npz`，S1–S3未启动。
- 定点修复：新增`_torch_mask_to_numpy`，通过CPU `uint8`值列表重建当前NumPy类数组；只替换两个PyTorch掩码的转换边界，不改变统计、选带或训练逻辑。边界测试先失败后通过，SID及launcher相关24项聚焦测试全部通过，`py_compile`与`git diff --check`通过。
- S0状态：`ARTIFACTS_COMPLETE`。clean aggregate TX为90.1402%，Strict UDU为86.09%；`leo_clear_weak` overall/Strict UDU为78.47%/72.55%，`leo_low_elev_weak`为75.65%/69.86%，`leo_rain_weak`为75.29%/69.27%。结果仅作为同checkpoint基线，不是SID收益。
- 2026-08-22 11:39:51（Asia/Hong_Kong）再次直连复核：GPU0、1、2、4、5、6、7无计算进程，GPU3仅1个既有进程；正式run/log root仍不存在，release、ManySig与CORE90 checkpoint均可见。按上述实际GPU分配进入`LAUNCHING`，不使用GPU3，不干预既有PID335489。

## 科学停止与晋级规则

- 技术停止只允许：协议/数据角色/seed/场景错误、错误release或CWD、输出碰撞、进程归属不清、确定性同类预prediction异常至少重复两次、无法产生checkpoint或独立prediction闭合。
- 低性能不停止实验，只进入同row分析。
- S3晋级门槛：clean下降不超过0.3pp；LEO均值至少提升1pp；Strict UDU至少提升1pp；LEO floor至少提升0.5pp；`rescued>harmed`；RX probe相对下降至少20%且TX probe不下降。
- 在真实prediction与独立scorer闭合前，不把本次发布描述为性能改善。

## 预期artifact

- P0：`spectral_identifiability.json`、`spectral_identifiability.csv`、`spectral_identifiability.png`、`sid_mask.npz`
- S0：`final_eval.json`、`final_eval.txt`
- S1–S3：`final_ssdg.pth`、`metrics_epoch.csv`、`metrics_epoch.jsonl`、`phase1_terminal_status.json`、`independent_final_eval/final_eval.json`、`independent_final_eval/final_eval.txt`
- 所有完成训练的候选必须保留checkpoint身份、clean和三种LEO_WEAK逐场景结果。

## 发布后待追加

- P0完成证据、各row PID/CWD/cmdline/GPU/log增长证据
- prediction闭合后的同row指标、异常、解释与下一候选决定
