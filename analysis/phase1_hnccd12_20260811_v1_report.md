# Phase1 HNCCD 12臂训练实验报告

状态：`LOCAL_VERIFIED / P0=0 / P1=0 / N607_PRELAUNCH_PENDING / NO_PERFORMANCE_RESULT`

## 1.实验身份、目标与声明边界

- 实验ID：`phase1_hnccd12_20260811_v1`
- 日期：2026-08-11
- 操作方：Codex主控；N607唯一Runner待交接
- 实现commit：`b6afc5a3e19ae3146dd6afcfe8a90abff35f3cbb`
- 候选：P1-HNCCD（head-nullspace cross-covariance decorrelation）
- 对照：C为GeoSat-C `training_final_only`共同续训；G只在同一共同路径上增加固定`0.02L_HNCCD`
- 目标：检验source-L的单LEO表征中，exact-head行空间坐标与其正交残差在receiver×class条件下的线性交叉协方差，是否是影响已知类LEO稳定性和后冻结连续几何的可证伪因素
- 声明边界：本实验不训练unknown、不访问真实unknown、不构成FAR、注册授权、Phase2、Phase3或多卫星协同能力声明。进程落地、技术receipt或checkpoint生成均不等于性能结果

P1-RCMMC已经在完整42步中永久拒绝，本轮不得重命名、拼接、调参或复活RCMMC及更早机制。HNCCD是新的单一结构原语，不读取clean特征做辅助对齐，也不训练Gaussian proxy分数。

## 2.冻结机制

仅对source-known-train L中已有单LEO `feat_joint`计算：

```text
W=model.id_backbone.cls_head.head.weight∈R^(4×160)
L=chol(WW^T)
Q=W^T L^(-T)
u=T(z_leo), ||z||=0时T(z)=0
h=Q^T u
b=u-Qh
C_rc=(H-Hbar)^T(B-Bbar)/n_rc
L_HNCCD=(1/28)Σ_(r,c)||C_rc||_F²
L_G=L_base+0.02L_HNCCD
```

冻结合同：

- source receiver slot固定7个、local class固定4类，共28个cell；
- `n_rc<2`为可微零贡献，始终除28，不按occupied/active cell重归一；
- `W`必须有限、full-row-rank；AMP外FP32 Cholesky和triangular solve失败即fail-closed；
- 禁止`pinv`、epsilon、降秩或fallback；
- C辅助严格N/A/0；G只增加HNCCD；
- `0≤L_HNCCD≤1`，固定`lambda_hnccd=0.02`使原始辅助上界不超过0.02；
- 每个clear/low/rain scene首个正项批需raw-unscaled VJP：LEO `feat_joint`、shared encoder和exact `W`均finite/nonzero；clean与head bias均None或数值零；
- 公共`L_base→W`保持live；
- raw VJP后只允许一次正常AMP backward/unscale/step-update；finite和AMP-skip两支都在下一forward前释放本批所有连图root，禁止第二次backward/unscale/forward、`gc`或`empty_cache`。

## 3.权限、公平与资源

- 训练辅助只读`source_known_train L`及sealed source receiver slot；
- U零iterate/forward/loss/backward/optimizer；
- V、proxy、held、target、day、fold和score对训练、校准、选模、停止与状态均零反馈；
- C/G同一GeoSat-C warm-start、physical row/order、seed、sampler、40E、新AdamW、AMP、single-LEO、三scene和共同`L_base`；
- 无额外model/head forward、view、epoch、参数、EMA、prototype或跨批feature cache；
- 可微FP32只允许当前批`Q[d,4]`、`h[B,4]`、`b[B,d]`和逐cell `[4,d]`统计；禁止`B×d²`或`B×28×d²`；
- 每共同物理batch记录一条峰显存与step-time观测；terminal要求数量逐批闭合，观测只作资源证据，不得反选候选。

## 4.本地文件、SHA与验证

|文件|SHA256|用途|
|---|---|---|
|`analysis/phase1_hnccd_design_20260811.md`|`886a2784ad338d64ef91f2cb028925e37ddfdf7bc42bcf497823d72937f98ea0`|冻结设计与追踪|
|`code/cvsrffi/phase1_hnccd.py`|`7d2d80d69892a42a14cbd32cda82ba001652f04e0976dba765f6da430fee4a9b`|公式、绑定、VJP、AMP、receipt和terminal|
|`code/SSDG/train_ssdg.py`|`7d7674cb0782880e0efaf7cedf3f04b372e373709cc43d093fc41e3246328f2d`|共同forward、G loss、AMP、资源、图释放和终态接线|
|`code/tests/test_phase1_hnccd.py`|`25378f427a0f67037aa7b52a28985ba7bad63172c05fc168ccdd4d927164580f`|29项正负与集成测试|
|`code/scripts/launch_phase1_hnccd12_20260811.sh`|`889d998fbf7612d3c95ea0d6f0b88dc3d05c9e2bbd81c2a25c5c7fe753604c71`|冻结12臂launcher，Git mode100755|

官方Conda hook激活`ssr-gpu`后串行验证：

- `python -m py_compile`：core、trainer、test通过；
- HNCCD focused：29 passed；
- HNCCD+HSCF+RCMMC共享回归：64 passed；
- `train_ssdg.py --help`识别3个HNCCD CLI；
- launcher `bash -n`通过；
- launcher dry-run精确12行、C=6、G=6、40E=12、G `lambda=.02`=6；
- normal/overflow AMP测试均只有一次backward和一次unscale；
- 无GC saved-tensor/root释放测试通过；
- terminal资源负测覆盖空列表、少一项、布尔/负峰值、NaN/负step-time和selection反馈篡改；
- `git diff --check`通过；
- 独立actual-diff审查：`P0=0 / P1=0 / ALLOW`。

本地没有GeoSat-C 4类checkpoint原件；现有本地真实checkpoint均为6类head，禁止截取或重写后冒充严格smoke。唯一Runner必须在launch前对远端F1C真实4类checkpoint执行strict load、无query前向和HNCCD VJP smoke；失败则`launch=0`。

## 5.冻结12臂矩阵与GPU映射

|顺序|候选|fold|arm|GPU|source train TX|known-val TX|proxy TX|
|---:|---|---:|---|---:|---|---|---|
|1|F1C_HNCCD12|1|C|0|20-15,20-19,6-15,8-20|14-7|14-10|
|2|F5G_HNCCD12|5|G|0|14-10,14-7,20-15,20-19|8-20|6-15|
|3|F1G_HNCCD12|1|G|1|20-15,20-19,6-15,8-20|14-7|14-10|
|4|F5C_HNCCD12|5|C|1|14-10,14-7,20-15,20-19|8-20|6-15|
|5|F2C_HNCCD12|2|C|2|14-10,20-19,6-15,8-20|20-15|14-7|
|6|F6G_HNCCD12|6|G|2|14-7,20-15,20-19,6-15|14-10|8-20|
|7|F2G_HNCCD12|2|G|3|14-10,20-19,6-15,8-20|20-15|14-7|
|8|F6C_HNCCD12|6|C|3|14-7,20-15,20-19,6-15|14-10|8-20|
|9|F3C_HNCCD12|3|C|4|14-10,14-7,6-15,8-20|20-19|20-15|
|10|F3G_HNCCD12|3|G|5|14-10,14-7,6-15,8-20|20-19|20-15|
|11|F4C_HNCCD12|4|C|6|14-10,14-7,20-15,8-20|6-15|20-19|
|12|F4G_HNCCD12|4|G|7|14-10,14-7,20-15,8-20|6-15|20-19|

共同配置：`epochs=40`、`label_epochs=40`、`pseudo_epochs=0`、`seed=7281105`、`sat_view_seed=9281105`、`lr=0.0002`、`weight_decay=0.0001`、`batch_size=128`、`lambda_sat_cons=0.10`、`amp=true`、`max_grad_norm=5.0`、`checkpoint_selection=final_only`。每GPU最多2个训练进程。

## 6.N607发布预登记

- 仅普通账号`N607`，禁止管理员账号；
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`；
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd12_20260811_v1_b6afc5a3`；
- CWD：上述release的`code`目录；
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hnccd12_20260811_v1`；
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd12_20260811_v1`；
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd12_20260811_v1_launcher.out`；
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`；
- warm-start根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`；
- retry：`NO`；
- 启动所有权：唯一Runner；主控不得重复启动。

冻结唯一命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd12_20260811_v1_b6afc5a3/code && nohup env RUN_ID=phase1_hnccd12_20260811_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd12_20260811_v1_b6afc5a3/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hnccd12_20260811_v1_b6afc5a3/code/scripts/launch_phase1_hnccd12_20260811.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hnccd12_20260811_v1_launcher.out 2>&1 < /dev/null &
```

唯一Runner启动前硬门：

1. direct `tools\n607_ssh_preflight.ps1`通过，普通账号、项目根和8张GPU可见；
2. 从commit生成完整LF、无prefix Git archive，核对成员数、`code/code=0`、五冻结文件SHA与launcher mode；
3. 远端release/run/log/outer必须均不存在；
4. ManySig SHA与6个GeoSat-C checkpoint SHA存在且记录；
5. 远端`py_compile`、`train_ssdg.py --help`、`bash -n`、dry-run12(C6/G6)通过；
6. F1C真实checkpoint strict load：missing=0、unexpected=0、exact `W[4,160]`、前向`z=[128,160]`且finite；
7. 使用真实模型、真实F1C checkpoint和仅source-L合成metadata做HNCCD no-query smoke：positive cell存在，LEO/encoder/W raw VJP finite/nonzero，clean/bias None-or-zero，query rows opened=0；
8. GPU并发记录后才允许唯一命令调用1次。SSH超时先清理绑定本地客户端，再只读确认是否landed；不得重发。

## 7.技术健康、停止与工件

预期每臂：

- `final_ssdg.pth`；
- config、resource、heldout、training completion、terminal status；
- `phase1_hnccd_terminal_receipt.json`；
- 完整arm日志。

G臂额外闭合：

- 1200个HNCCD batch及三scene共同coverage；
- 每scene至少一个正项批与一次raw VJP；
- optimizer attempts=effective steps+raw-finite AMP skips；
- raw/material nonfinite计数为0；
- terminal连续skip为0、effective step>0；
- 每共同batch一条资源观测；
- graph-release failure stage不得出现。

预注册技术停止：

- P0协议、权限、checkout/hash、输出覆盖或receiver/class/order绑定错误；
- launcher-wide确定性故障；
- 至少2个不同arm在产生final前出现相同标准化异常指纹；
- Cholesky/full-rank、OOM、CUDA、argparse、路径/权限、SIGSEGV、零final或terminal闭合失败。

停止只依据技术健康，不读取accuracy、loss、floor或proxy表现。停止前绑定本run的PID/CWD/cmdline，仅处理本run进程并保留部分工件。技术失败记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重跑本ID。

## 8.完成后分析与Phase1条件

只有12/12臂技术闭合后，主控才读取同row性能。训练阶段先检查：

1. 无表征塌缩或HNCCD数值失稳；
2. known RX/class/day稳定；
3. 6fold C/G共同输入与checkpoint闭合；
4. G资源与AMP/图释放没有系统性异常；
5. 每fold输出真实`training_final_only` checkpoint。

训练技术闭合后才允许固定后冻结42步：

`12 clean export + 12 LEO/binding + 12 fixed400 proxy + 6 same-fold pair = 42`。

最终非补偿门保持：

- clean 6/6；
- LEO clear/low/rain共18/18四floor；
- 每fold三scene等权overall 6/6；
- global 18-cell overall；
- fixed400 proxy AUROC和u-gap逐fold严格正，6/6；
- F6重开F1–F5 raw原件；
- 任何门失败则永久拒绝HNCCD，不调参、不挑fold、不改名或拼接。

当前结果：

|矩阵|技术状态|性能状态|最终判定|
|---|---|---|---|
|F1C/F1G…F6C/F6G|待唯一N607 Runner|NO_PERFORMANCE_RESULT|PENDING|

## 9.已知风险

- exact head在训练中可能接近秩亏；本轮不加epsilon或fallback，Cholesky失败按技术故障闭合；
- 去相关可能通过head行空间迁移或残差塌缩达到低loss；这不是能力，由clean/LEO/fold/global/proxy非补偿门证伪；
- raw VJP和AMP保留图会增加瞬时显存；批尾无GC显式释放及C/G资源receipt用于验证，不用于选模；
- HNCCD不保证proxy或真实unknown改善；
- 本地缺少严格4类GeoSat-C checkpoint，远端启动前真实smoke是不可跳过的唯一外部正确性门。

