# D112-SEAM-qKNN真实G0与三轮回顾报告

状态：`DESIGN_FROZEN / LOCAL_VERIFIED / G0_PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

## 1.身份与目标

|字段|值|
|---|---|
|实验ID|`d112_seam_g0_real_f2644d82_20260802_r1`|
|日期|2026-08-02|
|operator|主agent：理论集成、数据与结果分析；Terra Max子agent：bundle与score实现|
|目标|在不调参、不读truth的前提下，验证D112-SEAM-qKNN是否在真实588条strict tap上形成非零、协议合法的函数变化|
|比较目标|同fold、同support、同query的精确M0 Student-t qKNN|
|设计版本|Git`f2644d82`，`analysis/d112_seam_qknn_theory_20260802.md`|

假设：六个旧类的目标support在Phase1固定的rank-3球面切空间中含有可由其他类LOO估计的共享接收机位移。连续收缩与单位质量双专家只替换旧类support密度的一部分，不增加旧类先验质量；新类及无效旧类逐列原样返回M0。

## 2.D109–D111三轮回顾

本回顾在D112实现／实验发布前完成。项目conversation index已于2026-08-02刷新至1238条记录，并以`D109 D110 D111 域适应 负收益 零功能 轻型`检索；正式判断仍以本地实验报告和完整artifact为准。

|轮次|真实性能／功能证据|结论|D112吸收|
|---|---|---|---|
|D109-SCRC|仅`LOCAL_VERIFIED`，未发布、无真实性能结果|用户转向理论优先后停止，不把实现完整度当性能|D112先冻结生成模型、量纲和可证伪边界|
|D110-SCPM＋US-qKNN|真实G0三K有23/40/96个argmax变化；source-held G1中DA_AT_BASE的H−2.7957pp、old BA−1.5873pp、seen-new−1.5873pp、old floor−10.0529pp|有功能不等于有收益；低方差维放大会强化receiver特异方向|拒绝对角逆方差放大；只用跨类共享方向，donor不一致连续降权|
|D111-r2|真实588行G0三K的anchor/rho/score/margin/argmax全部零；0/168类状态有正anchor质量|硬资格下界与单位球运输上界矛盾，结构性全回退|删除`chi/6B/eta/phi_max`性能gate，以Log/PT/Exp和闭式连续收缩替代|

回顾裁决：不回修D109、D110或D111，不扫描其参数，不挑receiver、K或局部正行。D112必须同时保护旧类适应和新类注册：新类保持M0单位质量，旧类收益只能由同row G1的old/new/H/floor/forgetting共同确认。

## 3.协议与方法锁

- `protocol_schema=p2_min_v1`；只读不可变Phase1 bundle和当前row合法support。
- query逐样本面对全部注册类；query fit/update/selection均为0；无truth、role、quota或global reassignment。
- Phase1资产为共同封存int8聚合：`g/q0/U/sigma0^(r)/sigma0^amb/vg^(r)/vg^amb/tau_h^(r)`及量化尺度／receipt；不含source行、单样本feature、ID、路径或可替换sidecar。
- 固定`p=160,r=3`。rank方差只用于`alpha`；ambient endpoint chord-MSE只用于`rho`；通过`R_pi→Exp`的闭式Jacobian迹做一阶delta-method换算。
- 无donor、row几何无效、bundle无效和非旧类均直接复制已有M0列；继承的M0锁必须`kernel_volume_gamma=1`，D112不得覆盖`nu/h_c/d_eff`。
- 不从G0选择rank、prior、rho、核、阈值或候选；不运行125参数矩阵。

## 4.实现、验证与Git状态

|项目|当前值|
|---|---|
|理论文件|`analysis/d112_seam_qknn_theory_20260802.md`|
|理论commit|`f2644d82`|
|理论复审|两名独立审查：`P0=0/P1=0 / MERGE / DESIGN_FROZEN`|
|bundle实现|`code/cvsrffi/stage2_d112_seam_bundle.py`；typed只读资产、content root绑定checkpoint/source/权限、formal fail-closed|
|source builder|`code/cvsrffi/stage2_d112_g0_source_bundle.py`；receipt绑定的588行Phase1聚合，先int8量化再构造bundle|
|score实现|`code/cvsrffi/stage2_d112_seam_qknn.py`；Log/PT/Exp/R_pi、strict LOO、Jacobian、ambient rho、unit-mass score|
|实现commit|`b33c4ccf`|
|G0入口|`code/scripts/run_d112_seam_g0_one_shot.py`；commit`02092316`|
|focused tests|`tests/test_stage2_d112_seam_qknn.py`、`tests/test_run_d112_seam_g0_one_shot.py`；`ssr-gpu`中14项通过；入口及三个实现文件`py_compile`通过|
|真实source smoke|固定tap＋receipt＋checkpoint绑定通过；`global_bundle_valid=true`、reason=`NONE`、`tau_h^(r)=0.0029421325`|
|实现级独立复审|首轮`P0=2/P1=4`已修；次轮`P0=0`且剩余3项P1已修；最终`GO / P0=0/P1=0`|

真实source smoke还得到：`sigma0_r=0.006759–0.071533`、`sigma0_amb=0.000408–0.002467`、`v_g_r=0.007699–0.081483`、`v_g_amb=0.000252–0.002284`。这些只证明Phase1资产非退化，不是query功能或性能结果。source aggregate SHA256为`348bc924785a26d72b76e394c00b6cc0b3a9fcdf9d8e5d7544e365ec245bdb70`，bundle content root为`461fe3337bd5c5698fa3e591be5ea1ad872fc63f237711f54c58f10414bd94d6`。

## 5.最小G0预登记

|字段|冻结值|
|---|---|
|输入|D106真实588行strict tap|
|输入SHA256|`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`|
|结构|28个receiver×day fold、6个旧类、K1/K5/K10|
|truth访问|禁止；只输出功能和资源receipt|
|功能审计|`positive_rho_count`、`||a-g||`、anchor/logit/margin差异；argmax只作附属诊断|
|继续条件|任一K形成合法非零函数即可进入一次冻结G1设计，不据此调参|
|关闭条件|三K在anchor、rho、logit和margin层面全部严格零，且存在可解释的结构原因|
|技术停止|P0协议／安全、错误输入／SHA／checkout、覆盖风险、非有限数、重复确定性异常或零prediction；禁止按accuracy/H/floor停止|

## 6.运行面（实现后补齐）

|字段|值|
|---|---|
|本地环境|`ssr-gpu`|
|本地CWD|`E:\type10-7\code\snapshots\cdom_scxmap_d92_glf_r1_wt`|
|本地精确命令|`python code/scripts/run_d112_seam_g0_one_shot.py --archive E:\type10-7\automation_reports\CV-SincNet\d111_r2_g0_real_5f371082_20260802_184927_r1\artifacts\input\strict_tap\d106_ls_strict_tap.npz --receipt E:\type10-7\automation_reports\CV-SincNet\d111_r2_g0_real_5f371082_20260802_184927_r1\artifacts\input\strict_tap\d106_ls_strict_tap.receipt.json --archive-sha256 48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 --run-id d112_seam_g0_real_f2644d82_20260802_r1 --output E:\type10-7\automation_reports\CV-SincNet\d112_seam_g0_real_f2644d82_20260802_r1\artifacts\local_exact_g0_r1\result.json`|
|N607环境|已知远端尚未找到同名`ssr-gpu`；D112本地G0通过前不处理该环境问题|
|N607 CWD／命令／GPU／PID|`NOT_LAUNCHED / TBD_AFTER_RELEASE_REVIEW`|
|输出／日志|`E:\type10-7\automation_reports\CV-SincNet\d112_seam_g0_real_f2644d82_20260802_r1\artifacts\local_exact_g0_r1\result.json`；入口拒绝覆盖既有输出|

预期artifact：一个不可覆盖的JSON，绑定输入、checkpoint、bundle/source root，并给出每K的ρ、information、donor、anchor、score、margin、argmax和资源receipt以及无truth声明。任何process landing或完成都不等于性能结果。

## 7.后续真实性能边界

G0只判断“方法是否真正作用”。若功能成立，下一步只运行一次冻结四臂G1：`M0/M_DA/M_HEAD/M_JOINT`，并在同row报告old before/after、seen-new、H、每类old floor、forgetting和negative tail。若G1显示稳定负收益，立即关闭D112并转入下一条理论路线；不追加seed、不扫描参数、不恢复125矩阵。
