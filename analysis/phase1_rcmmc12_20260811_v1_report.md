# Phase1 RCMMC 12臂训练实验报告

状态：LOCAL_VERIFIED / PREREGISTERED / P0=0 / P1=0 / NO_PERFORMANCE_RESULT

## 1.实验身份与目标

- 实验ID：phase1_rcmmc12_20260811_v1
- 日期：2026-08-11
- 操作方：Codex主控；N607唯一Runner待交接
- 实现commit：021a639ddfd4e485c179dce8d08e436dd8880c80
- 目标：运行冻结P1-RCMMC的6fold×C/G共12臂训练矩阵，检验source-L同物理clean→单LEO的receiver×local4 cell一、二阶totalized-feature矩约束是否技术闭合，并为后续固定42步后冻结评价生成12份final checkpoint。
- 对照：C为GeoSat-C training_final_only共同续训；G只在同一共同路径上增加P1-RCMMC。
- 声明边界：本报告的本地实现、进程落地和receipt闭合均不是性能结果；12/12完整同row工件返回前不读取或解释accuracy、loss、floor或proxy指标。本实验不构成真实unknown、真实在轨或Phase3能力声明。

## 2.冻结假设与方法

对source-known-train L中同一物理行的clean和单LEO feat_joint先做safe totalized-L2：非零行归一化，零行保持零。运行时只从sealed source-split receipt读取有序7个source receiver token并映射为slot；与local4组成固定28个cell。

对每个cell计算：

- μ=mean(T(z))
- Q=XᵀX/n，并在AMP外FP32显式对称
- D=2||μ_LEO-sg(μ_clean)||²+||Q_LEO-sg(Q_clean)||²_F
- L_RCMMC=(1/28)ΣD
- G：L_G=L_base+0.02L_RCMMC；C：辅助N/A/0

RCMMC是RCAT的cell-moment严格放松：RCAT=0推出RCMMC=0；cell内样本置换可使RCMMC=0而RCAT>0。它不被描述为与RCAT双向不可比，也不与ICMT、CAGM、RCRMD、RECTE、HSCF或其它旧loss拼接。

共同冻结条件：B=128、d=160、local4、3个LEO scene、40E、seed=7281105、sat_view_seed=9281105、新AdamW、AMP、同一sampler/order及单次clean+LEO forward。U零iterate/forward/update；V、proxy、held、target、day、fold、domain对训练、校准和选模零反馈。

## 3.本地版本与验证

|文件|工作树SHA256|用途|
|---|---|---|
|analysis/phase1_rcmmc_design_20260811.md|59965e4f0baf05976b1a5a92ffede8e33406b3cc5f6af1f9efdc6a763fcfa42d|冻结设计与追踪|
|code/cvsrffi/phase1_rcmmc.py|190c27d2334aa322b933e4db3c7861963bc33f76c436e4daa1ce2ea2572d52c3|公式、safe-zero、source绑定、VJP、receipt与terminal|
|code/SSDG/train_ssdg.py|933e4e990b24fdecbcf9492dda0a025adb4c2aa4071639bf9bb638b31e0102f4|共同训练路径接线|
|code/tests/test_phase1_rcmmc.py|66980be7dd61f19c226e1b5911436479d21024fca562b7ddc14c604dc9f32b87|focused正负测与集成核验|
|code/scripts/launch_phase1_rcmmc12_20260811.sh|9aaafd22a7f3e265ef05da6cdf23e0ccf92cb6b3655ac5bab22cbb7461c46e05|冻结12臂launcher|

本地均用官方Conda hook激活ssr-gpu后串行验证：

- py_compile：core、trainer、test通过；
- RCMMC focused：15 passed；
- RCMMC+HSCF+RECTE+RCAT+RCRMD+CAGM+ICMT联合：94 passed；
- train_ssdg.py --help：3个RCMMC参数存在；
- bash -n：通过；
- launcher dry-run：12行，C=6、G=6，旧候选enabled=true为0，每GPU不超过2臂；
- git diff --check：通过；
- 独立actual-diff复审：P0=0、P1=0、ALLOW；
- 已修复并负测clean-VJP收据缺失盲区：audit强制四参，receipt和terminal均重新验证clean/head None-or-zero与LEO/encoder finite-nonzero。

## 4.冻结矩阵与GPU映射

|顺序|候选|fold|arm|GPU|source train TX|known-val TX|proxy TX|
|---:|---|---:|---|---:|---|---|---|
|1|F1C_RCMMC12|1|C|0|20-15,20-19,6-15,8-20|14-7|14-10|
|2|F5G_RCMMC12|5|G|0|14-10,14-7,20-15,20-19|8-20|6-15|
|3|F1G_RCMMC12|1|G|1|20-15,20-19,6-15,8-20|14-7|14-10|
|4|F5C_RCMMC12|5|C|1|14-10,14-7,20-15,20-19|8-20|6-15|
|5|F2C_RCMMC12|2|C|2|14-10,20-19,6-15,8-20|20-15|14-7|
|6|F6G_RCMMC12|6|G|2|14-7,20-15,20-19,6-15|14-10|8-20|
|7|F2G_RCMMC12|2|G|3|14-10,20-19,6-15,8-20|20-15|14-7|
|8|F6C_RCMMC12|6|C|3|14-7,20-15,20-19,6-15|14-10|8-20|
|9|F3C_RCMMC12|3|C|4|14-10,14-7,6-15,8-20|20-19|20-15|
|10|F3G_RCMMC12|3|G|5|14-10,14-7,6-15,8-20|20-19|20-15|
|11|F4C_RCMMC12|4|C|6|14-10,14-7,20-15,8-20|6-15|20-19|
|12|F4G_RCMMC12|4|G|7|14-10,14-7,20-15,8-20|6-15|20-19|

共同配置：epochs=40、lr=0.0002、weight_decay=0.0001、batch_size=128、amp=true、max_grad_norm=5.0、checkpoint_selection=final_only。每GPU最多2臂。

## 5.N607发布预登记

- 账号：普通N607；禁止管理员账号
- Python：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
- 项目根：/home/szu2070436088/2510044040/CV-SincNet
- release：/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc12_20260811_v1_021a639d
- CWD：上述release的code目录
- run根：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcmmc12_20260811_v1
- log根：/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcmmc12_20260811_v1
- outer：/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcmmc12_20260811_v1_launcher.out
- 数据：/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl
- warm-start根：/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1
- retry：NO
- 启动所有权：唯一Runner；主控不得重复启动

冻结唯一命令：

    cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc12_20260811_v1_021a639d/code && nohup env RUN_ID=phase1_rcmmc12_20260811_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc12_20260811_v1_021a639d/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc12_20260811_v1_021a639d/code/scripts/launch_phase1_rcmmc12_20260811.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcmmc12_20260811_v1_launcher.out 2>&1 < /dev/null &

Runner在唯一启动前必须完成：direct preflight；Git archive成员SHA、mode、LF、无prefix和code/code=0；ManySig及6个GeoSat-C checkpoint SHA；远端py_compile/help/bash-n/dry-run12；release/run/log/outer目标不存在；GPU占用记录。SSH超时后先清理本地ssh.exe与N607/bridge TCP22，再只读确认是否landed；不得重发。

## 6.技术健康、停止与预期工件

每臂预期：

- final_ssdg.pth
- training completion receipt
- phase1_terminal_status.json
- phase1_rcmmc_terminal_receipt.json
- config、resource、heldout receipt
- 完整arm日志

G臂额外必须闭合：

- 三scene共同28/28 cell coverage且各有正D批；
- rcmmc_batches、rows、sumD、loss和common C/G账本闭合；
- 全程首个正D批的raw-unscaled VJP：LEO feat_joint与shared encoder finite-nonzero，clean feat_joint与exact head None-or-zero；
- source receiver只以有序SHA/count持久化，不出现raw receiver token、physical key、feature或矩阵；
- streamed FP32资源账本成立，无B×d²、B×28×d²或跨批cache。

预注册技术停止条件：

- P0协议、权限、checkout/hash或输出覆盖错误；
- launcher-wide确定性故障；
- 至少2个不同arm在产生final前出现相同标准化异常指纹；
- OOM、CUDA、argparse、路径/权限错误，或final/receipt零闭合。

停止只依据技术健康，不读取accuracy、loss、floor或proxy表现。停止前必须绑定本run的PID、CWD和cmdline，只处理本run进程并保留全部部分工件。健康失败记为STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT，本ID不重跑。

## 7.成功条件、分析边界与风险

技术成功条件：12/12臂自然完成、final与全部terminal/completion/resource/heldout/config工件齐全、C/G receipt合同通过、无统一技术错误指纹、GPU与SSH清理完成。

只有技术闭合后，主控才可读取同row训练表现并决定是否进入固定42步postfreeze。进入postfreeze后仍必须评价clean 6/6、LEO 18/18四floor、fold/global overall及fixed400 proxy AUROC和u-gap双门；所有门非补偿，局部最大值不能代替完整行。

当前结果表：

|候选|技术状态|性能状态|最终判定|
|---|---|---|---|
|F1C/F1G…F6C/F6G|待N607|NO_PERFORMANCE_RESULT|PENDING|

主要风险：

- 28个cell逐批FP32 XᵀX会增加计算与同步开销；不得据中间耗时或GPU利用率停止。
- safe-zero和空cell合法，但终态每scene必须28/28覆盖并有正D批。
- 若VJP、resource ledger或terminal收据失败，按技术失败处理，不得改公式、λ、fold、seed或矩阵后在同run ID重试。

## 8.Runner预启动证据（2026-08-11）

- 治理入口：根目录direct `tools\\n607_ssh_preflight.ps1`通过；普通N607账号`szu2070436088`，未使用admin或bridge；SSH进程与TCP22均已清零。
- Git/archive输入：实现commit=`021a639ddfd4e485c179dce8d08e436dd8880c80`；最终无prefix、LF-only full archive=`release/phase1_rcmmc12_20260811_v1_021a639d_lfnorm.tar`，SHA256=`50f06763940c276604e74f386a5d3ef900190e3788c31c360e12f0addd4d358b`，267294720B，4969 members（4349 files、620 dirs），`code/code=0`，4291个无NUL文本成员CR字节=0。五冻结成员SHA与预登记一致：design=`59965e4f0baf05976b1a5a92ffede8e33406b3cc5f6af1f9efdc6a763fcfa42d`、core=`190c27d2334aa322b933e4db3c7861963bc33f76c436e4daa1ce2ea2572d52c3`、train archive-LF=`7a0a431280d8f782a638a74491459f279be429f9b32926ef93b3fbdc31107b73`（worktree=`933e4e990b24fdecbcf9492dda0a025adb4c2aa4071639bf9bb638b31e0102f4`）、test=`66980be7dd61f19c226e1b5911436479d21024fca562b7ddc14c604dc9f32b87`、launcher=`9aaafd22a7f3e265ef05da6cdf23e0ccf92cb6b3655ac5bab22cbb7461c46e05`；launcher归档mode=`0664`（实现commit实际Git mode=`100644`，冻结命令显式用`bash`）。首次worktree-attributes CRLF包保留为未放行证据，未传输。
- 远端输入与覆盖门（预SCP快照）：ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；六个GeoSat-C checkpoint SHA逐项匹配预登记；release/run/log/outer启动前均`ABSENT`；8GPU均0%利用率、1MiB显存、无训练进程。SCP次数=`0`，launch次数=`0`，retry=`NO`。

## 9.Runner落地与启动前静态门（2026-08-11）

- 新Runner只读审计发现远端预存incoming归档`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcmmc12_20260811_v1_021a639d.incoming.tar`（mode=`0664`，267294720B），SHA=`50f06763940c276604e74f386a5d3ef900190e3788c31c360e12f0addd4d358b`与本地final archive一致；按规则未重复SCP（本Runner `SCP=0`）。
- release目录已存在且mode=`0775`，五冻结member均mode=`0664`且SHA匹配预登记（train为archive-LF SHA=`7a0a431280d8f782a638a74491459f279be429f9b32926ef93b3fbdc31107b73`）；远端归档4969 members、4349 files、621 dirs，`code/code=0`、prefix=0、文本4291/CR bytes=0、`__pycache__`=0。
- ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；F1C--F6C六个GeoSat-C checkpoint均存在并已取SHA；三文件`py_compile`=0、`train_ssdg.py --help`=0且含RCMMC flags、launcher `bash -n`=0、dry-run=12（C=6/G=6，GPU计数0:2/1:2/2:2/3:2/4:1/5:1/6:1/7:1，旧候选=0）。
- 真实F1C checkpoint no-query smoke通过：从checkpoint args解析`input_len=256,num_domains=14`，state load missing/unexpected=0，前向`z=[128,160]`，RCMMC rows=128/cells=28/positive-D=28，receipt schema=`cvs.phase1.rcmmc_receipt.v1`且VJP receipt闭合；LEO feat_joint与shared encoder finite-nonzero，exact classifier head与clean feat_joint None-or-zero，`rcmmc_loss`签名无query参数。一次错误的128/1默认构造仅产生shape mismatch，未进入训练且已用checkpoint args正确重做。
- 本次Runner启动前release/run/log/outer分别为PRESENT/ABSENT/ABSENT/ABSENT；无匹配训练进程，8 GPU均0%/1MiB；当前`launch=0`、`retry=NO`，SSH/TCP22每次短连后均清零。
