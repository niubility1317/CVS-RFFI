# Phase1 CAGM 12臂正式训练v1预登记报告

状态：`ARTIFACTS_COMPLETE / TECHNICAL_SUCCESS / NON_PROMOTABLE_P0_DISABLED / NO_PERFORMANCE_RESULT`

## Runner阶段记录（2026-08-10，单次发布链）

- 指定工作树`phase3_responsibility_20260807_wt`与实现commit`979f0bfd3ac1d8c2d99f360d2a74fbcf3a780e2c`已核对；仅保留既有未跟踪`conversation_index/`，未改动。
- direct `n607_ssh_preflight.ps1`通过；用户`N607`、项目根、8张GPU可见，SSH进程/TCP22在连接后均已清零。
- 远端release、run、log、outer及临时归档路径均确认不存在；ManySig与6个GeoSat-C checkpoint SHA与预登记完全匹配。GPU7仅见既有PID 608786、488MiB进程，按预登记SCB v4无关占用处理，不干预。
- 本地完整归档SHA=`3aa98c55335ff3ba7d9f51ad06fc87a2be0083c67feca2275854d9c23a49489d`、4905 members、五个目标成员和`code/code=0`核验通过；下一步仅执行一次SCP与远端解包，仍无性能结果。
- 唯一SCP已成功；远端归档大小`260515840`字节、SHA、4905 members、`code/code=0`及五个成员核验通过。已解包到冻结release，五个release成员SHA完全匹配，launcher mode=`775`。
- release内远端`py_compile`、`train_ssdg.py --help`、`bash -n`和dry-run矩阵均通过；dry-run恰好12臂、GPU/候选映射匹配预登记。训练启动调用次数仍为`0`。
- 启动前二次只读复核：run/log/outer仍不存在；GPU0–6各1MiB、GPU7为既有PID 608786约488MiB，目标run进程为0，资源规则满足。下一步严格执行第5节唯一命令一次。
- 第5节逐字启动命令调用`1`次；本地SSH通道约124秒超时，已识别并终止唯一残留本地SSH PID、确认TCP22清零，未重发。只读确认远端已落地：wrapper=`665612`、launcher=`665613`、PGID=`665611`，12个child PID均存在且CWD统一为冻结release/code，`pids.tsv`记录的GPU映射为0/1/2/3/4/5/6/7预登记表。
- 首波健康检查：run/log/outer存在；12个arm日志均已生成并增长（约123–165KB），GPU上12个训练进程加既有SCB PID 608786，未超过每GPU两进程规则；首批日志仅见配置标记，未见Traceback/OOM/CUDA异常。当前仍无性能结果，继续短连接监控。
- 终态检查：12/12 child已退出，12/12均具备`metrics_epoch`、CAGM config、training completion、terminal status、resource summary、heldout receipt和final checkpoint；12/12 terminal为`NON_PROMOTABLE_P0_DISABLED`、`exit_code=8`。C臂6/6为`CONTROL_ARM_NOT_APPLICABLE_COMMON_SEQUENCE_BOUND`且pass，G臂6/6 CAGM terminal contract pass并完成gradient audit；未见Traceback/OOM/CUDA/CAGM失败指纹。
- 资源回收：GPU0–6回到约1MiB，GPU7保留既有SCB PID 608786约488MiB；wrapper/launcher已退出，无目标run进程，SSH/TCP22已清零。未下载`.pth`或`.npz`。
- 小工件bundle：远端`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm12_20260810_v1/phase1_cagm12_20260810_v1_small_artifacts.tar`，`17336320`字节、122 members、SHA=`2c966cd1cd7358844f448abe5981f09606e5953cdd79e7a3ca4e2d31cbdfec23`、`.pth/.npz=0`；本地回收路径`E:\type10-7\automation_reports\CV-SincNet\phase1_cagm12_20260810_v1\artifacts\returned_small\phase1_cagm12_20260810_v1_small_artifacts.tar`，manifest SHA=`6c945404f991ca2132c9c422b6bda151d8b0c5bbcec481149e3c2ded2903f68c`。首次构建因outer路径误写退出，随后在同一路径追加正确outer并完成最终校验；未影响训练。

### Runner技术结果（不含性能字段）

|candidate|GPU|child PID|final checkpoint|metrics/config|completion/terminal/resource/heldout|CAGM contract|技术判定|
|---|---:|---:|---|---|---|---|---|
|F1C_CAGM12|0|665616|有|有|有|C控制pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F5G_CAGM12|0|665618|有|有|有|G pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F1G_CAGM12|1|665620|有|有|有|G pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F5C_CAGM12|1|665622|有|有|有|C控制pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F2C_CAGM12|2|665624|有|有|有|C控制pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F6G_CAGM12|2|665626|有|有|有|G pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F2G_CAGM12|3|665628|有|有|有|G pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F6C_CAGM12|3|665630|有|有|有|C控制pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F3C_CAGM12|4|665632|有|有|有|C控制pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F3G_CAGM12|5|665634|有|有|有|G pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F4C_CAGM12|6|665636|有|有|有|C控制pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F4G_CAGM12|7|665638|有|有|有|G pass|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|

日期：2026-08-10

## 1.目标与冻结比较

|字段|冻结值|
|---|---|
|run ID|`phase1_cagm12_20260810_v1`|
|目标|完成6个LOTO fold×C/G、40E、final-only共12臂训练，为后续一次性postfreeze 42步提供12个最终checkpoint|
|C|GeoSat-C continuation；保持共同clean＋单次LEO forward、`lambda_sat_cons=.10`、数据顺序、优化器和训练长度|
|G|在C上唯一增加固定`lambda_cagm=.02`的Clean-Anchored Class Geometry Matching|
|假设|用clean锚定的类内球面半径与类间质心Gram约束LEO表征，可能提高分类最差格并改善后冻结unknown proxy分离；不预先承诺性能改善|
|实现commit|`979f0bfd3ac1d8c2d99f360d2a74fbcf3a780e2c`|
|独立复核|`P0=0、P1=0、ALLOW`|
|当前声明|仅本地实现、窄回归和发布前字节证据；尚无N607训练或性能结果|

对每个source-L batch，令`z=feat_joint`，联合辅助掩码为`M=(||z_clean||>0)&(||z_leo||>0)`。精确零范数行只退出辅助几何项，仍完整进入共同base CE/KL。对`M`中的每类样本，定义归一化特征`h`、类质心方向`a_c`、类内角半径`r_c`与类间Gram`g_cd`：

```text
a_c = normalize(mean_i h_i)
r_c = mean_i(1 - h_i^T a_c)
g_cd = a_c^T a_d
L_CAGM = [sum_c(r_c^leo-sg(r_c^clean))^2
          + sum_c<d(g_cd^leo-sg(g_cd^clean))^2] / 10
L_G = L_base + 0.02 * L_CAGM
```

clean统计完全detach；每类辅助有效行必须`n_c>=2`。辅助梯度只允许进入LEO侧共享encoder；首次有效G batch必须证明raw unscaled encoder VJP有限且非零，同时exact classifier head辅助梯度为None或精确零。CAGM不新增forward、EMA、状态、阈值、重采样、梯度投影、RX/day/domain输入或U/V/proxy训练访问。

## 2.本地版本、字节与验证

|文件|作用|工作树SHA256|commit归档SHA256|
|---|---|---|---|
|`analysis/phase1_cagm_design_20260810.md`|冻结设计与追踪|`828429464fb62a2d363a4b9f1a5ee9d207933a7480e92ca43d4a5472b24f21c3`|同左|
|`code/cvsrffi/phase1_cagm.py`|公式、绑定、VJP与收据|`7420f5dda9f4407697d9223da69e5a6320a8b6b1f49641caa050b3a2680861ed`|同左|
|`code/SSDG/train_ssdg.py`|C/G训练接线|`f194997a027585ad6df3b9258f04174879121c36f9b66f6c247186ece9015c9e`|`c842c0d830e7c39d5005ab6dcccbf3b580848f79574f787eaa6c900e60c432b5`|
|`code/tests/test_phase1_cagm.py`|focused测试与真实lite_d smoke|`4c19b73176cfb9e9046929ca16cd8aa783d5fd6a23fef150caf1e7f5e5ba1025`|同左|
|`code/scripts/launch_phase1_cagm12_20260810.sh`|冻结12臂launcher|`7b145de310fef58ad0f932b8e59c64a45653cc4fa7ffeee2986daa6c89be29b9`|同左；Git mode=`100755`|

`train_ssdg.py`的差异仅为本地mixed-EOL工作树字节与commit规范化字节，不是算法差异。供Runner落地的完整无prefix归档为：

|字段|值|
|---|---|
|本地路径|`E:\type10-7\automation_reports\CV-SincNet\phase1_cagm12_20260810_v1\artifacts\phase1_cagm12_20260810_v1_979f0bfd_fulltree.tar`|
|SHA256|`3aa98c55335ff3ba7d9f51ad06fc87a2be0083c67feca2275854d9c23a49489d`|
|大小|`260515840`字节|
|members|`4905`；无额外prefix，预期`code/code=0`|

在`ssr-gpu`环境中已完成：

- `py_compile`通过；
- CAGM focused为`9 passed`；
- CAGM＋GD＋ICMT＋CB＋CP窄回归为`49 passed`，仅既有AMP弃用警告；
- launcher `bash -n`通过，dry-run精确展开12臂；
- `git diff --check`通过；
- 独立reviewer逐字复核5文件SHA、公式、C/G公平性、VJP范围、单次反传、权限与终态收据，结论`P0=0、P1=0、ALLOW`。

## 3.数据、训练与权限合同

ManySig固定为`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。split=`tx_rx_day_1_6_3`、seed=`7281105`、sat seed=`9281105`、L/U/V=`.07/.63/.30`、batch=`128`、lr=`2e-4`、weight decay=`1e-4`、label smoothing=`.01`、epochs=`40`、checkpoint selection=`final_only`。

训练更新与CAGM严格L-only：U可由共同trainer构建但不迭代、不forward；V只做C/G共同source-validation诊断，不进入loss、backward、optimizer、校准或选模；proxy/held不构建训练loader、不forward。C/G使用相同L-batch、共同clean与单次LEO forward，场景按`(epoch+batch_idx-2)%3`轮换`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。

6个GeoSat-C warm-start：

|fold|checkpoint SHA256|train TX|known V|proxy TX|
|---:|---|---|---|---|
|1|`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`|20-15,20-19,6-15,8-20|14-7|14-10|
|2|`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76`|14-10,20-19,6-15,8-20|20-15|14-7|
|3|`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`|14-10,14-7,6-15,8-20|20-19|20-15|
|4|`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`|14-10,14-7,20-15,8-20|6-15|20-19|
|5|`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`|14-10,14-7,20-15,20-19|8-20|6-15|
|6|`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`|14-7,20-15,20-19,6-15|14-10|8-20|

基线根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`。Runner启动前只做一次必要的ManySig、6 checkpoint、release成员和不可覆盖路径核验。

## 4.冻结矩阵与资源

|GPU|候选|
|---:|---|
|0|F1C、F5G|
|1|F1G、F5C|
|2|F2C、F6G|
|3|F2G、F6C|
|4|F3C|
|5|F3G|
|6|F4C|
|7|F4G；若SCB v4仍在，则GPU7总计算进程不得超过2|

所有C/G从各fold同一`training_final_only`checkpoint启动，使用相同class order、物理样本/批序列、seed、sampler、40E、新AdamW/AMP初态。G唯一额外项为`lambda_cagm=.02`。

## 5.N607路径与唯一启动命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v1_979f0bfd`|
|CWD|`<release>/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_cagm12_20260810_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm12_20260810_v1`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm12_20260810_v1_launcher.out`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；调用端超时或异常时先只读确认是否已landed，禁止重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v1_979f0bfd/code && nohup env RUN_ID=phase1_cagm12_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v1_979f0bfd/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cagm12_20260810_v1_979f0bfd/code/scripts/launch_phase1_cagm12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cagm12_20260810_v1_launcher.out 2>&1 < /dev/null &
```

## 6.发布、健康停止与预期工件

唯一Runner执行direct N607 preflight；启动前确认release、run、log、outer均不存在，核对完整归档、5个目标成员、ManySig、6个基线checkpoint、launcher mode与GPU占用。落地后只做`py_compile`、CLI help、`bash -n`和dry-run12，然后执行§5命令一次。

启动后记录wrapper、launcher、12个child PID、CWD、cmdline、run/log绑定、GPU映射和日志增长。只允许因路径/hash/覆盖风险、数据或类序漂移、协议违例、Traceback/OOM/CUDA、CAGM有限性/coverage/VJP失败、两个distinct arm同一确定性预测前异常或最终必要工件缺失而停止；不得按accuracy、loss趋势或任何性能值停止。触发时仅终止已证明属于本run的进程树，保留partial并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试。

技术成功预期每臂生成E40 final checkpoint、metrics、config、training completion、terminal status、resource和heldout receipt；G还需CAGM终态合同通过，C为N/A/0。`NON_PROMOTABLE_P0_DISABLED/exit8`是预期训练终态。Runner只回收日志、JSON/CSV、PID、completion和manifest等小工件，不下载`.pth`或特征NPZ，不读取或解释性能。

训练成功后另行冻结并执行统一postfreeze 42步；公平评价原则是沿用已签字的source-L-only totalized-L2 Gaussian-NLL、固定400 proxy、三场景LEO、F6原始工件重算及非补偿门，但必须在CAGM专用postfreeze设计中显式写出，不能隐式复用旧候选结论。

## 7.结果（技术终态；无性能）

|candidate|fold/arm|GPU|final checkpoint|terminal|postfreeze|当前结论|
|---|---|---:|---|---|---|---|
|F1C/F1G|1 C/G|0/1|12臂均有|12臂均为`NON_PROMOTABLE_P0_DISABLED/exit8`|未开始|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F2C/F2G|2 C/G|2/3|12臂均有|12臂均为`NON_PROMOTABLE_P0_DISABLED/exit8`|未开始|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F3C/F3G|3 C/G|4/5|12臂均有|12臂均为`NON_PROMOTABLE_P0_DISABLED/exit8`|未开始|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F4C/F4G|4 C/G|6/7|12臂均有|12臂均为`NON_PROMOTABLE_P0_DISABLED/exit8`|未开始|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F5C/F5G|5 C/G|1/0|12臂均有|12臂均为`NON_PROMOTABLE_P0_DISABLED/exit8`|未开始|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
|F6C/F6G|6 C/G|3/2|12臂均有|12臂均为`NON_PROMOTABLE_P0_DISABLED/exit8`|未开始|`TECHNICAL_SUCCESS / NO_PERFORMANCE_RESULT`|
