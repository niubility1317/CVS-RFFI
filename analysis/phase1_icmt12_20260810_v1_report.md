# Phase1 ICMT 12臂正式训练v1报告

状态：`PREREGISTERED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标、假设与比较对象

|字段|冻结值|
|---|---|
|run ID|`phase1_icmt12_20260810_v1`|
|目标|完成6个LOTO fold×C/G、40E、final-only共12臂训练，为后续一次性postfreeze 42步生成12个最终checkpoint|
|C|GeoSat-C continuation；保持共同clean＋单次LEO forward、`lambda_sat_cons=.10`、数据顺序、优化器和训练长度|
|G|在C上唯一增加固定`lambda_icmt=.05`的Independent-view Classwise Margin-Tail Tightening|
|假设|只用source-L标签收紧clean与LEO各自的类内低margin尾部，可能改善最差分类格与开放世界proxy分离；不预先声称针对RX/day或一定改善unknown拒识|
|永久淘汰规则|完整postfreeze任一非补偿门失败即`REJECT_P1_ICMT_PERMANENT`；不得用均值补偿floor或调参重试|
|实现commit|`08fb6a5282971a0ae73b9ab3d3d89935ffaa4bfa`|
|独立复核|设计与实际实现均为`P0=0、P1=0、MERGE/ALLOW`|
|当前声明|只有本地实现与测试证据；尚未产生N607训练或性能结果|

冻结公式为：

\[
m_i^v=\ell_{i,y_i}^v-\log\sum_{k\ne y_i}\exp(\ell_{i,k}^v),\qquad
\bar m_c^v=\frac1{n_c}\sum_{i:y_i=c}m_i^v,
\]

\[
L_{ICMT}=\frac18\sum_{v,c}\frac1{n_c}\sum_{i:y_i=c}
\left[\max\{0,\operatorname{sg}(\bar m_c^v)-m_i^v\}\right]^2,
\qquad L_G=L_{base}+0.05L_{ICMT}.
\]

`v∈{clean,leo}`；每类均值分母包含该类全部行；只有严格低于类均值的margin行贡献，tie为0；不按active行重归一，不使用EMA、q、quantile、temperature、RX/day/domain或跨view配对。

## 2.本地版本、文件与验证

|文件|作用|SHA256|
|---|---|---|
|`analysis/phase1_icmt_design_20260810.md`|冻结设计与追踪卡|`3cdbfe3edd6b93571f972e56dec7efd2e0b82b28ea98999ef6c0f9c9d91027be`|
|`code/cvsrffi/phase1_icmt.py`|ICMT公式、VJP与16格收据|`f81bd010949b2b0e2e95a3246e53359a2d430d492d01a9e0642b00974c526b33`|
|`code/SSDG/train_ssdg.py`|C/G训练接线与终态合同|`b1623d4ee30ed16c7f9181617a4a8c88776b601a8d48cf998a97da45896a85d9`|
|`code/tests/test_phase1_icmt.py`|公式、权限、收据、真实lite_d smoke|`60a3af504238a2d790e21924543046455399c0c487dbce1888f17e00590da985`|
|`code/scripts/launch_phase1_icmt12_20260810.sh`|冻结12臂launcher|`3dd1e66f1ace3017605c328016d36853f38d25936634207698be987fb69b7be6`|

在`ssr-gpu`环境中已完成：

- `py_compile`通过；
- ICMT、GD、CB、CP focused tests共`40 passed`，含真实`lite_d` no-query forward/backward smoke；
- launcher `bash -n`通过，dry-run精确展开12臂；
- `git diff --check`通过；
- launcher Git mode=`100755`；
- 工作树除既存未归属`conversation_index/`外干净，该目录未纳入提交。

## 3.数据、权限与训练合同

ManySig固定为`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，冻结SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。split=`tx_rx_day_1_6_3`、seed=`7281105`、sat seed=`9281105`、L/U/V=`.07/.63/.30`、batch=`128`、lr=`2e-4`、weight decay=`1e-4`、label smoothing=`.01`、epochs=`40`、checkpoint selection=`final_only`。

训练更新与ICMT严格L-only：U loader可以由共同trainer构建，但不迭代、不forward；V只做C/G共同source-validation诊断，不进入loss、backward、optimizer、校准或选模；proxy/held不构建训练loader、不forward。C/G使用相同L-batch、共同clean与单次LEO forward，场景严格按`(epoch+batch_idx-2)%3`轮换`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。

每个G辅助batch要求local4每类`n_c>=2`，两视图`z_id`与raw logits均有限。首个有效G batch必须记录ICMT对共享encoder和精确分类head的raw-unscaled VJP，二者均需finite且nonzero。训练收据闭合clean×4与LEO×4×3共16格的rows、active和finite计数；C的ICMT字段为N/A或0。

6个GeoSat-C warm-start为：

|fold|checkpoint|SHA256|train TX|known V|proxy TX|
|---|---|---|---|---|---|
|F1|`F1C_LOTO_CLSGeo12/final_ssdg.pth`|`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`|20-15,20-19,6-15,8-20|14-7|14-10|
|F2|`F2C_LOTO_CLSGeo12/final_ssdg.pth`|`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76`|14-10,20-19,6-15,8-20|20-15|14-7|
|F3|`F3C_LOTO_CLSGeo12/final_ssdg.pth`|`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`|14-10,14-7,6-15,8-20|20-19|20-15|
|F4|`F4C_LOTO_CLSGeo12/final_ssdg.pth`|`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`|14-10,14-7,20-15,8-20|6-15|20-19|
|F5|`F5C_LOTO_CLSGeo12/final_ssdg.pth`|`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`|14-10,14-7,20-15,20-19|8-20|6-15|
|F6|`F6C_LOTO_CLSGeo12/final_ssdg.pth`|`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`|14-7,20-15,20-19,6-15|14-10|8-20|

基线根为`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`；Runner启动前必须逐个核对完整SHA。

## 4.冻结矩阵与资源

|GPU|候选|
|---|---|
|0|F1C、F5G|
|1|F1G、F5C|
|2|F2C、F6G|
|3|F2G、F6C|
|4|F3C|
|5|F3G|
|6|F4C|
|7|F4G；若SCB仍运行，则GPU7总计算进程仍不得超过2|

所有C/G从各fold同一`training_final_only`checkpoint启动，使用相同class order、物理样本/批序列、seed、sampler、40E、新AdamW/AMP初态。G唯一额外项为`lambda_icmt=.05`，不得增加forward、重采样或训练长度。

## 5.N607不可覆盖路径与唯一启动命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt12_20260810_v1_08fb6a52`|
|CWD|`<release>/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_icmt12_20260810_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_icmt12_20260810_v1`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_icmt12_20260810_v1_launcher.out`|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；调用端超时或异常时先只读确认是否已landed，禁止重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt12_20260810_v1_08fb6a52/code && nohup env RUN_ID=phase1_icmt12_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt12_20260810_v1_08fb6a52/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt12_20260810_v1_08fb6a52/code/scripts/launch_phase1_icmt12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_icmt12_20260810_v1_launcher.out 2>&1 < /dev/null &
```

## 6.发布门、健康停止与预期工件

唯一Runner执行direct N607 preflight；启动前确认release、run、log、outer和临时archive路径均不存在，核对commit/archive/member、ManySig和6个基线SHA、launcher mode、GPU现状与每卡最多2个训练进程。落地后只做`py_compile`、CLI help、`bash -n`和dry-run12，然后执行§5命令一次。

启动后记录wrapper、launcher、12个child PID、CWD、cmdline、run/output/log绑定、GPU映射及日志增长。只允许因路径/hash/覆盖风险、split/head/class/data-order漂移、协议违例、Traceback/OOM/CUDA、ICMT有限性/coverage/VJP失败、两个distinct row同一预测前确定性异常或最终必要工件缺失而停止；不得按accuracy、loss趋势或任何性能值停止。若触发，只终止已证明属于本run的进程树，保留partial并标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试。

技术成功预期每臂生成E40 final checkpoint、metrics、config、training completion、terminal status、resource和heldout receipt；G还需ICMT终态合同通过，C为N/A/0。`NON_PROMOTABLE_P0_DISABLED/exit8`是预期训练终态，不是性能失败。只回收日志、JSON/CSV、PID、completion、manifest等小工件，不下载`.pth`或特征NPZ。训练完成后再独立实现并执行冻结postfreeze 42步，随后才可读取完整性能并作永久淘汰或晋级判断。

## 7.运行结果占位

|candidate|fold|arm|GPU|final checkpoint|terminal|postfreeze|最终结论|
|---|---:|---|---:|---|---|---|---|
|F1C/F1G|1|C/G|0/1|待运行|待运行|未开始|`NO_PERFORMANCE_RESULT`|
|F2C/F2G|2|C/G|2/3|待运行|待运行|未开始|`NO_PERFORMANCE_RESULT`|
|F3C/F3G|3|C/G|4/5|待运行|待运行|未开始|`NO_PERFORMANCE_RESULT`|
|F4C/F4G|4|C/G|6/7|待运行|待运行|未开始|`NO_PERFORMANCE_RESULT`|
|F5C/F5G|5|C/G|1/0|待运行|待运行|未开始|`NO_PERFORMANCE_RESULT`|
|F6C/F6G|6|C/G|3/2|待运行|待运行|未开始|`NO_PERFORMANCE_RESULT`|

## 8.运行段：RUNNING（2026-08-10）

Runner：`Luna/max`；角色边界为本run的N607落地、静态验证、唯一启动、短连接监控和小工件回收，不读取或解释性能，不改变算法、矩阵、超参或重试策略。

### 8.1落地身份与EOL字节映射

实现身份固定为commit`08fb6a5282971a0ae73b9ab3d3d89935ffaa4bfa`，预注册报告commit为`6f311930edbec9a583b3dc3278f9e5856be16f7f`。默认无prefix归档从该实现commit生成，远端实际使用归档为`263321600`字节、SHA256=`9d230a6542136055b666974e1f10fa7f9b9c723565bf83c1ff8454892ad4dcac`、`4893`个members、`code/code=0`；远端release成员集合与该归档逐字一致，无partial解包或额外代码层级。一次传输partial（`155287552`字节，SHA256=`d042626034b19034b4e09a9c6ee925302c1b0a00cac80abebe09ee4fbd6daf1f`）仅作为失败证据保留，未用于解包。

|文件|预注册local raw SHA（原冻结）|同一commit的reviewed LF-normalized SHA|远端release direct SHA|远端CRLF→LF校验|
|---|---|---|---|---|
|`analysis/phase1_icmt_design_20260810.md`|`3cdbfe3edd6b93571f972e56dec7efd2e0b82b28ea98999ef6c0f9c9d91027be`|`3cdbfe3edd6b93571f972e56dec7efd2e0b82b28ea98999ef6c0f9c9d91027be`|`601600bcc6e8912f77457da1e185d68b9db3467c33dc1bda7b08c20fb96ec1dd`|PASS|
|`code/cvsrffi/phase1_icmt.py`|`f81bd010949b2b0e2e95a3246e53359a2d430d492d01a9e0642b00974c526b33`|`f81bd010949b2b0e2e95a3246e53359a2d430d492d01a9e0642b00974c526b33`|`198a4036d0105d18321d14fe099e5737a49db6646b405a30f1e23100cb3f95cc`|PASS|
|`code/SSDG/train_ssdg.py`|`b1623d4ee30ed16c7f9181617a4a8c88776b601a8d48cf998a97da45896a85d9`|`31f8a920c3b470463029c89209f6261dbf4ff63fe63a2d9900fef0d95a2a6fd3`|`45ef46681b91bd7937b955f2dd4f7f462216b86048936a47083b63884ba9c87c`|PASS|
|`code/tests/test_phase1_icmt.py`|`60a3af504238a2d790e21924543046455399c0c487dbce1888f17e00590da985`|`60a3af504238a2d790e21924543046455399c0c487dbce1888f17e00590da985`|`3a5f4d3e4b0495f71fe38323d2303b533a675ce8e8eea533f54df51f9529db75`|PASS|
|`code/scripts/launch_phase1_icmt12_20260810.sh`|`3dd1e66f1ace3017605c328016d36853f38d25936634207698be987fb69b7be6`|`3dd1e66f1ace3017605c328016d36853f38d25936634207698be987fb69b7be6`|`3dd1e66f1ace3017605c328016d36853f38d25936634207698be987fb69b7be6`|PASS|

`train_ssdg.py`的预注册raw值`b1623d4e...`是本地mixed-EOL工作树字节；commit归档的canonical LF member为`31f8a920c3b470463029c89209f6261dbf4ff63fe63a2d9900fef0d95a2a6fd3`。远端direct SHA保留实际CRLF传输字节，五项均已按CRLF→LF规范化后与同一commit内容闭合；这只是EOL传输映射，不是算法或矩阵变化。canonical诊断归档（本地，未同步）SHA256=`24c222d9f0ef913df2967859778eaea66c4aecd7df95fc6b518b866b7d3659ba`，不覆盖当前release。

### 8.2落地路径与静态验证

远端release=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_icmt12_20260810_v1_08fb6a52`，CWD=`<release>/code`；run root、log root和outer在静态验证后仍为`ABSENT`。launcher远端mode=`775`、可执行位保持（Git mode=`100755`）。ManySig与6个GeoSat-C基线SHA在启动前均匹配冻结值；当时8卡无compute进程，未干预其它任务。

|检查|结果|
|---|---|
|远端`py_compile`（ICMT、train、test）|PASS|
|`train_ssdg.py --help`|PASS，92729字节，`--lambda_icmt`命中2处|
|launcher`bash -n`|PASS|
|launcher`--dry-run`|PASS，12行，SHA256=`9b8f39bed78508f876a309e3ffd3056b3134ddcd2c486fea04d5ce56dc84860c`|
|唯一exact命令调用次数（截至本段）|1（调用端超时后只读确认已落地，未重发）|

静态检查不构成训练或性能结果；唯一启动与健康证据见§8.3。

### 8.3唯一启动与首波健康证据

按§5冻结字符串执行exact命令一次（调用端SSH于约120秒超时，未重发）。超时后发现并终止唯一残留本地SSH客户端PID=`31848`，随后确认本地`ssh.exe=0`、N607 TCP22=`0`；远端命令已经落地并继续运行。

远端wrapper PID=`581240`、launcher PID=`581241`，二者CWD/cmdline均绑定release`.../releases/phase1_icmt12_20260810_v1_08fb6a52/code`与run ID；12个child与GPU绑定如下，均为唯一run-owned进程：

|PID|candidate|GPU|output/log绑定|
|---:|---|---:|---|
|581249|F1C_ICMT12|0|run/F1C_ICMT12；log/F1C_ICMT12.out|
|581256|F5G_ICMT12|0|run/F5G_ICMT12；log/F5G_ICMT12.out|
|581259|F1G_ICMT12|1|run/F1G_ICMT12；log/F1G_ICMT12.out|
|581263|F5C_ICMT12|1|run/F5C_ICMT12；log/F5C_ICMT12.out|
|581265|F2C_ICMT12|2|run/F2C_ICMT12；log/F2C_ICMT12.out|
|581268|F6G_ICMT12|2|run/F6G_ICMT12；log/F6G_ICMT12.out|
|581270|F2G_ICMT12|3|run/F2G_ICMT12；log/F2G_ICMT12.out|
|581272|F6C_ICMT12|3|run/F6C_ICMT12；log/F6C_ICMT12.out|
|581274|F3C_ICMT12|4|run/F3C_ICMT12；log/F3C_ICMT12.out|
|581276|F3G_ICMT12|5|run/F3G_ICMT12；log/F3G_ICMT12.out|
|581278|F4C_ICMT12|6|run/F4C_ICMT12；log/F4C_ICMT12.out|
|581280|F4G_ICMT12|7|run/F4G_ICMT12；log/F4G_ICMT12.out|

启动后首个短连接快照显示12个GPU compute各1个本run child（每卡两臂），12个candidate log均已增长；`pids.tsv`为2380字节、SHA256=`8319bd5b47b17b5cca70191341060e9d3a1787993fabba629e54536c4d7d1f6d`。outer启动日志当时为0字节且无错误marker；未读取任何accuracy、loss或其它性能字段。当前状态为`RUNNING / NO_PERFORMANCE_RESULT`，继续仅按技术健康规则短连接监控。
