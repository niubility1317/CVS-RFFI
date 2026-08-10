# Phase1 P1-RECTE 12臂训练实验报告

## 1.状态与目标

- 实验ID：`phase1_recte12_20260810_v1`
- 日期：2026-08-10
- 操作边界：主代理冻结候选、矩阵、版本和证据边界；唯一N607 Runner仅负责落地、启动、监控与小工件回收。
- 当前状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`
- 目标：从相同GeoSat-C final-only checkpoint继续训练，比较同折C控制臂与唯一增加P1-RECTE辅助项的G实验臂。
- 可证伪假设：source-known-train L中，同一物理样本clean→LEO真实类margin位移若在RX×local-class格之间长期出现相对低端尾部，则只上推较低格、允许共同位移的RECTE可能改善min-RX/min-class分类端点，同时不破坏后冻结proxy反退化端点。
- 声明边界：不得预称修复RX、day、proxy、真实unknown或Phase3；完整门通过也只能进入`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`。

## 2.冻结版本与本地证据

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`48a2c284b2cca8430320da16560748898ed3b9d5`
- 独立设计终裁：`P0=0/P1=0/ALLOW-DESIGN-FREEZE`
- 独立actual-diff终裁：`P0=0/P1=0/ALLOW`

|文件|工作树SHA256|用途|
|---|---|---|
|`analysis/phase1_recte_design_20260810.md`|`d83eb97c44b030348991514fe0c96da2227e3fd070f2099c5e5dd46e2c63205f`|冻结公式、权限、VJP与证据边界|
|`code/cvsrffi/phase1_recte.py`|`8e5b79666e3bc6270d724339b80a60430c3fd214e3112bca860cf87810364310`|固定28格/378无序pair损失、functional exact-head、收据与终态合同|
|`code/SSDG/train_ssdg.py`|`d5affe248b9a968307cbc986cd10296b887ad12a61f7955e43d827e659d46fd9`|共同训练路径接入|
|`code/tests/test_phase1_recte.py`|`23144af7f8f0a22f85d3f52b50abc8b7717b4ebb5d2de69b033bed10085a94b1`|公式、CosFace、权限、逐场景VJP与终态负测|
|`code/scripts/launch_phase1_recte12_20260810.sh`|`34a84733288ac1a0fb8b41b59403abf34b65129bb1b42ed1cb449c43324009f9`|冻结12臂launcher，Git mode100755|

本地`ssr-gpu`串行验证：

- `py_compile`：core、trainer和focused test通过。
- RECTE focused：`19 passed`；逐场景正例及缺失、篡改、重复scene审计负例均闭合。
- RECTE+RCAT+RCRMD+CAGM联合回归：`50 passed`，仅4条既有CosFace autocast弃用警告。
- 真实lite_d CosFace functional readout：与共同live logits逐元素相等；LEO`feat_joint`和shared encoder aux VJP finite/nonzero；exact-head aux VJP为None/zero。
- `bash -n`：通过。
- launcher dry-run：精确12臂，C=6、G=6，旧GD/ICMT/CAGM/RCRMD/RCAT路线启用数为0。
- `git diff --check`：通过。

Runner必须从该commit生成无prefix、LF-only完整Git archive，记录archive SHA、member SHA、成员数、`code/code=0`和launcher可执行位；不得直接复制Windows mixed-EOL工作树。

### 2.1 Runner落地与启动前静态证据（2026-08-10）

- Direct N607 preflight：通过；项目根可见，GPU0–6空闲，GPU7仅有既有SCB v5（主PID`958333`、worker PID`958466`，显存合计约845MiB），未干预其它实验。
- 本地Git archive：由实现commit`48a2c284b2cca8430320da16560748898ed3b9d5`生成，无prefix、4944 members（4324 regular）、261,509,120 bytes，SHA256=`d77cb7895c14ec07c7a5859244f83170c527102941b35a5565a7e276586c2dfa`；五冻结成员均按Git LF blob核对，launcher archive mode=`0775`。
- 远端archive：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte12_20260810_v1_48a2c284.tar`，SHA256/bytes/member count与本地一致；无绝对路径与commit-prefix；由隐藏incoming目录解包后原子落地release`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte12_20260810_v1_48a2c284`，未覆盖既有路径。
- 远端关键输入：ManySig SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`，bytes=`2359341461`。GeoSat-C final checkpoint（六个均bytes=`14977627`）：F1=`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`；F2=`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76d`；F3=`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`；F4=`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`；F5=`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`；F6=`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`。
- Release静态门：`code/code=0`（release/code存在且代码静态检查闭合）；关键五成员SHA与冻结值一致；`py_compile=0`（core/train/test，外置`PYTHONPYCACHEPREFIX`）；`train_ssdg.py --help=0`；`bash -n=0`；launcher dry-run=`0`、12行/12臂（C=6、G=6、旧GD/ICMT/CAGM/RCRMD/RCAT启用数=0）；release内`__pycache__`数量=0。
- 启动前状态：release已落地，run根与log根仍为`ABSENT`；每次SSH/SCP后本地`ssh.exe=0`、N607 TCP22 established=0。

## 3.方法与冻结配置

固定source receiver集合`R_s={0,1,2,3,4,5,6}`与local4类形成28格。对同一物理source-L行，clean原始logits整体stopgrad；LEO复用已计算的`feat_joint`，以当前exact `id_backbone.cls_head.head`的全部参数和buffer的detached clone做一次`torch.func.functional_call`。functional logits必须与共同live LEO logits逐元素相等，且不得写live head状态。

对格`a=(r,c)`：

`δ_a=mean(m_i^LEO)-mean(sg(m_i^clean))`。

对固定字典序无序pair`a<b`：

`q_ab=[sg(δ_b)-δ_a]_+²+[sg(δ_a)-δ_b]_+²`。

仅较低`δ`端接收aux梯度；任一端为空格时该pair贡献0。唯一实验项为：

`L_RECTE=(1/378)Σ_{a<b}A_aA_bq_ab`，`L_G=L_base+0.02L_RECTE`。

- 分母固定378，不按occupied或positive pair重归一。
- C保持共同`L_base`，RECTE aux、pair和VJP为N/A或0。
- G每个clear/low/rain scene在首个positive-tail批分别审计一次未缩放VJP：LEO`feat_joint`和shared encoder必须finite/nonzero，exact head aux必须None/zero。
- C/G共同绑定相同final-only warm-start、head/class order、source-L physical/RX/class/scene/order、seed/sampler、40 epochs、新AdamW空初态、AMP、clean+单LEO forward和共同`L_base`。
- RECTE只读source-known-train L的TX与physical-ID绑定RX；U可构建loader但零迭代/forward，V仅共同诊断零反馈，proxy/held/target/day/fold不进入新项、loss、状态或选模。
- functional head-only读出每个G批一次；不增加model/clean/LEO forward、样本、epoch、optimizer step、持久状态或缓存。

共同超参：epochs=40、batch=128、lr=2e-4、weight_decay=1e-4、label_smoothing=0.01、seed=7281105、sat_seed=9281105、`lambda_sat_cons=0.10`。

## 4.冻结矩阵与GPU

|GPU|候选1|候选2/并发边界|
|---:|---|---|
|0|F1C_RECTE12|F5G_RECTE12|
|1|F1G_RECTE12|F5C_RECTE12|
|2|F2C_RECTE12|F6G_RECTE12|
|3|F2G_RECTE12|F6C_RECTE12|
|4|F3C_RECTE12|—|
|5|F3G_RECTE12|—|
|6|F4C_RECTE12|—|
|7|F4G_RECTE12|既有SCB v5为另一实验，Runner只读核验且不得干预|

每fold C/G使用相同GeoSat-C checkpoint、source TX、known-validation TX和proxy TX。Runner不得改fold、receiver、TX、seed、lambda、场景、训练长度或GPU映射。GPU7启动前必须记录SCB实际进程、CWD和显存；RECTE作为第二个实验不得使该GPU超过两个实验，也不得停止或修改SCB。

## 5.N607冻结路径与唯一命令

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte12_20260810_v1_48a2c284`
- 精确CWD：`<release>/code`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- GeoSat-C根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_recte12_20260810_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte12_20260810_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte12_20260810_v1_launcher.out`

唯一启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte12_20260810_v1_48a2c284/code && nohup env RUN_ID=phase1_recte12_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte12_20260810_v1_48a2c284/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte12_20260810_v1_48a2c284/code/scripts/launch_phase1_recte12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte12_20260810_v1_launcher.out 2>&1 < /dev/null &
```

Runner只能调用一次。调用端超时后必须先清理本地残留SSH，并只读确认run/log/PID/CWD/cmdline是否已landed；严禁盲目重发。retry=`NO`。

### 5.1 唯一启动与初始技术健康（2026-08-10）

- 唯一启动调用：报告§5原样命令调用1次。SSH通道约34秒超时（exit124），未重发；本地残留SSH PID`22288`经命令行核对后定向清理，之后`ssh.exe=0`、TCP22 established=0。远端只读确认已landed。
- wrapper/launcher：wrapper PID`1085062`、launcher PID`1085063`；CWD均为`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_recte12_20260810_v1_48a2c284/code`。outer launcher=`.../logs/phase1_recte12_20260810_v1_launcher.out`，初始bytes=`0`。
- 12臂PID/GPU绑定（`pids.tsv` 13行，12/12 alive）：F1C=`1085066`/GPU0，F5G=`1085068`/GPU0；F1G=`1085070`/GPU1，F5C=`1085072`/GPU1；F2C=`1085074`/GPU2，F6G=`1085076`/GPU2；F2G=`1085078`/GPU3，F6C=`1085080`/GPU3；F3C=`1085082`/GPU4；F3G=`1085084`/GPU5；F4C=`1085086`/GPU6；F4G=`1085088`/GPU7。
- GPU7并发边界：RECTE F4G与既有SCB v5主/worker`958333/958466`共存，仍为两项实验，未停止或修改SCB；其它GPU仅对应冻结RECTE双臂。
- 初始健康：12个arm日志与config receipt均已创建并增长；只读技术扫描未见`Traceback`、`RuntimeError`、OOM、argparse/unrecognized-argument、Permission denied或No such file错误指纹。当前仅记录技术运行状态，`NO_PERFORMANCE_RESULT`。

## 6.预期工件、健康规则与成功条件

每臂应生成`final_ssdg.pth`、metrics CSV/JSONL、config、training completion、terminal、heldout、resource和RECTE receipt。C应通过共同三场景×28格合同且aux为N/A/0；G应通过三场景×28格、positive-tail、functional equality、逐场景VJP和终态合同。P0 promotion默认禁用，预期终态为`NON_PROMOTABLE_P0_DISABLED`/exit8；工件与合同闭合时不视为技术失败。

仅在错误checkout/hash、覆盖风险、协议/P0违反、launcher-wide确定性故障，或至少两个distinct arm在终态工件前出现相同确定性异常指纹时停止本run。停止前必须核对run-root/CWD/cmdline/PID树，只停止本run并保留partial。不得读取accuracy、loss或任何性能值作早停依据。

Runner完成后只回收小日志、JSON/CSV、PID与manifest，不下载checkpoint/NPZ，不解释性能；交接状态先到`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。

## 7.后冻结判定边界

训练技术闭合后另行实现并运行固定42步：12 clean export+12 LEO/binding+12 proxy+6 same-fold pair。Gaussian只用L拟合，V/proxy零fit；使用float64 totalized-L2，保留zero且nonfinite fatal；proxy days/RX/seed/400、ManySig SHA和physical keys固定。

非补偿门：clean 6/6四floor不低于C−2pp；LEO 18/18四floor不低于C−2pp；每fold三场景overall和全18格overall均不低于C；每fold proxy AUROC增量>0且proxy−V mean-u gap增量>0，必须6/6。任一完整门失败即`REJECT_P1_RECTE_PERMANENT`。

## 8.Runner终态与小bundle

- 终态：12/12 child自然退出；12/12 `phase1_training_completion_receipt.json`、`phase1_terminal_status.json`、`phase1_recte_terminal_receipt.json`、`frozen_phase1_heldout_eval.json`、`phase1_resource_summary.json`、metrics CSV/JSONL、config receipt和`final_ssdg.pth`齐全；12个RECTE terminal receipt均为`NON_PROMOTABLE_P0_DISABLED`，对应exit8为预期非促销终态，不是技术失败。
- 无健康停止：错误指纹扫描为空；wrapper/launcher PID`1085062/1085063`均已退出；outer launcher bytes=`0`；GPU0–6回到空闲，GPU7仅保留既有SCB v5主/worker`958333/958466`（约845MiB），未干预SCB。
- 小bundle远端：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_recte12_20260810_v1/phase1_recte12_20260810_v1_runner_bundle.tar`，123 members、164485120 bytes、SHA256=`a4649e41c5d524399d392711bbbb60b55eb3977d47e26d14324b1a7d00c39d0c`；manifest=`phase1_recte12_20260810_v1_runner_manifest.txt`，21880 bytes、SHA256=`7c55ccd0b7148d3cdc263332cc4e5fb00a310165208ad32226f932188cf16180`。本地SCP目录：`E:\type10-7\automation_reports\CV-SincNet\phase1_recte12_20260810_v1\runner_bundle`，本地hash与远端一致；bundle禁入后缀`.pth/.npz/.pt/.npy`成员数=0，未下载任何权重或数组归档。
- 终态SSH/GPU清理：最后一次SCP后本地`ssh.exe=0`、N607 TCP22 established=0；GPU7 SCB保持运行，未停止或修改。

### 8.1 C/G逐场景技术合同

- C六臂（F1C–F6C）：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`均为28/28冻结cells；functional head aux readout=`0`，occupied/positive pair=`0/0`；`recte_terminal_contract_passed=true`。
- G五臂（F1G、F2G、F3G、F5G、F6G）：三场景均400/400 functional-equal batches；clear pair=`148347/148347`、low=`148535/148535`、rain=`148376/148376`；每场景`feat_joint`与shared encoder VJP finite/nonzero，exact-head aux none/nonzero=`1/0`；contract均通过。
- G臂F4G：low/rain同上；clear为occupied=`148347`、positive-tail=`148346`，属于一个合法zero-tie pair（冻结合同允许`0≤positive≤occupied`且每scene仍有positive tail），标记为`VALID_ZERO_TIE_PAIR / CONTRACT_PASS`，不重跑、不作性能解释。

## 9.逐臂技术结果表（不含性能解释）

|候选|fold/arm|GPU|PID|必需工件|terminal status|exit|final checkpoint SHA256|bytes|最终判定|
|---|---|---:|---:|---:|---|---:|---|---:|---|
|F1C_RECTE12|F1/C|0|1085066|10/10|NON_PROMOTABLE_P0_DISABLED|8|4b8c24733196644b0cf0839efc7a165ba79a268086eb1457920633ae9e2afd49|8481659|NO_PERFORMANCE_RESULT|
|F1G_RECTE12|F1/G|1|1085070|10/10|NON_PROMOTABLE_P0_DISABLED|8|fe0688d39d42b1746b3f2137e58764167395f09e14fd0bb397f8e4bc8b8e3d3a|9636347|NO_PERFORMANCE_RESULT|
|F2C_RECTE12|F2/C|2|1085074|10/10|NON_PROMOTABLE_P0_DISABLED|8|5cc84ad5984a4c45627228734caff0f6b8c9e0cb36830bf93bcbd33122c3b1bc|8481659|NO_PERFORMANCE_RESULT|
|F2G_RECTE12|F2/G|3|1085078|10/10|NON_PROMOTABLE_P0_DISABLED|8|d9971a30d7718fe75effb43cad0cc0995c660eaf0dc6e822cdbb9a6641d12a92|9636347|NO_PERFORMANCE_RESULT|
|F3C_RECTE12|F3/C|4|1085082|10/10|NON_PROMOTABLE_P0_DISABLED|8|1b143700a15e378c6961f4a232d2a25151caa448c3fb3300ae29e2555668577c|8481659|NO_PERFORMANCE_RESULT|
|F3G_RECTE12|F3/G|5|1085084|10/10|NON_PROMOTABLE_P0_DISABLED|8|53dc14892610ca9908170ab2d939a32beba7b3a2a3408f3fe4086b0b9b14c4d9|9636347|NO_PERFORMANCE_RESULT|
|F4C_RECTE12|F4/C|6|1085086|10/10|NON_PROMOTABLE_P0_DISABLED|8|e8940f6b1366b6ebfe7f00f2f0b5a6fb1c5d8dff41a7fdfdd7b1a05e2fd25ac7|8481659|NO_PERFORMANCE_RESULT|
|F4G_RECTE12|F4/G|7|1085088|10/10|NON_PROMOTABLE_P0_DISABLED|8|117f58eca2277c9dc212e1dbe8ffd07c879d9411fa4eed9080eb3f0e783d7172|9636347|NO_PERFORMANCE_RESULT|
|F5C_RECTE12|F5/C|1|1085072|10/10|NON_PROMOTABLE_P0_DISABLED|8|5e4df2aea0065e08a401826dc977094349a16958d3aaa5021c4c496efd67f98a|8481659|NO_PERFORMANCE_RESULT|
|F5G_RECTE12|F5/G|0|1085068|10/10|NON_PROMOTABLE_P0_DISABLED|8|fcb5406a56b23a65627bf556cc00f31371ff931a927aa78d3a06bed533f0d4a9|9636347|NO_PERFORMANCE_RESULT|
|F6C_RECTE12|F6/C|3|1085080|10/10|NON_PROMOTABLE_P0_DISABLED|8|9055408c27b034cf02b4a67b6ddaae1d9bf7dea36c27b6289fe99e6d946478c8|8481659|NO_PERFORMANCE_RESULT|
|F6G_RECTE12|F6/G|2|1085076|10/10|NON_PROMOTABLE_P0_DISABLED|8|728bea80024a758fe473ace53196233448553165091571b7d236140b51428416|9636347|NO_PERFORMANCE_RESULT|
