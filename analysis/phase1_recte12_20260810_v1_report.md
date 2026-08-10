# Phase1 P1-RECTE 12臂训练实验报告

## 1.状态与目标

- 实验ID：`phase1_recte12_20260810_v1`
- 日期：2026-08-10
- 操作边界：主代理冻结候选、矩阵、版本和证据边界；唯一N607 Runner仅负责落地、启动、监控与小工件回收。
- 当前状态：`RUNNING / NO_PERFORMANCE_RESULT`
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
- Release静态门：关键五成员SHA与冻结值一致；`py_compile=0`（core/train/test，外置`PYTHONPYCACHEPREFIX`）；`train_ssdg.py --help=0`；`bash -n=0`；launcher dry-run=`0`、12行/12臂（C=6、G=6、旧GD/ICMT/CAGM/RCRMD/RCAT启用数=0）；release内`__pycache__`数量=0。
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

## 8.结果表占位

|候选|机制|fold/arm|source RX/TX|K-shot|seed|训练技术状态|后冻结clean/LEO/proxy|最终判定|
|---|---|---|---|---:|---:|---|---|---|
|F1–F6 C/G|RECTE或共同控制|固定六fold配对|见launcher冻结数组|N/A|7281105|待Runner回填|未启动|待完整非补偿门|
