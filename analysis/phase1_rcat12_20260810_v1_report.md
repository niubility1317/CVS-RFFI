# Phase1 P1-RCAT 12臂训练实验报告

## 1.状态与目标

- 实验ID：`phase1_rcat12_20260810_v1`
- 日期：2026-08-10
- 操作角色：主代理冻结候选、矩阵和证据边界；唯一N607 Runner负责落地、启动、监控与小工件回收
- 当前状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`
- 目标：从相同GeoSat-C final-only checkpoint继续训练，比较同折C控制臂与唯一增加P1-RCAT辅助项的G实验臂。
- 可证伪假设：在source-L上保持同一物理样本clean→单LEO的特征角方向，可能减少分类head核空间中的卫星视图漂移，并改善后冻结totalized-L2 Gaussian几何，同时不破坏分类floor。
- 声明边界：不得预称修复RX/day、proxy、真实unknown或Phase3；完整门通过也只能进入`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`。

## 2.冻结版本与本地证据

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`97353bcbce4202c42a2e76e19c39b80f6fa3b645`
- 独立设计终裁：`P0=0/P1=0/ALLOW-DESIGN-FREEZE`
- 独立actual-diff终裁：`P0=0/P1=0/ALLOW`

|文件|工作树SHA256|用途|
|---|---|---|
|`analysis/phase1_rcat_design_20260810.md`|`52d3b95a1914aa1aa80ff7cf9d02d60e6c4b60dfea5e61d65cc2c9e8e0707b08`|冻结设计、权限与证据边界|
|`code/cvsrffi/phase1_rcat.py`|`4b9c0eec667478fff2071e56f193e1133e5aad205de0d3b82a8e99e24109441d`|固定28格loss、收据、VJP与终态合同|
|`code/SSDG/train_ssdg.py`|`359cc53d39b96ca3b59016cc4ead21a4ef8328b2d2e0d527f76a24b11e5a7f6a`|共同训练路径接入|
|`code/tests/test_phase1_rcat.py`|`1ac9a9282a4a4804336bff2f5619bb0bdcdc2464013efd7b9e7dfb9629797eb1`|公式、权限、VJP、终态和lite_d测试|
|`code/scripts/launch_phase1_rcat12_20260810.sh`|`ca5e1c885882d6a2cc07d2b4428d2a6120a2d055320c32baf13482cbc48b9e87`|冻结12臂launcher，Git mode100755|

本地`ssr-gpu`串行验证：

- `py_compile`：通过。
- RCAT+RCRMD+CAGM聚焦回归：`31 passed`，其中RCAT为11项。
- 真实lite_d图：`positive_q=8`；`feat_joint`VJP范数`0.004565`；shared encoder VJP范数`0.368112`；RCAT对exact head参数为None/zero；共同sat-cons KL对exact head为finite/nonzero。
- `bash -n`：通过。
- launcher dry-run：精确12臂，C=6、G=6；GD/ICMT/CAGM/RCRMD等旧路线均显式关闭。
- `git diff --check`：通过。

Runner必须从上述commit生成无prefix、LF-only完整归档，逐项记录archive SHA、member SHA、成员数、`code/code=0`和launcher可执行位；不得直接复制Windows mixed-EOL工作树作为release。

## 3.方法与冻结配置

对任一特征向量定义`T(z)=z/||z||_2`（`||z||_2>0`），并令`T(0)=0`。对共同L batch中同一物理样本的clean和单LEO视图：

`q_i=||T(z_i^LEO)-sg(T(z_i^clean))||_2^2`。

对固定source receiver集合`R_s={0,1,2,3,4,5,6}`和local4类，`g_rc`为空格时为可微零，否则为该格全部行的`mean(q_i)`：

`L_RCAT=(1/28)Σ_{r∈R_s,c∈local4}g_rc`，`L_G=L_base+0.02L_RCAT`。

- 固定分母28；不按occupied cell、positive q或valid行重归一。
- zero行保留在共同base和RCAT定义中；nonfinite feature/norm/T/q/loss在backward前fatal。
- C/G共同：相同final-only warm-start、head/class order、物理样本和批顺序、seed/sampler、clean+单LEO forward、三场景轮转、40 epochs、新AdamW空初态、AMP和`L_base`。
- C：RCAT关闭、`lambda=0`，只封存共同coverage；aux、positive q与VJP为N/A或0。
- G：RCAT开启、`lambda=0.02`；只读source-L的TX标签和physical-ID绑定RX；首个每场景正q批要求LEO`feat_joint`与shared encoder的raw aux VJP finite/nonzero；exact head aux VJP应为N/A/None-or-zero。
- U不forward；V仅C/G共同只读诊断且不回流；proxy/held/target/day/fold不进入新项、loss、状态或选模。
- 每场景7×4=28格，三场景终态84格；不增加forward、模型参数、状态、缓存、样本、epoch或optimizer step。

共同超参：epochs=40、batch=128、lr=2e-4、weight_decay=1e-4、label_smoothing=0.01、seed=7281105、sat_seed=9281105、`lambda_sat_cons=0.10`。

RCAT与共同sat-cons KL并非同一约束：对非共线`u∈ker(W)`，可出现KL为0而RCAT为正；对`z_LEO=a z_clean`且`a>0,a≠1`，RCAT为0而一般KL为正。RCAT约束全部角方向漂移，其相对KL的新可识别部分是head-nullspace方向。

## 4.冻结矩阵与GPU

|GPU|候选1|候选2/并行边界|
|---:|---|---|
|0|F1C_RCAT12|F5G_RCAT12|
|1|F1G_RCAT12|F5C_RCAT12|
|2|F2C_RCAT12|F6G_RCAT12|
|3|F2G_RCAT12|F6C_RCAT12|
|4|F3C_RCAT12|—|
|5|F3G_RCAT12|—|
|6|F4C_RCAT12|—|
|7|F4G_RCAT12|既有SCB v5构建为另一实验，Runner只读核验且不得干预|

每fold C/G使用相同GeoSat-C checkpoint、source TX、known-validation TX和proxy TX。Runner不得改fold、receiver、TX、seed、lambda、场景、训练长度或GPU映射。按项目规则GPU7的RCAT是与SCB并行的第二个实验；启动前必须记录实际显存、进程/CWD和总并发，不得干预SCB。

## 5.N607冻结路径与唯一命令

- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 预计release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat12_20260810_v1_97353bcb`
- 精确CWD：`<release>/code`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- GeoSat-C根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcat12_20260810_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcat12_20260810_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcat12_20260810_v1_launcher.out`

唯一启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat12_20260810_v1_97353bcb/code && nohup env RUN_ID=phase1_rcat12_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat12_20260810_v1_97353bcb/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat12_20260810_v1_97353bcb/code/scripts/launch_phase1_rcat12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcat12_20260810_v1_launcher.out 2>&1 < /dev/null &
```

Runner只能调用一次。调用端超时必须先清理本地残留SSH并只读确认run/log/PID/CWD/cmdline是否已landed，严禁盲目重发。retry=`NO`。

## 6.预期工件与技术停止

每臂应生成：`final_ssdg.pth`、metrics CSV/JSONL、config、training completion、terminal、heldout、resource和RCAT receipt。C应通过共同84格合同且aux为N/A/0；G应通过84格、positive q、VJP和终态合同。由于P0 promotion默认禁用，预期终态为`NON_PROMOTABLE_P0_DISABLED`/exit8；工件与合同闭合时不视为技术失败。

仅在错误checkout/hash、覆盖风险、协议/P0违反、launcher-wide确定性故障，或至少两个distinct arm在终态工件前出现相同确定性异常指纹时，停止本run。停止前必须核对run-root/CWD/cmdline/PID树，只停止本run并保留partial。不得读取accuracy、loss或任何性能值作早停依据。

Runner完成后仅回收小日志、JSON/CSV、PID与manifest，不下载checkpoint/NPZ，不解释性能；交接状态先到`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。

## 7.后冻结判定边界

训练技术闭合后另行实现并运行固定42步：12 clean export+12 LEO/binding+12 proxy+6 same-fold pair。Gaussian只用L拟合，V/proxy零fit；使用float64的同一totalized-L2分段规则，保留zero且nonfinite fatal；proxy days/RX/seed/400、ManySig SHA和physical keys固定。

非补偿门：clean 6/6四floor不低于C−2pp；LEO 18/18四floor不低于C−2pp；每fold三场景overall和全18格overall均不低于C；每foldproxy AUROC增量>0且proxy−V mean-u gap增量>0，必须6/6。任一完整门失败即`REJECT_P1_RCAT_PERMANENT`。

## 8.Runner执行收据（2026-08-10）

- 交接状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`；本Runner仅执行冻结实现、落地、启动、监控和小工件回收，不读取或解释性能。
- 实现commit：`97353bcbce4202c42a2e76e19c39b80f6fa3b645`；预登记report commit：`e6e26b318e06cfbd5993d4d612fa07c1c8c69f38`。
- 精确Git archive：无prefix、LF-only，`261171200`B，tar成员`4932`（文件`4312`、目录`620`），archive SHA256=`4b242d60f6dcee74d6e764ecb46a93562f0d3fa1cb1852a0805c41a0fadd2bbd`；`code/code*`成员数=`0`；code下`.py/.sh`文本`1347`个、CRLF=`0`；launcher mode=`775`。

|archive member|SHA256|mode|
|---|---|---:|
|`analysis/phase1_rcat_design_20260810.md`|`52d3b95a1914aa1aa80ff7cf9d02d60e6c4b60dfea5e61d65cc2c9e8e0707b08`|`664`|
|`code/cvsrffi/phase1_rcat.py`|`4b9c0eec667478fff2071e56f193e1133e5aad205de0d3b82a8e99e24109441d`|`664`|
|`code/SSDG/train_ssdg.py`|`4148ddd823d11cc17efa7dca23960a777311f001a34c508cd5e25006db61a593`|`664`|
|`code/tests/test_phase1_rcat.py`|`1ac9a9282a4a4804336bff2f5619bb0bdcdc2464013efd7b9e7dfb9629797eb1`|`664`|
|`code/scripts/launch_phase1_rcat12_20260810.sh`|`ca5e1c885882d6a2cc07d2b4428d2a6120a2d055320c32baf13482cbc48b9e87`|`775`|

工作树中的`train_ssdg.py`SHA=`359cc53d39b96ca3b59016cc4ead21a4ef8328b2d2e0d527f76a24b11e5a7f6a`为Windows CRLF smudge字节；release严格使用上述Git archive LF blob（`4148ddd...`），未对科学代码做修补。

本地静态门（`ssr-gpu`串行）：实现core/train/test`py_compile`通过；`train_ssdg.py --help`通过；`bash -n`通过；launcher dry-run精确12行（C=6、G=6、旧候选启用行=0）。远端release复核同样通过，且release内`__pycache__`目录=`0`。

远端只读preflight与落地：ManySig SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；release/run/log/outer启动前均`ABSENT`；GPU7既有SCB的compute PID=`958333,958466`（约845MiB），启动前后均未干预。archive远端SHA与本地一致，release成员数和5个目标member SHA/mode一致。

唯一启动：报告§5命令调用`1`次，`retry=NO`。调用端约45秒返回`-1`且无成功文本；按规则未重发。随后只读确认launcher已落地：父PID=`997307`，CWD=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcat12_20260810_v1_97353bcb/code`，`pids.tsv`为13行（含表头），12个子PID均绑定该CWD和冻结命令；父launcher及12个子PID自然退出。GPU映射与实际PID如下：

|PID|fold/arm|GPU|epochs/final_only|
|---:|---|---:|---|
|997310|F1C|0|40/true|
|997312|F5G|0|40/true|
|997314|F1G|1|40/true|
|997316|F5C|1|40/true|
|997318|F2C|2|40/true|
|997320|F6G|2|40/true|
|997322|F2G|3|40/true|
|997324|F6C|3|40/true|
|997326|F3C|4|40/true|
|997328|F3G|5|40/true|
|997330|F4C|6|40/true|
|997332|F4G|7|40/true|

技术健康与工件：12/12日志达到`E040/040`；严格`Traceback/RuntimeError/argparse/OOM/SIGSEGV/NaN`异常指纹均为`0`。曾有一次粗grep命中`leo_grad_nonfinite=0`遥测字段，复核后不属于异常。9类必需工件均为12/12：`final_ssdg.pth`、`metrics_epoch.csv`、`metrics_epoch.jsonl`、`phase1_rcat_config_receipt.json`、`phase1_training_completion_receipt.json`、`phase1_terminal_status.json`、`phase1_rcat_terminal_receipt.json`、`frozen_phase1_heldout_eval.json`、`phase1_resource_summary.json`。所有终态为`NON_PROMOTABLE_P0_DISABLED`/exit8，未转为性能结果。

RCAT合同逐臂复核：C=`6/6`通过，三场景各28格common覆盖、`rcat_total_rows=0`、`rcat_positive_q=0`、RCAT aux/audit/scenes为空或N/A；G=`6/6`通过，三场景各28格、`rcat_batches=1200`、`rcat_total_rows=153600`、`rcat_positive_q>0`，`feat_joint_leo`与shared encoder raw VJP finite/nonzero，exact head aux为none/zero，终态合同通过。未读取或报告accuracy/loss等性能字段。

12个final checkpoint仅记录远端SHA，均未下载：

|arm|bytes|远端SHA256|
|---|---:|---|
|F1C|9480955|`d96eccecb65cbae8d29751fa4a4217171c211afffbe51129c3d8362458bd0f5c`|
|F1G|10697851|`e4b64bd1d0037bab42f46b2df497c6a9fb5063683999960fb0e29a8ca7a11e99`|
|F2C|9480955|`1834b4d5dda556fb9a2a7084596d23532ae8b0c6c17c3a77488b3c9ef7ca1e9f`|
|F2G|10697851|`2f0628254d1361c1ac14a7acfd67459b429dd55fc52f92e09d62a015c5aaa6ad`|
|F3C|9480955|`3f0f3049c33647b6d0d7f5ceb45ecaa6234dfbd483ecc876814fc4b407f823c4`|
|F3G|10697851|`84e2e67c853b8cb009af05ad9ee429898a2a6a95de3f19c01048166f0e917c6f`|
|F4C|9480955|`40529f92d629b645a2cc5f2df850b3fa9471ccfde388f8b37008ad73fddda799`|
|F4G|10697851|`1aeae68b3c74ba28ba1b0ac31aa74ba6d10d74ba5dd2cc24fa121445370352a9`|
|F5C|9480955|`30dac552d1a52a8f7bab70d7d73b90a764b57589038db07b3890411fc7ac98c5`|
|F5G|10697851|`d0f1b138955e43ef3eb0f1727edf8d23c901bc36e2364ec74e99b7dea23a7f41`|
|F6C|9480955|`a116db53be8d3b19d719f32a457b91b626514eb4af3631b86c8f79edcffbf424`|
|F6G|10697851|`3e08a692a723b73e4503316f60106d7e185d2893ed6a84eeb810e65facd6b0a1`|

回收与清理：远端manifest含73个小文件行和12个`NO_DOWNLOAD` final行，SHA256=`55dcc9793dbedb0535d13037d4b788c1a3a62f2766ffeea8f2c2c8de3c5eed0c`，12766B；小工件bundle不含`.pth/.npz`，SHA256=`baa406d29556d275bcd305d476a3d9c1c2c40ae1d8cd1d7dc7cae499eb1e38f6`，4966400B。bundle和manifest已回收到`E:\type10-7\automation_reports\CV-SincNet\phase1_rcat12_20260810_v1\artifacts\`，本地SHA与远端一致。每次SSH/SCP后本地`ssh.exe=0`、TCP22`ESTABLISHED=0`；SCB进程未受影响。
