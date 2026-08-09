# Phase1 P1-RCRMD 12臂训练实验报告

## 1.状态与目标

- 实验ID：`phase1_rcrmd12_20260810_v1`
- 日期：2026-08-10
- 操作角色：主代理冻结方法与矩阵；唯一N607 Runner负责落地、启动、监控和工件回收
- 当前状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`
- 目标：以GeoSat-C final-only checkpoint为共同起点，比较严格同折的C控制臂与仅增加P1-RCRMD辅助损失的G实验臂。
- 假设：source-L中receiver×class等权的正clean→LEO margin-drop二阶矩，可能减少接收机条件退化。
- 声明边界：该方法不是分位数tail估计；不得声称已修复RX/day、proxy、真实unknown或Phase3。即使完整门通过也只能进入`pending-main`复核。

## 2.冻结版本与本地验证

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`d99eb9391cf5d0de33f17722bba04f9d0bb3fe57`
- 独立实际diff复审：初审唯一P1为float32批loss与Python cell账本的过严对账；最小修复后终裁`P0=0/P1=0`。
- 修改文件：

|文件|SHA256|用途|
|---|---|---|
|`analysis/phase1_rcrmd_design_20260810.md`|`0affa6fed13769f56369270574411528fce7579c0dea05d1cb1b443c313a1954`|冻结设计、权限和证据边界|
|`code/cvsrffi/phase1_rcrmd.py`|`3968eaa1228621d607be7f4cc6008afeff82cc88fdf983e6ff6ae424f2d7e64b`|固定1/28损失、收据、VJP和终态合同|
|`code/SSDG/train_ssdg.py`|`4e076f044cabc23ec66e5e6866d54f977837b79723838cf331aaafd8aa8fc5c6`|共同训练路径接入|
|`code/tests/test_phase1_rcrmd.py`|`a4753b5629cb0fc157e25214721932a77c489218b6fe7edecb8275f129f61996`|公式、权限、账本、lite_d和launcher测试|
|`code/scripts/launch_phase1_rcrmd12_20260810.sh`|`ee5e91f39737beabfe054aebbdbf8ad94832f6c32d0bc2a7d25e55c89a7200cb`|冻结12臂launcher，Git mode100755|

表中为本地工作树原始字节SHA。Git commit/archive的LF成员中，仅`code/SSDG/train_ssdg.py`因本地mixed-EOL而使用SHA`89280a76f32bee8c7f9df19aef0e447f047938ada35e3e6c56123b54d14794e6`；其余四项与表中相同。远端release必须按commit/archive成员SHA核验，不得把EOL差异误判为算法漂移。

本地`ssr-gpu`串行验证：

- `py_compile`：通过。
- RCRMD+CAGM+ICMT+GD+CB+CP聚焦回归：`60 passed`。
- `bash -n`：通过。
- launcher dry-run：12臂，6C+6G，旧候选激活数0。
- `git diff --check`：通过。
- float32账本回归：`75.0000057220459`与cell账本`75.0`合法闭合；`+1`实质漂移仍fail-closed。

### 2.1 Runner前置证据（2026-08-10）

- direct `n607_ssh_preflight.ps1`通过；本地SSH进程与N607 TCP22均为0。
- d99完整无prefix archive：`artifacts/phase1_rcrmd12_20260810_v1_d99eb939_fulltree.tar`；SHA256=`c2d86c524710d6590d459eca02bc156c4330840af29b6066f839cfbb7f788c66`，大小=`260823040`字节，`4918` members，`code/code=0`。归档使用`core.autocrlf=false`生成，确保冻结LF成员。
- 归档成员SHA/mode：design=`0affa6fed13769f56369270574411528fce7579c0dea05d1cb1b443c313a1954`；core=`3968eaa1228621d607be7f4cc6008afeff82cc88fdf983e6ff6ae424f2d7e64b`；train=`89280a76f32bee8c7f9df19aef0e447f047938ada35e3e6c56123b54d14794e6`；test=`a4753b5629cb0fc157e25214721932a77c489218b6fe7edecb8275f129f61996`；launcher=`ee5e91f39737beabfe054aebbdbf8ad94832f6c32d0bc2a7d25e55c89a7200cb`、mode=`775`。
- 远端只读目标核验：release/run/log/outer及`/tmp/phase1_rcrmd12_20260810_v1_d99eb939_fulltree.tar`均ABSENT；ManySig=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；6个GeoSat-C final SHA与v2签字报告完全匹配。
- 资源快照：GPU0–6各约1MiB、GPU7约498MiB；仅既有SCB v4 PID608786（约488MiB），未干预；目标RCRMD进程为0。
- 下一步：仅执行一次SCP落地并做远端archive/member/static门；静态门失败不得启动，retry=`NO`。

### 2.2 远端落地与静态门

- SCP次数=`1`，远端release已创建且未覆盖既有路径；archive SHA/size/member/code-code和五成员SHA/mode均闭合。
- 远端`py_compile`（core/train/test）、`train_ssdg.py --help`、`bash -n`均通过；launcher dry-run=`12`行（C=`6`、G=`6`、旧候选激活=`0`、RUN_ID=`12`、epochs40=`12`）。
- 静态门通过后唯一启动命令已登记于§5；launch调用次数仍为`0`，retry=`NO`。

### 2.3 唯一启动与首波健康

- 按§5 exact command调用`1`次；SSH端超时，随后只清理本地残留ssh PID并确认N607 TCP22为0；未重发、retry=`NO`。
- wrapper PID=`764314`，launcher PID=`764315`；12个主child均绑定`CWD=<release>/code`、cmdline=`python train_ssdg.py --candidate_id <candidate> --run_id phase1_rcrmd12_20260810_v1`：F1C=`764318`/GPU0，F5G=`764320`/GPU0，F1G=`764322`/GPU1，F5C=`764324`/GPU1，F2C=`764326`/GPU2，F6G=`764328`/GPU2，F2G=`764330`/GPU3，F6C=`764332`/GPU3，F3C=`764334`/GPU4，F3G=`764336`/GPU5，F4C=`764338`/GPU6，F4G=`764340`/GPU7。
- 首波目录/日志：candidate dirs=`12`、`.out`=`12`；技术异常marker计数均为0（Traceback、RuntimeError、OOM、CUDA、unrecognized arguments）。
- GPU快照：GPU0–3约90%–93%且5.1–5.4GiB；GPU4–7约17%–18%且2.6–3.2GiB；GPU7含SCB v4 PID608786与F4G，合计2个计算进程；未干预SCB。
- 当前仅报告技术健康，不读取或解释accuracy/loss/性能，继续短连接监控至12臂自然终态。

### 2.4 首个短连接监控

- launcher PID=`764315`仍在，12/12主child仍存活；12/12 config receipt已生成，final/completion/terminal/resource/heldout/receipt尚未到达自然终态。
- logs累计约`1538174`字节且持续增长；技术异常指纹计数=`0`；GPU0–2约93%，GPU3约39%，GPU4–7约20%–38%，未超过每GPU两进程。

### 2.5 中段监控

- launcher仍存；main child=`9/12`；final=`7/12`、completion/terminal/resource/heldout/RCRMD-terminal receipt均=`3/12`、config=`12/12`。
- logs累计约`2501924`字节；Traceback/RuntimeError/OOM/CUDA/unrecognized均为0；GPU4–6已释放，GPU0–3与GPU7仍在合法占用范围。

### 2.6 后段监控

- main child=`4/12`、launcher仍存；12/12 final已生成，completion/terminal/resource/heldout/RCRMD-terminal receipt=`8/12`，config=`12/12`。
- logs累计约`2612414`字节；技术异常指纹仍为0；GPU4–6已空闲、GPU7仅保留SCB v4约498MiB，GPU0–3各约35%–42%。

## 8.最终技术交接（2026-08-10）

- 12/12主child、launcher和wrapper均自然退出；技术异常指纹为0。每臂`NON_PROMOTABLE_P0_DISABLED`、exit8、`promotion_ready=false`、heldout=`COMPLETE`，该终态是预登记P0禁用语义，不是技术失败，也不构成性能结果。
- RCRMD合同：C=`6/6` common 3场景×28 cells、`rcrmd_batches/rows/active_q/loss_sum=0/0/0/0`、aux/VJP为N/A或0、terminal pass；G=`6/6` common 3场景×28 cells、G 3场景×28 cells、`rcrmd_batches=1200`、`rcrmd_total_rows=153600`、active_q>0、首个active raw encoder/exact head VJP完成、terminal pass。receipt schema=`cvs.phase1.rcrmd_receipt.v1`。

### 8.1 final checkpoint技术身份（远端SHA，只读计算，未下载）

|candidate|GPU|child PID|size(bytes)|SHA256|
|---|---:|---:|---:|---|
|F1C_RCRMD12|0|764318|9469243|`44a70c0404ab90d37fa37963977b7ad5063aee283e669cb4811f67b523f2a682`|
|F5G_RCRMD12|0|764320|10682875|`08c9aa034142942ec8ec4aaad938d056c1b846d2cb9479261a7c102671753eae`|
|F1G_RCRMD12|1|764322|10682875|`cae272233557e3f9eac0a428615460d2f0d73a4c71b268a948aeef80b5a82037`|
|F5C_RCRMD12|1|764324|9469243|`ff6d5077c439859dfc3ce7c46cfdefa135c568b6bc6c51a5eb0da52d70f93907`|
|F2C_RCRMD12|2|764326|9469243|`b02e41ca1fda183fe12ed0d480cc5a76495dbadfd9cbeb46e31dd217aba98b6d`|
|F6G_RCRMD12|2|764328|10682875|`b7acbb06dd2f459586fc2dab964ff1f51160ebc01cd52279882691fe96aec19a`|
|F2G_RCRMD12|3|764330|10682875|`e1271cf95dc183150fa0f854a7d02163bf4455f53dc5caa0d7a66c3775cc1863`|
|F6C_RCRMD12|3|764332|9469243|`8a17b326d9c85256f0f8cb52f8302054955c705e9a89a40f6210885683c3de69`|
|F3C_RCRMD12|4|764334|9469243|`4aa6595caefa05ce079499bacadf1d10ced884abd47fb42b2cb98bf138086570`|
|F3G_RCRMD12|5|764336|10682875|`0048892d092edf65f2b08543cb7c050ddab607f0883b30e5cde63baf47fe0fb9`|
|F4C_RCRMD12|6|764338|9469243|`7c722da043e40216f98505620155876e5dcc8f1f86a0ef2c93345cf6acf9271e`|
|F4G_RCRMD12|7|764340|10682875|`51dcfca5773b22215393e049a432c14e56a77691fdc33cad05218690e06ea34c`|

### 8.2 工件、bundle与清理

- 远端每臂均有final、`metrics_epoch.csv`、config、training completion、terminal、resource、heldout和RCRMD receipt：各=`12/12`；日志=`12`，`pids.tsv`=`1`，outer=`0`字节；性能CSV未读取或解释。
- 回收bundle：`artifacts/returned_small/phase1_rcrmd12_20260810_v1_small_artifacts.tar.gz`，121 members、16,566,855B、SHA256=`f5c0d053541853b9d8fba22e9adff6a1967df5ba2da9baec49b873e05aae34cc`，无`.pth/.npz/.pt/.npy`；manifest=`phase1_rcrmd12_20260810_v1_small_artifacts.manifest.txt`，17,526B、SHA256=`5c5242237bd55ac4b34bfda89500a4e66f2aab03b63bcf04cf7ec7eb431fd97e`。
- 远端保留release、run/log、compressed bundle、manifest和全部checkpoint；已移除我创建的`/tmp`静态门临时文件及未压缩252MB bundle；未删除任何训练日志、JSON/CSV、checkpoint或SCB v4。
- 最终资源：训练PID=`0`，launcher/wrapper=`0`；GPU0–6约1MiB，GPU7仅既有SCB v4 PID608786约488MiB；本地SSH进程/TCP22=`0`。未启动postfreeze。

## 3.方法与冻结配置

[
m_i^v=ell_{i,y_i}^v-operatorname{logsumexp}_{k
e y_i}ell_{i,k}^v,quad
q_i=[operatorname{sg}(m_i^{clean})-m_i^{leo}]_+^2
]

对冻结source receiver集合`R_s={0,1,2,3,4,5,6}`与local4 TX类：

[
g_{rc}=0;(n_{rc}=0),quad g_{rc}=rac{1}{n_{rc}}sum_{iin I_{rc}}q_i;(n_{rc}>0)
]

[
L_{RCRMD}=rac{1}{4|R_s|}sum_{r,c}g_{rc}=rac{1}{28}sum_{r,c}g_{rc},quad
L_G=L_{base}+0.02L_{RCRMD}
]

- C/G共同：同一`training_final_only`warm-start、物理L样本和批顺序、seed/sampler、clean+单次LEO forward、三场景轮转、40 epochs、新AdamW/AMP初态和`L_base`。
- C：RCRMD关闭，`lambda=0`，仅封存共同coverage；aux/active/loss/VJP为N/A或0。
- G：RCRMD开启，`lambda=0.02`；只读source-L的TX标签与physical-ID绑定RX；不增加forward、样本、epoch、state或重采样。
- 非有限logit/margin/q/g/loss均fail-closed；q=0逐行合法；不删行、不加eps。
- 每场景28格，三场景终态84格；至少一个active q；首个active batch要求共享encoder和exact head raw VJP均finite/nonzero。
- 优化loss不变；账本对账冻结为`32×float32 eps×max(1,|batch|,|cell|)`。

共同超参：epochs=40、batch=128、lr=2e-4、weight_decay=1e-4、label_smoothing=0.01、seed=7281105、sat_seed=9281105、`lambda_sat_cons=0.10`。

## 4.冻结矩阵与GPU

|GPU|候选1|候选2|
|---:|---|---|
|0|F1C_RCRMD12|F5G_RCRMD12|
|1|F1G_RCRMD12|F5C_RCRMD12|
|2|F2C_RCRMD12|F6G_RCRMD12|
|3|F2G_RCRMD12|F6C_RCRMD12|
|4|F3C_RCRMD12|—|
|5|F3G_RCRMD12|—|
|6|F4C_RCRMD12|—|
|7|F4G_RCRMD12|SCB v4既有进程，合计不超过2个训练进程|

每fold C/G使用相同GeoSat-C checkpoint、source TX、known-validation TX和proxy TX。Runner不得改fold、receiver、TX、seed、λ、场景、训练长度或GPU映射。

## 5.N607冻结路径与命令

- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 预计release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd12_20260810_v1_d99eb939`
- 精确CWD：`<release>/code`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- GeoSat-C根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_rcrmd12_20260810_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd12_20260810_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd12_20260810_v1_launcher.out`

唯一启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd12_20260810_v1_d99eb939/code && nohup env RUN_ID=phase1_rcrmd12_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd12_20260810_v1_d99eb939/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_rcrmd12_20260810_v1_d99eb939/code/scripts/launch_phase1_rcrmd12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_rcrmd12_20260810_v1_launcher.out 2>&1 < /dev/null &
```

Runner只能调用一次；调用端超时先只读确认是否landed，严禁盲目重发。

## 6.预期工件与技术停止规则

每臂应生成：

- `final_ssdg.pth`
- `metrics_epoch.csv`和JSONL
- config、training completion、terminal、heldout、resource和RCRMD receipt
- C臂共同coverage合同通过且aux为N/A/0；G臂84格、active、VJP和终态合同通过
- 由于P0 promotion默认禁用，预期终态为`NON_PROMOTABLE_P0_DISABLED`/exit8；若工件和合同闭合，该退出语义不视为技术失败。

仅在下列情况停止本run：错误checkout/hash、覆盖风险、协议/P0违反、launcher-wide确定性故障，或至少两个distinct arm在产生终态工件前出现相同确定性异常指纹。停止前必须核CWD/cmdline/PID树，只停止本run并保留partial。不得按accuracy、loss或任何性能值早停。retry默认`NO`。

Runner完成后回收小工件与日志，不下载checkpoint/NPZ，不读性能；状态先到`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。

## 7.后冻结判定（训练完成后另行执行）

固定42步：12 clean export+12 LEO/binding+12 proxy+6 same-fold pair。Gaussian只用L拟合，V/proxy零fit；totalized-L2保留零向量，nonfinite fatal；proxy days/RX/seed/400和ManySig SHA/physical keys固定。

非补偿门：

- clean 6/6四floor不低于C−2pp；
- LEO 18/18四floor不低于C−2pp；
- 每fold三场景overall与全18格overall均不低于C；
- 每foldproxy AUROC增量>0且proxy−V的mean-u gap增量>0，必须6/6。

任一完整门失败即`REJECT_P1_RCRMD_PERMANENT`；均值不得补偿floor，分类端点与proxy端点不得互相补偿。
