# Phase1 HSCF 12臂v2训练实验报告

状态：`ARTIFACTS_COMPLETE / TECHNICALLY_CLOSED / P0=0 / P1=0 / NO_PERFORMANCE_RESULT`

## 1.实验身份与目标

- 实验ID：`phase1_hscf12_20260811_v2`
- 日期：2026-08-11
- 操作方：Codex主控；N607唯一Runner待交接
- 实现commit：`6e1a032f1f2f9dc3c196767c8070db6fea357506`
- 目标：仅修复v1中6个G臂共同出现的AMP技术故障，再运行原冻结HSCF 6fold×C/G矩阵。v2不改变HSCF公式、`λ=0.02`、数据、fold、seed、epoch、optimizer、forward或GPU映射。
- 对照：C为GeoSat-C`training_final_only`共同续训；G只在同一共同路径上增加P1-HSCF。
- 声明边界：进程落地、技术完成、receipt闭合均不等于性能结果。完整12臂同row工件返回前不读取或解释性能；本实验不构成unknown、真实开放集或Phase3能力声明。

## 2.v1故障与v2假设

v1运行`phase1_hscf12_20260810_v1`中6C完整，6G在`post_backward_combined_gradient_nonfinite`以同一`HSCF_AUX_GRADIENT_OR_HEAD_PATH_FAILURE`终止。根因是HSCF分支在`scaler.unscale_(optimizer)`后遇到非有限scaled梯度即报错，没有执行GradScaler标准`step()+update()`的skip/backoff。

v2的可证伪技术假设为：

1. material loss与raw combined-loss VJP有限时，scaled overflow可由标准GradScaler跳过该optimizer step并降低scale，AdamW state不得推进；
2. raw或material梯度非有限仍立即fail-closed；
3. 后续有效optimizer step清零连续skip；终态连续skip未清零或全程零有效step拒绝；
4. 为异常raw VJP暂时保留的autograd图，在本批receipt、日志和telemetry物化后、下一次forward前显式释放；不得跨批保留saved tensor。

上述修复只处理运行健全性，不是方法超参或性能选择规则。

## 3.冻结方法与公平合同

`B=128`、`K=4`、`D=512`，对同物理source-L批的clean/单LEO raw local4 logits做class中心化和batch中心化：

`L_HSCF=(1/512)Σ_i||r_i^LEO-sg(r_i^clean)||²`，G臂固定`L_G=L_base+0.02L_HSCF`。

共同约束：

- C/G同GeoSat-C final-only warm-start、physical batch/order、seed、sampler、40E、新AdamW、AMP、clean+单LEO forward及clear/low/rain循环；
- 仅source-known L进入HSCF；U零iterate/forward/loss/backward/optimizer；V/proxy/held/target对训练、校准和选模零反馈；
- C辅助N/A/0；G每个scene需正项与raw VJP闭合；
- 不增加view、model forward、epoch、持久模型state/cache或GPU并发；
- postfreeze仍为固定42步，且只有v2训练12/12技术闭合后才允许启动。

## 4.本地版本、文件与验证

|文件|SHA256|用途|
|---|---|---|
|`analysis/phase1_hscf_design_20260810.md`|`c4e5911df8559c370aaee2cd40f5e6524bbc5b324220f0cd194515b512e2a7df`|AMP恢复与graph-release追踪卡|
|`code/cvsrffi/phase1_hscf.py`|`b1851976563a859637b40209f69f812f187fc5187047afc95bb0e5c33781b721`|overflow分类、skip/backoff、receipt与图释放helper|
|`code/SSDG/train_ssdg.py`|`efd3885208da61b42ce56865eeb7b5d0e710f99a2ebac1a3845e35d57253573b`|训练接线、逐批事件、批尾graph-release|
|`code/tests/test_phase1_hscf.py`|`421772f678acc3c1cd6f2ff043bd9ef025cb5b4a505d9e4d333967576f0f2c8f`|AMP、receipt、saved-tensor和集成负测|
|`code/scripts/launch_phase1_hscf12_20260810.sh`|`03dd8505657eca3249a0ee38368ddb93a43fd05ce1f4b41964b4dd9af72ac33f`|未修改的冻结12臂launcher；v2通过显式RUN_ID隔离|

本地验证均在官方Conda hook激活`ssr-gpu`后串行完成：

- `python -m py_compile`：HSCF core、trainer、test通过；
- `pytest -q code/tests/test_phase1_hscf.py`：20 passed；
- HSCF+RECTE+RCAT+RCRMD+CAGM联合：70 passed；
- 真实CUDA GradScaler：`65536→32768`，overflow批参数与AdamW state不推进；
- finite/overflow saved-tensor弱引用：释放后且任何GC前旧token为0；
- `train_ssdg.py --help`退出0；
- launcher `bash -n`通过，dry-run=12，C=6、G=6；
- `git diff --check`通过；
- 独立actual-diff复审：`P0=0 / P1=0 / ALLOW`。

## 5.冻结矩阵与GPU映射

|顺序|候选|fold|arm|GPU|source train TX|known-val TX|proxy TX|
|---:|---|---:|---|---:|---|---|---|
|1|F1C_HSCF12|1|C|0|20-15,20-19,6-15,8-20|14-7|14-10|
|2|F5G_HSCF12|5|G|0|14-10,14-7,20-15,20-19|8-20|6-15|
|3|F1G_HSCF12|1|G|1|20-15,20-19,6-15,8-20|14-7|14-10|
|4|F5C_HSCF12|5|C|1|14-10,14-7,20-15,20-19|8-20|6-15|
|5|F2C_HSCF12|2|C|2|14-10,20-19,6-15,8-20|20-15|14-7|
|6|F6G_HSCF12|6|G|2|14-7,20-15,20-19,6-15|14-10|8-20|
|7|F2G_HSCF12|2|G|3|14-10,20-19,6-15,8-20|20-15|14-7|
|8|F6C_HSCF12|6|C|3|14-7,20-15,20-19,6-15|14-10|8-20|
|9|F3C_HSCF12|3|C|4|14-10,14-7,6-15,8-20|20-19|20-15|
|10|F3G_HSCF12|3|G|5|14-10,14-7,6-15,8-20|20-19|20-15|
|11|F4C_HSCF12|4|C|6|14-10,14-7,20-15,8-20|6-15|20-19|
|12|F4G_HSCF12|4|G|7|14-10,14-7,20-15,8-20|6-15|20-19|

共同配置：`epochs=40`、`seed=7281105`、`sat_view_seed=9281105`、`lr=0.0002`、`weight_decay=0.0001`、`batch_size=128`、`amp=true`、`max_grad_norm=5.0`、`checkpoint_selection=final_only`。每GPU最多2臂。

## 6.N607发布预登记

- 普通账号目标：`N607`；禁止管理员账号。
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf12_20260811_v2_6e1a032f`
- CWD：上述release的`code`目录
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hscf12_20260811_v2`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf12_20260811_v2`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf12_20260811_v2_launcher.out`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- warm-start根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`
- retry：`NO`
- 启动所有权：唯一Runner；主控不得重复启动。

冻结唯一命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf12_20260811_v2_6e1a032f/code && nohup env RUN_ID=phase1_hscf12_20260811_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf12_20260811_v2_6e1a032f/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf12_20260811_v2_6e1a032f/code/scripts/launch_phase1_hscf12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf12_20260811_v2_launcher.out 2>&1 < /dev/null &
```

Runner落地前必须完成：direct preflight、Git archive/成员SHA/mode/LF/code-code0、ManySig SHA、6个GeoSat-C checkpoint SHA、远端py_compile/help/bash-n/dry-run12、run/log/outer不存在、GPU占用记录。SSH超时后先清理本地ssh/TCP22，再只读确认是否landed；不得重发。

## 7.技术健康、停止与预期工件

预期每臂：

- `final_ssdg.pth`
- training completion receipt
- terminal receipt
- `phase1_hscf_receipt.json`
- config、resource、heldout receipt
- 完整训练日志

G臂额外必须闭合：

- 三scene共同B128/local4/denom512；
- raw-finite overflow时pre/post scale下降且optimizer state不推进；
- `attempts=effective_steps+raw_finite_skips=hscf_batches`；
- raw/material非有限计数为0；
- 终态连续skip为0且effective step>0；
- 三sceneHSCF正项及raw VJP；
- graph-release failure stage不得出现。

预注册技术停止：

- P0协议/权限/checkout/hash/输出覆盖错误；
- launcher-wide确定性故障；
- 至少2个不同arm在产生final前出现相同标准化异常指纹；
- OOM、CUDA、argparse、路径/权限错误或零prediction/final闭合失败。

停止只依据技术健康，不读取accuracy、loss、floor或proxy表现。停止前必须绑定本run的PID/CWD/cmdline，先温和终止，仅处理本run进程并保留全部部分工件。健康失败记为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不得重跑本ID。

## 8.完成后分析边界

只有12/12臂技术工件和完整日志匹配后，主控才可读取同row性能并决定是否进入HSCF 42步postfreeze。届时必须补充每fold C/G、clean、三LEO scene、min-class/min-RX/min-day、fold/global overall及proxy双门；任何局部最大值不得替代完整行或非补偿门。

当前结果表：

|候选|技术状态|性能状态|最终判定|
|---|---|---|---|
|F1C/F1G…F6C/F6G|待N607 v2|NO_PERFORMANCE_RESULT|PENDING|

## 9.风险与检查重点

- v1的6G共同故障可能是可恢复scaled overflow，也可能暴露raw combined梯度问题；v2通过异常raw VJP直接区分。
- 显式graph-release清理64个冻结路径别名；独立AST和无GC weakref测试已闭合，但真实双view训练仍需观察显存与receipt。
- v2不得因AMP skip数量或中间性能调参；公式、scale策略和matrix均冻结。
- v2若技术失败，保留证据并回本地修具体缺陷；不得把失败run转为性能实验。

## 10.Runner技术收口（2026-08-11）

状态：`ARTIFACTS_COMPLETE / TECHNICALLY_CLOSED / NO_PERFORMANCE_RESULT`。唯一Runner完成一次release SCP、一次精确启动和一次技术bundle SCP；`retry=NO`。启动SSH在124秒超时，按规则仅清理绑定的本地`ssh.exe` PID=`14756`并只读确认已落地，未重发。两次prelaunch探针中止均为机械验证器路径/计数错误，未改变代码、矩阵、命令、SCP或启动次数。

### 10.1归档、发布与输入闭合

- 最终归档：`automation_reports/CV-SincNet/phase1_hscf12_20260811_v2/release/phase1_hscf12_20260811_v2_6e1a032f_final.tar`；SHA256=`0760e1dad9f6aa0be5f30c3b23ee9401d9b4a158e9a7111200ecd2aab5480acd`，`266977280`字节，4957 members（4337 files、620 dirs），无顶层prefix、`code/code=0`、文本CR字节=0。`code/SYNC_MANIFEST.txt`仅在归档层做CR→LF规范化；科学代码与commit未改动。关键归档成员：design=`c4e5911df8559c370aaee2cd40f5e6524bbc5b324220f0cd194515b512e2a7df`，core=`b1851976563a859637b40209f69f812f187fc5187047afc95bb0e5c33781b721`，train archive-LF=`eae726be9e8b332eeda1b36a45cce999f3e96e1e2a3520eec78533fe493985ff`（工作树换行SHA=`efd3885208da61b42ce56865eeb7b5d0e710f99a2ebac1a3845e35d57253573b`），test=`421772f678acc3c1cd6f2ff043bd9ef025cb5b4a505d9e4d333967576f0f2c8f`，launcher=`03dd8505657eca3249a0ee38368ddb93a43fd05ce1f4b41964b4dd9af72ac33f`，mode=`775`。
- 远端incoming归档SHA/字节数与本地一致；release静态门`py_compile`、`train_ssdg.py --help`（95932 bytes，HSCF命中9）、`bash -n`和dry-run=`12`（C=6、G=6）均通过。release落地后无pycache；训练期间自然生成5个runtime`__pycache__`，按证据保留。
- ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。GeoSat-C checkpoint SHA：F1=`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`，F2=`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76d`，F3=`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`，F4=`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`，F5=`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`，F6=`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`。

### 10.2运行绑定与技术结果

|候选|arm/GPU|child PID|终态|final/完成/terminal/resource/heldout/config|HSCF技术计数|判定|
|---|---|---:|---|---|---|---|
|F1C_HSCF12|C/0|1320599|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|aux=N/A；contract=true|技术闭合|
|F5G_HSCF12|G/0|1320601|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|batches=1200；effective/attempts=1197/1200；raw-finite skip=3；raw/material nonfinite=0；scale↓=optimizer unchanged=3；persistent=false；3×scene400/51200/positive400；audit=true|技术闭合|
|F1G_HSCF12|G/1|1320603|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|batches=1200；effective/attempts=1197/1200；raw-finite skip=3；raw/material nonfinite=0；scale↓=optimizer unchanged=3；persistent=false；3×scene400/51200/positive400；audit=true|技术闭合|
|F5C_HSCF12|C/1|1320605|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|aux=N/A；contract=true|技术闭合|
|F2C_HSCF12|C/2|1320607|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|aux=N/A；contract=true|技术闭合|
|F6G_HSCF12|G/2|1320609|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|batches=1200；effective/attempts=1197/1200；raw-finite skip=3；raw/material nonfinite=0；scale↓=optimizer unchanged=3；persistent=false；3×scene400/51200/positive400；audit=true|技术闭合|
|F2G_HSCF12|G/3|1320614|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|batches=1200；effective/attempts=1198/1200；raw-finite skip=2；raw/material nonfinite=0；scale↓=optimizer unchanged=2；persistent=false；3×scene400/51200/positive400；audit=true|技术闭合|
|F6C_HSCF12|C/3|1320617|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|aux=N/A；contract=true|技术闭合|
|F3C_HSCF12|C/4|1320619|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|aux=N/A；contract=true|技术闭合|
|F3G_HSCF12|G/5|1320627|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|batches=1200；effective/attempts=1197/1200；raw-finite skip=3；raw/material nonfinite=0；scale↓=optimizer unchanged=3；persistent=false；3×scene400/51200/positive400；audit=true|技术闭合|
|F4C_HSCF12|C/6|1320632|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|aux=N/A；contract=true|技术闭合|
|F4G_HSCF12|G/7|1320636|exit=8`NON_PROMOTABLE_P0_DISABLED`|1/1/1/1/1/1|batches=1200；effective/attempts=1198/1200；raw-finite skip=2；raw/material nonfinite=0；scale↓=optimizer unchanged=2；persistent=false；3×scene400/51200/positive400；audit=true|技术闭合|

12/12 child PID均自然退出，12/12 arm日志存在；技术异常扫描（Traceback、RuntimeError、argparse、OOM、CUDA、SIGSEGV、HSCF异常指纹）和graph-release failure扫描均为空。所有GPU最终为0%利用率、1MiB显存。`performance_result_available=false`、`promotion_ready=false`；本节不读取或解释accuracy、loss、floor、proxy或metrics_epoch。

### 10.3证据bundle与远端保留

- 远端技术bundle：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf12_20260811_v2/phase1_hscf12_20260811_v2_technical_evidence.tar`；本地回收于`automation_reports/CV-SincNet/phase1_hscf12_20260811_v2/retrieved/phase1_hscf12_20260811_v2_technical_evidence.tar`；SHA256=`045eb94c1200f00e9a7779882558ad03de2c06e1479dbb87b53da53e4c657d5f`，`26071040`字节，98 members；manifest=`d190b9901860ee5616e9d442d7a99ffd29649e8ac02b8930ff39a306e8866c7d`，14026 bytes。bundle含12 arm日志、12臂技术JSON、pids.tsv和manifest，排除`metrics_epoch*`、`.pth/.npz/.pt/.npy`；outer为其父目录的0字节文件，SHA=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`，远端保留。
- 远端release、run、log、incoming归档、checkpoint和technical bundle均保留；未删除或覆盖任何既有数据、权重、日志或未归属文件。实验仅达到技术收口，不触发postfreeze或性能分析。
