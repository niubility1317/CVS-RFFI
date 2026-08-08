# Phase1 CCPC-LEO六折C/G v3有限零梯度修复报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`STOPPED_EARLY_TECHNICAL_AUDIT_DEFECT / NO_PERFORMANCE_RESULT`

证据边界：`PHASE1_SOURCE_ONLY_OPEN_WORLD_READY_REPRESENTATION_NON_CONFIRMATORY`

## 1.目标与可证伪假设

实验ID：`phase1_ccpc_leo12_20260809_v3`。时间：2026-08-09。operator：Codex主控；N607唯一runner：专属实验子代理。

假设：在不使用RX/domain对齐、proxy/held训练、拒识head或阈值的前提下，LEO anchor对detached clean同类bank的class-conditional paired contrastive（CCPC）能提高跨接收场景的身份表示稳定性，同时保持clean known精度。C为同一GeoSat-C checkpoint的无CCPC续训；G只增加固定`T=0.12、lambda=0.02`的CCPC loss。

v2技术停止的直接原因不是loss非有限或断图，而是把有限零梯度合法驻点误判为技术故障。v3只修监控与回执：finite-zero累计后继续；gradient None和nonfinite仍fail-closed；G终态必须`nonzero_batches>=1 && nonfinite_batches==0`。loss、lambda、T、数据、AMP、seed、checkpoint、矩阵和GPU映射均不变。

## 2.冻结矩阵与资源

| Fold | train TX | known-validation TX | proxy TX | C/G GPU |
|---|---|---|---|---|
| F1 | 20-15,20-19,6-15,8-20 | 14-7 | 14-10 | 0/1 |
| F2 | 14-10,20-19,6-15,8-20 | 20-15 | 14-7 | 2/3 |
| F3 | 14-10,14-7,6-15,8-20 | 20-19 | 20-15 | 4/5 |
| F4 | 14-10,14-7,20-15,8-20 | 6-15 | 20-19 | 6/7 |
| F5 | 14-10,14-7,20-15,20-19 | 8-20 | 6-15 | 1/0 |
| F6 | 14-7,20-15,20-19,6-15 | 14-10 | 8-20 | 3/2 |

GPU0:F1C+F5G；GPU1:F1G+F5C；GPU2:F2C+F6G；GPU3:F2G+F6C；GPU4:F3C；GPU5:F3G；GPU6:F4C；GPU7:F4G。每卡最多2个训练任务。

固定40epoch、seed=7281105、sat_view_seed=9281105、AdamW、AMP、final-only；C/G按fold严格载入同一`phase1_loto_clsgeo12_20260808_v1/F*C_LOTO_CLSGeo12/final_ssdg.pth`模型权重，并新建optimizer/AMP/RNG状态。

## 3.本地版本与验证

implementation commit：`90fac195606a8598cf2c734a950b1967e2e778b7`。

| 文件 | SHA256 | 用途 |
|---|---|---|
| `code/SSDG/train_ssdg.py` | `a78d2c2fe39e352001b45cfa21ebed248fc1cc0dec42dccc77e8e004fc598807` | 梯度三态审计、失败/终态receipt |
| `code/cvsrffi/phase1_ccpc_leo.py` | `134b83868392e3e866438bca7a9f6f641e42c5300b521867854a0ad57fd39906` | frozen CCPC loss与receipt helper |
| `code/tests/test_phase1_ccpc_leo.py` | `85bfb2ac91f21d870f9afb37f5238e792756c250e82c5d30e75496e5731ea4a9` | focused正负测试 |
| `code/scripts/launch_phase1_ccpc_leo12_20260809.sh` | `e4d39695f171e1449cddf090e91b47f30b28bc692c4a23eb89aac9c39bf4469e` | 冻结12任务launcher |
| `analysis/phase1_ccpc_leo_design_20260809.md` | `158b6d556be2ac408937d45eb4ac8d95be9bd19f03f6f6644f3adcff0a9c6eff` | 设计追溯 |

`ssr-gpu`串行验证：py_compile通过；`pytest -q code/tests/test_phase1_ccpc_leo.py`为15 passed；`bash -n`通过；launcher dry-run为12条；`git diff --check`通过。独立复核：`APPROVE / Critical=0 / Important=0`。writer失败不会遮蔽原CCPC异常，marker不含路径、原错误文本或数据。

## 4.N607冻结路径与唯一命令

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v3_90fac195`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v3`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v3`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v3.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

```text
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v3_90fac195/code && nohup setsid env RUN_ID=phase1_ccpc_leo12_20260809_v3 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v3_90fac195/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python WISIG_PKL=/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl GEOSAT_CKPT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1 RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccpc_leo12_20260809_v3 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v3 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccpc_leo12_20260809_v3_90fac195/code/scripts/launch_phase1_ccpc_leo12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccpc_leo12_20260809_v3.launch.out 2>&1 < /dev/null & echo $!
```

## 5.健康门、预期artifact与停止规则

启动后核launcher PID/CWD/cmdline、12 child与GPU映射、日志增长、`CONFIG-CCPC-LEO`、E001及梯度三态字段。禁用项、未执行项或0/0统计的N/A `nan`不触发停止。

技术停止仅接受：协议/路径/hash/overwrite错误；真实loss/模型输出/梯度非有限；gradient None；Traceback/OOM/CUDA故障；至少2个不同任务同一确定性异常；或预注册无进展。有限零梯度不停止。停止前必须绑定精确run树；只停止本run，保留partial；不得重启、调参或远端修码。retry=`NO`。

每任务预期：`final_ssdg.pth`、metrics CSV/JSONL、config receipt、terminal receipt、heldout/resource receipt；G终态receipt必须证明至少1个非零梯度批且0个非有限批。log根预期`pids.tsv`、`completion.tsv`与12 stdout。

训练闭环后才执行独立postfreeze：clean known、source proxy连续排序、三LEO场景同physical C/G配对；proxy/held/LEO均零fit、零校准、零选参。性能门仍为clean known 6/6四项G-C>=-2pp、LEO 18/18四项G-C>=-2pp且总体改善、proxy排序同向和artifact闭环；任一失败即`REJECT_CCPC_LEO_NO_RETRY`，不得进入Phase3。

## 6.运行终态（2026-08-09）

v3按冻结命令仅启动一次，复用commit=`90fac195606a8598cf2c734a950b1967e2e778b7`的无prefix归档；wrapper PID=`4030420`、launcher PID=`4030421`，12个child的C/G与GPU绑定记录在`artifacts/pids.tsv`和`artifacts/manifest.json`。归档SHA=`dfed9477ba427fd0943194bf322541e2a41ab9128a6b084f061402bc68ec069c`（261212160B），远端五文件archive SHA按LF口径记录；worktree与archive差异仅为换行，不涉及远端改码。

首波确认12个`CONFIG-CCPC-LEO`与日志增长。`sat_cos=nan`、禁用domain的`dom=nan%`、未执行test的`overall_tx=nan%(0/0)`等N/A占位被正确忽略；有限零梯度也按v3规则继续。F1G（E002）、F3G（E013）、F4G（E005）分别写出`ccpc_failure_receipt.json`，均为`error_fingerprint=CCPC_LEO_GRADIENT_NONFINITE`、`failure_stage=post_backward_leo_gradient_audit`、`ccpc_grad_nonfinite_batches=1`且`ccpc_grad_zero_batches=0`，随后抛出同一明确异常：`CCPCLEORuntimeError: CCPC-LEO fail-closed: paired LEO feature gradient is non-finite`。该指纹满足至少两个任务系统性技术停止条件；未发现日志中的真实`loss_total`/`loss_ccpc_leo`非有限值。

停止前核对roots=`{4030420,4030421}`及精确run树，bound count=83；TERM后仅roots短暂残留且无训练子进程，随后reap，最终v3进程数为0。未触碰无关任务、未重试、未重启、未远端修码。停止后GPU0–7均0%/1MiB；SSH/SCP短连接已断开，TCP22无残留。

只回收小证据至`E:\type10-7\automation_reports\CV-SincNet\phase1_ccpc_leo12_20260809_v3\artifacts`：原始54文件（39 run小文件含3失败回执、12 stdout、pids/completion/outer），另生成本地`manifest.json`，当前共55文件；未下载checkpoint、dataset或大型数组。`completion.tsv`为header-only，outer为空；run在停止时无`final_ssdg.pth/latest_ssdg.pth`。本run标记`NO_PERFORMANCE_RESULT`，不用于性能比较、晋级或Phase3结论；retry=NO。

## 7.主控技术复判与下一步

独立复核确认，v3的停止证据是有效的runner触发证据，但不是CCPC方法数值失败证据。代码在`scaler.scale(total_loss).backward()`之后读取`retain_grad()`得到的非parameter中间张量`ccpc_leo_feature.grad`；`scaler.unscale_(optimizer)`只还原optimizer参数梯度，不会还原该中间梯度。因此receipt中的`NONFINITE`表示“被GradScaler放大后的中间梯度溢出”，不能证明未缩放的`d(lambda*L_CCPC)/dz_leo`非有限，也不能据此REJECT方法。v3重新归类为`STOPPED_EARLY_TECHNICAL_AUDIT_DEFECT / NO_PERFORMANCE_RESULT`。

按两轮发布修复上限，不直接重跑完整12任务。下一步冻结一个新run ID的6fold G-only、15epoch真实checkpoint one-shot：在scaled backward前用`torch.autograd.grad(lambda*loss_ccpc, z_leo)`审计未缩放且CCPC专属的梯度，并同时记录参数梯度有限性和optimizer step；loss、lambda、T、AMP、数据、checkpoint和seed均不变。该one-shot只决定技术可运行性，不读取性能、不执行postfreeze；通过后才允许重新发布完整C/G矩阵。
