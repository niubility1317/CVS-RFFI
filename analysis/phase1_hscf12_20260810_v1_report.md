# Phase1 P1-HSCF 12臂训练实验报告

## 1.状态与目标

- 实验ID：`phase1_hscf12_20260810_v1`
- 预登记日期：2026-08-11；ID沿用2026-08-10冻结设计与launcher命名，不代表已提前运行。
- 当前状态：`PREREGISTERED / LOCAL_VERIFIED / NOT_LAUNCHED / NO_PERFORMANCE_RESULT`
- 操作边界：主代理冻结候选、矩阵、版本和证据边界；唯一N607 Runner只负责归档落地、一次启动、短连接监控和小工件回收。
- 目标：从相同GeoSat-C final-only checkpoint继续训练，比较同折C控制臂与唯一增加P1-HSCF辅助项的G实验臂。
- 可证伪假设：source-known-train L中，同一物理样本clean→单LEO的local4 head-contrast若发生跨样本相对构型压缩或旋转，固定双中心化保真项可能在维持分类端点的同时减少连续Gaussian几何退化。
- 声明边界：不得预称改善proxy、RX/day、真实unknown或Phase3；训练完成只形成技术工件，必须另经固定42步后冻结非补偿门判定。

## 2.冻结版本与本地证据

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`86ec70d2b7f1c1401ea24983a6659431f9706c3d`
- 科学实现commit：`c60a501496c5e84769a32cb12b8f6611ee594c76`
- 独立设计终裁：`P0=0/P1=0/ALLOW-DESIGN-FREEZE`
- 独立actual-diff终裁：`P0=0/P1=0/ALLOW本地Git发布`

|文件|工作树SHA256|用途|
|---|---|---|
|`analysis/phase1_hscf_design_20260810.md`|`9caac99e078e772c7b0f2253ea85d284ed75cfbfe75fafb6c5b59c1784a8cd0d`|冻结公式、权限、VJP和证据边界|
|`code/cvsrffi/phase1_hscf.py`|`806592e38b5dac9c704731fb9f537e3b9e27cc1a47b099d7401c1f06f8075a79`|固定B=128/K=4/分母512损失、收据与终态合同|
|`code/SSDG/train_ssdg.py`|`6d585f7213cf4770fe88c9a0100375f847427ae551ffef870528170b63b7d127`|共同训练路径接入|
|`code/tests/test_phase1_hscf.py`|`a414821bf34f6d4474b5834ac0d854649e2e0bcafeb47b58baabe715469c87c2`|公式、零集、权限、VJP和终态负测|
|`code/scripts/launch_phase1_hscf12_20260810.sh`|`03dd8505657eca3249a0ee38368ddb93a43fd05ce1f4b41964b4dd9af72ac33f`|冻结12臂launcher，Git mode100755|

本地`ssr-gpu`串行验证：

- `py_compile`：core、trainer和focused test通过。
- HSCF focused：`14 passed`。
- HSCF+RECTE+RCAT+RCRMD+CAGM联合回归：`64 passed`，仅4条既有CosFace autocast弃用警告。
- 合成`128×4`原始VJP：LEO shared encoder和exact head weight finite/nonzero；clean路径None/zero；可选bias为zero。
- `train_ssdg.py --help`识别`--phase1_hscf_frozen_mode`、`--phase1_hscf_enabled`、`--lambda_hscf`。
- `bash -n`通过；launcher dry-run精确12臂，C=6、G=6、epochs=40，G启用6臂、C关闭6臂，旧机制均关闭。
- `git diff --check`通过；Git工作树仅保留既有`conversation_index/`未跟踪项。

Runner必须从最终commit生成无prefix、LF-only完整Git archive，记录archive SHA、member SHA、成员数、`code/code=0`和launcher可执行位；不得直接复制Windows mixed-EOL工作树。

## 3.方法与冻结配置

对同一物理source-L批的既有clean与单LEO raw local4 logits，令`K=4`、`B0=128`、`P=I_4-11^T/4`：

`a_i^v=Pℓ_i^v`，`r_i^v=a_i^v-(1/128)Σ_j a_j^v`，`v∈{C,L}`。

唯一G辅助项为：

`L_HSCF=(1/512)Σ_i||r_i^L-sg(r_i^C)||²`，`L_G=L_base+0.02L_HSCF`。

- `B!=128`、`K!=4`或任意非有限值均在backward前fail-closed；不按active样本或场景重归一。
- clean辅助路径完全stop-gradient；LEO exact head weight与shared encoder保持live。首个正批原始VJP必须finite/nonzero；clean辅助VJP必须None/zero，head bias因双中心化预期zero。
- C辅助、正项和VJP为N/A/0；G每个clear/low/rain scene至少有一个正项见证和一次VJP审计。
- C/G共同绑定相同final-only warm-start、head/class order、physical batch/order、seed/sampler、40E、新AdamW空初态、AMP、clean+单LEO forward和共同`L_base`。
- HSCF只读source-known-train L；U零iterate/forward，V/proxy/held/target/day/fold对新项、loss、状态、校准和选模零反馈。
- 不新增view、模型、持久state/cache、样本、epoch、optimizer step或GPU并发；增量为`128×4`临时张量和`O(BK)`归约。

共同超参：epochs=40、batch=128、lr=2e-4、weight_decay=1e-4、label_smoothing=0.01、seed=7281105、sat_seed=9281105、`lambda_sat_cons=0.10`。

## 4.冻结矩阵与GPU

|GPU|候选1|候选2|
|---:|---|---|
|0|F1C_HSCF12|F5G_HSCF12|
|1|F1G_HSCF12|F5C_HSCF12|
|2|F2C_HSCF12|F6G_HSCF12|
|3|F2G_HSCF12|F6C_HSCF12|
|4|F3C_HSCF12|—|
|5|F3G_HSCF12|—|
|6|F4C_HSCF12|—|
|7|F4G_HSCF12|—|

每fold C/G使用相同GeoSat-C checkpoint、source TX、known-validation TX和proxy TX。Runner不得改fold、receiver、TX、seed、lambda、场景、训练长度或GPU映射；启动前只读记录实际GPU占用，每GPU不得超过两个训练实验。

## 5.N607冻结路径与唯一命令

- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf12_20260810_v1_86ec70d2`
- 精确CWD：`<release>/code`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- GeoSat-C根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_hscf12_20260810_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf12_20260810_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf12_20260810_v1_launcher.out`

唯一启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf12_20260810_v1_86ec70d2/code && nohup env RUN_ID=phase1_hscf12_20260810_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf12_20260810_v1_86ec70d2/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_hscf12_20260810_v1_86ec70d2/code/scripts/launch_phase1_hscf12_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_hscf12_20260810_v1_launcher.out 2>&1 < /dev/null &
```

Runner只能调用一次。调用端超时后先清理本地残留SSH，再只读确认run/log/PID/CWD/cmdline是否已landed；严禁盲目重发。retry=`NO`。

## 6.预期工件、健康规则与成功条件

每臂应生成`final_ssdg.pth`、metrics CSV/JSONL、config、training completion、terminal、heldout、resource和HSCF receipt。C应通过共同三场景合同且aux为N/A/0；G应通过三场景正项、VJP和终态合同。P0 promotion默认禁用，预期终态为`NON_PROMOTABLE_P0_DISABLED`/exit8；工件和合同闭合时不视为技术失败。

仅在错误checkout/hash、覆盖风险、协议/P0违反、launcher-wide确定性故障，或至少两个distinct arm在终态工件前出现相同确定性异常指纹时停止本run。停止前必须核对run-root/CWD/cmdline/PID树，只停止本run并保留partial。不得读取accuracy、loss或任何性能值作早停依据。

Runner完成后只回收小日志、JSON/CSV、PID和manifest，不下载checkpoint/NPZ，不解释性能；交接状态先到`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。

## 7.后冻结判定边界

训练技术闭合后另行实现并运行固定42步：12 clean export+12 LEO/binding+12 proxy+6 same-fold pair。Gaussian只用L拟合，V/proxy零fit；使用float64 totalized-L2，保留zero且nonfinite fatal；proxy days/RX/seed/400、ManySig SHA和physical keys固定。

非补偿门：clean 6/6四floor不低于C−2pp；LEO 18/18四floor不低于C−2pp；每fold三场景overall和全18格overall均不低于C；每fold proxy AUROC增量>0且proxy−V mean-u gap增量>0，必须6/6。任一完整门失败即`REJECT_P1_HSCF_PERMANENT`；完整通过也只可进入`PHASE1_ADVANCEMENT_CANDIDATE_PENDING_MAIN_REVIEW`。
