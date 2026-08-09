# Phase1 GD-ProtoNLL postfreeze v2报告

状态：`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`

日期：2026-08-10

## 1.目标、唯一修复与证据

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll_postfreeze_20260810_v2`|
|目标|对同一GD-ProtoNLL v3 12个final checkpoint完整重算42步并形成6折aggregate|
|v1终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；12 GD-clean、12 LEO、12 proxy完成，F1–F5 pair生成，F6在零范数几何处停止；旧pair不得沿用|
|权威诊断|253440行clean feature无nonfinite；唯一zero为F6C `source_validation_known` 1/16800；所有L与proxy zero=0|
|唯一数学修复|totalized L2：正范数`z/||z||₂`，精确零范数映射为零向量；全部L/V/proxy行保留，nonfinite仍fatal|
|不变项|exporter、L-only Gaussian、ddof1、class-equal pooled variance、.9/.1 shrink、1e-6 floor、完整NLL、u、分类/proxy非补偿门、训练root和42步矩阵|
|实现commit|`62d0cf905785c4fac7dff98bf824ff28de20d523`|
|独立复核|`P0=0、P1=0、ALLOW`；pycompile、focused10、bash-n、dry-run42、diff-check及zero-L/zero-V手算均通过|

文件SHA：evaluator=`0612acea58d61d1bca40d54ae1a404df700900f4ecf9900f191104dfd1627409`；tests=`d7367c6525127e315785c79790eb22d7942558f77f95f3fca26439387f6b4b9d`；launcher=`52d50aaab5292840b94faf4d6c7f687315e7ee65a42b0953e0da4711404ee7f3`；design=`30e4bd1ecfb920efc7bb8426ee5238badb28159adba19a5f6d7b7426cb48195c`。exporter沿用v1字节`c21d84a76c5448e0c8414389222ad764575820f011b05784982d313499f71580`。

zero的u完全由L-only Gaussian决定，不人为奖励、删除或施加固定惩罚，也不预判方向。每折receipt须封存C/G×L/V/proxy的total/positive/zero/nonfinite/retained/dropped与L逐类计数，且F6拒绝v1 schema、跨root或旧prior。

## 2.固定数据、门与42步

训练root只读：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3`；ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。专用clean严格重建L/U/V，只forward L、V、proxy，U forward=0；clean分类V、Gaussian fit L、known u V、unknown u proxy。LEO为source-only 1600行三场景。

硬门保持：clean 6/6四floor≥C−2pp；LEO 18格四floor≥C−2pp；每折三场景等权overall Δ≥0；18格等权overall Δ≥0；逐折6/6同时满足ΔAUROC>0与Δ(proxy mean u−V mean u)>0。任一失败即`REJECT_GD_PROTO_NLL_PERMANENT`，不重试、不调参。

42步固定为12 GD-clean＋12 LEO＋12 proxy＋6 pair；GPU矩阵0=`F1C＋F5G`、1=`F1G＋F5C`、2=`F2C＋F6G`、3=`F2G＋F6C`、4=`F3C`、5=`F3G`、6=`F4C`、7=`F4G`。

## 3.N607路径与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v2_62d0cf90`|
|CWD|`<release>/code`|
|postfreeze root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll_postfreeze_20260810_v2`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v2`|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v2_launcher.out`|
|retry|NO|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v2_62d0cf90/code && nohup env POSTFREEZE_RUN_ID=phase1_gd_proto_nll_postfreeze_20260810_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v2_62d0cf90/code TRAIN_RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v3 bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll_postfreeze_20260810_v2_62d0cf90/code/scripts/launch_phase1_gd_proto_nll_postfreeze_20260810.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll_postfreeze_20260810_v2_launcher.out 2>&1 < /dev/null &
```

## 4.发布与停止

唯一runner先direct preflight；新release/run/log/outer必须ABSENT，v1只读保留；核ManySig、12 final、archive/member、远端pycompile/help/bash-n/dry-run42和每卡≤2后唯一启动。caller超时仅只读确认，不重发。

只因路径/hash/覆盖、split/head/checkpoint/arm/root、Traceback/OOM/CUDA、确定性异常、缺输出或零pair停止，绝不按性能数值提前停。完成后核12 clean NPZ、12 LEO NPZ、12 proxy JSON/CSV、6个v2 pair JSON与F6 aggregate；不下载NPZ/checkpoint，只回收小artifact。runner只记录原始verdict，不解释性能。

## 5.落地、静态核验与启动前状态（2026-08-10 01:07 CST）

- direct `N607` preflight通过（01:06:35 CST）；项目根、GPU与服务器时间可见。GPU0–6各约1MiB，GPU7约498MiB；SCB v3 PID=`420208`约488MiB，未见v1/v3活动进程；本地每次SSH/SCP后`ssh.exe=0`且N607/bridge TCP22为0。
- 新v2 release、run、log、outer及临时包在落地前均为`ABSENT`；v1 release/run仅只读保留；v3 training root保持12个`final_ssdg.pth`。ManySig远端SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；6个GeoSat-C baseline full SHA与冻结值一致：F1C=`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`、F2C=`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76`、F3C=`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`、F4C=`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`、F5C=`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`、F6C=`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`。
- 由实现commit=`62d0cf905785c4fac7dff98bf824ff28de20d523`生成无prefix archive：本地`263168000` bytes、SHA256=`93a66efe43776abc7d8e8223281a387d780c3338842bd79c7450e46fbffe910f`、`4885` members、`code/code=0`；远端临时包同SHA/大小/成员数并解包至新release。远端LF member SHA：pair=`1e8aeefe5d5a3ce65dbc800d3f4510a397ab5702396cc4c64f6bbe97f6c0cc0c`、GD-clean exporter=`6f75e19352ec400a5f64ee900ec476f09894974ccdf12b243eaf60a18b175874`、launcher=`52d50aaab5292840b94faf4d6c7f687315e7ee65a42b0953e0da4711404ee7f3`、test=`dd1c5c42ce0fd6c79e749a041d371fe024f51f025ea24ca767e686d28769ec03`、design=`2544c2954b5b93262a7793b737d55c76141aab509c737af5667cd673f360e923`。
- 远端静态核验通过：相关Python模块/测试`py_compile`、GD exporter/pair/proxy `--help`、launcher `bash -n`；`--dry-run`精确42行，GD-clean=12、LEO=12、proxy=12、pair=6，dry-run bytes=`55470`、SHA256=`9a7fbebc38966f2a9a541bc7eb93de2b73db7072ec0a891c7fcd2eaa88660bb0`。静态临时文件已删除并核验`ABSENT`。

启动前唯一命令尚未执行；随后严格按本报告§3显式`bash`命令调用一次，caller返回`-1`后仅只读确认已landed，未重发。

## 6.唯一启动、42步技术闭合与小工件（2026-08-10）

- 严格按§3命令显式调用`bash`一次；caller返回`-1`，但只读确认已落地，未重发。wrapper PID=`539235`、launcher PID=`539236`、run PGID=`539234`；`candidate_pids.tsv`记录12个primary child：GPU0=`539239(F1C),539240(F5G)`、GPU1=`539241(F1G),539242(F5C)`、GPU2=`539243(F2C),539244(F6G)`、GPU3=`539245(F2G),539247(F6C)`、GPU4=`539251(F3C)`、GPU5=`539252(F3G)`、GPU6=`539256(F4C)`、GPU7=`539259(F4G)`。每个CWD/cmdline均绑定v2 release与v2 run；12 child及launcher/runner均自然退出，未执行kill；GPU0–6回到空闲基线，GPU7仅保留SCB PID=`420208`约488MiB。
- 42步完整技术输出：GD-clean NPZ=`12`（留N607、未下载）、LEO NPZ=`12`（留N607、未下载）、proxy JSON/CSV=`12/12`、v2 pair JSON=`6`；6个pair均为`cvs.phase1.gd_proto_nll_postfreeze_pair.v2`，matrix/output/training root binding均通过，F6 aggregate存在且`gates.technical_binding.passed=true`；outer launcher=`0` bytes，技术异常指纹=`0`。未读取或解释任何性能字段。
- NPZ只读技术诊断（未下载NPZ、未输出特征值）：12个GD-clean均`21120×160`，nonfinite行总数=`0`；仅`F6C_GD_PROTO_NLL12`有1个zero-norm行，归属`source_validation_known`（`16800`行、zero=`1`；`labeled_fit`=`3920/0`、`proxy_unknown`=`400/0`），该行按冻结totalized L2保留为零向量，未触发技术失败门。
- 仅回收50-member小bundle（无`.npz/.pth`）与诊断JSON；bundle=`3864595` bytes、SHA256=`1dc723c196f945d4f090804eafebc80d60e571eedb540a083e086e2f2d2629a8`；诊断JSON=`10333` bytes、SHA256=`09d8e01e040e4adfc480bea46784c3d6922e7ebc7aea7f8f8b381fa7033099c9`。逐项清单`remote_artifact_sha256.tsv`共53 rows、`8807` bytes、SHA256=`9a5b4b8a2b7686439b388977a6164277d771b7f17d1a9ff5ae744bd139dc627a`；`completion.tsv`=`2097` bytes、SHA256=`51b97b58a5d393caab9ec419147fc7efc14a4c4e78e4672e951a481cf4680975`；`manifest.json`=`7762` bytes、SHA256=`680bfc69a0fda435540ef2a2f7beffbb0f419d867a49831ff399b4c94aaf15d6`；`runner_handoff.json`=`2758` bytes、SHA256=`022adadf44ca7c11e69d6ee6d2ef5d0b84f78bf9d7521c3c14c0a5806a65fb93`。
- 远端临时archive/list/static/bundle/diagnostic均已删除并核验`ABSENT`；run-owned进程=`0`；本地每次SSH/SCP后`ssh.exe=0`、N607/bridge TCP22=`0`。本run只报告技术闭合，无性能结论；runner不再启动新run。
