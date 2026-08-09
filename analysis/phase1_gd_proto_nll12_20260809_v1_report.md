# Phase1 GD-ProtoNLL 12臂正式训练v1报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

日期：2026-08-09

## 1.目标、假设与版本

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll12_20260809_v1`|
|目标|验证L-only、class×LEO scenario滞后EMA重加权是否能在不改变C共同基座的前提下改善困难LEO决策风险|
|C/G差异|C为原GeoSat-C 40E continuation；G唯一增加`.10*L_GD`，两臂使用同一checkpoint、sampler、逐批LEO场景序列、共同`lambda_sat_cons=.10`和final-only策略|
|实现commit|`6465a7f33abb730ae58de4f6e0bec5181f128d0a`|
|Git工作树|`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`，分支`codex/phase3-responsibility-20260807`|
|独立复核|`P0=0、P1=0、ALLOW`|
|声明|训练技术与最终checkpoint生成；本run不读取性能、不作晋级结论，postfreeze另行一次性执行|

## 2.冻结机制与验证

G每个L辅助批必须包含local4四类。三场景沿C既有`(epoch+batch_idx-2)%3`序列轮转；旧`q_t`计算`L_GD=3*sum_c q_t[c,s]*ell_c`，反传后以`beta=.05`更新四个当前格的loss EMA，再对全12格执行`softmax(eta=1)`。焦点指数`gamma=1`、原型余弦scale`16`；feature/head逐行范数必须有限且大于0。首个有效G批仅审计未缩放base与`.10L_GD`在共享encoder及exact head上的梯度norm/cos，不据此调参。Gaussian ProtoNLL函数已实现但formal训练不调用，也不读取V/proxy。

|文件|SHA256|
|---|---|
|`code/cvsrffi/phase1_gd_proto_nll.py`|`71DCFFC9D1BED35BB746C67EB25A9A7C79F31C1DDFA0289E0CC803F17FBEF57D`|
|`code/SSDG/train_ssdg.py`|`815545AA383C4E666EB9295B4A147B8870DB122DEFCA0F84A1F11B48D5250A46`|
|`code/tests/test_phase1_gd_proto_nll.py`|`45EB9972674E9307646EEA054E38D29B4A46689D85FBE81A0B656E73D767EDC5`|
|`code/scripts/launch_phase1_gd_proto_nll12_20260809.sh`|`50F2472C2F71A9FC15DFDD02729107FACFCB68F165FB5CAC872386EFB65C5601`|
|`analysis/phase1_gd_proto_nll_design_20260809.md`|`F28A2C8E20C8D4E91B010FD4B5852C3A73C0AF5CC2C7334A4A7E7EFF3B705F35`|

`ssr-gpu`验证：`py_compile`PASS；GD＋CB＋CP focused`29 passed`，包含lite_d无query前向／反向；`bash -n`PASS；dry-run精确12臂；`git diff --check`PASS。

## 3.数据、起点与矩阵

ManySig固定路径：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。split=`tx_rx_day_1_6_3`，seed=`7281105`，L／U／V=`.07／.63／.30`；formal候选关闭U训练，GD只读L。每折source train为4TX，known source-val与proxy TX由launcher冻结；V只在final-only评分，不进optimizer／EMA／DRO／几何拟合。

|fold baseline|路径后缀|SHA256|
|---|---|---|
|F1C|`F1C_LOTO_CLSGeo12/final_ssdg.pth`|`4d515204f2cea62c5b82313a01b722b3b3d13a3e4fe647ff4b723b69e8a0c040`|
|F2C|`F2C_LOTO_CLSGeo12/final_ssdg.pth`|`29c7d7ca31d80d90d7c0235fa234707b05866914dc0acdae5c44505af1bbd76d`|
|F3C|`F3C_LOTO_CLSGeo12/final_ssdg.pth`|`39c6cdd65aade504efdea956db02cc5e762aee299a9e9319c07ed6fb839434b7`|
|F4C|`F4C_LOTO_CLSGeo12/final_ssdg.pth`|`32d956f44f60844471ba2ef04526c5f40cad0f8bc8acb7249be6035aa85005e4`|
|F5C|`F5C_LOTO_CLSGeo12/final_ssdg.pth`|`2b9381546878b19e7e8e2106a82b0d0a4672a3012ef79bd7f28eadfd03b75a9f`|
|F6C|`F6C_LOTO_CLSGeo12/final_ssdg.pth`|`573ca9d039a8c854f9c0927b5b5c303ab8eeaf527ccd42cd0d764b81e630de6f`|

GPU矩阵：0=`F1C＋F5G`；1=`F1G＋F5C`；2=`F2C＋F6G`；3=`F2G＋F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。若preflight时GPU7仍有SCB v3技术构建，则F4G使GPU7总计算进程为2，仍在默认上限内；任一卡将超过2则不启动。

## 4.N607路径与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v1_6465a7f3`|
|CWD|`<release>/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v1`，启动前ABSENT|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v1`，启动前ABSENT|
|outer log|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v1_launcher.out`，启动前ABSENT|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；caller超时后只读确认landed，不得重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v1_6465a7f3/code && nohup env RUN_ID=phase1_gd_proto_nll12_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v1_6465a7f3/code /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v1_6465a7f3/code/scripts/launch_phase1_gd_proto_nll12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v1_launcher.out 2>&1 < /dev/null &
```

## 5.健康门、期望工件与终态

唯一runner执行direct preflight；仅direct TCP／SSH不可用时使用一次verified bridge。落地前核release／run／log／outer均ABSENT，ManySig及6个baseline SHA，archive/member SHA，`py_compile`、CLI`--help`、`bash -n`及dry-run12。启动后记录launcher、12 child PID、CWD/cmdline、GPU矩阵、日志增长与进程数。

每臂期望40个epoch JSONL／41行CSV、`final_ssdg.pth`、config、GD terminal、training completion、terminal status、resource及heldout receipt。C合同应为`CONTROL_ARM_NOT_APPLICABLE`；G须有1200 batches、153600 rows、12格覆盖、每批EMA状态更新、首批raw梯度审计和`terminal_contract_passed=true`。最终`NON_PROMOTABLE_P0_DISABLED/exit8`是预期P0 gate，不是技术失败；本run不读取其性能字段。

停止仅由路径／hash／覆盖、协议或类序漂移、Traceback／OOM／CUDA／nonfinite、failure receipt、两行同一确定性异常、零epoch／零checkpoint或成员不全触发；不得依据accuracy、loss趋势或任何性能值停止。触发后停止新分派并只终止已严格绑定本run的进程树，保留partial，标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，不重试。

完成后回收小日志、JSON／CSV、PID、completion与manifest；不下载checkpoint或NPZ。更新本报告根与Git镜像，仅由主控在完整postfreeze后解释性能。

## 6.Runner预检与release封存（2026-08-09）

直连`N607`只读预检通过：用户`szu2070436088`、主机`dell-DSS8440`、项目根可见、8卡可见。启动前release、run、log和outer均为`ABSENT`。GPU7已有SCB v3技术构建PID=`420208`，CWD为`.../releases/phase1_single_control_bundle_v1_build_20260809_v3_3d6b739a`，显存=`488MiB`；其余卡各约`1MiB`。F4G启动后GPU7计算进程总数上限为2，本run不得超过该上限；不得中止或影响SCB v3。

ManySig与6个GeoSat-C baseline远端SHA分别与§3完全匹配。实现commit=`6465a7f33abb730ae58de4f6e0bec5181f128d0a`已用无prefix`git archive`封存；本地archive=`phase1_gd_proto_nll12_20260809_v1_6465a7f3.tar`，大小=`262952960` bytes，SHA256=`08b1d785df403f59e11cb68fb542629ed6724eaf0e1a3872aeb810e854fbffef`，成员数=`4877`，`code/code/*`=`0`。archive内LF成员SHA：`phase1_gd_proto_nll.py=c7af287f05cf8fbc69c615ad3f8ea8c74ab906a564e49728dbeb932cdb1e4c74`、`train_ssdg.py=7587bd7b8b1c4dedbfb5007a27ece91ec2f52d9ac1b525db2be3f59bc062ec90`、`test_phase1_gd_proto_nll.py=fbf5dac628416244fe58857555e313158377fc320a22ff6d2621da01bebccec3`、`launch_phase1_gd_proto_nll12_20260809.sh=50f2472c2f71a9fc15dfdd02729107facfcb68f165fb5cac872386efb65c560`、`phase1_gd_proto_nll_design_20260809.md=ccdd484d43da6559cf0ebc1c309ac51efd8a27381126676c80803f3fd3137a0f`。Windows工作树SHA保留于§2，差异仅归档LF/工作树CRLF表示，不在N607远端改码。

release已在N607落地并完成静态验证：远端临时archive与本地SHA、262952960 bytes一致；release无`code/code`。远端成员LF SHA与上列archive成员完全一致；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m py_compile cvsrffi/phase1_gd_proto_nll.py SSDG/train_ssdg.py tests/test_phase1_gd_proto_nll.py`、`train_ssdg.py --help`（包含GD参数）、`bash -n scripts/launch_phase1_gd_proto_nll12_20260809.sh`均PASS。`DRY_RUN=1 bash scripts/launch_phase1_gd_proto_nll12_20260809.sh --dry-run`精确输出12行，dry-run SHA256=`1cc3285ef8b2d5dfcc9ecbc36ef824fe8c0cdd21009575a5d00d9a7b2d59cb95`，含GPU7臂。

## 7.唯一启动终态与失败证据（2026-08-09）

按§4精确命令于`22:52`唯一detached启动一次。outer log立即写入权限错误：`env: .../code/scripts/launch_phase1_gd_proto_nll12_20260809_v1_6465a7f3/code/scripts/launch_phase1_gd_proto_nll12_20260809.sh: Permission denied`。根因是冻结commit中launcher Git mode=`100644`（archive解包后远端mode=`664`），因此命令未进入launcher；未发生代码、参数、矩阵或远端release修改。launcher PID、12 child PID、run root、log root、`pids.tsv`均不存在；outer SHA256=`cdf3cc59c87439f181c18b300c43fe75a6dab57e0b282404ac9acabb1fd6817a`（178 bytes）已回收至本地artifacts。SCB v3 PID=`420208`仍为GPU7唯一计算进程（488MiB），GPU0–6保持1MiB，未中止或影响SCB。

本run在启动前失败，故不存在12×E40、final checkpoint、metrics/config、GD terminal/training/terminal/resource/heldout、C合同或G的batches/rows/12cells/state/raw-gradient/terminal结果；这些字段标记为`NOT_PRODUCED`，不构成性能结果。未下载任何`.pth`或`.npz`。本地仅保留实现archive、outer failure log及结构化runner证据；远端临时archive已按原SHA校验后删除，release与outer failure log原样保留。retry=`NO`；不得把该run重标为训练结果，也不得从中作性能、晋级或方法结论。

本地artifact与逐项SHA：archive=`phase1_gd_proto_nll12_20260809_v1_6465a7f3.tar`（262952960 bytes，`08b1d785df403f59e11cb68fb542629ed6724eaf0e1a3872aeb810e854fbffef`）；`outer_launcher_failure.out`（178 bytes，`cdf3cc59c87439f181c18b300c43fe75a6dab57e0b282404ac9acabb1fd6817a`）；`completion.tsv`（184 bytes，`b46af6f56bfbea84aba8580620609b85cd374b4fc3b2235a46fc923f4568e488`）；`runner_handoff.json`（854 bytes，`cf287752cb1143353302c3ec4fdea31596fe72812b032d12834500894efa34d1`）；`manifest.json`（2219 bytes，`6052bd277b098ecb962c36f96ee1091320d272f44ccde2f136caf0310ca3d40b`）。manifest记录archive成员SHA、远端临时包删除、SSH/TCP22均为0以及未产生checkpoint/NPZ。Git镜像路径：`analysis/phase1_gd_proto_nll12_20260809_v1_report.md`。
