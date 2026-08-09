# Phase1 GD-ProtoNLL 12臂正式训练v2报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

日期：2026-08-09

## 1.目标、差异与版本

|字段|冻结值|
|---|---|
|run ID|`phase1_gd_proto_nll12_20260809_v2`|
|目标|执行与v1完全相同的P1-GD-ProtoNLL 6折C/G、40E、final-only 12臂矩阵|
|v1终态|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`；唯一命令因launcher文件mode 100644而`Permission denied`，0 child、0 run/log root|
|唯一修复|新命令显式使用`bash launch_phase1_gd_proto_nll12_20260809.sh`；不改代码、参数、数据、矩阵、GPU、停止门或结果门|
|实现commit|`6465a7f33abb730ae58de4f6e0bec5181f128d0a`|
|Git工作树|`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`|
|科学复核|沿用冻结实现的`P0=0、P1=0、ALLOW`；本轮是机械启动修复，不重开科学评审|
|声明|本run只完成训练技术闭环，不读取性能，不作晋级结论；postfreeze另行执行|

实现与本地验证保持不变：core SHA`71DCFFC9D1BED35BB746C67EB25A9A7C79F31C1DDFA0289E0CC803F17FBEF57D`；train`815545AA383C4E666EB9295B4A147B8870DB122DEFCA0F84A1F11B48D5250A46`；test`45EB9972674E9307646EEA054E38D29B4A46689D85FBE81A0B656E73D767EDC5`；launcher`50F2472C2F71A9FC15DFDD02729107FACFCB68F165FB5CAC872386EFB65C5601`；design`F28A2C8E20C8D4E91B010FD4B5852C3A73C0AF5CC2C7334A4A7E7EFF3B705F35`。本地`py_compile`、29 focused、lite_d no-query、`bash -n`、dry-run12及diff-check均PASS。

## 2.冻结数据、方法与矩阵

ManySig=`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`；6个GeoSat-C final checkpoint及SHA、每折TX映射、L/U/V、seed和非补偿门与[v1报告](E:/type10-7/automation_reports/CV-SincNet/phase1_gd_proto_nll12_20260809_v1/report.md)完全相同。C/G只差G的`.10*L_GD`；每批local4、三场景共同序列、12格EMA/coverage、首批raw gradient合同不变。

GPU矩阵固定：0=`F1C＋F5G`；1=`F1G＋F5C`；2=`F2C＋F6G`；3=`F2G＋F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。preflight须记录SCB v3在GPU7的既有进程；加入F4G后任一卡计算进程不得超过2。

## 3.N607路径与唯一命令

|字段|冻结值|
|---|---|
|release|`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v2_6465a7f3`，启动前ABSENT|
|CWD|`<release>/code`|
|run root|`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gd_proto_nll12_20260809_v2`，启动前ABSENT|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v2`，启动前ABSENT|
|outer|`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v2_launcher.out`，启动前ABSENT|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|retry|NO；caller超时后只读核验landed，禁止重复launch|

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v2_6465a7f3/code && nohup env RUN_ID=phase1_gd_proto_nll12_20260809_v2 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v2_6465a7f3/code bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gd_proto_nll12_20260809_v2_6465a7f3/code/scripts/launch_phase1_gd_proto_nll12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gd_proto_nll12_20260809_v2_launcher.out 2>&1 < /dev/null &
```

## 4.执行、健康门与工件

唯一runner重新执行direct preflight；仅direct TCP／SSH不可用时使用一次verified bridge。核新release/run/log/outer均ABSENT、ManySig和6 checkpoint SHA、archive/member SHA、SCB进程及每卡≤2、远端`py_compile`／help／`bash -n`／dry-run12后唯一启动。v1路径与outer只读保留，不覆盖、不续跑。

立即记录launcher及12 child的PID/CWD/cmdline/GPU。停止仅依据路径/hash/覆盖、协议/类序、Traceback/OOM/CUDA/nonfinite/failure receipt、两行同一异常、零checkpoint或成员不全；绝不依据性能。预期每臂E40、final checkpoint、40行JSONL/41行CSV及config/GD terminal/training/terminal/resource/heldout receipt；C为`CONTROL_ARM_NOT_APPLICABLE`，G需1200 batches、153600 rows、12 cells、EMA逐批、raw gradient和terminal pass。最终`NON_PROMOTABLE_P0_DISABLED/exit8`是预期P0 gate。

成功后只回收小日志、JSON/CSV、PID、completion和manifest，不下载checkpoint/NPZ；更新root与Git镜像报告，清SSH/SCP/TCP22。任何技术失败标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`并且不重试。

## 5.Runner预检、落地与静态验证（启动前）

2026-08-09直连`N607`预检通过；v2 release、run、log、outer启动前均为`ABSENT`，v1 release及outer仅只读保留。ManySig SHA=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`，6个GeoSat-C baseline SHA与§2及v1一致。GPU7既有SCB v3 PID=`420208`、显存=`488MiB`，GPU0–6各约`1MiB`；F4G加入后每卡计算进程上限2满足。

复用v1本地无prefix archive（实现commit=`6465a7f33abb730ae58de4f6e0bec5181f128d0a`）：262952960 bytes，SHA256=`08b1d785df403f59e11cb68fb542629ed6724eaf0e1a3872aeb810e854fbffef`，成员无`code/code`。v2远端新release解包后5个LF成员SHA保持：`phase1_gd_proto_nll.py=c7af287f05cf8fbc69c615ad3f8ea8c74ab906a564e49728dbeb932cdb1e4c74`、`train_ssdg.py=7587bd7b8b1c4dedbfb5007a27ece91ec2f52d9ac1b525db2be3f59bc062ec90`、`test_phase1_gd_proto_nll.py=fbf5dac628416244fe58857555e313158377fc320a22ff6d2621da01bebccec3`、`launch_phase1_gd_proto_nll12_20260809.sh=50f2472c2f71a9fc15dfdd02729107facfcb68f165fb5cac872386efb65c560`、`phase1_gd_proto_nll_design_20260809.md=ccdd484d43da6559cf0ebc1c309ac51efd8a27381126676c80803f3fd3137a0f`。远端`py_compile`、`train_ssdg.py --help`、`bash -n`、精确12行dry-run均PASS；dry-run SHA=`a04f2c49e4cce24edac7a3c2d5655a905c2d66d08b9089fa21180a02a477bd24`。v2 launcher mode=`664`，正式命令显式调用`bash`，不修改release代码。
## 6.唯一启动、技术停止与工件回收（无性能结论）

按§3唯一命令启动一次。调用方连接在30秒后超时；仅作只读落地核验，未重复启动。识别到的本地SSH客户端PID=`26712`与本次命令对应，已关闭并确认`ssh.exe=0`、N607及bridge的TCP22连接均为0。落地核验得到wrapper PID=`457010`、launcher PID=`457011`、run PGID=`457009`；12个primary child均已创建：GPU0=`F1C:457014、F5G:457016`，GPU1=`F1G:457018、F5C:457020`，GPU2=`F2C:457022、F6G:457024`，GPU3=`F2G:457026、F6C:457028`，GPU4=`F3C:457030`，GPU5=`F3G:457032`，GPU6=`F4C:457034`，GPU7=`F4G:457036`。启动后每卡计算进程数不超过2；GPU7为SCB v3 PID=`420208`加F4G，SCB未触碰。

首波技术日志在F1G的E009出现确定性失败receipt，fingerprint=`GD_PROTO_NLL_NONFINITE`，异常为`cvsrffi.phase1_gd_proto_nll.GDProtoNLLRuntimeError: P1-GD-ProtoNLL feature/head contains non-finite or zero L2 norm before raw cosine`，failure stage为`satellite_feat_joint_binding_or_gd_proto_nll_loss`。该receipt SHA256=`95560bffd5ee5d1157cc785485af2589f30539e3d0837118a644933385222de7`。这是预注册的nonfinite技术停止条件；不读取loss/accuracy或任何性能字段。

停止前逐进程解析PGID=`457009`的CWD与cmdline，确认全部绑定v2 release/run；随后执行精确`kill -TERM -- -457009`，2秒后run-owned残余进程数为0。最终GPU0–6各约1MiB，GPU7仅SCB PID=`420208`约488MiB；v2进程为0。v2 release、run、log和outer均保留，v1 release/outer只读未改。未产生或回收checkpoint/`.pth`/`.npz`。

只回收小工件：远端bundle 50 members、10782720 bytes、SHA256=`e7e50d8f13ae2b8b78a128967c72e4820e453a50ca30db93d39959edcfa9b276`；包含12个日志、PID/config/failure receipt及部分JSON/JSONL/CSV。逐项SHA清单`remote_artifact_sha256.tsv`为52行、8115 bytes、SHA256=`be5224b2210c19a97a5660bd1c8d8b14c9401c81836bfc7369db38f919cad3cd`；其中outer launcher为0 bytes、SHA256=`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。completion.tsv为344 bytes、SHA256=`611b4a011543b3c78fa888a7b504c2b486fc7c8ac5b127791dbf844762a80717`；runner_handoff.json为1510 bytes、SHA256=`1f61b72bb72c4de38d0ea12019bce3a46d0fc6ef26e80769b6ac0e4634dfc1b5`；manifest.json为1879 bytes、SHA256=`1869cbf6e7934f8cbd351099fb9a7d9975f6d4d26f283602529ecd98f013b354`。远端临时archive、small bundle和file list均已删除并核验不存在。

本run终态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，`launch_count=1`、`retry=NO`。预期的12×E40、final、metrics/config、GD terminal/training/terminal/resource/heldout、G的1200 batches/153600 rows/12 cells/raw-gradient/terminal pass均因技术停止而`NOT_PRODUCED`；已回收的partial logs/JSON/CSV仅作技术证据，不作性能分析、比较或晋级结论。
